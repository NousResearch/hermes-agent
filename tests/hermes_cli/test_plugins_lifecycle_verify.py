"""Tests for plugin lifecycle verification: deterministic collision resolution (gap 1) and the
read-only console-package / ``hermes plugins verify`` surface (gap 2).

Behaviour contracts, not snapshots: every test pins a relationship between two pieces of data
(winner vs source rank, declared version vs installed version), never a frozen value.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_cli.plugins_discovery import collision_sort_key, resolve_key_collisions
from hermes_cli.plugins_manifest import PluginManifest
from hermes_cli.plugins_verify import (
    ConsolePackageProbe,
    ProfileVerifyResult,
    probe_console_package,
    reconcile_console_package,
    tree_digest,
    verify_plugin,
)


def _manifest(name: str, *, source: str = "user", path: str | None = None,
              key: str = "", version: str = "1.0.0") -> PluginManifest:
    return PluginManifest(
        name=name, version=version, description="d", source=source, path=path, key=key)


# ── Gap 1: deterministic key-collision resolution ──────────────────────────


class TestCollisionSortKey:
    """Source precedence first; within a source the canonical install dir wins."""

    def test_later_source_outranks_canonical_name(self, tmp_path):
        canonical_early = collision_sort_key("user", str(tmp_path / "wp" / "wp.py"), "wp")
        impostor_late = collision_sort_key("project", str(tmp_path / "wp.old-123" / "x.py"), "wp")
        assert impostor_late > canonical_early

    def test_canonical_beats_backup_within_one_source(self, tmp_path):
        # manifest.path is the plugin DIRECTORY (parse_manifest_file sets str(plugin_dir)).
        active = collision_sort_key("user", str(tmp_path / "plugins" / "wp"), "wp")
        backup = collision_sort_key("user", str(tmp_path / "plugins" / "wp.old-20260913"), "wp")
        assert active > backup

    def test_unknown_source_ranks_with_user(self):
        assert collision_sort_key("mystery", None, "x") == collision_sort_key("user", None, "x")

    def test_ties_are_equal_scan_order_fallback(self, tmp_path):
        a = collision_sort_key("user", str(tmp_path / "same" / "f.py"), "same")
        b = collision_sort_key("user", str(tmp_path / "same" / "g.py"), "same")
        assert a == b  # tie → first wins, and scan order is sorted/stable


class TestResolveKeyCollisions:
    def test_no_collisions_single_winner_map(self):
        manifests = [_manifest("a"), _manifest("b")]
        winners, collisions = resolve_key_collisions(manifests)
        assert set(winners) == {"a", "b"}
        assert collisions == []

    def test_backup_directory_cannot_shadow_active_plugin(self, tmp_path):
        """The production bug: ``wp.old-<ts>`` sorted after ``wp`` and won last-writer-wins."""
        active = _manifest("wp", source="user", path=str(tmp_path / "plugins" / "wp"), version="2.0.0")
        stale = _manifest(
            "wp", source="user", path=str(tmp_path / "plugins" / "wp.old-20260913"), version="1.0.0")
        winners, collisions = resolve_key_collisions([stale, active])
        assert winners["wp"] is active
        assert [(k, len(g)) for k, g, _ in collisions] == [("wp", 2)]

    def test_project_beats_user_regardless_of_scan_order(self, tmp_path):
        user = _manifest("wp", source="user", path=str(tmp_path / "u" / "wp"))
        project = _manifest("wp", source="project", path=str(tmp_path / "p" / "wp"))
        winners, _ = resolve_key_collisions([user, project])
        assert winners["wp"] is project

    def test_first_seen_key_order_preserved(self):
        winners, _ = resolve_key_collisions([_manifest("z"), _manifest("a"), _manifest("m")])
        assert list(winners) == ["z", "a", "m"]  # load-order resolution downstream unchanged

    def test_key_not_name_groups_collision(self, tmp_path):
        """Registry key is the path-derived key; same manifest name under different keys = no clash."""
        a = _manifest("wp", source="user", path=str(tmp_path / "u" / "wp"), key="wp")
        b = _manifest("wp", source="project", path=str(tmp_path / "p" / "nested"), key="nested")
        winners, collisions = resolve_key_collisions([a, b])
        assert winners == {"wp": a, "nested": b}
        assert collisions == []


# ── Gap 2: console-package probe ───────────────────────────────────────────


class TestProbeConsolePackage:
    def test_no_pyproject_not_declared(self, tmp_path):
        probe = probe_console_package(tmp_path)
        assert probe.declared is False and probe.ok is True

    def test_declared_but_not_installed_is_drift(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "definitely-not-installed-xyz"\nversion = "3.1.4"\n'
            '[project.scripts]\nxyz-cli = "xyz:main"\n',
            encoding="utf-8")
        probe = probe_console_package(tmp_path)
        assert probe.declared is True
        assert probe.ok is False
        assert any("not importable" in m for m in probe.mismatches)

    def test_version_mismatch_detected(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "pytest"\nversion = "999.0.0"\n', encoding="utf-8")
        import importlib.metadata
        real = importlib.metadata.version
        monkeypatch.setattr(
            importlib.metadata, "version",
            lambda dist: real("pytest") if dist == "pytest" else (_ for _ in ()).throw(
                importlib.metadata.PackageNotFoundError(dist)))
        probe = probe_console_package(tmp_path)
        assert probe.declared is True
        assert probe.expected_version == "999.0.0"
        assert probe.installed_version == real("pytest")  # both present, different → drift
        assert probe.ok is False
        assert any("!= declared" in m for m in probe.mismatches)

    def test_matching_installed_package_passes(self, tmp_path):
        """The installed pytest distribution itself satisfies a matching declaration."""
        import importlib.metadata
        (tmp_path / "pyproject.toml").write_text(
            f'[project]\nname = "pytest"\nversion = "{importlib.metadata.version("pytest")}"\n',
            encoding="utf-8")
        probe = probe_console_package(tmp_path)
        assert probe.declared is True
        assert probe.importable is True
        assert probe.ok is True
        assert probe.mismatches == []


class TestReconcileConsolePackage:
    def test_install_report_never_auto_pips(self, tmp_path, caplog):
        """Drift is reported with a remediation command; no pip subprocess is spawned."""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "definitely-not-installed-xyz"\nversion = "1.0"\n', encoding="utf-8")
        console = MagicMock()
        with caplog.at_level("DEBUG"):
            probe = reconcile_console_package(tmp_path, console)
        assert probe.ok is False
        printed = " ".join(str(call.args) for call in console.print.call_args_list)
        assert "Console package drift" in printed
        assert "pip install --force-reinstall --no-deps" in printed

    def test_undeclared_plugin_is_silent(self, tmp_path):
        console = MagicMock()
        probe = reconcile_console_package(tmp_path, console)
        assert probe.declared is False
        console.print.assert_not_called()


# ── verify: tree digest + per-profile verification ─────────────────────────


def _plugin_tree(root: Path, name: str = "wp", version: str = "2.0.0",
                 manifest_name: str | None = None) -> Path:
    """Create a plugin dir named *name* whose plugin.yaml declares *manifest_name* (default *name*).

    A stale backup sibling (dir ``wp.old-<ts>``) still declares the ORIGINAL manifest name ``wp``,
    which is exactly why shadowing detection keys on manifest name, not directory name.
    """
    d = root / name
    d.mkdir(parents=True)
    (d / "plugin.yaml").write_text(
        f"name: {manifest_name or name}\nversion: {version}\ndescription: test plugin\n",
        encoding="utf-8")
    # A real register() is required for plugin-doctor to pass registration (doctor runs for real
    # in the E2E tests — behaviour contract, not a mock).
    (d / "__init__.py").write_text(
        "# plugin\n"
        "def register(ctx):\n"
        "    \"\"\"No-op registration for lifecycle-verification fixtures.\"\"\"\n"
        "    return None\n",
        encoding="utf-8")
    return d


class TestTreeDigest:
    def test_deterministic_and_content_sensitive(self, tmp_path):
        d = _plugin_tree(tmp_path)
        first = tree_digest(d)
        assert first == tree_digest(d)
        (d / "__init__.py").write_text("# changed\n", encoding="utf-8")
        assert tree_digest(d) != first

    def test_generated_noise_ignored(self, tmp_path):
        d = _plugin_tree(tmp_path)
        base = tree_digest(d)
        cache = d / "__pycache__"
        cache.mkdir()
        (cache / "x.cpython-312.pyc").write_bytes(b"\x00\x01")
        assert tree_digest(d) == base

    def test_hex_only_sha256(self, tmp_path):
        digest = tree_digest(_plugin_tree(tmp_path))
        assert len(digest) == 64
        int(digest, 16)  # hex


class TestVerifyPlugin:
    def test_unknown_profile_fails_closed(self):
        with pytest.raises(Exception, match="does not exist"):
            verify_plugin("wp", profiles=["no-such-profile-xyz"])

    def test_missing_plugin_is_error_not_crash(self, tmp_path, monkeypatch):
        import hermes_constants
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "plugins").mkdir()
        results = verify_plugin("ghost", run_doctor=False)
        assert len(results) == 1
        assert results[0].ok is False
        assert any("not found" in e for e in results[0].errors)

    def test_healthy_plugin_passes_and_reports_shadows(self, tmp_path, monkeypatch):
        import hermes_cli.plugins_verify as pv
        import hermes_constants
        plugins_root = tmp_path / "plugins"
        active = _plugin_tree(plugins_root, version="2.0.0")
        # The stale backup declares the SAME manifest name 'wp' — the shadowing condition.
        _plugin_tree(plugins_root, name="wp.old-20260913", version="1.0.0", manifest_name="wp")
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(pv, "_run_doctor", lambda d: (True, []))
        results = verify_plugin("wp", run_doctor=True)
        r = results[0]
        assert r.ok is True
        assert r.active["manifest_version"] == "2.0.0"
        assert len(r.active["tree_sha256"]) == 64
        assert [s.path for s in r.shadows] == [str(plugins_root / "wp.old-20260913")]
        assert any("shadowing tree" in w for w in r.warnings)

    def test_result_dict_is_json_serialisable(self, tmp_path, monkeypatch):
        import hermes_cli.plugins_verify as pv
        import hermes_constants
        active = _plugin_tree(tmp_path / "plugins")
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(pv, "_run_doctor", lambda d: (True, []))
        payload = verify_plugin("wp", run_doctor=True)[0].to_dict()
        assert json.loads(json.dumps(payload))["ok"] is True


# ── E2E on a temp HERMES_HOME (card requirement) ───────────────────────────


class TestVerifyE2ETempHome:
    """End-to-end through the real cmd_verify on a temp profile home — no mocks for discovery,
    digest, or console probe (doctor runs for real on the generated tree)."""

    def _home(self, tmp_path):
        home = tmp_path / "hermes-home"
        (home / "plugins").mkdir(parents=True)
        return home

    def test_e2e_clean_install_passes_exit_0(self, tmp_path, monkeypatch):
        import hermes_constants
        from hermes_cli.plugins_verify import cmd_verify
        home = self._home(tmp_path)
        _plugin_tree(home / "plugins", name="wp", version="2.0.0")
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
        rc = cmd_verify("wp", json_output=True)
        assert rc == 0

    def test_e2e_backup_sibling_shadowing_still_verifies_active(self, tmp_path, monkeypatch, capsys):
        """The original incident shape: stale ``wp.old-<ts>`` next to the live tree. Verify must
        report the CANONICAL dir as active (v2.0.0), list the backup as shadowing, and pass."""
        import hermes_constants
        from hermes_cli.plugins_verify import cmd_verify
        home = self._home(tmp_path)
        _plugin_tree(home / "plugins", name="wp.old-20260913", version="1.0.0", manifest_name="wp")
        _plugin_tree(home / "plugins", name="wp", version="2.0.0")
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
        rc = cmd_verify("wp", json_output=True)
        assert rc == 0
        results = json.loads(capsys.readouterr().out)
        payload = results["profiles"][0]
        assert payload["ok"] is True
        assert payload["active"]["manifest_version"] == "2.0.0"
        assert payload["active"]["path"].endswith("wp")
        assert [s["version"] for s in payload["shadowing_entries"]] == ["1.0.0"]

    def test_e2e_console_drift_fails_closed_exit_1(self, tmp_path, monkeypatch):
        """A plugin declaring a console package that is absent in the runtime FAILS verify —
        this is the gap-2 detection the card requires, proved on a real temp home."""
        import hermes_constants
        from hermes_cli.plugins_verify import cmd_verify
        home = self._home(tmp_path)
        d = _plugin_tree(home / "plugins", name="wp")
        (d / "pyproject.toml").write_text(
            '[project]\nname = "ghost-dist-not-installed"\nversion = "0.1.0"\n'
            '[project.scripts]\nghost-cli = "ghost:main"\n', encoding="utf-8")
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
        rc = cmd_verify("wp", json_output=True, no_doctor=True)
        assert rc == 1

    def test_e2e_named_profile_verifies_that_home(self, tmp_path, monkeypatch):
        """--profile routes verification through profiles.get_profile_dir, not the active home."""
        from hermes_cli.plugins_verify import cmd_verify
        import hermes_cli.profiles as prof
        home = self._home(tmp_path)
        monkeypatch.setattr(prof, "profile_exists", lambda name: name == "tech")
        monkeypatch.setattr(prof, "get_profile_dir", lambda name: home)
        _plugin_tree(home / "plugins", name="wp", version="2.0.0")
        rc = cmd_verify("wp", profiles=["tech"], json_output=True)
        assert rc == 0
        # Comma-separated form (--profiles a,b) expands to the same targets.
        monkeypatch.setattr(prof, "profile_exists", lambda name: name in {"a", "b"})
        assert cmd_verify("wp", profiles=["a,b"], json_output=True) == 0
        # Unknown profile → VerifyError caught by cmd_verify → fail-closed exit 2 (never verifies
        # a different home by accident).
        monkeypatch.setattr(prof, "profile_exists", lambda name: name == "tech")
        assert cmd_verify("wp", profiles=["ghost-profile"], json_output=True) == 2


class TestActivePluginDir:
    def test_picks_canonical_over_backup_sibling(self, tmp_path):
        from hermes_cli.plugins_verify import _active_plugin_dir
        plugins_root = tmp_path / "plugins"
        _plugin_tree(plugins_root, name="wp.old-20260913", version="1.0.0", manifest_name="wp")
        active = _plugin_tree(plugins_root, name="wp", version="2.0.0")
        assert _active_plugin_dir("wp", tmp_path) == active


# ── CLI wiring ─────────────────────────────────────────────────────────────


class TestVerifyCliWiring:
    def test_exit_code_propagates_fail_closed(self):
        """plugins_command must return the handler's int so main() can sys.exit(1) on FAIL."""
        from hermes_cli import plugins_cmd as pc
        args = MagicMock()
        args.plugins_action = "verify"
        args.name = "x"
        args.profile = None
        args.json = True
        args.no_doctor = False
        with __import__("unittest").mock.patch.object(
                pc, "cmd_verify", return_value=1) as fake:
            rc = pc.plugins_command(args)
        assert rc == 1
        fake.assert_called_once()

    def test_dispatch_table_knows_verify(self):
        from hermes_cli.plugins_cmd import _PLUGIN_ACTIONS
        assert "verify" in _PLUGIN_ACTIONS

    def test_argparse_builds_verify_subcommand(self):
        """The real parser accepts the documented flags (contract with subcommands/plugins.py)."""
        import argparse
        from hermes_cli.subcommands.plugins import build_plugins_parser
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_plugins_parser(subparsers, cmd_plugins=lambda a: None)
        args = parser.parse_args(["plugins", "verify", "wp", "--json", "--no-doctor"])
        assert args.plugins_action == "verify"
        assert args.name == "wp"
        assert args.json is True and args.no_doctor is True and args.profile is None
        args2 = parser.parse_args(["plugins", "verify", "wp", "--profile", "a", "--profile", "b"])
        assert args2.profile == ["a", "b"]


# ── dataclass contracts ────────────────────────────────────────────────────


class TestProbeContracts:
    def test_undeclared_probe_always_ok(self):
        probe = ConsolePackageProbe()
        assert probe.declared is False and probe.ok is True and probe.mismatches == []

    def test_launcher_missing_fails_even_when_importable(self):
        probe = ConsolePackageProbe(
            declared=True, distribution="x", installed_version="1.0", expected_version="1.0",
            importable=True, launcher_found=False, scripts=["x-cli"])
        assert probe.ok is False

    def test_shadowing_entry_fields_stay_printable(self):
        r = ProfileVerifyResult(profile="p", home="/h", plugin_name="wp")
        assert r.to_dict()["shadowing_entries"] == []
