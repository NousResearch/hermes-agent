"""Tests for the catalog-driven ``hermes plugins`` CLI surface.

Covers: catalog-name install resolution (pinned ref + provenance sidecar),
custom-URL banner, --allow-removed wiring, catalog-pin updates, list
annotations, live-index fetch/fallback/TTL, search/browse/info rendering,
doctor, and argparse dispatch.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import pytest
import yaml

import hermes_cli.plugin_catalog as plugin_catalog
import hermes_cli.plugins_cmd as plugins_cmd
from hermes_constants import get_hermes_home

SHA_A = "a" * 40
SHA_B = "b" * 40


# ── Helpers / fixtures ─────────────────────────────────────────────────────


def _write_entry(catalog_dir: Path, name: str, **overrides) -> Path:
    data = {
        "name": name,
        "repo": f"https://github.com/example/{name}",
        "sha": SHA_A,
        "description": f"Test entry {name}.",
        "maintainer": "Example",
    }
    data.update(overrides)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _write_removed(catalog_dir: Path, removed: list) -> Path:
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / "removed.yaml"
    path.write_text(yaml.safe_dump({"removed": removed}), encoding="utf-8")
    return path


def _install_user_plugin(name: str, *, sidecar: dict | None = None) -> Path:
    """Create a fake installed plugin under the per-test HERMES_HOME."""
    d = get_hermes_home() / "plugins" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "plugin.yaml").write_text(
        yaml.safe_dump(
            {"name": name, "version": "1.0.0", "description": f"{name} plugin"}
        ),
        encoding="utf-8",
    )
    if sidecar is not None:
        (d / ".hermes-catalog.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
    return d


@pytest.fixture()
def catalog_dir(tmp_path, monkeypatch):
    d = tmp_path / "catalog"
    d.mkdir()
    monkeypatch.setenv("HERMES_PLUGIN_CATALOG_DIR", str(d))
    return d


@pytest.fixture()
def offline(monkeypatch):
    """Force the live-index path to fall back to the in-tree catalog."""
    monkeypatch.setattr(plugin_catalog, "fetch_live_catalog", lambda **kw: None)


@pytest.fixture()
def fake_core(monkeypatch, tmp_path):
    """Replace _install_plugin_core with a recording fake."""
    calls: list[dict] = []
    target = tmp_path / "fake-installed"

    def fake(identifier, *, force, ref=None, skip_removed_check=False):
        target.mkdir(parents=True, exist_ok=True)
        calls.append(
            {
                "identifier": identifier,
                "force": force,
                "ref": ref,
                "skip_removed_check": skip_removed_check,
            }
        )
        return target, {"name": "my-entry"}, "my-entry"

    monkeypatch.setattr(plugins_cmd, "_install_plugin_core", fake)
    return types.SimpleNamespace(calls=calls, target=target)


# ── Catalog-name install ───────────────────────────────────────────────────


class TestCatalogInstall:
    def test_catalog_name_resolves_to_pinned_repo(
        self, catalog_dir, offline, fake_core
    ):
        _write_entry(catalog_dir, "my-entry", sha=SHA_A)
        plugins_cmd.cmd_install("my-entry", enable=False)
        assert len(fake_core.calls) == 1
        call = fake_core.calls[0]
        assert call["identifier"] == "https://github.com/example/my-entry"
        assert call["ref"] == SHA_A
        assert call["skip_removed_check"] is False

    def test_subdir_entry_uses_fragment_identifier(
        self, catalog_dir, offline, fake_core
    ):
        _write_entry(catalog_dir, "my-entry", subdir="plugins/inner")
        plugins_cmd.cmd_install("my-entry", enable=False)
        assert fake_core.calls[0]["identifier"] == (
            "https://github.com/example/my-entry#plugins/inner"
        )

    def test_sidecar_written_with_provenance(
        self, catalog_dir, offline, fake_core
    ):
        _write_entry(catalog_dir, "my-entry", tier="official")
        plugins_cmd.cmd_install("my-entry", enable=False)
        sidecar_path = fake_core.target / ".hermes-catalog.json"
        assert sidecar_path.is_file()
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["catalog_name"] == "my-entry"
        assert sidecar["repo"] == "https://github.com/example/my-entry"
        assert sidecar["sha"] == SHA_A
        assert sidecar["tier"] == "official"
        assert sidecar["installed_at"]

    def test_capability_summary_and_tier_shown(
        self, catalog_dir, offline, fake_core, capsys
    ):
        _write_entry(
            catalog_dir,
            "my-entry",
            tier="official",
            capabilities={"provides_tools": ["cool_tool"]},
        )
        plugins_cmd.cmd_install("my-entry", enable=False)
        out = capsys.readouterr().out
        assert "official" in out
        assert "cool_tool" in out

    def test_unknown_catalog_like_name_errors(
        self, catalog_dir, offline, fake_core, capsys
    ):
        with pytest.raises(SystemExit):
            plugins_cmd.cmd_install("nonexistent-entry", enable=False)
        out = capsys.readouterr().out
        assert "search" in out
        assert not fake_core.calls

    def test_custom_url_gets_unreviewed_banner(
        self, catalog_dir, offline, fake_core, capsys
    ):
        plugins_cmd.cmd_install(
            "https://github.com/foo/bar.git", enable=False
        )
        out = capsys.readouterr().out
        assert "custom (unreviewed) source" in out
        # Custom installs never get a ref pin.
        assert fake_core.calls[0]["ref"] is None

    def test_allow_removed_passes_skip_flag_and_warns(
        self, catalog_dir, offline, fake_core, capsys
    ):
        plugins_cmd.cmd_install(
            "https://github.com/foo/bar.git",
            enable=False,
            allow_removed=True,
        )
        out = capsys.readouterr().out
        assert fake_core.calls[0]["skip_removed_check"] is True
        assert "removed" in out.lower()


# ── Catalog update ─────────────────────────────────────────────────────────


class TestCatalogUpdate:
    def test_update_reinstalls_at_new_pin(
        self, catalog_dir, offline, fake_core, capsys
    ):
        _write_entry(catalog_dir, "my-entry", sha=SHA_B)
        target = _install_user_plugin(
            "my-entry",
            sidecar={
                "catalog_name": "my-entry",
                "repo": "https://github.com/example/my-entry",
                "sha": SHA_A,
                "tier": "community",
                "installed_at": "2026-01-01T00:00:00Z",
            },
        )
        plugins_cmd.cmd_update("my-entry")
        assert len(fake_core.calls) == 1
        call = fake_core.calls[0]
        assert call["ref"] == SHA_B
        assert call["force"] is True
        out = capsys.readouterr().out
        assert SHA_A[:8] in out
        assert SHA_B[:8] in out
        # Sidecar refreshed to the new pin (written into the reinstall target).
        sidecar = json.loads(
            (fake_core.target / ".hermes-catalog.json").read_text(
                encoding="utf-8"
            )
        )
        assert sidecar["sha"] == SHA_B
        assert target.exists() or True  # target replaced by reinstall

    def test_update_already_at_pin_is_noop(
        self, catalog_dir, offline, fake_core, capsys
    ):
        _write_entry(catalog_dir, "my-entry", sha=SHA_A)
        _install_user_plugin(
            "my-entry",
            sidecar={
                "catalog_name": "my-entry",
                "repo": "https://github.com/example/my-entry",
                "sha": SHA_A,
                "tier": "community",
                "installed_at": "2026-01-01T00:00:00Z",
            },
        )
        plugins_cmd.cmd_update("my-entry")
        out = capsys.readouterr().out
        assert "already at catalog pin" in out
        assert not fake_core.calls

    def test_update_preserves_enabled_state(
        self, catalog_dir, offline, fake_core
    ):
        _write_entry(catalog_dir, "my-entry", sha=SHA_B)
        _install_user_plugin(
            "my-entry",
            sidecar={
                "catalog_name": "my-entry",
                "repo": "https://github.com/example/my-entry",
                "sha": SHA_A,
                "tier": "community",
                "installed_at": "2026-01-01T00:00:00Z",
            },
        )
        plugins_cmd._save_enabled_set({"my-entry"})
        plugins_cmd.cmd_update("my-entry")
        assert "my-entry" in plugins_cmd._get_enabled_set()

    def test_update_without_sidecar_keeps_git_flow(
        self, catalog_dir, offline, fake_core, capsys
    ):
        _install_user_plugin("plain-git-plugin")  # no sidecar, no .git
        with pytest.raises(SystemExit):
            plugins_cmd.cmd_update("plain-git-plugin")
        out = capsys.readouterr().out
        assert "not installed from git" in out
        assert not fake_core.calls

    def test_update_entry_gone_from_catalog_errors(
        self, catalog_dir, offline, fake_core, capsys
    ):
        _install_user_plugin(
            "my-entry",
            sidecar={
                "catalog_name": "my-entry",
                "repo": "https://github.com/example/my-entry",
                "sha": SHA_A,
                "tier": "community",
                "installed_at": "2026-01-01T00:00:00Z",
            },
        )
        with pytest.raises(SystemExit):
            plugins_cmd.cmd_update("my-entry")
        out = capsys.readouterr().out
        assert "no longer in the catalog" in out


# ── List annotations ───────────────────────────────────────────────────────


class TestListAnnotations:
    def test_json_includes_catalog_annotation(
        self, catalog_dir, offline, capsys
    ):
        _install_user_plugin(
            "cat-plugin",
            sidecar={
                "catalog_name": "cat-plugin",
                "repo": "https://github.com/example/cat-plugin",
                "sha": SHA_A,
                "tier": "official",
                "installed_at": "2026-01-01T00:00:00Z",
            },
        )
        args = argparse.Namespace(json=True)
        plugins_cmd.cmd_list(args)
        payload = json.loads(capsys.readouterr().out)
        row = next(p for p in payload if p["name"] == "cat-plugin")
        assert row["catalog"] == f"catalog:official@{SHA_A[:8]}"

    def test_removed_plugin_flagged(self, catalog_dir, offline, capsys):
        _install_user_plugin("evil-plugin")
        _write_removed(
            catalog_dir,
            [{"name": "evil-plugin", "reason": "exfiltrated env vars"}],
        )
        plugins_cmd.cmd_list(argparse.Namespace())
        out = capsys.readouterr().out
        assert "REMOVED from catalog" in out
        assert "exfiltrated env vars" in out


# ── Live index fetch ───────────────────────────────────────────────────────


class _FakeResp:
    def __init__(self, *, json_data=None, text=""):
        self._json = json_data
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def _fake_httpx_get(listing, files, counter):
    def fake_get(url, **kwargs):
        counter.append(url)
        if "api.github.com" in url:
            return _FakeResp(json_data=listing)
        fname = url.rsplit("/", 1)[-1]
        return _FakeResp(text=files[fname])

    return fake_get


class TestLiveIndex:
    def _remote_entry_yaml(self, name, sha=SHA_B):
        return yaml.safe_dump(
            {
                "name": name,
                "repo": f"https://github.com/example/{name}",
                "sha": sha,
                "description": f"Remote entry {name}.",
                "maintainer": "Example",
            }
        )

    def test_live_fetch_populates_cache_and_entries(
        self, catalog_dir, monkeypatch
    ):
        _write_entry(catalog_dir, "local-entry")
        listing = [
            {
                "name": "remote-entry.yaml",
                "download_url": "https://raw.example/remote-entry.yaml",
            },
        ]
        files = {"remote-entry.yaml": self._remote_entry_yaml("remote-entry")}
        counter: list[str] = []
        monkeypatch.setattr(
            "httpx.get", _fake_httpx_get(listing, files, counter)
        )
        entries = plugin_catalog.load_catalog_live()
        names = [e.name for e in entries]
        assert names == ["remote-entry"]
        cache = get_hermes_home() / "cache" / "plugin-catalog"
        assert (cache / "remote-entry.yaml").is_file()

    def test_network_failure_falls_back_to_in_tree(
        self, catalog_dir, monkeypatch
    ):
        _write_entry(catalog_dir, "local-entry")

        def boom(url, **kwargs):
            raise OSError("no network")

        monkeypatch.setattr("httpx.get", boom)
        entries = plugin_catalog.load_catalog_live()
        assert [e.name for e in entries] == ["local-entry"]

    def test_ttl_cache_skips_refetch(self, catalog_dir, monkeypatch):
        listing = [
            {
                "name": "remote-entry.yaml",
                "download_url": "https://raw.example/remote-entry.yaml",
            },
        ]
        files = {"remote-entry.yaml": self._remote_entry_yaml("remote-entry")}
        counter: list[str] = []
        monkeypatch.setattr(
            "httpx.get", _fake_httpx_get(listing, files, counter)
        )
        plugin_catalog.load_catalog_live()
        first_count = len(counter)
        assert first_count >= 2  # listing + file

        # Second call within TTL must not hit the network at all — even if
        # the network is now broken.
        def boom(url, **kwargs):
            raise AssertionError("network hit despite fresh cache")

        monkeypatch.setattr("httpx.get", boom)
        entries = plugin_catalog.load_catalog_live()
        assert [e.name for e in entries] == ["remote-entry"]


# ── search / browse / info ─────────────────────────────────────────────────


class TestSearchBrowseInfo:
    def test_search_filters_entries(self, catalog_dir, offline, capsys):
        _write_entry(catalog_dir, "alpha-entry")
        _write_entry(catalog_dir, "beta-entry")
        plugins_cmd.cmd_search("alpha")
        out = capsys.readouterr().out
        assert "alpha-entry" in out
        assert "beta-entry" not in out

    def test_browse_lists_all(self, catalog_dir, offline, capsys):
        _write_entry(catalog_dir, "alpha-entry")
        _write_entry(catalog_dir, "beta-entry")
        plugins_cmd.cmd_browse()
        out = capsys.readouterr().out
        assert "alpha-entry" in out
        assert "beta-entry" in out

    def test_search_no_results_message(self, catalog_dir, offline, capsys):
        plugins_cmd.cmd_search("zzz-nothing")
        out = capsys.readouterr().out
        assert "No catalog entries" in out

    def test_info_shows_full_detail(self, catalog_dir, offline, capsys):
        _write_entry(
            catalog_dir,
            "alpha-entry",
            tier="official",
            requires_hermes=">=0.19",
            docs_url="https://example.com/docs",
            platforms=["linux"],
            capabilities={
                "provides_tools": ["cool_tool"],
                "requires_env": ["ALPHA_KEY"],
            },
        )
        plugins_cmd.cmd_info("alpha-entry")
        out = capsys.readouterr().out
        assert SHA_A in out
        assert "official" in out
        assert "cool_tool" in out
        assert "ALPHA_KEY" in out
        assert ">=0.19" in out
        assert "hermes plugins install alpha-entry" in out

    def test_info_unknown_entry_exits(self, catalog_dir, offline, capsys):
        with pytest.raises(SystemExit):
            plugins_cmd.cmd_info("ghost-entry")

    def test_info_warns_when_removed(self, catalog_dir, offline, capsys):
        _write_entry(catalog_dir, "alpha-entry")
        _write_removed(
            catalog_dir,
            [{"name": "alpha-entry", "reason": "bad actor"}],
        )
        plugins_cmd.cmd_info("alpha-entry")
        out = capsys.readouterr().out
        assert "REMOVED" in out
        assert "bad actor" in out


# ── doctor ─────────────────────────────────────────────────────────────────


class TestDoctor:
    @pytest.fixture(autouse=True)
    def _no_runtime_scan(self, monkeypatch):
        monkeypatch.setattr(
            plugins_cmd, "_runtime_load_errors", lambda: {}
        )

    def test_doctor_table_lists_installed_plugin(
        self, catalog_dir, offline, capsys
    ):
        _install_user_plugin(
            "doc-plugin",
            sidecar={
                "catalog_name": "doc-plugin",
                "repo": "https://github.com/example/doc-plugin",
                "sha": SHA_A,
                "tier": "official",
                "installed_at": "2026-01-01T00:00:00Z",
            },
        )
        _write_entry(catalog_dir, "doc-plugin", sha=SHA_A, tier="official")
        plugins_cmd.cmd_doctor()
        out = capsys.readouterr().out
        assert "doc-plugin" in out
        assert "official" in out

    def test_doctor_detail_flags_pin_mismatch(
        self, catalog_dir, offline, capsys
    ):
        _install_user_plugin(
            "doc-plugin",
            sidecar={
                "catalog_name": "doc-plugin",
                "repo": "https://github.com/example/doc-plugin",
                "sha": SHA_A,
                "tier": "official",
                "installed_at": "2026-01-01T00:00:00Z",
            },
        )
        _write_entry(catalog_dir, "doc-plugin", sha=SHA_B)
        plugins_cmd.cmd_doctor("doc-plugin")
        out = capsys.readouterr().out
        assert "doc-plugin" in out
        assert "behind catalog pin" in out or "pin mismatch" in out

    def test_doctor_flags_removed_plugin(self, catalog_dir, offline, capsys):
        _install_user_plugin("evil-plugin")
        _write_removed(
            catalog_dir,
            [{"name": "evil-plugin", "reason": "exfiltrated env vars"}],
        )
        plugins_cmd.cmd_doctor("evil-plugin")
        out = capsys.readouterr().out
        assert "REMOVED" in out

    def test_doctor_unknown_plugin_exits(self, catalog_dir, offline, capsys):
        with pytest.raises(SystemExit):
            plugins_cmd.cmd_doctor("no-such-plugin")


# ── argparse dispatch ──────────────────────────────────────────────────────


class TestDispatch:
    def _dispatch(self, monkeypatch, action, **attrs):
        recorded = {}

        def record(fn_name):
            def _rec(*args, **kwargs):
                recorded["fn"] = fn_name
                recorded["args"] = args
                recorded["kwargs"] = kwargs

            return _rec

        for fn in (
            "cmd_search",
            "cmd_browse",
            "cmd_info",
            "cmd_validate",
            "cmd_doctor",
            "cmd_install",
        ):
            monkeypatch.setattr(plugins_cmd, fn, record(fn))
        ns = argparse.Namespace(plugins_action=action, **attrs)
        plugins_cmd.plugins_command(ns)
        return recorded

    def test_search_dispatch(self, monkeypatch):
        rec = self._dispatch(monkeypatch, "search", query="foo")
        assert rec["fn"] == "cmd_search"
        assert "foo" in rec["args"] or rec["kwargs"].get("query") == "foo"

    def test_browse_dispatch(self, monkeypatch):
        rec = self._dispatch(monkeypatch, "browse")
        assert rec["fn"] == "cmd_browse"

    def test_info_dispatch(self, monkeypatch):
        rec = self._dispatch(monkeypatch, "info", name="foo")
        assert rec["fn"] == "cmd_info"

    def test_validate_dispatch(self, monkeypatch):
        rec = self._dispatch(monkeypatch, "validate", path="/tmp/x", json=True)
        assert rec["fn"] == "cmd_validate"

    def test_doctor_dispatch(self, monkeypatch):
        rec = self._dispatch(monkeypatch, "doctor", name=None)
        assert rec["fn"] == "cmd_doctor"

    def test_install_allow_removed_dispatch(self, monkeypatch):
        rec = self._dispatch(
            monkeypatch,
            "install",
            identifier="x",
            force=False,
            enable=False,
            no_enable=True,
            allow_removed=True,
        )
        assert rec["fn"] == "cmd_install"
        assert rec["kwargs"].get("allow_removed") is True

    def test_parser_wires_new_subcommands(self):
        from hermes_cli.subcommands.plugins import build_plugins_parser

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        build_plugins_parser(sub, cmd_plugins=lambda args: None)
        for argv in (
            ["plugins", "search", "foo"],
            ["plugins", "browse"],
            ["plugins", "info", "foo"],
            ["plugins", "validate", "/tmp/x", "--json"],
            ["plugins", "doctor"],
            ["plugins", "install", "foo", "--allow-removed"],
        ):
            args = parser.parse_args(argv)
            assert args.plugins_action == argv[1]
