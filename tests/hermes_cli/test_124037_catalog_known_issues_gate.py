"""#124037: plugin catalog entries that document known traps must gate installs.

The catalog (``plugin-catalog/hindsight.yaml``) already documents in prose
that ``local_embedded`` mode is unsupported on PM-managed Hermes with the
current pin (it calls the retired lazy-install path for ``hindsight-all`` and
loops update takeovers), yet ``hermes plugins install`` and the dashboard
install path accept the entry with no gate.

This adds a machine-readable ``known_issues`` list to catalog entries and a
fail-closed gate: non-interactive installs refuse; interactive installs print
the issues and require explicit confirmation.
"""

import pytest
from pathlib import Path

# Local-env workaround (#124037 tests): the checkout's venv python3 is a
# symlink into <home>/hermes-agent/.hermes-runtime/..., so stdlib zoneinfo's
# first import (sysconfig._safe_realpath(sys.executable)) trips the
# real-home IO guard after it installs. Importing here — during collection,
# before any autouse guard fixture — caches the module. CI pythons are not
# symlinked into the home, so this is a no-op there.
import zoneinfo  # noqa: F401  (cached for the moa-loop fixture)

from hermes_cli.plugin_catalog import PluginCatalogEntry, entry_from_mapping, load_catalog


def _entry(**overrides):
    base = dict(
        name="hindsight",
        repo="https://github.com/vectorize-io/hindsight",
        sha="176f8c2de1369f569c489b831d143b78128b5535",
        tier="community",
        category="memory",
    )
    base.update(overrides)
    return entry_from_mapping(base, "test-entry")


# ── catalog parsing ──────────────────────────────────────────────────────────


class TestCatalogParsing:
    def test_entry_from_mapping_parses_known_issues(self):
        entry = _entry(known_issues=["First issue.", "Second issue."])
        assert entry is not None
        assert entry.known_issues == ["First issue.", "Second issue."]

    def test_entry_from_mapping_missing_known_issues_defaults_empty(self):
        entry = _entry()
        assert entry is not None
        assert entry.known_issues == []

    def test_entry_to_dict_round_trips_known_issues(self):
        entry = _entry(known_issues=["Trap."])
        assert entry.to_dict()["known_issues"] == ["Trap."]

    def test_live_catalog_hindsight_declares_known_issues(self):
        entries = load_catalog()
        hindsight = next((e for e in entries if e.name == "hindsight"), None)
        assert hindsight is not None, "hindsight entry must exist in plugin-catalog/"
        assert hindsight.known_issues, (
            "hindsight.yaml must declare known_issues for its documented local-embedded trap"
        )


# ── cmd_install gate ─────────────────────────────────────────────────────────


class TestCmdInstallGate:
    def test_non_tty_refuses_when_entry_has_known_issues(self, monkeypatch):
        from hermes_cli.plugins_cmd import cmd_install

        entry = _entry(known_issues=["Local embedded mode is not supported on PM-managed Hermes."])
        monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.looks_like_catalog_name", lambda _id: True)
        monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.resolve_catalog_name", lambda _id, _c: entry)
        monkeypatch.setattr("hermes_cli.plugins_cmd._is_tty", lambda: False)
        installed = []
        monkeypatch.setattr(
            "hermes_cli.plugins_cmd_catalog.install_catalog_entry",
            lambda *a, **k: installed.append(1),
        )

        with pytest.raises(SystemExit) as exc_info:
            cmd_install("hindsight")
        assert exc_info.value.code == 1
        assert installed == [], "install must not proceed without explicit confirmation"

    def test_tty_yes_proceeds(self, monkeypatch):
        from hermes_cli.plugins_cmd import cmd_install

        entry = _entry(known_issues=["Trap."])
        monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.looks_like_catalog_name", lambda _id: True)
        monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.resolve_catalog_name", lambda _id, _c: entry)
        monkeypatch.setattr("hermes_cli.plugins_cmd._is_tty", lambda: True)
        monkeypatch.setattr("hermes_cli.plugins_cmd._ask_yes", lambda _p, **k: True)
        monkeypatch.setattr(
            "hermes_cli.plugins_cmd_catalog.install_catalog_entry",
            lambda *a, **k: (Path("/tmp/hindsight-target"), {"requires_env": []}, "hindsight"),
        )
        monkeypatch.setattr("hermes_cli.plugins_cmd._display_after_install", lambda *a, **k: None)
        monkeypatch.setattr(
            "hermes_cli.plugins_cmd_catalog.raise_if_removed", lambda *a: None)
        monkeypatch.setattr("hermes_cli.plugins_cmd._resolve_git_url",
                            lambda _id: ("https://github.com/vectorize-io/hindsight.git", ""))
        monkeypatch.setattr("hermes_cli.plugins_cmd._looks_like_plugin_dir", lambda _t: True)
        monkeypatch.setattr("hermes_cli.plugins_cmd_install._prompt_plugin_env_vars", lambda _m, _c: None)
        monkeypatch.setattr("hermes_cli.plugins_cmd._python_dependency_summary", lambda *a, **k: None)
        monkeypatch.setattr("pm.workspace.enabled_plugin_dirs", lambda: [])

        cmd_install("hindsight", force=True, enable=False, no_deps=True)  # confirmed → should not raise

    def test_tty_no_cancels(self, monkeypatch):
        from hermes_cli.plugins_cmd import cmd_install

        entry = _entry(known_issues=["Trap."])
        monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.looks_like_catalog_name", lambda _id: True)
        monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.resolve_catalog_name", lambda _id, _c: entry)
        monkeypatch.setattr("hermes_cli.plugins_cmd._is_tty", lambda: True)
        monkeypatch.setattr("hermes_cli.plugins_cmd._ask_yes", lambda _p, **k: False)
        installed = []
        monkeypatch.setattr(
            "hermes_cli.plugins_cmd_catalog.install_catalog_entry", lambda *a, **k: installed.append(1))

        with pytest.raises(SystemExit) as exc_info:
            cmd_install("hindsight")
        assert exc_info.value.code == 1
        assert installed == []

    def test_custom_source_install_is_not_gated(self, monkeypatch):
        from hermes_cli.plugins_cmd import cmd_install

        monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.looks_like_catalog_name", lambda _id: False)
        monkeypatch.setattr("hermes_cli.plugins_cmd._resolve_git_url",
                            lambda _i: ("https://github.com/x/y.git", None))
        monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.raise_if_removed", lambda *a: None)
        monkeypatch.setattr("hermes_cli.plugins_cmd._install_plugin_core",
                            lambda *a, **k: (Path("/tmp/target"), {"name": "y"}, "y"))
        monkeypatch.setattr("hermes_cli.plugins_cmd._looks_like_plugin_dir", lambda _t: False)
        monkeypatch.setattr("hermes_cli.plugins_cmd._is_tty", lambda: False)
        monkeypatch.setattr("hermes_cli.plugins_cmd._display_after_install", lambda *a, **k: None)
        monkeypatch.setattr("hermes_cli.plugins_cmd_install._prompt_plugin_env_vars", lambda _m, _c: None)
        monkeypatch.setattr("pm.workspace.enabled_plugin_dirs", lambda: [])

        home = "/tmp/hermes-test-home-124037-custom"
        monkeypatch.setattr("hermes_cli.plugins_cmd._plugins_dir", lambda: None)

        cmd_install("owner/repo")  # 不应 raise —— custom source 不过 known-issue 闸


# ── dashboard install gate ───────────────────────────────────────────────────


class TestDashboardInstallGate:
    def test_dashboard_refuses_known_issues_entry(self, monkeypatch):
        from hermes_cli.plugins_cmd_install import dashboard_install_plugin

        entry = _entry(known_issues=["Trap on managed installs."])
        monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.get_live_catalog_entry", lambda _n: entry)
        installed = []
        monkeypatch.setattr(
            "hermes_cli.plugins_cmd_catalog.install_catalog_entry", lambda *a, **k: installed.append(1))

        result = dashboard_install_plugin("", force=False, enable=True, catalog_name="hindsight")
        assert result["ok"] is False
        assert "known issues" in result["error"]
        assert installed == [], "dashboard install must refuse known-issues entries"

    def test_dashboard_unaffected_without_known_issues(self, monkeypatch):
        from hermes_cli.plugins_cmd_install import dashboard_install_plugin

        entry = _entry()
        monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.get_live_catalog_entry", lambda _n: entry)
        monkeypatch.setattr("hermes_cli.plugins_cmd._resolve_git_url",
                            lambda _i: ("https://github.com/vectorize-io/hindsight.git", None))
        monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.raise_if_removed", lambda *a: None)
        monkeypatch.setattr(
            "hermes_cli.plugins_cmd_catalog.install_catalog_entry",
            lambda *a, **k: (Path("/tmp/hindsight-target"), {"manifest": 1}, "hindsight"))
        monkeypatch.setattr("hermes_cli.plugins_cmd._python_dependency_summary", lambda _t, w: None)
        monkeypatch.setattr("hermes_cli.plugins_cmd._set_plugin_enabled", lambda *a, **k: None)
        monkeypatch.setattr("hermes_cli.plugins_cmd._missing_env_specs", lambda _m: [])
        monkeypatch.setattr("hermes_cli.plugins_activation.activate_plugin_now", lambda _n: {})

        result = dashboard_install_plugin("", force=False, enable=False, catalog_name="hindsight")
        assert result.get("ok") is True