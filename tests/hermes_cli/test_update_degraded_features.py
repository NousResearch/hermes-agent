"""Configured-feature observability: an update that completes with a degraded
platform must say so where operators look, and `pm doctor` must flag it.

The dependency sync prints warnings for configured features whose SDK install
failed, but the completion line still reads ``✓ Update complete!`` and PM's
package audit stays green while the gateway cannot load the platform on
restart. Regression for #146281.
"""
from __future__ import annotations

import json

import pytest

from hermes_cli import update_receipt


def _latest(home):
    return json.loads((home / "logs/update_receipts/latest.json").read_text())


def _maintenance_stubs(monkeypatch, tmp_path):
    """Neutralize every post-update side effect; keep the completion path real."""
    from hermes_cli import update_cmd, update_cmd_maint as maint
    from hermes_cli import update_windows_acl

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(update_windows_acl, "repair_windows_home_acl", lambda: None)
    for name in ("_verify_and_restore_state_dbs_post_update", "_invalidate_live_plugin_catalog_caches",
                 "_print_bundled_skills_sync_report", "_sync_profiles_after_update"):
        monkeypatch.setattr(maint, name, lambda name=name: None)
    monkeypatch.setattr(update_cmd, "_check_and_apply_config_migration", lambda **kwargs: None)
    monkeypatch.setattr("hermes_cli.gitlock.fetch_full_commit_graph", lambda *a, **kw: False)
    monkeypatch.setattr("hermes_cli.model_catalog.seed_cache_from_checkout", lambda root: False)
    return maint


def test_degraded_configured_features_reach_receipt_and_completion_line(tmp_path, monkeypatch, capsys):
    # A platform whose SDK install failed is not tail work (the tail cannot install it),
    # but it must surface on the receipt, as a follow-up line, and inside the one
    # completion line operators read — otherwise a broken Matrix reads as a clean update.
    maint = _maintenance_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr("hermes_cli.main_install_repair._configured_features_missing_deps",
                        lambda: [("Matrix", "Run `hermes setup` to install Matrix support", "matrix"),
                                 ("MCP servers", "Run `hermes pm install` to install MCP support", "mcp")])
    verdicts: list[str] = []
    monkeypatch.setattr(maint, "_print_verified_update_completion",
                        lambda message: verdicts.append(message) or True)
    monkeypatch.setattr(maint, "_print_post_update_notices_and_self_heals", lambda: None)

    update_receipt.begin_update_receipt()
    owed: list = []
    assert maint._run_post_update_maintenance(
        assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
        had_desktop_app_before_update=False, pre_update_version="0.1.0", followups=owed) is True
    assert owed == []
    assert len(verdicts) == 1
    assert verdicts[0].startswith("✓")
    assert "2 configured features degraded: Matrix, MCP servers" in verdicts[0]
    assert "Update follow-up 'features'" in capsys.readouterr().out
    update_receipt.finalize_update_receipt("success")
    followups = {f["step"]: f["reason"] for f in _latest(tmp_path)["followups"]}
    assert "Matrix, MCP servers" in followups["features"]


def test_clean_update_completion_line_unchanged_without_degraded_features(tmp_path, monkeypatch, capsys):
    # The degraded suffix is conditional: a fully healthy update keeps the exact
    # completion line downstream parsers and docs pin.
    maint = _maintenance_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr("hermes_cli.main_install_repair._configured_features_missing_deps", list)
    verdicts: list[str] = []
    monkeypatch.setattr(maint, "_print_verified_update_completion",
                        lambda message: verdicts.append(message) or True)
    monkeypatch.setattr(maint, "_print_post_update_notices_and_self_heals", lambda: None)

    update_receipt.begin_update_receipt()
    owed: list = []
    assert maint._run_post_update_maintenance(
        assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
        had_desktop_app_before_update=False, pre_update_version=None, followups=owed) is True
    assert "degraded" not in verdicts[0]
    update_receipt.finalize_update_receipt("success")
    assert _latest(tmp_path).get("followups", []) == []