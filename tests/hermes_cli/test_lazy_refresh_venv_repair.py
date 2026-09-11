"""Tests for lazy-backend refresh venv repair (#57828 / #58004)."""

from __future__ import annotations

from types import SimpleNamespace

import hermes_cli.main as m
import hermes_cli.main_install_repair as hermes_cli_main_install_repair
import pytest


def test_refresh_failure_reports_pm_error(monkeypatch, capsys):
    import importlib
    ensure = importlib.import_module("pm.client")
    def fail(*args, **kwargs):
        raise RuntimeError("resolution failed")
    monkeypatch.setattr(ensure, "sync_venv", fail)
    assert m._refresh_active_lazy_features(["matrix"]) is False
    assert "resolution failed" in capsys.readouterr().out


def test_refresh_uses_pre_rebuild_snapshot_when_provided(monkeypatch):
    import importlib
    ensure = importlib.import_module("pm.client")
    calls = []
    monkeypatch.setattr(ensure, "sync_venv", lambda extras, **kwargs: calls.append((extras, kwargs)))
    assert m._refresh_active_lazy_features(["telegram"]) is True
    assert calls == [(["telegram"], {"explicit": True})]


def test_cmd_update_repairs_before_refreshing_dependency_inputs(tmp_path, monkeypatch):
    """The current-checkout repair must bypass PM's matching-stamp shortcut."""
    from hermes_cli import update_cmd

    (tmp_path / ".git").mkdir()
    snapshot = ["platform.telegram"]
    refresh_calls = []

    class SyncReached(Exception):
        pass

    def fake_run(cmd, **kwargs):
        if "rev-parse" in cmd:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        if "rev-list" in cmd:
            return SimpleNamespace(returncode=0, stdout="0\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_sync_raises(extras=None, *, explicit=False, repair=False):
        refresh_calls.append((extras, explicit, repair))
        raise SyncReached

    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(m, "_capture_active_lazy_features", lambda: snapshot.copy())
    monkeypatch.setattr(m, "_is_windows", lambda: False)
    monkeypatch.setattr(hermes_cli_main_install_repair, "_is_windows", lambda: False)
    monkeypatch.setattr(m, "_run_pre_update_backup", lambda args: None)
    monkeypatch.setattr(m, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(m, "_resume_windows_gateways_after_update", lambda state: None)
    monkeypatch.setattr(update_cmd, "_discard_lockfile_churn", lambda *args: None)
    monkeypatch.setattr(m, "_get_origin_url", lambda *args: "https://github.com/NousResearch/hermes-agent.git")
    monkeypatch.setattr(m, "_resolve_update_branch", lambda args: "main")
    monkeypatch.setattr(m, "_stash_local_changes_if_needed", lambda *args: None)
    monkeypatch.setattr(update_cmd, "_invalidate_update_cache", lambda: None)
    monkeypatch.setattr(
        update_cmd, "_venv_core_imports_healthy", lambda: (False, "broken")
    )
    monkeypatch.setattr(update_cmd, "_write_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(m.subprocess, "run", fake_run)
    import pm
    monkeypatch.setattr(pm, "sync_venv", fake_sync_raises)

    args = SimpleNamespace(
        yes=True,
        force=False,
        force_venv=False,
        no_backup=True,
        backup=False,
        branch=None,
    )
    with pytest.raises(SyncReached):
        update_cmd._cmd_update_impl(args, gateway_mode=False)

    # Repair must bypass freshness before the normal update can refresh inputs.
    assert refresh_calls == [(None, False, True)]
