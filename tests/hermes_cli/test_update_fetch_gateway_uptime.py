"""Windows source updates keep an existing gateway serving through Git transport.

Regression for a fetch stall that left the gateway paused for the entire network
timeout. The real update dispatcher is exercised with a synthetic Git transport;
no live gateway or remote repository is touched.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import main, update_cmd
from hermes_cli import update_cmd_windows, update_pause_record


pytestmark = pytest.mark.platforms("windows")


@pytest.fixture
def updater_at_fetch(monkeypatch, tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    events = []

    monkeypatch.setattr(main, "PROJECT_ROOT", root)
    monkeypatch.setattr(update_cmd, "git_operation_in_progress", lambda root: None)
    monkeypatch.setattr("hermes_cli.gitlock.release_dead_index_lock", lambda root: False)
    monkeypatch.setattr(update_cmd._check, "clear_git_debris", lambda root: None)
    monkeypatch.setattr(update_cmd._check, "report_pack_tidy", lambda root: None)
    monkeypatch.setattr(update_cmd._check, "tracking_refspec", lambda *args: "main")
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *args: SimpleNamespace(
        gw_input_fn=None, assume_yes=True, no_gateway_restart=False,
        pre_update_version="before", discard_local_changes=False, keep_stash=False,
        switch_branch=False))
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda args: None)
    monkeypatch.setattr(main, "_run_pre_update_backup", lambda args: None)
    monkeypatch.setattr(update_cmd, "_record_pre_update_backup_outcome", lambda *args: None)
    monkeypatch.setattr(update_cmd, "_record_snapshot_stage", lambda *args: None)
    monkeypatch.setattr(update_cmd, "_record_stop", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "_desktop_packaged_executable", lambda root: None)
    monkeypatch.setattr(main, "_desktop_dist_exists", lambda root: False)
    monkeypatch.setattr(main, "_installed_desktop_apps", lambda: [])
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda **kwargs: (False, ["git"], False))
    monkeypatch.setattr(update_cmd, "_source_completion_request", lambda *args: {
        "receipt": {"update_id": "fixture"}, "windows_resume": args[3]})
    monkeypatch.setattr(main, "_resolve_update_branch", lambda args: "main")
    monkeypatch.setattr(update_cmd, "_source_update_channel", lambda args: "main")
    monkeypatch.setattr(main, "_warn_orphaned_update_autostashes", lambda *args: None)
    monkeypatch.setattr(update_cmd, "_heal_stale_shallow_checkout", lambda *args: None)
    monkeypatch.setattr("hermes_cli.gitlock.repair_broken_shallow_boundaries", lambda root: 0)
    monkeypatch.setattr("hermes_cli.gitlock.prune_stale_shallow_grafts", lambda root: 0)
    monkeypatch.setattr("hermes_cli.gitlock.fetch_with_partial_clone_recovery",
                        lambda run, git_cmd, args, root: run(git_cmd, args))
    # The test runs on Windows and supplies the discovered-gateway precondition.
    monkeypatch.setattr(update_cmd, "_defer_windows_gateway_pause_for_fetch", lambda: True,
                        raising=False)
    return events


def test_successful_fetch_precedes_gateway_pause(updater_at_fetch, monkeypatch):
    events = updater_at_fetch

    class ReachedPause(Exception):
        pass

    def pause():
        events.append("pause")
        raise ReachedPause

    def git_run(git_cmd, args, **kwargs):
        assert args[0] == "fetch"
        events.append("fetch")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(main, "_pause_windows_gateways_for_update", pause)
    monkeypatch.setattr(update_cmd, "_git_run", git_run)
    with pytest.raises(ReachedPause):
        update_cmd._cmd_update_impl(SimpleNamespace(branch="main"), gateway_mode=False)
    assert events == ["fetch", "pause"]


def test_failed_fetch_does_not_pause_a_serving_gateway(updater_at_fetch, monkeypatch):
    events = updater_at_fetch
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update",
                        lambda: events.append("pause"))

    def git_run(git_cmd, args, **kwargs):
        assert args[0] == "fetch"
        events.append("fetch")
        return subprocess.CompletedProcess(args, 124, "", "transport timed out")

    monkeypatch.setattr(update_cmd, "_git_run", git_run)
    with pytest.raises(SystemExit) as result:
        update_cmd._cmd_update_impl(SimpleNamespace(branch="main"), gateway_mode=False)
    assert result.value.code == 1
    assert events == ["fetch"]


def test_checkout_cleanup_and_handoff_wait_for_pause(updater_at_fetch, monkeypatch):
    events = updater_at_fetch
    request = {}

    class ReachedBranch(Exception):
        pass

    monkeypatch.setattr("atexit.register", lambda *args: None)
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update",
                        lambda: events.append("pause") or {"resume_needed": True})
    monkeypatch.setattr(update_cmd, "_git_run", lambda git_cmd, args, **kwargs:
                        events.append("fetch") or subprocess.CompletedProcess(args, 0, "", ""))
    monkeypatch.setattr(update_cmd, "_discard_lockfile_churn",
                        lambda *args, **kwargs: events.append("lockfile"))
    monkeypatch.setattr(update_cmd, "_normalize_managed_eol",
                        lambda *args, **kwargs: events.append("eol"))
    monkeypatch.setattr(update_cmd, "_source_completion_request", lambda *args:
                        request)
    monkeypatch.setattr(update_cmd, "_current_branch_name", lambda *args, **kwargs:
                        (_ for _ in ()).throw(ReachedBranch))
    with pytest.raises(ReachedBranch):
        update_cmd._cmd_update_impl(SimpleNamespace(branch="main"), gateway_mode=False)
    assert events == ["fetch", "pause", "lockfile", "eol"]
    assert request["windows_resume"] == {"resume_needed": True}


def test_pause_deferral_requires_serving_gateway_and_no_orphan(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(update_pause_record, "orphans", lambda: [])
    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways",
                        lambda: ({}, [], set(), [123]))
    assert update_cmd_windows._defer_windows_gateway_pause_for_fetch()
    monkeypatch.setattr(update_pause_record, "orphans", lambda: [("old", {})])
    assert not update_cmd_windows._defer_windows_gateway_pause_for_fetch()
    monkeypatch.setattr(update_pause_record, "orphans", lambda: [])
    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways",
                        lambda: ({}, [], set(), []))
    assert not update_cmd_windows._defer_windows_gateway_pause_for_fetch()
    def uncertain():
        raise RuntimeError("gateway ownership could not be determined")
    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways", uncertain)
    with pytest.raises(RuntimeError, match="ownership"):
        update_cmd_windows._defer_windows_gateway_pause_for_fetch()
