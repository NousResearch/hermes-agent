"""Published-head regressions from #56787's October review."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_auto as auto, update_auto_state as state
from hermes_cli import update_auto_precheck as precheck


@pytest.fixture
def contexts(tmp_path, monkeypatch):
    root = tmp_path / "home"
    named = root / "profiles" / "work"
    named.mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n")
    (named / "config.yaml").write_text("{}\n")
    install = Path(precheck.__file__).resolve().parents[1]
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "host-state"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    receipts = root / "logs" / "update_receipts"
    return state.AutoUpdateContext(install, root, receipts), state.AutoUpdateContext(install, named, receipts)


@pytest.mark.parametrize("pending_recovery", [False, True])
def test_no_update_run_skips_child_and_backup_unless_recovery_is_owed(contexts, monkeypatch, pending_recovery):
    from hermes_cli import backup

    context, _ = contexts
    status = {**state.default_status(context), "enabled": True}
    if pending_recovery:
        (context.home / "fleet_restart_pending").write_text("unfinished restart")
    calls = []
    monkeypatch.setattr(auto, "_configured_spec", lambda *args: SimpleNamespace(schedule="04:00", plan_times=[]))
    monkeypatch.setattr(auto.scheduler, "scheduled_action", lambda *args: "run")
    monkeypatch.setattr(auto, "check_update", lambda *args: {"updateAvailable": False, "currentSha": "current"})
    monkeypatch.setattr(auto, "_run", lambda *args: calls.append("child") or 0)
    monkeypatch.setattr(backup, "create_pre_update_backup", lambda *a, **kw: calls.append("backup"))
    with state.operation_lock(context):
        assert auto._scheduled(context, status, SimpleNamespace(scheduler_identity=context.identity)) == 0
    assert calls == (["child"] if pending_recovery else [])
    if not pending_recovery:
        saved = state.read_status(context)
        assert saved["status"] == "up_to_date"
        assert saved["terminalReceipt"] is None
        assert saved["outcomeSource"] == "availability_check"


def test_failed_availability_check_does_not_launch_updater(contexts, monkeypatch):
    context, _ = contexts
    status = {**state.default_status(context), "enabled": True}
    monkeypatch.setattr(auto, "_configured_spec", lambda *args: SimpleNamespace(schedule="04:00", plan_times=[]))
    monkeypatch.setattr(auto.scheduler, "scheduled_action", lambda *args: "run")

    def fail_check(*args):
        raise ValueError("could not verify target")

    monkeypatch.setattr(auto, "check_update", fail_check)
    monkeypatch.setattr(auto, "_run", lambda *args: pytest.fail("updater must not run after a failed check"))
    with state.operation_lock(context), pytest.raises(ValueError, match="could not verify target"):
        auto._scheduled(context, status, SimpleNamespace(scheduler_identity=context.identity))
    assert state.read_status(context)["status"] == "check_failed"
