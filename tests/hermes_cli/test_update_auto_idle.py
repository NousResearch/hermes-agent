"""Scheduled updates wait for idle gateways instead of moving the checkout under live turns."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_auto as auto, update_auto_idle as idle, update_auto_state as state
from hermes_cli import update_auto_precheck as precheck


@pytest.fixture
def context(tmp_path, monkeypatch):
    root = tmp_path / "home"
    root.mkdir()
    (root / "config.yaml").write_text("updates:\n  auto:\n    idle_wait_minutes: 0.005\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "host-state"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    install = Path(precheck.__file__).resolve().parents[1]
    ctx = state.AutoUpdateContext(install, root, root / "logs" / "update_receipts")
    monkeypatch.setattr(auto, "_configured_spec", lambda *a: SimpleNamespace(schedule="04:00", plan_times=[]))
    monkeypatch.setattr(auto.scheduler, "scheduled_action", lambda *a: "run")
    monkeypatch.setattr(auto, "check_update", lambda *a: {"updateAvailable": True, "currentSha": "old"})
    monkeypatch.setattr(auto, "recovery_needed", lambda *a: False)
    return ctx


def _slot(ctx, monkeypatch, answers):
    """Run one scheduled slot against a gateway whose ``active-work`` answers come from ``answers``."""
    calls = []
    monkeypatch.setattr(idle, "live_work", lambda home: answers.pop(0) if len(answers) > 1 else answers[0])
    monkeypatch.setattr(auto, "_run", lambda *a: calls.append("updater") or 0)
    status = {**state.default_status(ctx), "enabled": True}
    with state.operation_lock(ctx):
        assert auto._scheduled(ctx, status, SimpleNamespace(scheduler_identity=ctx.identity)) == 0
    return calls


def test_a_slot_skips_the_update_while_a_turn_is_still_running(context, monkeypatch):
    turn = {"active_agents": 1, "active_work": [{"kind": "chat", "session": "agent:main:telegram:dm:42"}]}
    assert _slot(context, monkeypatch, [turn]) == []
    saved = state.read_status(context)
    assert saved["status"] == "deferred_busy"
    assert saved["busy"] == ["default: chat turn agent:main:telegram:dm:42"]


def test_a_slot_updates_once_the_gateway_goes_idle(context, monkeypatch):
    turn = {"active_agents": 1, "active_work": [{"kind": "chat", "session": "s"}]}
    assert _slot(context, monkeypatch, [turn, {"active_agents": 0, "active_work": []}]) == ["updater"]
