"""Edge-case tests for restart handoff and degraded/rollback verification (BM-5)."""

from __future__ import annotations

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from tests.hermes_cli.test_gateway_migrate_backends_apply import FakeHost, _GATEWAY_PID, _args

_NEW_GATEWAY_PID = _GATEWAY_PID + 1000


class _SequenceRestartHost(FakeHost):
    """FakeHost whose gateway_pid() returns a scripted sequence of PIDs."""

    def __init__(self, home, *, pids, state="running", preflight_pid=_GATEWAY_PID):
        super().__init__(home, gateway_installed=True, gateway_active=True)
        self._pids = list(pids)
        self._state = state
        self._preflight_pid = preflight_pid
        self.order: list[str] = []

    def gateway_state(self):
        st = super().gateway_state()
        st["running_pid"] = self._preflight_pid if self.gw["active"] else None
        return st

    def gateway_pid(self):
        self.order.append("pid")
        if len(self._pids) > 1:
            return self._pids.pop(0)
        return self._pids[0]

    def gateway_status(self, host, port):
        self.order.append("status")
        if self._pids[0] != _NEW_GATEWAY_PID:
            return None
        return {"gateway_state": self._state, "auth_required": True, "auth_providers": ["basic"]}

    def pid_listens_on(self, pid, port):
        return pid in (_GATEWAY_PID, _NEW_GATEWAY_PID) and self._webapp_on_9119()


def test_verify_budget_is_at_least_300s():
    """Verify timeout must cover a busy multi-profile gateway boot (BM-5)."""
    assert engine.VERIFY_SECONDS >= 300.0


def test_degraded_gateway_passes_verify_while_still_enforcing_webapp_checks(home, monkeypatch):
    monkeypatch.setattr(engine, "RESTART_HANDOFF_SECONDS", 5.0, raising=False)
    host = _SequenceRestartHost(home, pids=[_GATEWAY_PID, _NEW_GATEWAY_PID], state="degraded")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    assert engine.status_payload(host)["state"] == "done"


def test_transient_none_before_restart_falls_back_to_preflight_pid(home, monkeypatch):
    """If gateway_pid() transiently returns None before restart, fall back to the preflight PID
    recorded in the manifest so the still-draining old PID is not mistaken for a replacement."""
    monkeypatch.setattr(engine, "RESTART_HANDOFF_SECONDS", 5.0, raising=False)
    # 1st read (before restart): None -> falls back to preflight _GATEWAY_PID.
    # 2nd/3rd reads (handoff wait): still _GATEWAY_PID (draining).
    # 4th read: _NEW_GATEWAY_PID (replacement landed).
    host = _SequenceRestartHost(
        home,
        pids=[None, _GATEWAY_PID, _GATEWAY_PID, _NEW_GATEWAY_PID],
        preflight_pid=_GATEWAY_PID,
    )

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    assert host.order[:4] == ["pid", "pid", "pid", "pid"]
    assert "status" not in host.order[:4]
    assert engine.status_payload(host)["state"] == "done"


def test_missing_before_pid_waits_for_live_pid_and_warns_cleanly_on_timeout(home, monkeypatch, capsys):
    """When both pre-restart and preflight PIDs are None, handoff waits for any live PID,
    warns without 'PID None' if none appears, and rolls back when verify fails."""
    monkeypatch.setattr(engine, "RESTART_HANDOFF_SECONDS", 0.0, raising=False)
    host = _SequenceRestartHost(home, pids=[None], preflight_pid=None)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK
    out = capsys.readouterr().out
    assert "⚠ gateway did not hand over within 0s — verifying anyway" in out
    assert "PID None" not in out
    assert host.old["active"] is True


def test_missing_before_pid_proceeds_once_live_pid_appears(home, monkeypatch):
    monkeypatch.setattr(engine, "RESTART_HANDOFF_SECONDS", 5.0, raising=False)
    # 1st read (before restart): None (preflight_pid also None).
    # 2nd read (handoff wait): still None (gateway restarting).
    # 3rd read: _NEW_GATEWAY_PID -> handoff succeeds before verify probes /api/status.
    host = _SequenceRestartHost(home, pids=[None, None, _NEW_GATEWAY_PID], preflight_pid=None)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    assert host.order[:3] == ["pid", "pid", "pid"]
    assert "status" not in host.order[:3]
    assert engine.status_payload(host)["state"] == "done"


class _RequestedRestartHost(FakeHost):
    """The old PID keeps draining until ``gateway_verb("restart")`` is requested; after it, the PID lookup
    reads the old PID, misses once (the old process gone, the new one not published yet), reads the old
    PID again, and only then the replacement."""

    def __init__(self, home):
        super().__init__(home, gateway_installed=True, gateway_active=True)
        self.order: list[str] = []
        self.after_request = [_GATEWAY_PID, None, _GATEWAY_PID]
        self.replaced = False

    def gateway_verb(self, verb):
        super().gateway_verb(verb)
        self.order.append(verb)

    def gateway_pid(self):
        self.order.append("pid")
        if "restart" in self.order and not self.after_request:
            self.replaced = True
        if self.replaced:
            return _NEW_GATEWAY_PID
        return self.after_request.pop(0) if "restart" in self.order else _GATEWAY_PID

    def sleep(self, seconds):
        assert "restart" in self.order, "waited for a replacement before the restart was requested"

    def gateway_status(self, host, port):
        self.order.append("status")
        if not self.replaced:
            return None
        return {"gateway_state": "running", "auth_required": True, "auth_providers": ["basic"]}

    def pid_listens_on(self, pid, port):
        return pid == _NEW_GATEWAY_PID and self._webapp_on_9119()


def test_handoff_starts_after_the_restart_request_and_outlasts_a_missed_pid_lookup(home, monkeypatch):
    monkeypatch.setattr(engine, "RESTART_HANDOFF_SECONDS", 5.0)
    host = _RequestedRestartHost(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    between = host.order[host.order.index("restart") + 1:host.order.index("status")]
    assert between == ["pid"] * 4  # old, missed (None), old, replacement: then verify


@pytest.mark.parametrize("bad_state", ["draining", "starting", "startup_failed", None])
def test_non_serving_gateway_states_fail_verify_and_roll_back(home, monkeypatch, bad_state):
    monkeypatch.setattr(engine, "RESTART_HANDOFF_SECONDS", 5.0, raising=False)
    host = _SequenceRestartHost(home, pids=[_GATEWAY_PID, _NEW_GATEWAY_PID], state=bad_state)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK
    assert host.old["active"] is True


def test_degraded_gateway_still_rolls_back_when_webapp_page_is_broken(home, monkeypatch):
    """Accepting gateway_state='degraded' never bypasses the Webapp page or listener checks."""
    monkeypatch.setattr(engine, "RESTART_HANDOFF_SECONDS", 5.0, raising=False)
    host = _SequenceRestartHost(home, pids=[_GATEWAY_PID, _NEW_GATEWAY_PID], state="degraded")
    host.bundle_broken = True

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK
    assert host.old["active"] is True


def test_plan_verify_step_title_uses_verify_seconds(monkeypatch, tmp_path):
    """Plan copy derives its verify timeout from shared VERIFY_SECONDS instead of hardcoding 90 s."""
    from hermes_cli import gateway_migrate_backends_timeouts as timeouts
    from hermes_cli.gateway_migrate_backends_plan import BackendPlan

    assert engine.VERIFY_SECONDS == timeouts.VERIFY_SECONDS
    plan = BackendPlan(home=str(tmp_path), new_host="0.0.0.0", new_port=9119)
    verify_step = next(step for step in plan.step_titles() if step["name"] == "verify")
    assert f"within {int(timeouts.VERIFY_SECONDS)} s" in verify_step["title"]

    monkeypatch.setattr(timeouts, "VERIFY_SECONDS", 420.0)
    updated = next(step for step in plan.step_titles() if step["name"] == "verify")
    assert "within 420 s" in updated["title"]
