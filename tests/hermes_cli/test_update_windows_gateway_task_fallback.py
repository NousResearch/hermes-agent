"""Regression coverage for the Windows post-update gateway relaunch fallback.

The relaunch the updater verifies is a CHILD OF THE UPDATER process, so a parent
Job Object can kill it during updater teardown (#48820) no matter how healthy the
install is. The failure message told the user to recover with `schtasks /Run`,
which works because Task Scheduler launches the gateway outside any Job Object --
so when a task is registered, the updater should just do that itself instead of
reporting a failure the user has to fix by hand.
"""

import pytest

from hermes_cli import update_cmd_windows as ucw


class _Recorder:
    """Stands in for gateway_windows so the fallback's decisions are observable."""

    def __init__(self, registered=True, run_code=0):
        self.registered = registered
        self.run_code = run_code
        self.runs = []

    def is_task_registered(self):
        return self.registered

    def get_task_name(self):
        return "Hermes_Gateway"

    def _exec_schtasks(self, args):
        self.runs.append(tuple(args))
        return (self.run_code, "SUCCESS", "")


def test_retry_launches_the_registered_task(monkeypatch):
    """A registered task is the reliable relaunch path and must actually be run."""
    rec = _Recorder()
    monkeypatch.setattr(
        "hermes_cli.gateway_windows.is_task_registered", rec.is_task_registered, raising=False)
    monkeypatch.setattr(
        "hermes_cli.gateway_windows.get_task_name", rec.get_task_name, raising=False)
    monkeypatch.setattr(
        "hermes_cli.gateway_windows._exec_schtasks", rec._exec_schtasks, raising=False)

    assert ucw._retry_via_scheduled_task() is True
    assert rec.runs == [("/Run", "/TN", "Hermes_Gateway")]


def test_retry_is_a_noop_without_a_registered_task(monkeypatch):
    """No task => nothing to fall back to; the caller reports the honest failure."""
    rec = _Recorder(registered=False)
    monkeypatch.setattr(
        "hermes_cli.gateway_windows.is_task_registered", rec.is_task_registered, raising=False)
    monkeypatch.setattr(
        "hermes_cli.gateway_windows._exec_schtasks", rec._exec_schtasks, raising=False)

    assert ucw._retry_via_scheduled_task() is False
    assert rec.runs == []


def test_retry_never_raises_when_schtasks_fails(monkeypatch):
    """A failing schtasks must degrade to False, never break the update path."""
    rec = _Recorder(run_code=1)
    monkeypatch.setattr(
        "hermes_cli.gateway_windows.is_task_registered", rec.is_task_registered, raising=False)
    monkeypatch.setattr(
        "hermes_cli.gateway_windows.get_task_name", rec.get_task_name, raising=False)
    monkeypatch.setattr(
        "hermes_cli.gateway_windows._exec_schtasks", rec._exec_schtasks, raising=False)

    assert ucw._retry_via_scheduled_task() is False


def test_verify_falls_back_before_giving_up(monkeypatch):
    """The verification gate must try the Scheduled Task before declaring failure."""
    import hermes_cli.gateway_windows as gw

    attempts = []

    def fake_wait(*, timeout_s, all_profiles=False):
        attempts.append(timeout_s)
        return [4321] if len(attempts) > 1 else []   # dead on the first poll, alive after the fallback

    monkeypatch.setattr(gw, "_wait_for_gateway_ready", fake_wait, raising=False)
    monkeypatch.setattr(gw, "_write_start_attestation", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ucw, "_retry_via_scheduled_task", lambda: True)

    ucw._verify_relaunched_gateways_alive({}, {}, [])

    assert len(attempts) == 2, "expected a re-poll after the Scheduled Task fallback"


def test_verify_still_fails_when_nothing_recovers(monkeypatch):
    """Fail-closed: no gateway and no fallback must still raise, not print a false success."""
    import hermes_cli.gateway_windows as gw

    monkeypatch.setattr(gw, "_wait_for_gateway_ready", lambda **k: [], raising=False)
    monkeypatch.setattr(ucw, "_retry_via_scheduled_task", lambda: False)

    with pytest.raises(RuntimeError, match="not verified alive"):
        ucw._verify_relaunched_gateways_alive({}, {}, [])
