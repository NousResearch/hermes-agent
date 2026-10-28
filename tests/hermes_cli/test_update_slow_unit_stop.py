"""A systemd unit whose stop outlives the updater's ``systemctl`` timeout.

systemd keeps running a job after its client gives up, so a timed-out ``stop``/``start`` is
unknown until the unit's state is read: the pause must hold while the gateway is gone, and the
restart must count a unit that finished coming up on its own job.
"""
import subprocess
import types

import pytest

from hermes_cli import update_cmd_fleet as fleet
from hermes_cli import update_cmd_fleet_unit_budget as budget
from hermes_cli import update_cmd_posix_pause as pause

UNIT = "hermes-gateway"
SCOPE = ["systemctl", "--user"]
MANAGE = [*SCOPE, "--no-ask-password"]


class _SlowUnit:
    """A unit whose queued job finishes ``job_s`` (fake) seconds after it is issued, into
    ``settled`` (ActiveState, MainPID)."""

    def __init__(self, monkeypatch, *, job_s: float, settled: tuple[str, int]):
        self.now, self.job_s, self.settled, self.done_at, self.verbs = 0.0, job_s, settled, None, []
        clock = types.SimpleNamespace(monotonic=lambda: self.now, sleep=self._sleep)
        monkeypatch.setattr(budget, "time", clock, raising=False)  # base has no clock here
        monkeypatch.setattr(fleet, "_systemctl", self.systemctl)

    def _sleep(self, seconds):
        self.now += seconds

    def _state(self):
        if self.done_at is not None and self.now >= self.done_at:
            return self.settled[0], self.settled[1], ""
        return "deactivating", 0, "7"

    def systemctl(self, cmd, *, timeout):
        verb = next((v for v in ("stop", "start", "restart", "reset-failed", "show", "is-active") if v in cmd), None)
        if verb in ("stop", "start", "restart"):
            self.verbs.append(verb)
            self.done_at = self.now + self.job_s
            self.now += timeout  # the client waits out its timeout, the job keeps running
            raise subprocess.TimeoutExpired(cmd, timeout)
        active, main_pid, job = self._state()
        if verb == "is-active":
            return subprocess.CompletedProcess(cmd, 0, active, "")
        prop = next((c for c in cmd if c.startswith("--property=")), "")
        if "TimeoutStopUSec" in prop:
            return subprocess.CompletedProcess(cmd, 0, "TimeoutStopUSec=90s\nTimeoutStartUSec=90s", "")
        if prop == "--property=MainPID":
            return subprocess.CompletedProcess(cmd, 0, str(main_pid), "")
        return subprocess.CompletedProcess(
            cmd, 0, f"ActiveState={active}\nSubState=x\nMainPID={main_pid}\nJob={job}", "")


@pytest.mark.parametrize("serving", [False, True], ids=["gateway-gone", "gateway-serving"])
def test_pause_holds_while_systemd_finishes_a_slow_stop(monkeypatch, serving):
    """The incident: the drained gateway already exited, ``systemctl stop`` timed out at 120 s while
    systemd needed ~181 s (SIGTERM wait + final-SIGKILL wait) for a stuck cgroup child. Base raised,
    and the pause's rollback ``start`` queued behind the stop and booted the gateway mid-swap."""
    unit = _SlowUnit(monkeypatch, job_s=181.0, settled=("active", 4242) if serving else ("failed", 0))
    monkeypatch.setattr(pause, "_alive", lambda pid, ct: False)
    entry = {"kind": "systemd", "scope": "user", "unit": UNIT, "pid": 1234, "ct": None}
    if serving:
        with pytest.raises(pause.PauseRefused):
            pause._stop_unit(entry)
    else:
        pause._stop_unit(entry)
    assert unit.verbs == ["stop"]


def test_restart_counts_a_unit_whose_start_job_outlived_the_client(monkeypatch):
    """After the drain, ``systemctl start`` queues behind the unit's still-running stop and times
    out; systemd then brings the gateway up on the new code. That is a restart, not a failure."""
    unit = _SlowUnit(monkeypatch, job_s=190.0, settled=("active", 5555))
    calls = iter([subprocess.CompletedProcess([], 0, "active", ""), subprocess.CompletedProcess([], 0, "42", "")])
    real = unit.systemctl
    monkeypatch.setattr(fleet, "_systemctl", lambda cmd, *, timeout: next(calls) if not unit.verbs and (
        "is-active" in cmd or "--property=MainPID" in cmd) else real(cmd, timeout=timeout))
    monkeypatch.setattr(fleet, "_systemd_unit_owned_by_update", lambda scope_cmd, svc: True)
    monkeypatch.setattr(fleet, "_repair_unit_without_fatal_exit_park", lambda svc, scope: None)
    monkeypatch.setattr(fleet, "_drain_or_signal_gateway_for_update", lambda *a, **kw: True)
    restarted, failed = [], []
    fleet._restart_one_systemd_gateway_unit(
        UNIT, scope="user", scope_cmd=SCOPE, drain_budget=45, _manage_cmd_cache={"user": MANAGE},
        restarted_services=restarted, failed_or_stale_units=failed,
    )
    assert (restarted, failed, unit.verbs) == ([UNIT], [], ["start"])
