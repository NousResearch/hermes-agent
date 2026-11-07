"""systemd unit timing budgets for the update's gateway restarts (``update_cmd_fleet`` sibling).

How long a restart client must wait (the unit's stop + start budgets) and how long an ``is-active``
poller must outwait the unit's ``RestartSec`` cooldown. Both read unit properties through
``update_cmd_fleet._systemctl`` (late-imported: that is the seam tests patch).
"""

from __future__ import annotations

import subprocess
import time
from contextlib import suppress

# poll() takes signed 32-bit milliseconds; keep headroom for rounding in communicate().
_SYSTEMCTL_RESTART_TIMEOUT_MAX = (2**31 - 1) // 1000 - 1


def _unit_stop_start_budgets(scope_cmd: list, svc_name: str) -> tuple[float, float]:
    """``(TimeoutStopUSec, TimeoutStartUSec)`` in seconds. Unknown or infinite limits use
    systemd's usual 90s per phase so automation stays bounded."""
    from gateway.shutdown_forensics import parse_systemd_duration_to_us
    from hermes_cli.update_cmd_fleet import _systemctl

    budgets = {"TimeoutStopUSec": 90.0, "TimeoutStartUSec": 90.0}
    try:
        show = _systemctl(
            scope_cmd + ["show", svc_name, "--property=TimeoutStopUSec,TimeoutStartUSec"],
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return budgets["TimeoutStopUSec"], budgets["TimeoutStartUSec"]
    if show.returncode == 0:
        for line in (show.stdout or "").splitlines():
            key, _, raw = line.partition("=")
            if key in budgets:
                # The shared parser returns None for infinity/unrecognized units.
                try:
                    raw = raw.strip()
                    duration = int(raw) if raw.isascii() and raw.isdigit() else parse_systemd_duration_to_us(raw)
                    if duration is not None and duration > 0:
                        budgets[key] = duration / 1_000_000
                except (ValueError, OverflowError):
                    pass
    return budgets["TimeoutStopUSec"], budgets["TimeoutStartUSec"]


def systemd_restart_timeout(scope_cmd: list, svc_name: str, *, start_only: bool = False) -> float:
    """Outwait the unit's stop + start budgets, not just the systemctl client.

    A client timeout does not cancel the manager's queued restart: ``run_unit_job`` then
    waits for that job instead of calling the restart failed.
    """
    stop, start = _unit_stop_start_budgets(scope_cmd, svc_name)
    return min((0.0 if start_only else stop) + start + 15.0, _SYSTEMCTL_RESTART_TIMEOUT_MAX)


#: A wait on a unit's own job never runs past this, whatever its timeouts say.
_UNIT_JOB_HARD_CAP_S = 900.0
#: Exit status ``run_unit_job`` reports when systemd's job for the unit is still running.
UNIT_JOB_STILL_RUNNING = 124
_TRANSITIONAL_STATES = ("activating", "deactivating", "reloading", "refreshing")


def unit_job_budget(scope_cmd: list, svc_name: str, verb: str) -> float:
    """How long systemd may legitimately take to finish *verb* on the unit: TimeoutStopUSec
    twice (the SIGTERM wait, then the final-SIGKILL wait systemd bounds by the same value; a
    cgroup child stuck in D state takes both), plus TimeoutStartUSec for a start, plus slack."""
    stop, start = _unit_stop_start_budgets(scope_cmd, svc_name)
    budget = 2 * stop + (start if verb in ("start", "restart") else 0.0) + 30.0
    return min(budget, _UNIT_JOB_HARD_CAP_S)


def unit_job_state(scope_cmd: list, svc_name: str) -> dict[str, str]:
    """``ActiveState``/``SubState``/``MainPID``/``Job`` of the unit; ``{}`` when unreadable."""
    from hermes_cli.update_cmd_fleet import _systemctl

    try:
        show = _systemctl(scope_cmd + ["show", svc_name, "--property=ActiveState,SubState,MainPID,Job"], timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if show.returncode != 0:
        return {}
    return dict(line.partition("=")[::2] for line in (show.stdout or "").splitlines() if "=" in line)


def _settled(state: dict[str, str]) -> bool:
    return state.get("ActiveState") not in (None, "", *_TRANSITIONAL_STATES) and not state.get("Job")


def run_unit_job(manage_cmd: list, scope_cmd: list, verb: str, svc_name: str, *, timeout: float):
    """``systemctl <verb> <unit>``, where a client timeout is UNKNOWN, not failed.

    The timeout cancels nothing: systemd keeps running the job (a ``start`` queued behind a
    slow stop included), so judging it failed reports a restart that then happens, and a
    rollback ``start`` queues behind the stop and boots the unit later, unasked. Past the
    timeout this waits for the unit's job to settle, bounded by ``unit_job_budget`` from the
    call, and reports the settled state: exit 0 when the unit reached what *verb* asked for
    (``active`` for start/restart, stopped for stop), else ``UNIT_JOB_STILL_RUNNING`` while
    the job still runs, 1 when it settled elsewhere. A call that returns in time is untouched.
    """
    from hermes_cli.update_cmd_fleet import _systemctl

    started = time.monotonic()
    try:
        return _systemctl(manage_cmd + [verb, svc_name], timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        argv = exc.cmd
    deadline = started + max(unit_job_budget(scope_cmd, svc_name, verb), timeout)
    state = unit_job_state(scope_cmd, svc_name)
    while not _settled(state) and time.monotonic() < deadline:
        time.sleep(1.0)
        state = unit_job_state(scope_cmd, svc_name)
    waited, active = time.monotonic() - started, state.get("ActiveState", "unknown")
    if not _settled(state):
        print(f"  ⏳ systemctl {verb} {svc_name} timed out after {timeout:.0f}s; systemd's job is still "
              f"running after {waited:.0f}s ({active}/{state.get('SubState', '?')})")
        return subprocess.CompletedProcess(argv, UNIT_JOB_STILL_RUNNING, "", f"systemd's {verb} job still running")
    reached = active == "active" if verb in ("start", "restart") else active in ("inactive", "failed")
    print(f"  ⏳ systemctl {verb} {svc_name} timed out after {timeout:.0f}s; systemd finished the job "
          f"after {waited:.0f}s ({active})")
    return subprocess.CompletedProcess(argv, 0 if reached else 1, "", "" if reached else f"unit settled {active}")


_RESTART_SEC_UNITS = (("ms", 0.001), ("us", 0.000001), ("min", 60.0), ("s", 1.0))


def service_restart_sec(scope_cmd_: list, svc_name_: str, default: float = 0.0) -> float:
    """Read the unit's ``RestartUSec`` in seconds. ``is-active`` pollers must wait
    >= RestartSec + slack or they give up *during* the cooldown and misreport."""
    from hermes_cli.update_cmd_fleet import _systemctl

    try:
        _show = _systemctl(scope_cmd_ + ["show", svc_name_, "--property=RestartUSec", "--value"], timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return default
    raw = (_show.stdout or "").strip()
    # Values like "30s", "100ms", "1min 30s", "infinity"; on any miss return default.
    if not raw or raw == "infinity":
        return default
    total = 0.0
    matched = False
    for part in raw.split():
        for _suf, _mult in _RESTART_SEC_UNITS:
            if part.endswith(_suf):
                with suppress(ValueError):
                    total += float(part[: -len(_suf)]) * _mult
                    matched = True
                break
    return total if matched else default
