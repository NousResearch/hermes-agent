"""systemd unit timing budgets for the update's gateway restarts (``update_cmd_fleet`` sibling).

How long a restart client must wait (the unit's stop + start budgets) and how long an ``is-active``
poller must outwait the unit's ``RestartSec`` cooldown. Both read unit properties through
``update_cmd_fleet._systemctl`` (late-imported: that is the seam tests patch).
"""

from __future__ import annotations

import subprocess
from contextlib import suppress

# poll() takes signed 32-bit milliseconds; keep headroom for rounding in communicate().
_SYSTEMCTL_RESTART_TIMEOUT_MAX = (2**31 - 1) // 1000 - 1


def systemd_restart_timeout(scope_cmd: list, svc_name: str, *, start_only: bool = False) -> float:
    """Outwait the unit's stop + start budgets, not just the systemctl client.

    A client timeout does not cancel the manager's queued restart. Unknown or
    infinite limits use systemd's usual 90s per phase so automation stays bounded.
    Custom ExecStop chains or EXTEND_TIMEOUT_USEC can still exceed this budget;
    genuine timeouts must continue through the existing per-unit failure path.
    """
    from gateway.shutdown_forensics import parse_systemd_duration_to_us
    from hermes_cli.update_cmd_fleet import _systemctl

    budgets = {"TimeoutStartUSec": 90.0}
    if not start_only:
        budgets["TimeoutStopUSec"] = 90.0
    try:
        show = _systemctl(
            scope_cmd + ["show", svc_name, "--property=TimeoutStopUSec,TimeoutStartUSec"],
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return sum(budgets.values()) + 15.0
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
    return min(sum(budgets.values()) + 15.0, _SYSTEMCTL_RESTART_TIMEOUT_MAX)


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
