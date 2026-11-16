"""Which runtime records name the gateway that holds ``gateway.lock`` right now.

``gateway.status.get_running_pid_identity_strict`` validates the PID file and the
lock record against one live process. The PID file is advisory beside a HELD
lock: a launch-service gateway keeps serving after ``gateway.pid`` was unlinked
(#110166), and ``gateway run --replace`` rewrites the lock but can leave the
previous gateway's PID file behind (#143331). That leftover names a different
process instance (its own start time) that is gone or whose PID was recycled,
and comparing it with the lock aborted every following ``hermes update`` at the
Windows gateway pause stage until the file was removed by hand.

A PID file claiming the lock holder's own start time under another PID is not a
leftover but garbled state, so it still reaches the comparison and fails closed;
so does a PID file naming another LIVE instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def strict_identity_records(pid_path: Path, lock_record: Optional[dict[str, Any]],
                            pid_exists: bool) -> tuple:
    """Records the strict identity check must agree on; a previous gateway's PID file drops out."""
    from gateway import status

    if not pid_exists:
        return (lock_record,)
    pid_record = status._read_pid_record(pid_path)
    if pid_record and lock_record and names_a_previous_gateway(pid_record, lock_record):
        return (lock_record,)
    return (pid_record, lock_record)


def names_a_previous_gateway(pid_record: dict[str, Any], lock_record: dict[str, Any]) -> bool:
    """True when *pid_record* is another instance than the lock holder and that instance is gone."""
    from gateway import status

    pid = status._pid_from_record(pid_record)
    recorded, held = pid_record.get("start_time"), lock_record.get("start_time")
    if pid is None or pid <= 0 or recorded is None or held is None:
        return False
    try:
        if status._start_times_agree(held, recorded):
            return False
        if not status._pid_exists(pid):
            return True
        live = status._get_process_start_time(pid)
        return live is not None and not status._start_times_agree(live, recorded)
    except (TypeError, ValueError):
        return False
