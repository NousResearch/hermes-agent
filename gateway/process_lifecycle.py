"""Gateway process lifecycle primitives shared by migration and profile orchestration."""
from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

GatewayStopStatus = Literal[
    "absent",
    "refused",
    "stopped",
    "force-stopped",
    "already-stopped",
    "error",
]


@dataclass(frozen=True)
class GatewayProcessStopResult:
    """Outcome of stopping the gateway recorded by one profile PID file."""

    status: GatewayStopStatus
    home: Path
    pid: int | None = None
    error: str | None = None

    @property
    def stopped(self) -> bool:
        return self.status in {"stopped", "force-stopped", "already-stopped", "absent"}


def wait_then_force_kill(
    pids: Sequence[int],
    start_times: dict[int, float | None],
    *,
    wait: float = 10.0,
) -> bool:
    """Wait after graceful termination, then force-kill surviving process incarnations.

    Returns True when every PID exited during the graceful wait.
    """
    from gateway.status import _pid_exists, terminate_pid
    from runtime.process_identity import get_process_start_time

    for _ in range(int(wait / 0.5)):
        time.sleep(0.5)
        if not any(_pid_exists(pid) for pid in pids):
            return True
    for pid in pids:
        if _pid_exists(pid):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                terminate_pid(
                    pid,
                    force=True,
                    expected_start_time=start_times.get(pid, get_process_start_time(pid)),
                )
    return False


def stop_gateway_process(profile_dir: Path) -> GatewayProcessStopResult:
    """Stop the gateway recorded by profile_dir/gateway.pid without printing.

    The PID record HERMES_HOME stamp is authoritative: a poisoned record that names
    another profile is refused rather than signalling the foreign gateway.
    """
    profile_dir = Path(profile_dir)
    pid_file = profile_dir / "gateway.pid"
    if not pid_file.exists():
        return GatewayProcessStopResult("absent", profile_dir)

    pid: int | None = None
    try:
        raw = pid_file.read_text(encoding="utf-8").strip()
        data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
        pid = int(data["pid"])
        from gateway.status import recorded_gateway_home_conflicts, terminate_pid
        from runtime.process_identity import get_process_start_time

        if recorded_gateway_home_conflicts(data, expected_home=profile_dir):
            return GatewayProcessStopResult("refused", profile_dir, pid)

        expected_start_time = data.get("start_time")
        if expected_start_time is None:
            expected_start_time = get_process_start_time(pid)
        terminate_pid(pid)
        graceful = wait_then_force_kill([pid], {pid: expected_start_time})
        return GatewayProcessStopResult(
            "stopped" if graceful else "force-stopped",
            profile_dir,
            pid,
        )
    except (ProcessLookupError, PermissionError):
        return GatewayProcessStopResult("already-stopped", profile_dir, pid)
    except Exception as exc:
        return GatewayProcessStopResult("error", profile_dir, pid, str(exc))
