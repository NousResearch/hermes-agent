"""Windows gateway watchdog for crash recovery and cron continuity.

The Scheduled Task runs one hidden tick every two minutes. Machines that cannot
create Scheduled Tasks launch the same module as a hidden login-time loop. The
watchdog never treats an intentional ``gateway stop`` as a crash and exits once
the gateway service is uninstalled.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
import traceback
from pathlib import Path


_LOG_MAX_LINES = 1000
_DEFAULT_INTERVAL_SECONDS = 120.0


def _resolve_log_path() -> Path:
    """Return the watchdog log path under the current HERMES_HOME.

    Imported lazily so a missing/broken import doesn't crash the watchdog.
    Falls back to ~/.hermes/logs/ if HERMES_HOME is unavailable.
    """
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    except Exception:
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    log_dir = Path(home) / "logs"
    return log_dir / "gateway-watchdog.log"


def _log(msg: str) -> None:
    """Append a timestamped line to the watchdog log (best-effort).

    A logging failure must never crash the watchdog.
    """
    try:
        log_path = _resolve_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def _truncate_log() -> None:
    """Keep the watchdog log bounded (last _LOG_MAX_LINES lines).

    Cheap per-invocation maintenance — file is small and reads sequentially.
    """
    try:
        log_path = _resolve_log_path()
        if not log_path.exists():
            return
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        if len(lines) <= _LOG_MAX_LINES:
            return
        tail = lines[-_LOG_MAX_LINES :]
        tmp = log_path.with_suffix(log_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.writelines(tail)
        tmp.replace(log_path)
    except Exception:
        pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="hermes_cli.gateway_watchdog",
        description="Windows watchdog for automatic gateway respawn on crash.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously for the Startup-folder fallback.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=_DEFAULT_INTERVAL_SECONDS,
        help="Seconds between checks in loop mode.",
    )
    return parser.parse_args(argv)


def _gateway_is_alive() -> tuple[bool, int | None]:
    """Probe whether a gateway is running for the current HERMES_HOME.

    Returns (alive, pid). When alive is True, pid is the running gateway PID.
    When alive is False, pid is None.

    Conservative on errors: if we can't import probes, assume gateway is alive
    (don't force-respawn on import failures).
    """
    try:
        from gateway.status import get_running_pid
    except Exception as exc:
        _log(f"probe import failure: {exc!r}")
        return (True, None)

    try:
        pid = get_running_pid(cleanup_stale=False)
    except Exception as exc:
        _log(f"get_running_pid raised: {exc!r}")
        return (True, None)
    return (pid is not None, pid)


def _respawn() -> int | None:
    """Spawn a fresh detached gateway using CREATE_BREAKAWAY_FROM_JOB.

    Return the PID on success, None on failure.

    The respawn uses gateway_windows._spawn_detached() which applies
    CREATE_BREAKAWAY_FROM_JOB so the new gateway survives if the watchdog
    or its parent job object dies.
    """
    try:
        from hermes_cli import gateway_windows
    except Exception as exc:
        _log(f"gateway_windows import failure: {exc!r}")
        return None
    try:
        return gateway_windows._spawn_detached()
    except Exception as exc:
        _log(f"_spawn_detached raised: {exc!r}\n{traceback.format_exc()}")
        return None


def _service_is_installed() -> bool:
    """Return whether this profile still has gateway persistence installed."""
    try:
        from hermes_cli import gateway_windows

        return bool(gateway_windows.is_installed())
    except Exception as exc:
        _log(f"service install probe failed: {exc!r}")
        return False


def _was_intentionally_stopped() -> bool:
    """Return whether the operator explicitly stopped the gateway."""
    try:
        from gateway.status import read_runtime_status

        status = read_runtime_status() or {}
    except Exception as exc:
        _log(f"runtime status probe failed: {exc!r}")
        return True
    return status.get("gateway_state") == "stopped"


def _run_once() -> bool:
    """Run one health check.

    Returns ``False`` only when the service is no longer installed, which tells
    the Startup-folder loop to exit. All other outcomes keep the loop alive.
    """
    if not _service_is_installed():
        _log("gateway service is not installed; watchdog exiting")
        return False

    alive, _pid = _gateway_is_alive()
    if alive:
        return True
    if _was_intentionally_stopped():
        return True

    # Recheck immediately before spawning to close the common manual-start race.
    alive, _pid = _gateway_is_alive()
    if alive:
        return True

    _log("gateway is down; respawning")
    new_pid = _respawn()
    if new_pid is None:
        _log("respawn failed; will retry on next tick")
    else:
        _log(f"respawned gateway pid={new_pid}")
    return True


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for pythonw -m hermes_cli.gateway_watchdog.

    Always returns 0 so the schtasks parent never marks the task as failed.
    """
    try:
        args = _parse_args(argv)
    except SystemExit:
        # argparse exits for --help or bad args.
        # Return 0 so the schtasks parent does not mark the task as failed.
        return 0
    except Exception as exc:
        _log(f"argparse raised: {exc!r}")
        return 0

    _truncate_log()

    interval = max(float(args.interval), 1.0)
    while True:
        try:
            keep_running = _run_once()
        except Exception as exc:
            _log(f"watchdog uncaught: {exc!r}\n{traceback.format_exc()}")
            keep_running = True
        if not args.loop or not keep_running:
            break
        time.sleep(interval)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
