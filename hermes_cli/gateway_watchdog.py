"""Windows gateway watchdog — auto-respawn crashed gateway for cron continuity.

On Windows, the gateway runs the cron scheduler as an internal thread. When the
gateway crashes, cron stops firing. This watchdog runs as a separate Scheduled
Task (every N minutes) and respawns the gateway if it detects it's down.

This is the Windows equivalent of systemd's Restart=always, ensuring cron jobs
continue to fire even after gateway crashes (via auto-respawn).

CLI:
    pythonw -m hermes_cli.gateway_watchdog [--profile NAME]

Exit codes:
    0   always (watchdog is best-effort; never crash the schtasks parent)

Logging:
    All actions are logged to $HERMES_HOME/logs/gateway-watchdog.log.
    The file is truncated to the last 1000 lines on each invocation.

Fixes: #41662 (Windows gateway cron scheduler circular dependency)
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import traceback
from pathlib import Path


_LOG_MAX_LINES = 1000


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
    parser.add_argument("--profile", default=None, help="Hermes profile name")
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


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for pythonw -m hermes_cli.gateway_watchdog.

    Always returns 0 so the schtasks parent never marks the task as failed.
    """
    try:
        _parse_args(argv)
    except SystemExit:
        # argparse exits with 2 on --help / bad args. Return 0 so the
        # schtasks parent doesn't mark the task as failed.
        return 0
    except Exception as exc:
        _log(f"argparse raised: {exc!r}")
        return 0

    _truncate_log()

    try:
        alive, pid = _gateway_is_alive()
        if alive:
            # Gateway is healthy. Stay quiet to keep the log small.
            return 0

        # Gateway is down. Attempt respawn.
        _log("gateway is down; respawning")
        new_pid = _respawn()
        if new_pid is None:
            _log("respawn failed; will retry on next tick")
        else:
            _log(f"respawned gateway pid={new_pid}")
    except Exception as exc:
        _log(f"watchdog uncaught: {exc!r}\n{traceback.format_exc()}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
