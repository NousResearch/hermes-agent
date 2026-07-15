"""Shared gateway restart constants and bounded recovery helpers."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows imports this module but cannot use this handoff.
    fcntl = None

from hermes_cli.config import DEFAULT_CONFIG
from hermes_constants import get_hermes_home
from utils import atomic_json_write

# EX_TEMPFAIL from sysexits.h — used to ask the service manager to restart
# the gateway after a graceful drain/reload path completes.
GATEWAY_SERVICE_RESTART_EXIT_CODE = 75

# EX_CONFIG from sysexits.h — fatal configuration error (e.g. token
# collision, no messaging platforms).  The s6 finish script translates
# this into exit 125 (permanent failure) so the supervisor stops
# restarting the gateway.  See #51228.
GATEWAY_FATAL_CONFIG_EXIT_CODE = 78

DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["restart_drain_timeout"]
)

# A resumed recovery turn must not immediately restart the gateway again.
GATEWAY_RECOVERY_RESTART_COOLDOWN_SECONDS = 300.0
_RECOVERY_RESTART_STATE_FILE = ".gateway_recovery_restart.json"
_RECOVERY_RESTART_LOCK_FILE = ".gateway_recovery_restart.lock"
_recovery_restart_lock = threading.Lock()


def parse_restart_drain_timeout(raw: object) -> float:
    """Parse a configured drain timeout, falling back to the shared default."""
    try:
        value = float(raw) if str(raw or "").strip() else DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    return max(0.0, value)


def is_gateway_restart_command(command: str) -> bool:
    """Return whether *command* is the literal supported recovery intent."""
    return isinstance(command, str) and command.strip(" ") == "hermes gateway restart"


def _is_cron_session() -> bool | None:
    try:
        from gateway.session_context import get_session_env

        return bool(get_session_env("HERMES_CRON_SESSION", ""))
    except Exception:
        return None


def _service_names() -> tuple[str, str]:
    from hermes_cli.gateway import get_launchd_label, get_service_name

    return get_service_name(), get_launchd_label()


def _is_current_supervisor() -> bool:
    service_name, launchd_label = _service_names()
    pid = str(os.getpid())
    if sys.platform == "linux":
        if not os.environ.get("INVOCATION_ID"):
            return False
        for scope in ([], ["--user"]):
            try:
                result = subprocess.run(
                    ["systemctl", *scope, "show", service_name, "--property=MainPID", "--value"],
                    capture_output=True, text=True, timeout=2, check=False,
                )
            except OSError:
                continue
            if result.returncode == 0 and result.stdout.strip() == pid:
                return True
        return False
    if sys.platform == "darwin" and os.environ.get("XPC_SERVICE_NAME", "0") == launchd_label:
        try:
            result = subprocess.run(
                ["launchctl", "list", launchd_label],
                capture_output=True, text=True, timeout=2, check=False,
            )
        except OSError:
            return False
        return result.returncode == 0 and result.stdout.split(maxsplit=1)[0:1] == [pid]
    return False


def _recovery_state_path() -> Path:
    return get_hermes_home() / _RECOVERY_RESTART_STATE_FILE


def _recovery_lock_path() -> Path:
    return get_hermes_home() / _RECOVERY_RESTART_LOCK_FILE


@contextmanager
def _locked_recovery_state():
    """Serialize admission across gateway threads and sibling processes."""
    with _recovery_restart_lock:
        lock_path = _recovery_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_recovery_state() -> dict[str, object]:
    try:
        with _recovery_state_path().open(encoding="utf-8") as state_file:
            state = json.load(state_file)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _is_cooldown_active(state: dict[str, object], session_id: str, now: float) -> bool:
    if state.get("session_id") != session_id or state.get("outcome") != "scheduled":
        return False
    try:
        scheduled_at = float(state["scheduled_at"])
    except (KeyError, TypeError, ValueError):
        return True  # Malformed successful state fails closed rather than looping.
    return now - scheduled_at < GATEWAY_RECOVERY_RESTART_COOLDOWN_SECONDS


def request_gateway_recovery_restart(
    *, session_id: str | None, task_id: str | None, reason: str = "agent_recovery"
) -> tuple[bool, str]:
    """Schedule one audited, supervised recovery restart for an active session.

    This receives metadata from the terminal tool invocation, not from prompt
    text. Only the literal canonical command can reach this helper.
    """
    if not isinstance(session_id, str) or not session_id:
        return False, "Blocked: gateway recovery restart requires an active session."
    if _is_cron_session() is not False:
        return False, "Blocked: cron or unknown session context cannot restart the gateway."
    if sys.platform not in {"darwin", "linux"} or not hasattr(signal, "SIGUSR1"):
        return False, "Gateway recovery restart is supported only by launchd or systemd."
    if not _is_current_supervisor():
        return False, "Blocked: gateway recovery restart requires its active launchd or systemd service."

    now = time.time()
    state = {
        "reason": reason,
        "session_id": session_id,
        "task_id": task_id or None,
        "attempted_at": now,
        "outcome": "pending",
    }
    try:
        with _locked_recovery_state():
            if _is_cooldown_active(_load_recovery_state(), session_id, now):
                return False, "Blocked: gateway recovery restart cooldown is active for this session."
            atomic_json_write(_recovery_state_path(), state, indent=None, mode=0o600)
            try:
                os.kill(os.getpid(), signal.SIGUSR1)
            except OSError as exc:
                state["outcome"] = "delivery_failed"
                state["failure"] = str(exc)
                atomic_json_write(_recovery_state_path(), state, indent=None, mode=0o600)
                return False, f"Could not deliver gateway recovery restart request: {exc}"
            state["outcome"] = "scheduled"
            state["scheduled_at"] = time.time()
            atomic_json_write(_recovery_state_path(), state, indent=None, mode=0o600)
    except OSError as exc:
        return False, f"Could not record gateway recovery restart: {exc}"
    return True, "Gateway recovery restart accepted and scheduled through its supervisor."
