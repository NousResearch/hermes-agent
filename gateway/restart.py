"""Shared gateway restart constants and parsing helpers."""

import os
import shlex
import signal
import sys
import subprocess

from hermes_cli.config import DEFAULT_CONFIG

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


def parse_restart_drain_timeout(raw: object) -> float:
    """Parse a configured drain timeout, falling back to the shared default."""
    try:
        value = float(raw) if str(raw or "").strip() else DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    return max(0.0, value)


def _service_names() -> tuple[str, str]:
    from hermes_cli.gateway import get_launchd_label, get_service_name

    return get_service_name(), get_launchd_label()


def is_gateway_restart_command(command: str) -> bool:
    """Return whether *command* is an exact gateway restart command."""
    if not isinstance(command, str) or any(char in command for char in ";|&$`()<>"):
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    if parts == ["hermes", "gateway", "restart"]:
        return True
    if not parts:
        return False

    service_name, launchd_label = _service_names()
    executable = parts[0].rsplit("/", 1)[-1]
    if executable == "systemctl" and "restart" in parts:
        action = parts.index("restart")
        return parts[action + 1:] in ([service_name], [f"{service_name}.service"])
    if executable == "launchctl" and ("restart" in parts or "kickstart" in parts):
        action = "restart" if "restart" in parts else "kickstart"
        target = parts[parts.index(action) + 1:]
        return len(target) == 1 and target[0].endswith(f"/{launchd_label}")
    return False


def _is_cron_session() -> bool | None:
    try:
        from gateway.session_context import get_session_env

        return bool(get_session_env("HERMES_CRON_SESSION", ""))
    except Exception:
        return None


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


def request_approved_gateway_restart_handoff() -> tuple[bool, str]:
    """Deliver a restart request to the existing supervised gateway handler."""
    cron_session = _is_cron_session()
    if cron_session is not False:
        return False, "Blocked: cron or unknown session context cannot restart the gateway."
    if sys.platform not in {"darwin", "linux"} or not hasattr(signal, "SIGUSR1"):
        return False, "Gateway self-restart handoff is supported only by launchd or systemd."
    if not _is_current_supervisor():
        return False, "Blocked: gateway self-restart requires its active launchd or systemd service."

    try:
        os.kill(os.getpid(), signal.SIGUSR1)
    except OSError as exc:
        return False, f"Could not deliver gateway restart request: {exc}"
    return True, "Gateway restart request delivered to the supervised gateway."
