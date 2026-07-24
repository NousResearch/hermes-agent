"""Local OpenViking server process management.

This module deliberately owns only process mechanics. Provisioning belongs to
``quick_local.py`` and connection/profile semantics remain with the provider.
Keeping the original ``Popen`` handle is the sole ownership proof: configured
endpoints and deployment markers never authorize Hermes to stop a process.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from hermes_cli._subprocess_compat import (
    windows_detach_flags,
    windows_detach_flags_without_breakaway,
)

__all__ = [
    "AUTOSTART_TIMEOUT_SECONDS",
    "DEFAULT_PORT",
    "LocalServerStartResult",
    "PROCESS_STOP_TIMEOUT_SECONDS",
    "defer_owned_shutdown",
    "endpoint_bind",
    "resolve_server_config_path",
    "server_command",
    "start_local_server",
    "stop_owned_process",
    "wait_for_health",
]

logger = logging.getLogger(__name__)

DEFAULT_PORT = 1933
AUTOSTART_TIMEOUT_SECONDS = 60.0
PROCESS_STOP_TIMEOUT_SECONDS = 10.0
TASK_WAIT_TIMEOUT_SECONDS = 300.0
TASK_POLL_INTERVAL_SECONDS = 0.5

_SERVER_LOG_RELATIVE_PATH = Path("logs") / "openviking-server.log"
_REAPER_LOG_RELATIVE_PATH = Path("logs") / "openviking-reaper.log"


@dataclass(frozen=True)
class LocalServerStartResult:
    """Result of starting one local OpenViking child process."""

    process: Optional[subprocess.Popen]
    message: str

    @property
    def started(self) -> bool:
        return self.process is not None


def resolve_server_config_path(
    *,
    env: Optional[dict[str, str]] = None,
    home: Optional[Path] = None,
) -> Path:
    """Resolve the separately managed OpenViking server configuration."""

    env_values = os.environ if env is None else env
    configured = env_values.get("OPENVIKING_CONFIG_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    return (Path.home() if home is None else home) / ".openviking" / "ov.conf"


def server_command() -> Optional[str]:
    """Return the installed OpenViking server executable, if available."""

    discovered = shutil.which("openviking-server")
    if discovered:
        return discovered
    executable_name = "openviking-server.exe" if is_windows() else "openviking-server"
    alongside_python = Path(sys.executable).parent / executable_name
    if alongside_python.is_file():
        return str(alongside_python)
    return None


def endpoint_bind(endpoint: str) -> tuple[str, int]:
    """Return the host/port a local server should bind for an endpoint."""

    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    return parsed.hostname or "127.0.0.1", parsed.port or DEFAULT_PORT


def start_detached_process(
    command: list[str],
    *,
    stdin,
    **kwargs,
) -> subprocess.Popen:
    """Start a child using Hermes' established cross-platform conventions."""

    if not is_windows():
        return subprocess.Popen(
            command,
            stdin=stdin,
            start_new_session=True,
            **kwargs,
        )

    try:
        return subprocess.Popen(
            command,
            stdin=stdin,
            creationflags=windows_detach_flags(),
            **kwargs,
        )
    except OSError:
        # Some Windows parents are already inside a job object that rejects
        # CREATE_BREAKAWAY_FROM_JOB. Match the gateway's proven fallback.
        return subprocess.Popen(
            command,
            stdin=stdin,
            creationflags=windows_detach_flags_without_breakaway(),
            **kwargs,
        )


def start_local_server(
    endpoint: str,
    *,
    hermes_home: Path,
    config_path: Optional[Path] = None,
) -> LocalServerStartResult:
    """Start OpenViking for a local endpoint and return exact ownership."""

    executable = server_command()
    if not executable:
        return LocalServerStartResult(
            None,
            "openviking-server was not found. Install OpenViking, then retry.",
        )

    try:
        host, port = endpoint_bind(endpoint)
    except ValueError as exc:
        return LocalServerStartResult(
            None,
            f"Could not parse local OpenViking URL: {exc}",
        )

    log_path = hermes_home / _SERVER_LOG_RELATIVE_PATH
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [executable]
        if config_path is not None:
            command.extend(["--config", str(config_path)])
        command.extend(["--host", host, "--port", str(port)])
        with log_path.open("ab") as log_file:
            process = start_detached_process(
                command,
                stdout=log_file,
                stderr=log_file,
                stdin=subprocess.DEVNULL,
            )
    except Exception as exc:
        return LocalServerStartResult(
            None,
            f"Could not start openviking-server: {exc}",
        )

    return LocalServerStartResult(
        process,
        f"Started openviking-server on {host}:{port} in the background. Logs: {log_path}",
    )


def wait_for_health(
    endpoint: str,
    health_check: Callable[[str], tuple[bool, str]],
    *,
    timeout_seconds: float = AUTOSTART_TIMEOUT_SECONDS,
    cancel_event=None,
    should_stop=None,
    monotonic=None,
    sleep=None,
) -> bool:
    """Poll a caller-supplied health check until success, timeout, or cancel."""

    if monotonic is None or sleep is None:
        import time

        monotonic = monotonic or time.monotonic
        sleep = sleep or time.sleep

    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        if (
            (cancel_event is not None and cancel_event.is_set())
            or (should_stop is not None and should_stop())
        ):
            return False
        healthy, _message = health_check(endpoint)
        if healthy:
            return True
        sleep(0.5)
    return False


def stop_owned_process(
    process: subprocess.Popen,
    *,
    timeout_seconds: float = PROCESS_STOP_TIMEOUT_SECONDS,
) -> bool:
    """Stop exactly one child process owned through its ``Popen`` handle."""

    try:
        if process.poll() is not None:
            return True
        process.terminate()
        process.wait(timeout=timeout_seconds)
        return True
    except subprocess.TimeoutExpired:
        logger.warning(
            "Owned OpenViking server did not stop within %.1f seconds; force-stopping it",
            timeout_seconds,
        )
    except Exception as exc:
        logger.warning("Could not stop owned OpenViking server cleanly: %s", exc)
        return False

    try:
        process.kill()
        process.wait(timeout=timeout_seconds)
        return True
    except subprocess.TimeoutExpired:
        logger.error(
            "Owned OpenViking server still did not exit after force-stop within %.1f seconds",
            timeout_seconds,
        )
        return False
    except Exception as exc:
        logger.error("Could not force-stop owned OpenViking server: %s", exc)
        return False


def defer_owned_shutdown(
    process: subprocess.Popen,
    *,
    hermes_home: Path,
    endpoint: str,
    headers: dict[str, str],
    task_ids: set[str],
) -> bool:
    """Hand exact ownership to the detached, work-aware shutdown helper."""

    reaper: Optional[subprocess.Popen] = None
    try:
        import psutil

        create_time = psutil.Process(process.pid).create_time()
        log_path = hermes_home / _REAPER_LOG_RELATIVE_PATH
        log_path.parent.mkdir(parents=True, exist_ok=True)
        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": None,
            "stderr": None,
            "close_fds": True,
        }
        reaper_command = [sys.executable, str(Path(__file__).with_name("reaper.py"))]
        if is_windows():
            from hermes_cli.gateway_windows import _resolve_detached_python

            pythonw, _venv_dir, extra_pythonpath = _resolve_detached_python(
                sys.executable
            )
            reaper_command[0] = pythonw
            if extra_pythonpath:
                env = dict(os.environ)
                pythonpath_entries = list(extra_pythonpath)
                if env.get("PYTHONPATH"):
                    pythonpath_entries.append(env["PYTHONPATH"])
                env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
                popen_kwargs["env"] = env

        payload = {
            "endpoint": endpoint,
            "pid": process.pid,
            "create_time": create_time,
            "task_ids": sorted(task_ids),
            "headers": headers,
            "wait_timeout": TASK_WAIT_TIMEOUT_SECONDS,
            "poll_interval": TASK_POLL_INTERVAL_SECONDS,
            "stop_timeout": PROCESS_STOP_TIMEOUT_SECONDS,
        }
        with log_path.open("ab") as log_file:
            popen_kwargs["stdout"] = log_file
            popen_kwargs["stderr"] = log_file
            reaper = start_detached_process(reaper_command, **popen_kwargs)
            if reaper.stdin is None:
                raise RuntimeError("shutdown helper stdin was not created")
            reaper.stdin.write(json.dumps(payload).encode("utf-8"))
            reaper.stdin.close()
        return True
    except Exception as exc:
        if reaper is not None:
            try:
                if reaper.stdin is not None and not reaper.stdin.closed:
                    reaper.stdin.close()
                reaper.terminate()
            except Exception:
                pass
        logger.warning("Could not defer managed OpenViking shutdown: %s", exc)
        return False


def is_windows() -> bool:
    return os.name == "nt"
