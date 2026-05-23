"""Local Hypura Harness daemon management for the Hermes CLI."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from hermes_cli.config import load_config
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18794
HEALTH_CHECK_PATH = "/health"
LEGACY_STATUS_PATH = "/status"
LEGACY_STATUS_TIMEOUT = 5.0
_FALSE_ENV_VALUES = {"0", "false", "no", "off", "disabled"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAEMON_SCRIPT = (
    PROJECT_ROOT
    / "vendor"
    / "openclaw-mirror"
    / "extensions"
    / "hypura-harness"
    / "scripts"
    / "harness_daemon.py"
)


def _env_flag_disabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _FALSE_ENV_VALUES


def _auto_start_disabled_by_env() -> bool:
    return _env_flag_disabled("HYPURA_HARNESS_AUTO_START") or _env_flag_disabled(
        "HERMES_HARNESS_AUTO_START"
    )


def _harness_config() -> dict[str, Any]:
    try:
        config = load_config()
    except Exception:
        logger.debug("Failed to load config for harness settings", exc_info=True)
        return {}

    harness_cfg = config.get("harness", {})
    if not isinstance(harness_cfg, dict):
        return {}
    return harness_cfg


def _coerce_port(value: Any) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return DEFAULT_PORT
    if 1 <= port <= 65535:
        return port
    return DEFAULT_PORT


def get_harness_host() -> str:
    """Return the configured harness host."""
    host = os.getenv("HYPURA_HARNESS_HOST") or _harness_config().get("host")
    if not host:
        return DEFAULT_HOST
    return str(host).strip() or DEFAULT_HOST


def get_harness_port() -> int:
    """Return the configured harness port."""
    port = os.getenv("HYPURA_HARNESS_PORT") or _harness_config().get("port")
    return _coerce_port(port)


def get_harness_url() -> str:
    """Return the local harness base URL."""
    return f"http://{get_harness_host()}:{get_harness_port()}"


def get_harness_script_path() -> Path:
    """Return the configured daemon script path, or the bundled default."""
    script = os.getenv("HYPURA_HARNESS_SCRIPT") or _harness_config().get("script_path")
    if script:
        return Path(str(script)).expanduser().resolve()
    return DEFAULT_DAEMON_SCRIPT


def _harness_log_path() -> Path:
    logs_dir = get_hermes_home() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "harness.log"


def is_harness_running(timeout: float = 1.0) -> bool:
    """Check whether the local harness responds on its lightweight health endpoint."""
    base_url = get_harness_url()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{base_url}{HEALTH_CHECK_PATH}")
        if response.status_code == 200:
            return True
        if response.status_code != 404:
            return False
    except Exception:
        return False

    try:
        with httpx.Client(timeout=max(timeout, LEGACY_STATUS_TIMEOUT)) as client:
            response = client.get(f"{base_url}{LEGACY_STATUS_PATH}")
        return response.status_code == 200
    except Exception:
        return False


def _start_command(script_path: Path) -> list[str]:
    cfg = _harness_config()
    configured_python = os.getenv("HYPURA_HARNESS_PYTHON") or cfg.get("python")
    if configured_python:
        return [str(configured_python), str(script_path)]

    if (script_path.parent / "pyproject.toml").is_file() and shutil.which("uv"):
        return ["uv", "run", "python", str(script_path)]

    return [sys.executable or "python", str(script_path)]


def _build_env(script_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    scripts_dir = str(script_path.parent)
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = scripts_dir + os.pathsep + current_pythonpath
    else:
        env["PYTHONPATH"] = scripts_dir
    return env


def start_harness_daemon(wait_seconds: float = 30.0) -> bool:
    """Start the local harness daemon in the background."""
    if is_harness_running():
        logger.info("Hypura Harness is already running at %s", get_harness_url())
        return True

    script_path = get_harness_script_path()
    if not script_path.is_file():
        logger.error("Harness daemon script not found: %s", script_path)
        return False

    cmd = _start_command(script_path)
    env = _build_env(script_path)
    log_path = _harness_log_path()
    creationflags = 0
    popen_kwargs: dict[str, Any] = {}
    if platform.system() == "Windows":
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        popen_kwargs["start_new_session"] = True

    try:
        log_file = log_path.open("ab")
    except Exception:
        logger.exception("Failed to open harness log file: %s", log_path)
        return False

    try:
        subprocess.Popen(
            cmd,
            cwd=str(script_path.parent),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            creationflags=creationflags,
            **popen_kwargs,
        )
    except Exception:
        logger.exception("Failed to launch harness daemon")
        return False
    finally:
        log_file.close()

    deadline = time.monotonic() + max(0.0, wait_seconds)
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if is_harness_running():
            logger.info("Hypura Harness started at %s", get_harness_url())
            return True

    logger.warning(
        "Harness daemon launched but did not respond within %.1f seconds; log: %s",
        wait_seconds,
        log_path,
    )
    return False


def _connection_port(conn: Any) -> int | None:
    laddr = getattr(conn, "laddr", None)
    if laddr is None:
        return None
    port = getattr(laddr, "port", None)
    if port is not None:
        return port
    try:
        return laddr[1] if len(laddr) > 1 else None
    except TypeError:
        return None


def _process_connections(proc):
    if hasattr(proc, "net_connections"):
        return proc.net_connections(kind="inet")
    return proc.connections(kind="inet")


def stop_harness_daemon(timeout: float = 3.0) -> bool:
    """Stop any process listening on the configured harness port."""
    was_running = is_harness_running()

    try:
        import psutil
    except Exception:
        logger.exception("psutil is required to stop the harness daemon")
        return not was_running

    port = get_harness_port()
    found_processes: list[Any] = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if any(_connection_port(conn) == port for conn in _process_connections(proc)):
                found_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    for proc in found_processes:
        try:
            logger.info("Stopping harness process %s", proc.pid)
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not found_processes:
        return not was_running

    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        live_processes = []
        for proc in found_processes:
            try:
                if proc.is_running():
                    live_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not live_processes and not is_harness_running():
            return True
        time.sleep(0.25)

    for proc in found_processes:
        try:
            if proc.is_running():
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return not is_harness_running()


def restart_harness_daemon() -> bool:
    """Restart the harness daemon."""
    stopped = stop_harness_daemon()
    time.sleep(1.0)
    return stopped and start_harness_daemon()


def ensure_harness_running() -> None:
    """Start the harness only when config/env explicitly allow auto-start."""
    if _auto_start_disabled_by_env():
        logger.debug("Hypura Harness auto-start disabled by environment.")
        return

    cfg = _harness_config()
    if not cfg.get("enabled", True):
        return

    if cfg.get("auto_start", True) and not is_harness_running():
        logger.info("Auto-starting Hypura Harness...")
        start_harness_daemon()


def _print_harness_help() -> None:
    print("Usage: hermes harness <start|stop|restart|status>")


def harness_command(args) -> int:
    """Route `hermes harness` subcommands."""
    action = getattr(args, "harness_action", None)
    if action is None:
        _print_harness_help()
        return 2

    url = get_harness_url()
    script_path = get_harness_script_path()

    if action == "status":
        if is_harness_running():
            print(f"Hypura Harness: ONLINE at {url}")
            return 0
        print(f"Hypura Harness: OFFLINE at {url}")
        if not script_path.is_file():
            print(f"Daemon script not found: {script_path}")
        return 1

    if action == "start":
        print(f"Starting Hypura Harness at {url}...")
        if start_harness_daemon():
            print("Hypura Harness started.")
            return 0
        print("Failed to start Hypura Harness.")
        print(f"Daemon script: {script_path}")
        print(f"Log file: {_harness_log_path()}")
        return 1

    if action == "stop":
        print("Stopping Hypura Harness...")
        if stop_harness_daemon():
            print("Hypura Harness stopped.")
            return 0
        print("Failed to stop Hypura Harness.")
        return 1

    if action == "restart":
        print("Restarting Hypura Harness...")
        if restart_harness_daemon():
            print("Hypura Harness restarted.")
            return 0
        print("Failed to restart Hypura Harness.")
        print(f"Daemon script: {script_path}")
        print(f"Log file: {_harness_log_path()}")
        return 1

    _print_harness_help()
    return 2


def _run_harness_command(args) -> None:
    raise SystemExit(harness_command(args))


def register_harness_subparser(subparsers) -> Any:
    """Register the top-level `hermes harness` command."""
    parser = subparsers.add_parser(
        "harness",
        help="Manage the local Hypura Harness daemon",
        description="Start, stop, restart, or check the local Hypura Harness daemon.",
    )
    harness_sub = parser.add_subparsers(dest="harness_action")
    harness_sub.add_parser("start", help="Start the harness daemon")
    harness_sub.add_parser("stop", help="Stop the harness daemon")
    harness_sub.add_parser("restart", help="Restart the harness daemon")
    harness_sub.add_parser("status", help="Show harness status")
    parser.set_defaults(func=_run_harness_command)
    return parser
