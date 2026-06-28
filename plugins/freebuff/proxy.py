"""Manage the local OpenAI-compatible Freebuff proxy (freebuff2api)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from . import token as token_bridge

FREEBUFF2API_GIT_SHA = "1fc766db9fcbba6a20c06661168c7d280d6275e2"
FREEBUFF2API_GIT_URL = (
    f"git+https://github.com/XxxXTeam/freebuff2api.git@{FREEBUFF2API_GIT_SHA}"
)
DEFAULT_PROXY_HOST = "127.0.0.1"
DEFAULT_PROXY_PORT = 8765


def _plugin_section() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:
        cfg = {}
    plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
    section = plugins.get("freebuff") if isinstance(plugins.get("freebuff"), dict) else {}
    return dict(section)


def proxy_pid_path() -> Path:
    return get_hermes_home() / "state" / "freebuff-proxy.pid"


def proxy_log_path() -> Path:
    return get_hermes_home() / "logs" / "freebuff-proxy.log"


def proxy_host_port() -> tuple[str, int]:
    cfg = _plugin_section()
    host = str(cfg.get("proxy_host") or DEFAULT_PROXY_HOST).strip() or DEFAULT_PROXY_HOST
    try:
        port = int(cfg.get("proxy_port") or DEFAULT_PROXY_PORT)
    except (TypeError, ValueError):
        port = DEFAULT_PROXY_PORT
    return host, port


def proxy_base_url() -> str:
    host, port = proxy_host_port()
    return f"http://{host}:{port}/v1"


def _read_env_key(name: str) -> str:
    return token_bridge._read_env_key(name)


def _proxy_installed() -> bool:
    try:
        import freebuff2api  # noqa: F401

        return True
    except ImportError:
        return False


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                return bool(ok and exit_code.value == STILL_ACTIVE)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok — POSIX-only fallback after nt branch above
        return True
    except OSError:
        return False


def proxy_status() -> dict[str, Any]:
    host, port = proxy_host_port()
    pid_path = proxy_pid_path()
    pid: int | None = None
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = None
    running = bool(pid and _pid_alive(pid))
    probe = probe_proxy() if running else {"ok": False, "error": "proxy not running"}
    return {
        "installed": _proxy_installed(),
        "running": running,
        "pid": pid,
        "host": host,
        "port": port,
        "base_url": proxy_base_url(),
        "log_path": str(proxy_log_path()),
        "git_sha": FREEBUFF2API_GIT_SHA,
        "probe": probe,
        "upstream_token_set": bool(token_bridge.resolve_upstream_token()),
        "proxy_api_key_set": bool(_read_env_key(token_bridge.ENV_PROXY_KEY)),
    }


def probe_proxy(*, timeout: float = 4.0) -> dict[str, Any]:
    host, port = proxy_host_port()
    local_key = _read_env_key(token_bridge.ENV_PROXY_KEY)
    headers = {"Accept": "application/json"}
    if local_key:
        headers["Authorization"] = f"Bearer {local_key}"
    url = f"http://{host}:{port}/healthz"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(body) if body.strip() else {}
            return {"ok": resp.status == 200, "status_code": resp.status, "body": payload}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        return {"ok": False, "status_code": exc.code, "error": detail or exc.reason}
    except Exception as exc:
        return {"ok": False, "status_code": None, "error": str(exc)}


def install_proxy(*, timeout: float = 300.0) -> dict[str, Any]:
    if _proxy_installed():
        return {"ok": True, "skipped": True, "action": "already_installed"}
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"freebuff2api @ {FREEBUFF2API_GIT_URL}",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"pip install timed out after {timeout}s", "cmd": cmd}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cmd": cmd}
    return {
        "ok": proc.returncode == 0 and _proxy_installed(),
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-1200:],
        "stderr_tail": (proc.stderr or "")[-1200:],
        "cmd": cmd,
        "installed": _proxy_installed(),
    }


def stop_proxy() -> dict[str, Any]:
    pid_path = proxy_pid_path()
    if not pid_path.is_file():
        return {"ok": True, "action": "not_running"}
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        pid_path.unlink(missing_ok=True)
        return {"ok": True, "action": "stale_pid_removed"}
    if _pid_alive(pid):
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
    pid_path.unlink(missing_ok=True)
    return {"ok": True, "action": "stopped", "pid": pid}


def start_proxy(*, force_restart: bool = False) -> dict[str, Any]:
    current = proxy_status()
    if current.get("running") and not force_restart:
        return {"ok": True, "already_running": True, **current}

    if force_restart:
        stop_proxy()

    upstream = token_bridge.resolve_upstream_token()
    if not upstream:
        sync = token_bridge.sync_upstream_token_to_env()
        if not sync.get("ok"):
            return sync
        upstream = token_bridge.resolve_upstream_token()
    if not upstream:
        return {
            "ok": False,
            "error": "Missing upstream Freebuff token (FREEBUFF_TOKEN or credentials.json)",
        }

    key_result = token_bridge.ensure_proxy_api_key()
    if not key_result.get("ok"):
        return key_result
    local_key = _read_env_key(token_bridge.ENV_PROXY_KEY)
    if not local_key:
        return {"ok": False, "error": "Failed to resolve FREEBUFF_PROXY_API_KEY"}

    if not _proxy_installed():
        installed = install_proxy()
        if not installed.get("ok"):
            return {
                "ok": False,
                "error": "freebuff2api is not installed",
                "install": installed,
            }

    host, port = proxy_host_port()
    log_path = proxy_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path = proxy_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["FREEBUFF_TOKEN"] = upstream
    env["FREEBUFF_API_KEY"] = local_key
    env["FREEBUFF_HOST"] = host
    env["FREEBUFF_PORT"] = str(port)
    env["FREEBUFF_LOG_LEVEL"] = str(env.get("FREEBUFF_LOG_LEVEL") or "INFO")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "freebuff2api.app:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    log_handle = open(log_path, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(get_hermes_home()),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    except OSError as exc:
        log_handle.close()
        return {"ok": False, "error": str(exc), "command": cmd}

    pid_path.write_text(str(proc.pid), encoding="utf-8")

    deadline = time.monotonic() + 25.0
    last_probe: dict[str, Any] = {"ok": False, "error": "starting"}
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log_handle.close()
            return {
                "ok": False,
                "error": f"proxy exited early with code {proc.returncode}",
                "log_path": str(log_path),
                "command": cmd,
            }
        last_probe = probe_proxy(timeout=2.0)
        if last_probe.get("ok"):
            log_handle.close()
            return {
                "ok": True,
                "pid": proc.pid,
                "base_url": proxy_base_url(),
                "probe": last_probe,
                "log_path": str(log_path),
                "command": cmd,
            }
        time.sleep(0.6)

    log_handle.close()
    return {
        "ok": False,
        "error": "proxy started but health check did not pass in time",
        "pid": proc.pid,
        "probe": last_probe,
        "log_path": str(log_path),
        "command": cmd,
    }
