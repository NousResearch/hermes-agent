from __future__ import annotations

import json
import os
import platform
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hermes_constants import get_hermes_home


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8888
DEFAULT_WAIT_SECONDS = 3.0


STATUS_SCHEMA = {
    "description": "Report local Unsloth Studio launcher and server status.",
    "type": "object",
    "properties": {
        "host": {"type": "string", "default": DEFAULT_HOST},
        "port": {"type": "integer", "default": DEFAULT_PORT},
        "probe_url": {"type": "boolean", "default": True},
    },
}

START_SCHEMA = {
    "description": "Start Unsloth Studio with the local unsloth CLI.",
    "type": "object",
    "properties": {
        "host": {
            "type": "string",
            "description": "Bind host. Defaults to loopback. Public hosts require confirm_public_host.",
            "default": DEFAULT_HOST,
        },
        "port": {"type": "integer", "default": DEFAULT_PORT},
        "wait_seconds": {
            "type": "number",
            "description": "Seconds to wait for HTTP readiness after launch.",
            "default": DEFAULT_WAIT_SECONDS,
        },
        "cwd": {
            "type": "string",
            "description": "Optional working directory for the launcher.",
        },
        "extra_args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Additional arguments appended after unsloth studio -H HOST -p PORT.",
        },
        "confirm_public_host": {
            "type": "boolean",
            "description": "Required when host is not loopback.",
            "default": False,
        },
    },
}

STOP_SCHEMA = {
    "description": "Stop the Unsloth Studio process recorded by the Hermes plugin.",
    "type": "object",
    "properties": {
        "pid": {
            "type": "integer",
            "description": "Optional explicit process id. Defaults to the plugin state file.",
        }
    },
}

INSTALL_INFO_SCHEMA = {
    "description": "Return official Unsloth Studio installation and launch commands.",
    "type": "object",
    "properties": {
        "local_only": {
            "type": "boolean",
            "description": "Return only local install commands, not cloud notebook links.",
            "default": False,
        }
    },
}


def to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _unsloth_exe() -> str | None:
    return os.environ.get("UNSLOTH_STUDIO_BIN") or _which("unsloth")


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def state_file() -> Path:
    return get_hermes_home() / "unsloth_studio_state.json"


def _read_state() -> dict[str, Any]:
    path = state_file()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state(payload: dict[str, Any]) -> None:
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _clear_state() -> None:
    try:
        state_file().unlink()
    except FileNotFoundError:
        pass


def _is_windows() -> bool:
    return os.name == "nt"


def _is_loopback(host: str) -> bool:
    value = (host or "").strip().lower()
    return value in {"127.0.0.1", "localhost", "::1"} or value.startswith("127.")


def _display_url(host: str, port: int) -> str:
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in browser_host and not browser_host.startswith("["):
        browser_host = f"[{browser_host}]"
    return f"http://{browser_host}:{port}"


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        if _is_windows():
            return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok - POSIX-only fallback when psutil is unavailable.
        return True
    except OSError:
        return False


def _http_probe(url: str, timeout: float = 2.0) -> dict[str, Any]:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return {
                "ok": 200 <= response.status < 500,
                "status_code": response.status,
                "url": url,
            }
    except HTTPError as exc:
        return {"ok": exc.code < 500, "status_code": exc.code, "url": url, "error": str(exc)}
    except (OSError, URLError) as exc:
        return {"ok": False, "url": url, "error": str(exc)}


def _url_ready(url: str, wait_seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while time.monotonic() <= deadline:
        if _http_probe(url, timeout=1.0)["ok"]:
            return True
        time.sleep(0.25)
    return False


def check_available() -> bool:
    return bool(_unsloth_exe())


def status_payload(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    host = str(values.get("host") or DEFAULT_HOST)
    port = int(values.get("port") or DEFAULT_PORT)
    state = _read_state()
    pid = int(state.get("pid") or 0) if state else 0
    state_host = str(state.get("host") or host)
    state_port = int(state.get("port") or port)
    url = str(state.get("url") or _display_url(host, port))
    running = _pid_alive(pid)
    payload: dict[str, Any] = {
        "ok": bool(_unsloth_exe()),
        "available": bool(_unsloth_exe()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
        },
        "paths": {
            "unsloth": _unsloth_exe(),
            "state_file": str(state_file()),
        },
        "server": {
            "pid": pid or None,
            "running": running,
            "host": state_host,
            "port": state_port,
            "url": url,
        },
        "notes": [],
    }
    if not _unsloth_exe():
        payload["notes"].append("The unsloth CLI was not found on PATH.")
    if values.get("probe_url", True):
        payload["server"]["http"] = _http_probe(url)
    return payload


def _popen_kwargs(cwd: str | Path | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "cwd": str(cwd) if cwd else None,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if _is_windows():
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    else:
        kwargs["start_new_session"] = True
    return kwargs


def start_studio(values: dict[str, Any]) -> dict[str, Any]:
    exe = _unsloth_exe()
    if not exe:
        return {
            "ok": False,
            "error": "The unsloth CLI was not found on PATH.",
            "install": install_info({}),
        }

    host = str(values.get("host") or DEFAULT_HOST)
    port = int(values.get("port") or DEFAULT_PORT)
    if not _is_loopback(host) and not values.get("confirm_public_host"):
        return {
            "ok": False,
            "confirmation_required": True,
            "reason": "Binding Unsloth Studio outside loopback exposes a local training UI.",
        }

    url = _display_url(host, port)
    existing = _read_state()
    existing_pid = int(existing.get("pid") or 0) if existing else 0
    if _pid_alive(existing_pid):
        return {
            "ok": True,
            "already_running": True,
            "pid": existing_pid,
            "url": existing.get("url") or url,
        }

    command = [exe, "studio", "-H", host, "-p", str(port)]
    command.extend(str(arg) for arg in values.get("extra_args") or [])
    cwd = values.get("cwd")
    try:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, **_popen_kwargs(cwd))
    except OSError as exc:
        return {"ok": False, "error": str(exc), "command": command}

    payload = {
        "pid": proc.pid,
        "host": host,
        "port": port,
        "url": url,
        "command": command,
        "cwd": str(cwd) if cwd else None,
        "started_at": time.time(),
    }
    _write_state(payload)
    wait_seconds = float(values.get("wait_seconds") or DEFAULT_WAIT_SECONDS)
    ready = _url_ready(url, wait_seconds) if wait_seconds > 0 else False
    return {
        "ok": True,
        "pid": proc.pid,
        "url": url,
        "ready": ready,
        "command": command,
        "state_file": str(state_file()),
    }


def _terminate_pid(pid: int) -> dict[str, Any]:
    if not _pid_alive(pid):
        return {"ok": True, "already_stopped": True, "pid": pid}
    if _is_windows():
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        return {
            "ok": result.returncode == 0,
            "pid": pid,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    os.kill(pid, signal.SIGTERM)
    return {"ok": True, "pid": pid, "signal": "SIGTERM"}


def stop_studio(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    state = _read_state()
    pid = int(values.get("pid") or state.get("pid") or 0)
    if not pid:
        return {"ok": False, "error": "No Unsloth Studio pid was provided or recorded."}
    result = _terminate_pid(pid)
    if result.get("ok") and int(state.get("pid") or 0) == pid:
        _clear_state()
    return result


def install_info(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    system = platform.system().lower()
    if "windows" in system:
        install = "irm https://unsloth.ai/install.ps1 | iex"
        update = install
        developer = [
            "git clone https://github.com/unslothai/unsloth.git",
            "cd unsloth",
            "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass",
            ".\\install.ps1 --local",
            "unsloth studio -p 8888",
        ]
    else:
        install = "curl -fsSL https://unsloth.ai/install.sh | sh"
        update = install
        developer = [
            "git clone https://github.com/unslothai/unsloth",
            "cd unsloth",
            "./install.sh --local",
            "unsloth studio -p 8888",
        ]
    payload: dict[str, Any] = {
        "ok": True,
        "platform": platform.system(),
        "install": install,
        "update": update,
        "launch": f"unsloth studio -H {DEFAULT_HOST} -p {DEFAULT_PORT}",
        "developer_install": developer,
        "notes": [
            "The Hermes plugin does not execute remote install scripts automatically.",
            "Default launch binds to loopback; use confirm_public_host for a public bind.",
        ],
    }
    if not values.get("local_only"):
        payload["colab_notebook"] = (
            "https://colab.research.google.com/github/unslothai/unsloth/"
            "blob/main/studio/Unsloth_Studio_Colab.ipynb"
        )
    return payload


def handle_slash(raw_args: str) -> str:
    parts = shlex.split(raw_args or "")
    action = parts[0] if parts else "status"
    if action == "status":
        return to_json(status_payload({}))
    if action == "install-info":
        return to_json(install_info({}))
    if action == "start":
        return to_json(start_studio({}))
    if action == "stop":
        return to_json(stop_studio({}))
    return to_json(
        {
            "ok": False,
            "error": "Supported /unsloth-studio actions: status, start, stop, install-info.",
        }
    )
