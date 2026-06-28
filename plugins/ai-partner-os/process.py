"""Process lifecycle helpers for AI Partner OS."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

STATE_NAME = "ai_partner_os_state.json"


def state_file() -> Path:
    return get_hermes_home() / STATE_NAME


def read_state() -> dict[str, Any]:
    path = state_file()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(payload: dict[str, Any]) -> None:
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def clear_state() -> None:
    try:
        state_file().unlink()
    except FileNotFoundError:
        pass


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        if os.name == "nt":
            return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok — POSIX-only fallback after psutil/nt branches above
        return True
    except OSError:
        return False


def start_exe(exe_path: Path, *, cwd: Path | None = None) -> dict[str, Any]:
    if not exe_path.is_file():
        return {"ok": False, "error": f"Executable not found: {exe_path}"}
    state = read_state()
    old_pid = int(state.get("pid") or 0)
    if pid_alive(old_pid):
        return {"ok": True, "already_running": True, "pid": old_pid, "exe": str(exe_path)}

    workdir = cwd or exe_path.parent
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        proc = subprocess.Popen(
            [str(exe_path)],
            cwd=str(workdir),
            creationflags=creationflags,
            close_fds=False,
        )
    except OSError as exc:
        return {"ok": False, "error": str(exc), "exe": str(exe_path)}

    payload = {
        "pid": proc.pid,
        "exe": str(exe_path),
        "cwd": str(workdir),
        "started_at": time.time(),
    }
    write_state(payload)
    return {"ok": True, "pid": proc.pid, "exe": str(exe_path), "cwd": str(workdir)}


def stop_exe(*, force: bool = False) -> dict[str, Any]:
    state = read_state()
    pid = int(state.get("pid") or 0)
    if not pid_alive(pid):
        clear_state()
        return {"ok": True, "stopped": False, "reason": "not_running"}

    try:
        import psutil  # type: ignore

        proc = psutil.Process(pid)
        if force:
            proc.kill()
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except ImportError:
        if os.name == "nt":
            cmd = ["taskkill", "/PID", str(pid), "/T"]
            if force:
                cmd.append("/F")
            subprocess.run(cmd, check=False)
        else:
            os.kill(pid, 9 if force else 15)
    except Exception as exc:
        if force and os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
        else:
            return {"ok": False, "error": str(exc), "pid": pid}

    clear_state()
    return {"ok": True, "stopped": True, "pid": pid, "force": force}
