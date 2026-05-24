"""Hermes native Hypura/VRChat/Ollama integration commands."""

from __future__ import annotations

import json
import os
import psutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_HYPURA_URL = "http://127.0.0.1:18794"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _hypura_script_path() -> Path:
    raw = os.getenv("HYPURA_HARNESS_SCRIPT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    from hermes_cli.harness import get_harness_script_path

    return get_harness_script_path()


def _pid_file() -> Path:
    hermes_home = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
    hermes_home.mkdir(parents=True, exist_ok=True)
    return hermes_home / "hypura-daemon.json"


def _log_file() -> Path:
    hermes_home = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
    hermes_home.mkdir(parents=True, exist_ok=True)
    return hermes_home / "hypura-daemon.log"


def _base_url() -> str:
    return os.getenv("HYPURA_HARNESS_URL", DEFAULT_HYPURA_URL).strip().rstrip("/")


def _http_json(path: str, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    url = f"{_base_url()}{path}"
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=body, method=method, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
            return json.loads(text) if text else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Hypura HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Hypura unreachable at {url}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Hypura request failed at {url}: {exc}") from exc


def _start_background(command: list[str], cwd: Path) -> int:
    log = _log_file().open("a", encoding="utf-8")
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
        proc = subprocess.Popen(command, cwd=str(cwd), creationflags=flags, stdout=log, stderr=log)
    else:
        proc = subprocess.Popen(command, cwd=str(cwd), start_new_session=True, stdout=log, stderr=log)
    return proc.pid


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    return psutil.pid_exists(pid)


def _write_pid(pid: int, command: list[str]) -> None:
    _pid_file().write_text(
        json.dumps({"pid": pid, "command": command}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_pid() -> dict[str, Any] | None:
    p = _pid_file()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def hypura_command(args: Any) -> None:
    action = getattr(args, "hypura_action", None)
    if action == "status":
        pid_info = _read_pid()
        daemon_meta = None
        if pid_info:
            daemon_meta = {
                "pid": pid_info.get("pid"),
                "alive": _is_pid_alive(int(pid_info.get("pid", 0))),
                "command": pid_info.get("command", []),
                "log_file": str(_log_file()),
            }
        try:
            status = _http_json("/status", "GET", timeout=10)
            if daemon_meta:
                status["_local_daemon"] = daemon_meta
            _print_json(status)
        except RuntimeError as exc:
            payload: dict[str, Any] = {"error": str(exc)}
            if daemon_meta:
                payload["_local_daemon"] = daemon_meta
            _print_json(payload)
            raise SystemExit(1)
        return

    if action == "start-daemon":
        script = _hypura_script_path()
        if not script.exists():
            raise SystemExit(f"Hypura daemon script not found: {script}")
        cmd = [sys.executable, str(script)]
        pid = _start_background(cmd, script.parent)
        _write_pid(pid, cmd)
        time.sleep(1.0)
        alive = _is_pid_alive(pid)
        if alive:
            print(f"Hypura daemon started (pid={pid})")
            print(f"URL: {_base_url()}")
            print(f"Log: {_log_file()}")
        else:
            print("Hypura daemon exited immediately.")
            print(f"Check log: {_log_file()}")
            raise SystemExit(1)
        return

    if action == "osc":
        payload = {"action": args.action, "payload": json.loads(args.payload) if args.payload else {}}
        _print_json(_http_json("/osc", "POST", payload, timeout=30))
        return

    if action == "speak":
        payload: dict[str, Any] = {"text": args.text}
        if args.emotion:
            payload["emotion"] = args.emotion
        _print_json(_http_json("/speak", "POST", payload, timeout=60))
        return

    if action == "run":
        payload = {"task": args.task, "model": args.model, "max_retries": args.max_retries}
        _print_json(_http_json("/run", "POST", payload, timeout=600))
        return

    if action == "scientist-run":
        payload = {
            "topic": args.topic or "",
            "num_ideas": args.num_ideas,
            "run_experiment": bool(args.run_experiment),
            "model": args.model,
        }
        _print_json(_http_json("/scientist/run", "POST", payload, timeout=600))
        return

    if action == "vrchat-chatbox":
        payload = {"action": "chatbox", "payload": {"text": args.message, "immediate": True, "sfx": False}}
        _print_json(_http_json("/osc", "POST", payload, timeout=30))
        return

    if action == "ollama-start":
        cmd = ["ollama", "serve"]
        pid = _start_background(cmd, _project_root())
        print(f"Ollama serve started (pid={pid})")
        if args.pull_model:
            subprocess.run(["ollama", "pull", args.pull_model], check=False)
        return

    if action == "ollama-status":
        url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/") + "/api/tags"
        req = Request(url, method="GET")
        with urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
            print(text)
        return

    raise SystemExit(
        "Usage: hermes hypura <status|start-daemon|osc|speak|run|scientist-run|vrchat-chatbox|ollama-start|ollama-status>"
    )
