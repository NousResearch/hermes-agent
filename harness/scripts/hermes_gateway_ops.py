#!/usr/bin/env python3
"""Operações seguras no gateway Hermes: status, logs, restart.

Saída JSON no stdout. Logs são sanitizados — nunca expõe tokens.
Restart exige --confirm (aprovação humana explícita).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)using api key:\s*[^\n]+"), "Using API key: [REDACTED]"),
    (re.compile(r"\bsk-or-v1[^\s\n]*"), "sk-or-v1[REDACTED]"),
    (re.compile(r"\bsk-[a-zA-Z0-9._\-]+(\.\.\.[a-zA-Z0-9]+)?\b"), "sk-[REDACTED]"),
    (re.compile(r"\bghp_[a-zA-Z0-9]{20,}\b"), "ghp_[REDACTED]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-+/=]{20,}\b", re.I), "Bearer [REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*['\"]?[^\s'\",]{8,}"), r"\1=[REDACTED]"),
    (re.compile(r"\beyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9._-]+\b"), "eyJ[REDACTED]"),
]


def _sanitize(text: str) -> str:
    out = text
    for pattern, repl in _SECRET_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def _default_hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / "AppData" / "Local" / "hermes"


def _default_repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _find_hermes_cmd(repo: Path) -> list[str]:
    """Resolve argv prefix for hermes CLI — prefer repo venv over PATH."""
    candidates = [
        repo / ".venv" / "Scripts" / "hermes.exe",
        repo / ".venv" / "Scripts" / "python.exe",
        repo / "venv" / "Scripts" / "hermes.exe",
        repo / "venv" / "Scripts" / "python.exe",
        Path.home() / "AppData/Local/hermes/hermes-agent/venv/Scripts/hermes.exe",
        Path.home() / "AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe",
        Path.home() / "AppData/Local/hermes/hermes-agent/.venv/Scripts/hermes.exe",
        Path.home() / "AppData/Local/hermes/hermes-agent/.venv/Scripts/python.exe",
    ]
    for exe in candidates:
        if not exe.is_file():
            continue
        if exe.name == "python.exe":
            return [str(exe), "-m", "hermes_cli.main"]
        return [str(exe)]

    which = shutil.which("hermes")
    if which:
        return [which]
    return ["hermes"]


def _run_hermes(cmd_prefix: list[str], args: list[str], *, timeout: int = 120) -> tuple[int, str, str]:
    full = cmd_prefix + args
    try:
        proc = subprocess.run(
            full,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=os.environ.copy(),
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s: {' '.join(full)}"
    except OSError as exc:
        return 127, "", str(exc)


def _read_gateway_state(home: Path) -> dict | None:
    path = home / "gateway_state.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _read_pid(home: Path) -> int | None:
    path = home / "gateway.pid"
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        return int(raw)
    except (OSError, ValueError, IndexError):
        return None


def _windows_service_hint(home: Path) -> dict:
    svc_dir = home / "gateway-service"
    cmd_file = svc_dir / "Hermes_Gateway.cmd"
    return {
        "service_dir": str(svc_dir) if svc_dir.is_dir() else None,
        "launcher_cmd": str(cmd_file) if cmd_file.is_file() else None,
        "scheduled_task": "Hermes_Gateway",
    }


def _tail_log(home: Path, lines: int) -> tuple[str, str | None]:
    log_path = home / "logs" / "gateway.log"
    if not log_path.is_file():
        return "", f"log not found: {log_path}"
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return "", str(exc)
    chunk = "\n".join(text.splitlines()[-lines:])
    return _sanitize(chunk), None


def action_status(home: Path, cmd_prefix: list[str]) -> dict:
    code, stdout, stderr = _run_hermes(cmd_prefix, ["gateway", "status"], timeout=90)
    state = _read_gateway_state(home)
    pid = _read_pid(home)
    platforms: dict[str, str] = {}
    gateway_state = None
    if state:
        gateway_state = state.get("gateway_state")
        raw_platforms = state.get("platforms") or {}
        if isinstance(raw_platforms, dict):
            for name, info in raw_platforms.items():
                if isinstance(info, dict):
                    platforms[str(name)] = str(info.get("state", "unknown"))

    alive = pid is not None or gateway_state == "running" or code == 0 and "running" in stdout.lower()
    summary_parts = []
    if gateway_state:
        summary_parts.append(f"gateway_state={gateway_state}")
    if pid:
        summary_parts.append(f"pid={pid}")
    if platforms:
        connected = [k for k, v in platforms.items() if v == "connected"]
        summary_parts.append(f"platforms_connected={','.join(connected) or 'none'}")

    cli_text = _sanitize((stdout + "\n" + stderr).strip())
    cli_summary = cli_text[-1500:] if len(cli_text) > 1500 else cli_text

    return {
        "ok": alive,
        "status": "; ".join(summary_parts) or cli_summary[:500],
        "cli_exit_code": code,
        "cli_output": cli_summary,
        "pid": pid,
        "gateway_state": gateway_state,
        "platforms": platforms,
        "hermes_home": str(home),
        "windows": _windows_service_hint(home) if sys.platform == "win32" else None,
        "log_tail": "",
    }


def action_logs(home: Path, *, lines: int, follow: bool) -> dict:
    if follow:
        return {
            "ok": False,
            "status": "follow not supported in JSON mode — use: hermes logs gateway --follow",
            "log_tail": "",
        }
    tail, err = _tail_log(home, lines)
    return {
        "ok": err is None,
        "status": "ok" if err is None else err,
        "log_tail": tail,
        "lines": lines,
        "log_path": str(home / "logs" / "gateway.log"),
    }


def action_restart(home: Path, cmd_prefix: list[str], *, confirm: bool) -> dict:
    if not confirm:
        return {
            "ok": False,
            "status": "restart blocked - pass --confirm after human approval",
            "log_tail": "",
        }

    before = action_status(home, cmd_prefix)
    code, stdout, stderr = _run_hermes(cmd_prefix, ["gateway", "restart"], timeout=180)
    time.sleep(3)
    after = action_status(home, cmd_prefix)

    ok = code == 0 and after.get("ok")
    return {
        "ok": ok,
        "status": "restarted" if ok else f"restart exit {code}",
        "cli_output": _sanitize((stdout + "\n" + stderr).strip()),
        "before": {k: before.get(k) for k in ("pid", "gateway_state", "platforms")},
        "after": {k: after.get(k) for k in ("pid", "gateway_state", "platforms", "status")},
        "log_tail": "",
    }


def _emit_json(payload: dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    try:
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
    except (AttributeError, OSError):
        print(json.dumps(payload, indent=2, ensure_ascii=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes gateway ops (status/logs/restart)")
    parser.add_argument(
        "action",
        choices=("status", "logs", "restart"),
        help="Operation to perform",
    )
    parser.add_argument(
        "--hermes-home",
        default="",
        help="HERMES_HOME override (default: env or %%LOCALAPPDATA%%\\hermes)",
    )
    parser.add_argument(
        "--repo-path",
        default="",
        help="hermes-agent checkout for CLI resolution",
    )
    parser.add_argument("--lines", type=int, default=50, help="Log tail lines (logs action)")
    parser.add_argument("--follow", action="store_true", help="Request follow (logs) — redirects to CLI hint")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required for restart — human approval",
    )
    args = parser.parse_args()

    home = Path(args.hermes_home).expanduser() if args.hermes_home else _default_hermes_home()
    repo = Path(args.repo_path).expanduser() if args.repo_path else _default_repo()
    cmd_prefix = _find_hermes_cmd(repo)

    if args.action == "status":
        result = action_status(home, cmd_prefix)
    elif args.action == "logs":
        result = action_logs(home, lines=max(1, args.lines), follow=args.follow)
    else:
        result = action_restart(home, cmd_prefix, confirm=args.confirm)

    out = {
        "action": args.action,
        "hermes_home": str(home),
        **result,
    }
    _emit_json(out)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
