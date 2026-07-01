"""LOCAL GSSAI startup-recovery dashboard plugin.

Mounted at /api/plugins/startup-recovery/ by the dashboard plugin system.

This v0.1 surface is read-only: it checks expected local Hermes Telegram
gateway tmux sessions, process presence, log/state metadata, bounded error
counts, and Telegram TCP connection count. It never sends Telegram messages,
prints tokens, returns full process arguments, or returns log contents.
"""

from __future__ import annotations

import os
import pwd
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()

ERROR_PATTERN = re.compile(r"Unauthorized|409 Conflict|Forbidden|Traceback|reconnect|error|critical|exception", re.I)
TELEGRAM_TCP_PATTERN = re.compile(r"149\.154\.16|91\.108\.")


@dataclass(frozen=True)
class GatewayProfile:
    profile: str
    bot: str
    sessions: tuple[str, ...]
    process_pattern: str


GATEWAYS: tuple[GatewayProfile, ...] = (
    GatewayProfile("ivan_bb", "@Ivanbigboy_bot", ("hermes-gateway", "default-gateway"), "hermes --profile ivan_bb gateway run"),
    GatewayProfile("storm", "@Storm_CCO_bot", ("storm-gateway",), "hermes --profile storm gateway run"),
    GatewayProfile("neo", "@Neo_GSSAI_CTO_bot", ("neo-gateway",), "hermes --profile neo gateway run"),
    GatewayProfile("gssai_admin", "@GSSAI_Admin_bot", ("gssai-admin-gateway",), "hermes --profile gssai_admin gateway run"),
)


def _real_user_home() -> Path:
    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        return Path.home()


def _hermes_root() -> Path:
    override = os.environ.get("STARTUP_RECOVERY_HERMES_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (_real_user_home() / ".hermes").resolve()


def _run_command(args: list[str], timeout: float = 2.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None


def _file_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {"exists": True, "size": stat.st_size, "mtime": stat.st_mtime, "mode": oct(stat.st_mode & 0o777)}


def _tmux_sessions() -> set[str]:
    proc = _run_command(["tmux", "ls"])
    if proc is None or proc.returncode != 0:
        return set()
    sessions: set[str] = set()
    for line in proc.stdout.splitlines():
        name = line.split(":", 1)[0].strip()
        if name:
            sessions.add(name)
    return sessions


def _process_present(pattern: str) -> bool:
    # Patterns are fixed local allowlist literals. Keep this guard if they ever
    # become configurable; pgrep receives list-form args, never shell strings.
    if "\n" in pattern:
        return False
    proc = _run_command(["pgrep", "-f", pattern])
    return bool(proc and proc.returncode == 0)


def _recent_error_count(path: Path, max_lines: int = 2000, max_bytes: int = 512 * 1024) -> int | None:
    if not path.is_file():
        return None
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
                handle.readline()
            lines = handle.read().decode("utf-8", errors="replace").splitlines()[-max_lines:]
    except OSError:
        return None
    return sum(1 for line in lines if ERROR_PATTERN.search(line))


def _telegram_tcp_count() -> int | None:
    proc = _run_command(["ss", "-tn"])
    if proc is None or proc.returncode != 0:
        return None
    return sum(1 for line in proc.stdout.splitlines() if TELEGRAM_TCP_PATTERN.search(line))


def _profile_status(gateway: GatewayProfile, tmux_session_names: set[str], hermes_root: Path) -> dict[str, Any]:
    profile_root = hermes_root / "profiles" / gateway.profile
    matched_session = next((session for session in gateway.sessions if session in tmux_session_names), None)
    log_file = profile_root / "logs" / "gateway.log"
    state_db = profile_root / "state.db"
    process_present = _process_present(gateway.process_pattern)
    tmux_present = matched_session is not None
    return {
        "profile": gateway.profile,
        "bot": gateway.bot,
        "expected_sessions": list(gateway.sessions),
        "matched_session": matched_session,
        "tmux_status": "OK" if tmux_present else "CHECK",
        "process_status": "OK" if process_present else "CHECK",
        "log_file": {"path": str(log_file), **_file_metadata(log_file), "recent_error_count_tail_2000": _recent_error_count(log_file)},
        "state_db": {"path": str(state_db), **_file_metadata(state_db)},
    }


@router.get("/sessions")
async def sessions() -> dict[str, Any]:
    allowed = {session for gateway in GATEWAYS for session in gateway.sessions}
    found = _tmux_sessions()
    return {
        "scope": "LOCAL GSSAI PROJECT ONLY",
        "policy": "read-only; allowlisted local tmux names only; no process args",
        "sessions": sorted(found & allowed),
    }


@router.get("/status")
async def status() -> dict[str, Any]:
    hermes_root = _hermes_root()
    tmux_session_names = _tmux_sessions()
    profiles = [_profile_status(gateway, tmux_session_names, hermes_root) for gateway in GATEWAYS]
    problems = sum(
        1
        for item in profiles
        if item["tmux_status"] != "OK" or item["process_status"] != "OK"
    )
    return {
        "scope": "LOCAL GSSAI PROJECT ONLY",
        "policy": "local status only; no Telegram sends; no secrets or log contents returned",
        "hermes_root": str(hermes_root),
        "profiles": profiles,
        "telegram_tcp_connections_detected": _telegram_tcp_count(),
        "overall": "OK" if problems == 0 else "CHECK",
        "problem_count": problems,
    }


@router.get("/logs/{profile}/recent")
async def recent_log_counts(profile: str) -> dict[str, Any]:
    allowed = {gateway.profile for gateway in GATEWAYS}
    if profile not in allowed:
        raise HTTPException(status_code=404, detail="Unknown local GSSAI gateway profile")
    log_file = _hermes_root() / "profiles" / profile / "logs" / "gateway.log"
    return {
        "scope": "LOCAL GSSAI PROJECT ONLY",
        "profile": profile,
        "log_file": str(log_file),
        "metadata": _file_metadata(log_file),
        "recent_error_count_tail_2000": _recent_error_count(log_file),
        "log_contents_returned": False,
    }
