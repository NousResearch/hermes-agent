"""Usage-aware command runner for long autonomous Hermes work.

This module is deliberately boring: it wraps a subprocess, detects provider
quota/rate-limit failures in stdout/stderr, persists a pause window under
``$HERMES_HOME/usage_guard/tasks/``, and exits with a retryable status. Cron or
any external scheduler can then call it periodically without burning model calls
while the provider is exhausted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home

RETRYABLE_EXIT_CODE = 75  # EX_TEMPFAIL: conventional "try again later" code.
DEFAULT_BACKOFF_MINUTES = (15, 30, 60, 120, 240)
RATE_LIMIT_PATTERNS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "quota exceeded",
    "quota_exceeded",
    "insufficient_quota",
    "429",
    "try again later",
    "retry after",
    "retry-after",
)


@dataclass(frozen=True)
class RateLimitPause:
    """Parsed pause recommendation from provider output."""

    reason: str
    resume_at: datetime
    source: str


@dataclass(frozen=True)
class GuardState:
    task_id: str
    status: str
    command: list[str]
    workdir: str
    attempts: int
    updated_at: datetime
    resume_at: Optional[datetime] = None
    reason: Optional[str] = None
    last_exit_code: Optional[int] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "command": self.command,
            "workdir": self.workdir,
            "attempts": self.attempts,
            "updated_at": _format_dt(self.updated_at),
            "resume_at": _format_dt(self.resume_at) if self.resume_at else None,
            "reason": self.reason,
            "last_exit_code": self.last_exit_code,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "GuardState":
        return cls(
            task_id=str(payload.get("task_id") or "default"),
            status=str(payload.get("status") or "unknown"),
            command=[str(part) for part in payload.get("command") or []],
            workdir=str(payload.get("workdir") or os.getcwd()),
            attempts=int(payload.get("attempts") or 0),
            updated_at=_parse_dt(payload.get("updated_at")) or _utc_now(),
            resume_at=_parse_dt(payload.get("resume_at")),
            reason=payload.get("reason"),
            last_exit_code=payload.get("last_exit_code"),
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_dt(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_dt(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _state_dir() -> Path:
    path = get_hermes_home() / "usage_guard" / "tasks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_task_id(task_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task_id.strip()).strip("-._")
    return safe or "default"


def state_path(task_id: str) -> Path:
    return _state_dir() / f"{_safe_task_id(task_id)}.json"


def load_state(task_id: str) -> Optional[GuardState]:
    path = state_path(task_id)
    if not path.exists():
        return None
    try:
        return GuardState.from_json(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def save_state(state: GuardState) -> Path:
    path = state_path(state.task_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def clear_state(task_id: str) -> bool:
    path = state_path(task_id)
    if path.exists():
        path.unlink()
        return True
    return False


def _extract_seconds(text: str) -> Optional[int]:
    patterns = (
        r"retry[- ]after[:= ]+(\d+)",
        r"try again in +(\d+)\s*(seconds?|secs?|s)\b",
        r"try again in +(\d+)\s*(minutes?|mins?|m)\b",
        r"try again in +(\d+)\s*(hours?|hrs?|h)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = int(match.group(1))
        unit = match.group(2).lower() if match.lastindex and match.lastindex >= 2 else "seconds"
        if unit.startswith(("h", "hr", "hour")):
            return value * 3600
        if unit.startswith(("m", "min")):
            return value * 60
        return value
    return None


def _extract_reset_time(text: str) -> Optional[datetime]:
    for pattern in (
        r"(?:reset|resets|reset_at|resume_at)[:= ]+([0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9:.+-]+Z?)",
        r"(?:reset|resets|reset_at|resume_at)[:= ]+([A-Z][a-z]{2},\s+\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+GMT)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        parsed = _parse_dt(match.group(1))
        if parsed:
            return parsed
        try:
            from email.utils import parsedate_to_datetime

            dt = parsedate_to_datetime(match.group(1))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
    return None


def _fallback_backoff(attempts: int, now: datetime) -> datetime:
    idx = min(max(attempts, 0), len(DEFAULT_BACKOFF_MINUTES) - 1)
    return now + timedelta(minutes=DEFAULT_BACKOFF_MINUTES[idx])


def detect_rate_limit(text: str, *, attempts: int = 0, now: Optional[datetime] = None) -> Optional[RateLimitPause]:
    """Return a pause window when command output looks quota/rate limited."""

    if not text:
        return None
    lowered = text.lower()
    if not any(pattern in lowered for pattern in RATE_LIMIT_PATTERNS):
        return None

    now = now or _utc_now()
    reset_at = _extract_reset_time(text)
    source = "reset-time"
    if reset_at is None:
        seconds = _extract_seconds(text)
        if seconds is not None:
            reset_at = now + timedelta(seconds=max(0, seconds))
            source = "retry-after"
        else:
            reset_at = _fallback_backoff(attempts, now)
            source = "exponential-backoff"

    return RateLimitPause(
        reason=_first_matching_reason(lowered),
        resume_at=reset_at,
        source=source,
    )


def _first_matching_reason(lowered_text: str) -> str:
    for pattern in RATE_LIMIT_PATTERNS:
        if pattern in lowered_text:
            return pattern
    return "rate limit"


def _combine_output(stdout: str, stderr: str) -> str:
    if stdout and stderr:
        return stdout + "\n" + stderr
    return stdout or stderr or ""


def _print_state(state: GuardState) -> None:
    print(f"task: {state.task_id}")
    print(f"status: {state.status}")
    print(f"attempts: {state.attempts}")
    print(f"workdir: {state.workdir}")
    if state.resume_at:
        print(f"resume_at: {_format_dt(state.resume_at)}")
    if state.reason:
        print(f"reason: {state.reason}")
    if state.last_exit_code is not None:
        print(f"last_exit_code: {state.last_exit_code}")
    if state.command:
        print("command: " + " ".join(state.command))


def _list_states() -> list[GuardState]:
    states: list[GuardState] = []
    for path in sorted(_state_dir().glob("*.json")):
        try:
            states.append(GuardState.from_json(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return states


def run_guarded_command(
    command: list[str],
    *,
    task_id: str,
    workdir: Optional[str] = None,
    now: Optional[datetime] = None,
    env: Optional[dict[str, str]] = None,
) -> int:
    """Run *command* unless the task is paused; persist pause/completion state."""

    if not command:
        print("usage-guard: no command supplied", file=sys.stderr)
        return 2

    now = now or _utc_now()
    task_id = _safe_task_id(task_id)
    existing = load_state(task_id)
    if existing and existing.status == "paused" and existing.resume_at and existing.resume_at > now:
        print(f"usage-guard: paused until {_format_dt(existing.resume_at)} ({existing.reason or 'rate limit'})")
        return RETRYABLE_EXIT_CODE

    attempts = existing.attempts if existing else 0
    cwd = str(Path(workdir or os.getcwd()).expanduser().resolve())
    started = GuardState(
        task_id=task_id,
        status="running",
        command=command,
        workdir=cwd,
        attempts=attempts,
        updated_at=now,
    )
    save_state(started)

    proc = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    output = _combine_output(proc.stdout, proc.stderr)
    pause = detect_rate_limit(output, attempts=attempts, now=now)
    if pause:
        paused = GuardState(
            task_id=task_id,
            status="paused",
            command=command,
            workdir=cwd,
            attempts=attempts + 1,
            updated_at=_utc_now(),
            resume_at=pause.resume_at,
            reason=f"{pause.reason} ({pause.source})",
            last_exit_code=proc.returncode,
        )
        save_state(paused)
        print(
            f"usage-guard: provider limit detected; paused until {_format_dt(pause.resume_at)} "
            f"[{paused.reason}]",
            file=sys.stderr,
        )
        return RETRYABLE_EXIT_CODE

    completed = GuardState(
        task_id=task_id,
        status="completed" if proc.returncode == 0 else "failed",
        command=command,
        workdir=cwd,
        attempts=0 if proc.returncode == 0 else attempts,
        updated_at=_utc_now(),
        last_exit_code=proc.returncode,
    )
    save_state(completed)
    return proc.returncode


def _normalize_remainder(command: Iterable[str]) -> list[str]:
    parts = list(command)
    if parts and parts[0] == "--":
        parts = parts[1:]
    return parts


def cmd_usage_guard(args: argparse.Namespace) -> int:
    action = getattr(args, "usage_guard_command", None)
    if action == "run":
        return run_guarded_command(
            _normalize_remainder(args.argv),
            task_id=args.id,
            workdir=args.workdir,
        )
    if action == "status":
        state = load_state(args.id)
        if not state:
            print(f"task: {_safe_task_id(args.id)}")
            print("status: none")
            return 1
        _print_state(state)
        return 0
    if action == "list":
        states = _list_states()
        if not states:
            print("no usage-guard tasks")
            return 0
        for index, state in enumerate(states):
            if index:
                print()
            _print_state(state)
        return 0
    if action == "clear":
        removed = clear_state(args.id)
        print(f"cleared: {_safe_task_id(args.id)}" if removed else f"not found: {_safe_task_id(args.id)}")
        return 0 if removed else 1
    print("usage: hermes usage-guard {run,status,list,clear}", file=sys.stderr)
    return 2


def _exit_cmd(args: argparse.Namespace) -> None:
    raise SystemExit(cmd_usage_guard(args))


def register_cli(parent: argparse.ArgumentParser) -> None:
    parent.set_defaults(func=_exit_cmd)
    subs = parent.add_subparsers(dest="usage_guard_command")

    run = subs.add_parser(
        "run",
        help="Run a command unless its task is paused by provider quota exhaustion",
        description=(
            "Wrap a long-running command, detect LLM provider rate-limit/quota output, "
            "persist the pause window, and exit 75 until it is safe to resume."
        ),
    )
    run.add_argument("--id", default="default", help="Stable task id for persisted pause state")
    run.add_argument("--workdir", default=None, help="Working directory for the command")
    run.add_argument("argv", nargs=argparse.REMAINDER, help="Command to run after --")
    run.set_defaults(func=_exit_cmd)

    status = subs.add_parser("status", help="Show persisted state for one guarded task")
    status.add_argument("--id", default="default", help="Task id")
    status.set_defaults(func=_exit_cmd)

    listing = subs.add_parser("list", help="List all persisted usage-guard task states")
    listing.set_defaults(func=_exit_cmd)

    clear = subs.add_parser("clear", help="Delete persisted state for one guarded task")
    clear.add_argument("--id", default="default", help="Task id")
    clear.set_defaults(func=_exit_cmd)
