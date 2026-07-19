"""Deterministic context handoff/checkpoint writer for Hermes.

This module is intentionally cheap and local-only: no model calls, bounded
SQLite reads, bounded git commands, and best-effort failure behavior for the
compression hook path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

_MAX_GIT_LINES = 80
_MAX_DB_ROWS = 80
_MAX_FIELD_CHARS = 2_000
_GIT_TIMEOUT_SECS = 1.5
_VERIFY_RE = re.compile(
    r"\b("
    r"pytest|python\s+-m\s+pytest|uv\s+run.*pytest|npm\s+(?:test|run\s+(?:test|build|typecheck|lint))|"
    r"node\s+--test|pnpm\s+(?:test|build|lint)|yarn\s+(?:test|build|lint)|"
    r"ruff|mypy|pyright|tsc|vite\s+build|cargo\s+test|go\s+test|swift\s+test"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HandoffResult:
    ok: bool
    path: Path
    repo_local: bool
    error: str | None = None


def create_handoff(
    *,
    cwd: str | os.PathLike[str] | None = None,
    session_id: str | None = None,
    task_goal: str | None = None,
    reason: str = "manual",
    now: datetime | None = None,
    state_db_path: str | os.PathLike[str] | None = None,
    hook_payload: dict[str, Any] | None = None,
) -> HandoffResult:
    """Write a deterministic handoff markdown file and return its location.

    Repo work writes ``CURRENT.md`` at the git root. Non-repo work falls back to
    ``$HERMES_HOME/handoffs/<timestamp>-<session>.md``. Exceptions are captured
    in the returned result so compression callers can stay best-effort.
    """
    timestamp = _coerce_now(now)
    workdir = _resolve_cwd(cwd)
    db_path = Path(state_db_path) if state_db_path is not None else get_hermes_home() / "state.db"
    payload = hook_payload or {}

    try:
        repo_root = _git_repo_root(workdir)
        repo_local = repo_root is not None
        target = repo_root / "CURRENT.md" if repo_root else _global_handoff_path(timestamp, session_id)

        git_info = _collect_git_info(repo_root) if repo_root else _empty_git_info()
        session_hints = _collect_session_hints(db_path, session_id)
        effective_goal = _first_nonempty(task_goal, session_hints.get("goal"), payload.get("goal"))
        verification_hints = _dedupe([*session_hints.get("verification_commands", []), *_payload_commands(payload)])

        content = _render_handoff(
            timestamp=timestamp,
            session_id=session_id,
            cwd=workdir,
            repo_root=repo_root,
            repo_local=repo_local,
            reason=reason,
            task_goal=effective_goal,
            git_info=git_info,
            verification_hints=verification_hints,
            hook_payload=payload,
        )
        _atomic_write(target, content)
        return HandoffResult(ok=True, path=target, repo_local=repo_local)
    except Exception as exc:  # best-effort contract for compression hook callers
        fallback = _global_handoff_path(timestamp, session_id)
        return HandoffResult(ok=False, path=fallback, repo_local=False, error=f"{type(exc).__name__}: {exc}")


def pre_context_compress_handoff(**kwargs: Any) -> HandoffResult:
    """Hook callback for ``pre_context_compress``.

    Never raises: compression must continue even if checkpointing fails.
    """
    try:
        return create_handoff(
            cwd=kwargs.get("cwd"),
            session_id=kwargs.get("session_id"),
            task_goal=kwargs.get("task_goal") or kwargs.get("goal"),
            reason="pre_context_compress",
            hook_payload=kwargs,
        )
    except Exception as exc:  # defensive belt-and-suspenders
        return HandoffResult(
            ok=False,
            path=_global_handoff_path(_coerce_now(None), kwargs.get("session_id")),
            repo_local=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def find_latest_handoff(*, cwd: str | os.PathLike[str] | None = None, hermes_home: Path | None = None) -> Path | None:
    """Return the best latest handoff for this workspace.

    Repo-local ``CURRENT.md`` wins when present because it is the source of
    truth for that checkout. Otherwise use the newest global handoff file.
    """
    workdir = _resolve_cwd(cwd)
    repo_root = _git_repo_root(workdir)
    if repo_root:
        current = repo_root / "CURRENT.md"
        if current.exists():
            return current
    home = hermes_home or get_hermes_home()
    handoff_dir = home / "handoffs"
    try:
        candidates = [p for p in handoff_dir.glob("*.md") if p.is_file()]
    except Exception:
        candidates = []
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def get_context_status(
    *,
    cwd: str | os.PathLike[str] | None = None,
    state_db_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return deterministic non-UI Context Governor status for Spend Watch style surfaces."""
    workdir = _resolve_cwd(cwd)
    repo_root = _git_repo_root(workdir)
    latest = find_latest_handoff(cwd=workdir)
    db_path = Path(state_db_path) if state_db_path is not None else get_hermes_home() / "state.db"
    burn = get_context_burn_summary(state_db_path=db_path)

    git_info = _collect_git_info(repo_root) if repo_root else _empty_git_info()
    changed_files = git_info.get("changed_files") or []
    latest_mtime = _safe_mtime(latest)
    changed_newer = False
    if repo_root and changed_files:
        for rel in changed_files:
            if rel == "CURRENT.md":
                continue
            candidate = repo_root / rel
            mtime = _safe_mtime(candidate)
            if latest_mtime is None or (mtime is not None and mtime > latest_mtime):
                changed_newer = True
                break

    high_burn = burn.get("avg_tokens_per_api_call", 0) >= 75_000 or burn.get("cache_read_percent", 0) >= 85.0
    handoff_recommended = bool(changed_newer or high_burn or (changed_files and latest is None))
    fresh_safe = bool(latest is not None and not changed_newer)
    if latest is None:
        reason = "no handoff found"
    elif changed_newer:
        reason = "changed files newer than latest handoff"
    elif high_burn:
        reason = "latest handoff exists; high context burn suggests fresh session is reasonable"
    else:
        reason = "latest handoff is current enough for deterministic resume"

    return {
        "cwd": str(workdir),
        "repo_root": str(repo_root) if repo_root else None,
        "latest_handoff_path": str(latest) if latest else None,
        "handoff_recommended": handoff_recommended,
        "fresh_session_safe": fresh_safe,
        "fresh_session_reason": reason,
        "git_status_summary": git_info.get("status_summary"),
        "changed_files": changed_files,
        "context_burn": burn,
    }


def get_context_burn_summary(
    *,
    state_db_path: str | os.PathLike[str] | None = None,
    days: int = 1,
) -> dict[str, Any]:
    db_path = Path(state_db_path) if state_db_path is not None else get_hermes_home() / "state.db"
    if not db_path.exists():
        return _empty_burn_summary(days)
    since = time.time() - max(days, 1) * 86_400
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.5)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS session_count,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                    COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
                    COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                    COALESCE(SUM(api_call_count), 0) AS api_call_count
                FROM sessions
                WHERE started_at >= ?
                """,
                (since,),
            ).fetchone()
        finally:
            conn.close()
    except Exception:
        return _empty_burn_summary(days)
    total = int(row["input_tokens"] or 0) + int(row["output_tokens"] or 0) + int(row["cache_read_tokens"] or 0) + int(row["cache_write_tokens"] or 0) + int(row["reasoning_tokens"] or 0)
    api_calls = int(row["api_call_count"] or 0)
    cache_read = int(row["cache_read_tokens"] or 0)
    return {
        "days": days,
        "session_count": int(row["session_count"] or 0),
        "total_tokens": total,
        "api_call_count": api_calls,
        "avg_tokens_per_api_call": round(total / api_calls, 2) if api_calls else 0,
        "cache_read_percent": round((cache_read / total) * 100, 2) if total else 0,
    }


def build_fresh_session_prompt(handoff_path: str | os.PathLike[str]) -> str:
    path = Path(handoff_path).expanduser()
    next_task = _extract_next_session_prompt(path)
    return (
        "[Hermes Context Governor fresh-session handoff]\n"
        f"Read this handoff file first: {path}\n\n"
        f"{next_task}\n\n"
        "Continue from the handoff, inspect git status before editing, preserve build quality, "
        "run relevant verification, and write/update a receipt with proof."
    )


def build_fresh_session_command(handoff_path: str | os.PathLike[str]) -> str:
    prompt = build_fresh_session_prompt(handoff_path)
    return f"hermes chat -q {shlex.quote(prompt)}"


def get_quality_report(
    *,
    state_db_path: str | os.PathLike[str] | None = None,
    days: int = 30,
) -> dict[str, Any]:
    """Compare fresh-from-handoff sessions against monster sessions.

    This is v4's deterministic baseline: no outcome claims beyond what state.db
    can show cheaply. Verification failures and user corrections are heuristic
    counters from recent message text.
    """
    db_path = Path(state_db_path) if state_db_path is not None else get_hermes_home() / "state.db"
    if not db_path.exists():
        return {"days": days, "handoff_sessions": _empty_quality_bucket(), "monster_sessions": _empty_quality_bucket()}
    since = time.time() - max(days, 1) * 86_400
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.5)
        conn.row_factory = sqlite3.Row
        try:
            sessions = conn.execute(
                """
                SELECT id, started_at, ended_at, input_tokens, output_tokens,
                       cache_read_tokens, cache_write_tokens, reasoning_tokens, api_call_count
                FROM sessions
                WHERE started_at >= ?
                """,
                (since,),
            ).fetchall()
            messages = conn.execute(
                """
                SELECT session_id, role, content, tool_name
                FROM messages
                WHERE session_id IN (SELECT id FROM sessions WHERE started_at >= ?)
                  AND COALESCE(active, 1) = 1
                ORDER BY id ASC
                """,
                (since,),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return {"days": days, "handoff_sessions": _empty_quality_bucket(), "monster_sessions": _empty_quality_bucket()}

    by_session: dict[str, list[sqlite3.Row]] = {}
    for msg in messages:
        by_session.setdefault(msg["session_id"], []).append(msg)

    handoff_rows = []
    monster_rows = []
    for session in sessions:
        sid = session["id"]
        session_messages = by_session.get(sid, [])
        total = _session_total_tokens(session)
        api_calls = int(session["api_call_count"] or 0)
        avg = total / api_calls if api_calls else 0
        first_user = next((_plain_text(m["content"]) or "" for m in session_messages if m["role"] == "user"), "")
        is_handoff = "Hermes Context Governor fresh-session handoff" in first_user or "Context Governor fresh-session handoff" in first_user
        enriched = {"session": session, "messages": session_messages}
        if is_handoff:
            handoff_rows.append(enriched)
        elif avg >= 75_000 or total >= 250_000:
            monster_rows.append(enriched)
    return {
        "days": days,
        "handoff_sessions": _quality_bucket(handoff_rows),
        "monster_sessions": _quality_bucket(monster_rows),
    }


def checkpoint_command(raw_args: str = "") -> str:
    """In-session slash command handler for ``/checkpoint``."""
    parts = raw_args.strip().split()
    verb = parts[0].lower() if parts else "create"
    if verb == "status":
        return _format_status(get_context_status())
    if verb == "latest":
        latest = find_latest_handoff()
        return str(latest) if latest else "No handoff found."
    if verb == "launch":
        latest = find_latest_handoff()
        return build_fresh_session_command(latest) if latest else "No handoff found to launch from."
    if verb == "quality":
        return _format_quality_report(get_quality_report())

    goal = raw_args.strip() or None
    result = create_handoff(task_goal=goal, reason="slash_checkpoint")
    if result.ok:
        scope = "repo" if result.repo_local else "global"
        return f"✓ Checkpoint written ({scope}): {result.path}"
    return f"⚠ Checkpoint failed: {result.error} (intended path: {result.path})"


def setup_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("args", nargs="*", help="create goal text, or one of: status, latest, launch, quality")
    parser.add_argument("--cwd", default=None, help="Directory to inspect (default: current working directory)")
    parser.add_argument("--session-id", default=None, help="Session id to record and inspect in state.db")
    parser.add_argument("--state-db", default=None, help="Path to Hermes state.db for cheap recent hints")
    parser.add_argument("--reason", default="cli_checkpoint", help="Reason label written to the handoff")
    parser.add_argument("--path", default=None, help="Explicit handoff path for launch")
    parser.add_argument("--execute", action="store_true", help="For launch: execute `hermes chat -q ...` instead of printing it")
    parser.add_argument("--days", type=int, default=30, help="For quality: lookback window in days")
    parser.set_defaults(func=cli_main)


def cli_main(args: argparse.Namespace) -> int:
    tokens = list(getattr(args, "args", []) or [])
    verb = tokens[0].lower() if tokens else "create"
    db_path = getattr(args, "state_db", None)
    cwd = getattr(args, "cwd", None)
    if verb == "status":
        print(_format_status(get_context_status(cwd=cwd, state_db_path=db_path)))
        return 0
    if verb == "latest":
        latest = find_latest_handoff(cwd=cwd)
        if latest:
            print(latest)
            return 0
        print("No handoff found.")
        return 1
    if verb == "launch":
        handoff_value = getattr(args, "path", None) or find_latest_handoff(cwd=cwd)
        if not handoff_value:
            print("No handoff found to launch from.")
            return 1
        handoff = Path(handoff_value)
        if not handoff.exists():
            print("No handoff found to launch from.")
            return 1
        command = build_fresh_session_command(handoff)
        if getattr(args, "execute", False):
            return subprocess.run(["hermes", "chat", "-q", build_fresh_session_prompt(handoff)]).returncode
        print(command)
        return 0
    if verb == "quality":
        print(_format_quality_report(get_quality_report(state_db_path=db_path, days=getattr(args, "days", 30))))
        return 0

    goal = " ".join(tokens).strip() or None
    result = create_handoff(
        cwd=cwd,
        session_id=getattr(args, "session_id", None),
        task_goal=goal,
        reason=getattr(args, "reason", "cli_checkpoint"),
        state_db_path=db_path,
    )
    if result.ok:
        scope = "repo-local" if result.repo_local else "global"
        print(f"Checkpoint written ({scope}): {result.path}")
        return 0
    print(f"Checkpoint failed: {result.error} (intended path: {result.path})")
    return 1


def _coerce_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _resolve_cwd(cwd: str | os.PathLike[str] | None) -> Path:
    path = Path(cwd or os.getcwd()).expanduser()
    try:
        return path.resolve()
    except Exception:
        return path.absolute()


def _run_git(repo_or_cwd: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_or_cwd), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_GIT_TIMEOUT_SECS,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_repo_root(cwd: Path) -> Path | None:
    out = _run_git(cwd, "rev-parse", "--show-toplevel")
    if not out:
        return None
    try:
        return Path(out).resolve()
    except Exception:
        return Path(out)


def _collect_git_info(repo_root: Path) -> dict[str, Any]:
    branch = _run_git(repo_root, "branch", "--show-current") or _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    status = _bounded_lines(_run_git(repo_root, "status", "--short") or "")
    changed = []
    for line in status:
        # porcelain v1 normally uses two status columns plus a separator, but
        # older/custom git output can collapse the separator for staged-only
        # paths ("M path"). Prefer the porcelain slice when present; otherwise
        # split on whitespace so we do not drop the first path character.
        if len(line) >= 3 and line[2].isspace():
            path = line[3:]
        else:
            parts = line.split(maxsplit=1)
            path = parts[1] if len(parts) == 2 else line
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path:
            changed.append(path)
    summary = "clean" if not status else f"{len(status)} changed path(s) shown"
    return {
        "branch": branch,
        "status_summary": summary,
        "status_lines": status,
        "changed_files": _dedupe(changed)[:_MAX_GIT_LINES],
    }


def _empty_git_info() -> dict[str, Any]:
    return {"branch": "(not a git repo)", "status_summary": "not a git repo", "status_lines": [], "changed_files": []}


def _collect_session_hints(db_path: Path, session_id: str | None) -> dict[str, Any]:
    if not session_id or not db_path.exists():
        return {"goal": None, "verification_commands": []}
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=0.5)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT role, content, tool_calls, tool_name
                FROM messages
                WHERE session_id = ? AND COALESCE(active, 1) = 1
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, _MAX_DB_ROWS),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return {"goal": None, "verification_commands": []}

    goal = None
    commands: list[str] = []
    for row in reversed(rows):
        role = row["role"]
        content = _plain_text(row["content"])
        if role == "user" and content and not content.lstrip().startswith("/"):
            goal = _truncate(content.replace("\n", " "), 240)
        commands.extend(_commands_from_tool_calls(row["tool_calls"]))
        if row["tool_name"] == "terminal":
            commands.extend(_commands_from_text(content))
    return {"goal": goal, "verification_commands": _dedupe([c for c in commands if _VERIFY_RE.search(c)])[:8]}


def _commands_from_tool_calls(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        calls = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, list):
        return []
    found: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        name = fn.get("name") if isinstance(fn, dict) else None
        if name != "terminal":
            continue
        args = fn.get("arguments") if isinstance(fn, dict) else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                continue
        if isinstance(args, dict) and isinstance(args.get("command"), str):
            found.append(_truncate(args["command"].strip(), 300))
    return found


def _commands_from_text(text: str | None) -> list[str]:
    if not text:
        return []
    commands: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _VERIFY_RE.search(stripped):
            commands.append(_truncate(stripped, 300))
    return commands


def _payload_commands(payload: dict[str, Any]) -> list[str]:
    hints = payload.get("verification_commands") or payload.get("verification_hints") or []
    if isinstance(hints, str):
        hints = [hints]
    if not isinstance(hints, Iterable):
        return []
    return [_truncate(str(x), 300) for x in hints if str(x).strip()]


def _render_handoff(
    *,
    timestamp: datetime,
    session_id: str | None,
    cwd: Path,
    repo_root: Path | None,
    repo_local: bool,
    reason: str,
    task_goal: str | None,
    git_info: dict[str, Any],
    verification_hints: list[str],
    hook_payload: dict[str, Any],
) -> str:
    ts = timestamp.isoformat().replace("+00:00", "Z")
    repo_display = f"`{repo_root}`" if repo_root else "`(not detected)`"
    changed_files = git_info.get("changed_files") or []
    status_lines = git_info.get("status_lines") or []
    payload_lines = _format_hook_payload(hook_payload)

    lines = [
        "# CURRENT — Hermes Context Governor Handoff",
        "",
        "This file is deterministic local state written before/around context loss. It contains no LLM analysis by default.",
        "",
        "## Snapshot",
        f"- Timestamp: `{ts}`",
        f"- Session ID: `{session_id or '(unknown)'}`",
        f"- CWD: `{cwd}`",
        f"- Repo root: {repo_display}",
        f"- Scope: {'repo-local CURRENT.md' if repo_local else 'global fallback'}",
        f"- Reason: {reason}",
        "",
        "## Current task / goal",
        _truncate(task_goal, _MAX_FIELD_CHARS) if task_goal else "(not available — start by reading git status and recent receipts)",
        "",
        "## Git state",
        f"- Branch: `{git_info.get('branch', 'unknown')}`",
        f"- Status summary: {git_info.get('status_summary', 'unknown')}",
        "",
        "### Changed files",
    ]
    if changed_files:
        lines.extend(f"- `{path}`" for path in changed_files)
    else:
        lines.append("- (none detected)")
    lines.extend(["", "### `git status --short` (bounded)"])
    if status_lines:
        lines.append("```text")
        lines.extend(status_lines)
        lines.append("```")
    else:
        lines.append("```text\n(clean or unavailable)\n```")

    lines.extend(["", "## Recent verification command hints"])
    if verification_hints:
        lines.extend(f"- `{cmd}`" for cmd in verification_hints)
    else:
        lines.append("- (none cheaply discovered)")

    if payload_lines:
        lines.extend(["", "## Compression trigger payload (bounded)"])
        lines.extend(payload_lines)

    lines.extend([
        "",
        "## Next-session prompt",
        "```text",
        "Resume this Hermes work from this handoff. Read CURRENT.md (or this global handoff) first, inspect git status, then continue the current task without assuming prior chat context. Preserve build quality: verify changed files, run the relevant focused tests/build commands, and write a receipt with proof before declaring success.",
        "```",
        "",
    ])
    return "\n".join(lines)


def _format_hook_payload(payload: dict[str, Any]) -> list[str]:
    keep = [
        "task_id",
        "platform",
        "model",
        "approx_tokens",
        "message_count",
        "context_length",
        "threshold_tokens",
        "compression_count",
        "focus_topic",
    ]
    lines = []
    for key in keep:
        if key in payload and payload[key] is not None:
            lines.append(f"- {key}: `{_truncate(str(payload[key]), 240)}`")
    return lines


def _global_handoff_path(timestamp: datetime, session_id: str | None) -> Path:
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    suffix = _safe_slug(session_id or "no-session")
    return get_hermes_home() / "handoffs" / f"{stamp}-{suffix}.md"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except Exception:
            pass


def _bounded_lines(text: str) -> list[str]:
    lines = text.splitlines()[:_MAX_GIT_LINES]
    return [_truncate(line, 500) for line in lines]


def _plain_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return _truncate(str(value), _MAX_FIELD_CHARS)
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, str):
                return parsed
            if isinstance(parsed, list):
                parts = []
                for item in parsed:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif isinstance(item, str):
                        parts.append(item)
                if parts:
                    return _truncate(" ".join(parts), _MAX_FIELD_CHARS)
        except Exception:
            pass
    return _truncate(stripped, _MAX_FIELD_CHARS)


def _truncate(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _safe_mtime(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except Exception:
        return None


def _empty_burn_summary(days: int) -> dict[str, Any]:
    return {
        "days": days,
        "session_count": 0,
        "total_tokens": 0,
        "api_call_count": 0,
        "avg_tokens_per_api_call": 0,
        "cache_read_percent": 0,
    }


def _extract_next_session_prompt(path: Path) -> str:
    default = "Resume from the latest Context Governor handoff."
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return default
    marker = "## Next-session prompt"
    idx = text.find(marker)
    if idx < 0:
        return default
    tail = text[idx + len(marker):]
    match = re.search(r"```(?:text)?\s*(.*?)```", tail, re.DOTALL | re.IGNORECASE)
    if match:
        return _truncate(match.group(1).strip(), 2_000) or default
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return _truncate(" ".join(lines[:6]), 2_000) or default


def _session_total_tokens(session: sqlite3.Row | dict[str, Any]) -> int:
    total = 0
    for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens"):
        try:
            total += int(session[key] or 0)
        except Exception:
            pass
    return total


def _empty_quality_bucket() -> dict[str, Any]:
    return {
        "session_count": 0,
        "total_tokens": 0,
        "api_call_count": 0,
        "avg_tokens_per_api_call": 0,
        "cache_read_percent": 0,
        "verification_failure_markers": 0,
        "user_correction_markers": 0,
        "avg_elapsed_minutes": 0,
    }


def _quality_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return _empty_quality_bucket()
    total_tokens = 0
    cache_read = 0
    api_calls = 0
    elapsed_minutes: list[float] = []
    verification_failures = 0
    corrections = 0
    for item in rows:
        session = item["session"]
        messages = item["messages"]
        total_tokens += _session_total_tokens(session)
        cache_read += int(session["cache_read_tokens"] or 0)
        api_calls += int(session["api_call_count"] or 0)
        started = float(session["started_at"] or 0)
        ended = float(session["ended_at"] or 0) if session["ended_at"] is not None else started
        if ended >= started and started > 0:
            elapsed_minutes.append((ended - started) / 60)
        for msg in messages:
            text = (_plain_text(msg["content"]) or "").lower()
            if msg["tool_name"] == "terminal" and any(token in text for token in ("failed", "error", "traceback", "exit_code")):
                verification_failures += 1
            if msg["role"] == "user" and any(token in text for token in ("actually", "not what", "wrong", "fix that", "you missed")):
                corrections += 1
    return {
        "session_count": len(rows),
        "total_tokens": total_tokens,
        "api_call_count": api_calls,
        "avg_tokens_per_api_call": round(total_tokens / api_calls, 2) if api_calls else 0,
        "cache_read_percent": round((cache_read / total_tokens) * 100, 2) if total_tokens else 0,
        "verification_failure_markers": verification_failures,
        "user_correction_markers": corrections,
        "avg_elapsed_minutes": round(sum(elapsed_minutes) / len(elapsed_minutes), 2) if elapsed_minutes else 0,
    }


def _format_status(status: dict[str, Any]) -> str:
    burn = status.get("context_burn") or {}
    return "\n".join([
        "Context Governor status",
        f"latest_handoff_path: {status.get('latest_handoff_path') or '(none)'}",
        f"handoff_recommended: {status.get('handoff_recommended')}",
        f"fresh_session_safe: {status.get('fresh_session_safe')}",
        f"fresh_session_reason: {status.get('fresh_session_reason')}",
        f"avg_tokens_per_api_call_1d: {burn.get('avg_tokens_per_api_call', 0)}",
        f"cache_read_percent_1d: {burn.get('cache_read_percent', 0)}",
        f"git_status_summary: {status.get('git_status_summary')}",
    ])


def _format_quality_report(report: dict[str, Any]) -> str:
    handoff = report.get("handoff_sessions") or {}
    monster = report.get("monster_sessions") or {}
    return "\n".join([
        f"Context Governor quality report ({report.get('days')}d)",
        "handoff_sessions:",
        f"  sessions: {handoff.get('session_count', 0)}",
        f"  avg_tokens_per_api_call: {handoff.get('avg_tokens_per_api_call', 0)}",
        f"  verification_failure_markers: {handoff.get('verification_failure_markers', 0)}",
        f"  user_correction_markers: {handoff.get('user_correction_markers', 0)}",
        "monster_sessions:",
        f"  sessions: {monster.get('session_count', 0)}",
        f"  avg_tokens_per_api_call: {monster.get('avg_tokens_per_api_call', 0)}",
        f"  verification_failure_markers: {monster.get('verification_failure_markers', 0)}",
        f"  user_correction_markers: {monster.get('user_correction_markers', 0)}",
    ])


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    slug = slug.strip(".-")
    return slug[:80] or "handoff"
