"""Event-driven notifier for human-facing Kanban blockers.

The plugin listens to the existing ``kanban_task_blocked`` lifecycle hook and
sends a one-shot message through ``hermes send`` when a task blocks for a human
decision/capability issue. It deliberately does not poll the board.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

HUMAN_BLOCK_KINDS = {"needs_input", "capability"}
DEFAULT_TARGETS = ["telegram"]
LEGACY_HUMAN_KEYWORDS = (
    "secret", "credential", "token", "password", "api key", "access",
    "permission", "dns", "cloudflare", "https", "domain", "choose",
    "decision", "user input", "human", "manual",
)

_LONG_SECRETISH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z0-9_\-]{28,}(?![A-Za-z0-9_])")
_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASS|KEY|CREDENTIAL)[A-Z0-9_]*)\s*=\s*([^\s,;]+)"
)
_URL_TOKEN_RE = re.compile(r"(?i)([?&](?:token|key|secret|password|access_token)=)[^&#\s]+")


def register(ctx) -> None:
    ctx.register_hook("kanban_task_blocked", _on_kanban_task_blocked)


def _plugin_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        entry = (((cfg.get("plugins") or {}).get("entries") or {}).get("kanban-block-notifier") or {})
        return entry if isinstance(entry, dict) else {}
    except Exception:
        return {}


def _as_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(part).strip() for part in value if str(part).strip()]
    return list(default)


def _state_db_path(config: dict[str, Any]) -> Path | None:
    raw = str(config.get("state_db") or "").strip()
    if raw:
        return Path(raw).expanduser()
    try:
        from hermes_cli import kanban_db as kb

        return kb.kanban_home() / "kanban" / "kanban-block-notifier.sqlite3"
    except Exception:
        # Persistent state must follow Hermes' profile/custom-home resolution.
        # If that resolver is unavailable, fail closed rather than writing to a
        # parallel ~/.hermes tree and weakening event-scoped deduplication.
        return None


def _ensure_state(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sent_block_notifications (
            dedupe_key TEXT NOT NULL,
            target TEXT NOT NULL,
            sent_at INTEGER NOT NULL,
            PRIMARY KEY (dedupe_key, target)
        )
        """
    )
    conn.commit()


def _mark_sent_once(db_path: Path, dedupe_key: str, target: str) -> bool:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path), timeout=30) as conn:
        _ensure_state(conn)
        try:
            conn.execute(
                "INSERT INTO sent_block_notifications (dedupe_key, target, sent_at) VALUES (?, ?, ?)",
                (dedupe_key, target, int(time.time())),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def _unmark_sent(db_path: Path, dedupe_key: str, target: str) -> None:
    try:
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            conn.execute(
                "DELETE FROM sent_block_notifications WHERE dedupe_key = ? AND target = ?",
                (dedupe_key, target),
            )
            conn.commit()
    except Exception:
        pass


def _sanitize(text: str, limit: int = 240) -> str:
    text = str(text or "").replace("\r", " ").strip()
    text = _ASSIGNMENT_SECRET_RE.sub(lambda m: f"{m.group(1)}=[redacted]", text)
    text = _URL_TOKEN_RE.sub(lambda m: f"{m.group(1)}[redacted]", text)
    text = _LONG_SECRETISH_RE.sub("[redacted]", text)
    text = re.sub(r"\s+", " ", text)
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _fetch_task(board: str | None, task_id: str):
    try:
        from hermes_cli import kanban_db as kb

        conn = kb.connect(board=board)
        try:
            return kb.get_task(conn, task_id)
        finally:
            conn.close()
    except Exception:
        return None


def _is_human_block(kind: str | None, reason: str, config: dict[str, Any]) -> bool:
    reason_l = (reason or "").casefold()
    if str(config.get("suppress_review_required", True)).lower() not in {"0", "false", "no", "off"}:
        if reason_l.startswith("review-required") or "review-required" in reason_l:
            return False
    notify_kinds = set(_as_list(config.get("notify_kinds"), sorted(HUMAN_BLOCK_KINDS)))
    if kind in notify_kinds:
        return True
    # Legacy/untyped blocks: notify only when the text strongly suggests a human-only input.
    return (not kind) and any(word in reason_l for word in LEGACY_HUMAN_KEYWORDS)


def _build_message(*, board: str | None, task_id: str, title: str, assignee: str | None, kind: str | None, reason: str, config: dict[str, Any]) -> str:
    safe_reason = _sanitize(reason, limit=int(config.get("reason_limit", 220) or 220))
    safe_title = _sanitize(title or task_id, limit=120)
    board_part = f"[{_sanitize(board or 'default', 64)}] "
    assignee_part = f" @{_sanitize(assignee, 48)}" if assignee else ""
    kind_part = _sanitize(kind or "blocked", 32)
    lines = [
        f"⏸ {board_part}Kanban {task_id}{assignee_part}: нужен ввод человека",
        f"Задача: {safe_title}",
        f"Тип блокера: {kind_part}",
    ]
    if safe_reason:
        lines.append(f"Причина: {safe_reason}")
    if _looks_secret_related(reason):
        secure_drop_url = str(config.get("secure_drop_url") or "").strip()
        if secure_drop_url:
            lines.append(f"Секреты не присылайте в чат; secure-drop: {secure_drop_url}")
        else:
            lines.append("Секреты не присылайте в чат; secure-drop не настроен.")
    return "\n".join(lines)


def _looks_secret_related(reason: str) -> bool:
    r = (reason or "").casefold()
    return any(word in r for word in ("secret", "token", "credential", "password", "api key", "ключ", "секрет", "парол"))


def _send(target: str, message: str, config: dict[str, Any]) -> None:
    hermes_exe = str(config.get("hermes_command") or os.environ.get("HERMES_CLI", "hermes"))
    timeout = int(config.get("send_timeout_seconds", 10) or 10)
    subprocess.run(
        [hermes_exe, "send", "--to", target, "--quiet"],
        input=message,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
        check=True,
    )


def _dedupe_key(
    board: str | None,
    task_id: str,
    kind: str | None,
    reason: str,
    *,
    event_id: Any = None,
    run_id: Any = None,
) -> str:
    board_key = board or "default"
    if event_id is not None:
        return f"{board_key}:{task_id}:event:{event_id}"
    if run_id is not None:
        return f"{board_key}:{task_id}:run:{run_id}"
    reason_hash = hashlib.sha256(_sanitize(reason, 500).encode("utf-8")).hexdigest()[:20]
    return f"{board_key}:{task_id}:{kind or 'legacy'}:{reason_hash}"


def _on_kanban_task_blocked(**kwargs: Any) -> None:
    task_id = str(kwargs.get("task_id") or "").strip()
    if not task_id:
        return
    config = _plugin_config()
    if str(config.get("enabled", True)).lower() in {"0", "false", "no", "off"}:
        return

    board = kwargs.get("board") or None
    reason = str(kwargs.get("reason") or "")
    task = _fetch_task(board, task_id)
    kind = getattr(task, "block_kind", None) or kwargs.get("kind") or None
    title = getattr(task, "title", "") or task_id
    assignee = getattr(task, "assignee", None) or kwargs.get("assignee") or None

    if not _is_human_block(kind, reason, config):
        return

    targets = _as_list(config.get("targets"), DEFAULT_TARGETS)
    if not targets:
        return
    message = _build_message(
        board=board,
        task_id=task_id,
        title=title,
        assignee=assignee,
        kind=kind,
        reason=reason,
        config=config,
    )
    state_db = _state_db_path(config)
    if state_db is None:
        logger.warning(
            "kanban-block-notifier: persistent state root unavailable; notification skipped"
        )
        return
    key = _dedupe_key(
        board,
        task_id,
        kind,
        reason,
        event_id=kwargs.get("event_id"),
        run_id=kwargs.get("run_id"),
    )

    for target in targets:
        if not _mark_sent_once(state_db, key, target):
            continue
        try:
            _send(target, message, config)
        except Exception as exc:
            _unmark_sent(state_db, key, target)
            # Never log the raw reason/message; it may mention credentials.
            logger.warning(
                "kanban-block-notifier: failed to send %s notification for task %s to %s: %s",
                kind or "blocked",
                task_id,
                target,
                exc.__class__.__name__,
            )
