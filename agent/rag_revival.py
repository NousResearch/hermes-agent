"""RAG Revival snapshots for compression handoffs.

When context compression drops older turns, write the same handoff summary to
an Obsidian vault folder grouped by Hermes profile/agent.  The note is meant as
a durable revival anchor: project files + Ágora messages + this note should be
enough for a future worker to continue after compaction.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

_DEFAULT_VAULT = Path.home() / "Documents" / "Obsidian Vault"
_DEFAULT_RELATIVE_DIR = Path("01 Projetos") / "Ágora" / "RAG Revival"
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _load_rag_config() -> dict[str, Any]:
    """Load RAG Revival settings from config.yaml, with env-var back-compat.

    Behavioural settings live in ``rag_revival.*`` in config.yaml. For
    transitional back-compat, ``HERMES_RAG_REVIVAL_ENABLED`` and
    ``HERMES_RAG_REVIVAL_DIR`` still override the matching config keys when
    explicitly set.
    """
    try:
        from hermes_cli.config import load_config
        cfg = (load_config() or {}).get("rag_revival", {})
    except Exception:
        cfg = {}
    return {
        "enabled": cfg.get("enabled", True),
        "vault_dir": cfg.get("vault_dir", "").strip(),
    }


def _rag_revival_enabled(config: dict[str, Any] | None = None) -> bool:
    cfg = config if config is not None else _load_rag_config()
    # Back-compat: env vars override config
    env_raw = os.environ.get("HERMES_RAG_REVIVAL_ENABLED")
    if env_raw is not None:
        return env_raw.strip().lower() not in {"0", "false", "no", "off"}
    return bool(cfg.get("enabled", True))


def _rag_revival_dir(config: dict[str, Any] | None = None) -> str:
    cfg = config if config is not None else _load_rag_config()
    # Back-compat: env vars override config
    env_raw = os.environ.get("HERMES_RAG_REVIVAL_DIR", "").strip()
    if env_raw:
        return env_raw
    return cfg.get("vault_dir", "") or str(_DEFAULT_RELATIVE_DIR)


def _vault_path() -> Path:
    raw = os.environ.get("OBSIDIAN_VAULT_PATH", "").strip()
    return Path(raw).expanduser() if raw else _DEFAULT_VAULT


def _safe_stem(value: str, fallback: str = "default") -> str:
    value = (value or "").strip() or fallback
    value = _SAFE_STEM_RE.sub("-", value).strip(".-_")
    return value or fallback


def _active_profile_name(agent: Any) -> str:
    for attr in ("profile", "profile_name", "agent_identity"):
        value = getattr(agent, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    env_profile = os.environ.get("HERMES_PROFILE", "").strip()
    if env_profile:
        return env_profile
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _message_excerpt(message: dict[str, Any], max_chars: int = 500) -> str:
    role = str(message.get("role") or "unknown")
    content = message.get("content")
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if text:
                    text_parts.append(str(text))
            elif isinstance(part, str):
                text_parts.append(part)
        text = " ".join(text_parts)
    else:
        text = str(content or "")
    text = redact_sensitive_text(re.sub(r"\s+", " ", text)).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return f"- **{role}**: {text}" if text else f"- **{role}**: [empty]"


def _recent_turns(messages: Iterable[dict[str, Any]], limit: int = 8) -> str:
    rows = list(messages)[-limit:]
    if not rows:
        return "- [none]"
    return "\n".join(_message_excerpt(m) for m in rows if isinstance(m, dict)) or "- [none]"


def _load_agora_status(profile: str) -> str:
    try:
        from hermes_constants import get_default_hermes_root
        db_path = get_default_hermes_root() / "agora.db"
        if not db_path.exists():
            return "- Ágora DB not found."
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            agent = conn.execute(
                "SELECT profile, state, current_task_id, current_step, status_text, last_heartbeat_at, pid, run_id "
                "FROM agora_agent_status WHERE profile = ?",
                (profile,),
            ).fetchone()
            notes: list[str] = []
            if agent:
                notes.append(
                    "- Agent status: "
                    f"state={agent['state']}, task={agent['current_task_id']}, "
                    f"step={agent['current_step']}, pid={agent['pid']}, run={agent['run_id']}."
                )
                if agent["status_text"]:
                    notes.append(f"- Status text: {agent['status_text']}")
            counts = conn.execute(
                "SELECT COUNT(*) total, "
                "SUM(CASE WHEN read_at IS NULL THEN 1 ELSE 0 END) unread, "
                "SUM(CASE WHEN ack_at IS NULL THEN 1 ELSE 0 END) unacked "
                "FROM agora_notifications WHERE recipient = ?",
                (profile,),
            ).fetchone()
            if counts and counts["total"]:
                notes.append(
                    f"- Notifications: total={counts['total']}, unread={counts['unread'] or 0}, unacked={counts['unacked'] or 0}."
                )
            return "\n".join(notes) if notes else "- No Ágora status for this profile."
        finally:
            conn.close()
    except Exception as exc:
        return f"- Ágora status unavailable: {type(exc).__name__}: {exc}"


def _build_note(
    *,
    agent: Any,
    profile: str,
    old_session_id: str,
    new_session_id: str,
    summary_text: str,
    messages: list[dict[str, Any]],
    approx_tokens: int | None,
    in_place: bool,
    focus_topic: str | None,
) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    model = getattr(agent, "model", "") or "unknown"
    provider = getattr(agent, "provider", "") or "unknown"
    platform = getattr(agent, "platform", "") or "cli"
    cwd = os.environ.get("TERMINAL_CWD") or os.getcwd()
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    board = os.environ.get("HERMES_KANBAN_BOARD", "").strip()
    workspace = os.environ.get("HERMES_KANBAN_WORKSPACE", "").strip()
    summary_text = redact_sensitive_text(summary_text.strip())
    frontmatter = {
        "created": time.strftime("%Y-%m-%d"),
        "type": "rag-revival",
        "profile": profile,
        "project": "Ágora",
        "session_id": new_session_id or old_session_id,
        "old_session_id": old_session_id,
        "model": model,
        "provider": provider,
        "tags": ["agora", "rag-revival", "compression", profile],
    }
    yamlish = "---\n" + "\n".join(
        f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v}"
        for k, v in frontmatter.items()
        if v is not None and v != ""
    ) + "\n---"
    return f"""{yamlish}

# RAG Revival — {profile} — {now}

## Purpose

Snapshot generated automatically around a Hermes context compression. Use this together with project files, Kanban task state, and Ágora channel history to continue work after compaction or restart.

## Runtime

- Profile: `{profile}`
- Platform: `{platform}`
- Model/provider: `{model}` / `{provider}`
- Session before compression: `{old_session_id or 'unknown'}`
- Session after compression: `{new_session_id or old_session_id or 'unknown'}`
- In-place compression: `{in_place}`
- Approx tokens before compression: `{approx_tokens if approx_tokens is not None else 'unknown'}`
- Focus topic: `{focus_topic or ''}`
- CWD: `{cwd}`
- Kanban board/task/workspace: `{board or '-'} / {task_id or '-'} / {workspace or '-'}`

## Ágora status at snapshot

{_load_agora_status(profile)}

## Compaction handoff summary

{summary_text}

## Recent live tail still in context

{_recent_turns(messages)}

## Revival checklist

1. Read this note first.
2. Open the current Kanban card with `kanban_show()` or `hermes kanban --board {board or '<board>'} show {task_id or '<task>'}`.
3. Read recent Ágora messages for the task/profile.
4. Inspect changed files/tests from the workspace before continuing.
5. Do not trust stale pending items unless the current Kanban/Ágora state still confirms them.
"""


def _note_path(out_dir: Path, new_session_id: str, old_session_id: str, agent: Any) -> Path:
    """Build a collision-resistant filename for the snapshot.

    Uses microseconds so that repeated compressions on the same session do not
    overwrite earlier revival notes.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    sid = _safe_stem(
        new_session_id or old_session_id or getattr(agent, "session_id", "") or "session"
    )
    return out_dir / f"{ts}-{sid}.md"


def write_rag_revival_snapshot(
    agent: Any,
    *,
    messages: list[dict[str, Any]],
    summary_text: str,
    old_session_id: str = "",
    new_session_id: str = "",
    approx_tokens: int | None = None,
    in_place: bool = False,
    focus_topic: str | None = None,
) -> Path | None:
    """Write a compression handoff note to the Obsidian vault.

    Best-effort by design: this function always catches its own errors and
    returns ``None`` on failure so that compression can never be blocked by an
    Obsidian/vault problem.
    """
    if not _rag_revival_enabled():
        return None
    try:
        vault = _vault_path()
        if not vault.exists():
            logger.debug("RAG Revival skipped: vault path does not exist: %s", vault)
            return None
        profile = _safe_stem(_active_profile_name(agent))
        rel = _rag_revival_dir()
        base_dir = vault / (Path(rel) if rel else _DEFAULT_RELATIVE_DIR)
        out_dir = base_dir / profile
        out_dir.mkdir(parents=True, exist_ok=True)
        path = _note_path(out_dir, new_session_id, old_session_id, agent)
        note = _build_note(
            agent=agent,
            profile=profile,
            old_session_id=old_session_id,
            new_session_id=new_session_id,
            summary_text=summary_text,
            messages=messages,
            approx_tokens=approx_tokens,
            in_place=in_place,
            focus_topic=focus_topic,
        )
        path.write_text(note, encoding="utf-8")
        logger.info("RAG Revival snapshot written: %s", path)
        return path
    except Exception as exc:
        logger.debug("RAG Revival snapshot failed (non-blocking): %s", exc)
        return None
