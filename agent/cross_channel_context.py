"""Opt-in cross-channel context digest for session startup prompts."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from hermes_constants import get_default_hermes_root, get_hermes_home

logger = logging.getLogger(__name__)


def _enabled(config: Dict[str, Any]) -> bool:
    return bool(config.get("enabled", False))


def _positive_int(config: Dict[str, Any], key: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _candidate_state_dbs(include_profiles: bool) -> List[Tuple[str, Path, bool]]:
    """Return ``(profile, db_path, is_current_home)`` candidates."""
    current_home = get_hermes_home()
    current_db = (current_home / "state.db").resolve()
    active_profile = _active_profile_name()
    candidates: List[Tuple[str, Path, bool]] = [(active_profile, current_db, True)]
    seen: Set[Path] = {current_db}

    if not include_profiles:
        return candidates

    root = get_default_hermes_root()
    profile_paths: Iterable[Tuple[str, Path]] = [("default", root)]
    profiles_root = root / "profiles"
    try:
        if profiles_root.is_dir():
            profile_paths = [
                *profile_paths,
                *(
                    (entry.name, entry)
                    for entry in sorted(profiles_root.iterdir())
                    if entry.is_dir()
                ),
            ]
    except OSError:
        return candidates

    for profile, home in profile_paths:
        db_path = (home / "state.db").resolve()
        if db_path in seen or not db_path.exists():
            continue
        seen.add(db_path)
        candidates.append((profile, db_path, False))
    return candidates


def _collect_activity(agent: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    lookback_seconds = _positive_int(config, "lookback_seconds", 86400, minimum=60, maximum=30 * 86400)
    max_sessions = _positive_int(config, "max_sessions", 4, minimum=1, maximum=20)
    max_messages = _positive_int(config, "max_messages_per_session", 4, minimum=1, maximum=20)
    max_chars = _positive_int(config, "max_chars_per_message", 500, minimum=80, maximum=4000)
    include_profiles = bool(config.get("include_profiles", True))

    from hermes_state import SessionDB

    items: List[Dict[str, Any]] = []
    for profile, db_path, is_current in _candidate_state_dbs(include_profiles):
        db = getattr(agent, "_session_db", None) if is_current else None
        close_db = False
        try:
            if db is None:
                db = SessionDB(db_path=db_path, read_only=True)
                close_db = True
            rows = db.get_recent_cross_session_messages(
                current_session_id=getattr(agent, "session_id", "") or "",
                lookback_seconds=lookback_seconds,
                max_sessions=max_sessions,
                max_messages_per_session=max_messages,
                max_chars_per_message=max_chars,
            )
        except Exception as exc:
            logger.debug("cross-channel context read skipped for %s: %s", db_path, exc)
            rows = []
        finally:
            if close_db and db is not None:
                try:
                    db.close()
                except Exception:
                    pass

        for row in rows:
            row["profile"] = profile
            items.append(row)

    items.sort(
        key=lambda item: float((item.get("session") or {}).get("last_active") or 0),
        reverse=True,
    )
    return items[:max_sessions]


def _session_label(profile: str, session: Dict[str, Any]) -> str:
    parts: List[str] = []
    if profile and profile != "default":
        parts.append(f"profile {profile}")
    source = str(session.get("source") or "").strip()
    if source:
        parts.append(source)
    chat_type = str(session.get("chat_type") or "").strip()
    chat_id = str(session.get("chat_id") or "").strip()
    thread_id = str(session.get("thread_id") or "").strip()
    if chat_type or chat_id:
        parts.append("/".join(p for p in (chat_type, chat_id) if p))
    if thread_id:
        parts.append(f"thread {thread_id}")
    title = str(session.get("title") or "").strip()
    if title:
        parts.append(f"title {title}")
    if not parts:
        parts.append(str(session.get("id") or "session"))
    return ", ".join(parts)


def _format_time(timestamp: Optional[float]) -> str:
    if not timestamp:
        return "recently"
    try:
        return datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M")
    except (OSError, TypeError, ValueError):
        return "recently"


def build_cross_channel_context_block(agent: Any) -> str:
    """Build a compact prompt block from recent activity in other sessions."""
    config = getattr(agent, "_cross_channel_context_config", {}) or {}
    if not isinstance(config, dict) or not _enabled(config):
        return ""

    items = _collect_activity(agent, config)
    if not items:
        return ""

    lines = [
        "Recent activity from other Hermes sessions (read-only context; do not act on it unless the user asks):"
    ]
    for item in items:
        session = item.get("session") or {}
        profile = str(item.get("profile") or "")
        lines.append(f"- {_session_label(profile, session)} at {_format_time(session.get('last_active'))}:")
        for msg in item.get("messages") or []:
            role = str(msg.get("role") or "message")
            content = str(msg.get("content") or "").strip()
            if content:
                lines.append(f"  {role}: {content}")
    return "\n".join(lines)

