"""Session topic segmentation runtime.

Topic state is durable in ``state.db`` while the cached system prompt remains
byte-stable.  The only model instruction is appended to the current user
message's ephemeral/``api_content`` tail, and the normal assistant response is
the only classifier call: no auxiliary or hidden LLM request is made.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Optional

logger = logging.getLogger("run_agent")

TOPIC_MESSAGE_FIELD = "_topic_id"
TOPIC_SESSION_CONFIG_KEY = "topic_segmentation_enabled"
_TOPIC_LINE_RE = re.compile(
    r"(?im)^[ \t]*TOPIC:[ \t]*([^\r\n]+?)[ \t]*\Z"
)
_TOPIC_WORD_RE = re.compile(r"[a-z0-9]+")
_MAX_TOPIC_TITLE_CHARS = 64


def _configured_default(config: Any) -> bool:
    session = config.get("session", {}) if isinstance(config, dict) else {}
    topics = session.get("topic_segmentation", {}) if isinstance(session, dict) else {}
    return bool(topics.get("enabled", False)) if isinstance(topics, dict) else False


def initialize_topic_segmentation(agent: Any, config: Any) -> None:
    """Resolve global + per-session enablement and restore the active topic.

    A persisted session override wins over the global default.  Every failure
    disables the feature for this agent rather than changing normal session
    loading or making a hidden model call.
    """
    enabled = _configured_default(config)
    agent._topic_segmentation_default = enabled
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    session_override: Optional[bool] = None
    if db is not None and session_id:
        try:
            override = db.get_session_model_config_value(
                session_id, TOPIC_SESSION_CONFIG_KEY, None
            )
            if isinstance(override, bool):
                session_override = override
                enabled = override
        except Exception:
            logger.warning(
                "Could not restore topic-segmentation state for session=%s; disabling it",
                session_id,
                exc_info=True,
            )
            enabled = False
    if session_override is not None and isinstance(
        getattr(agent, "_session_init_model_config", None), dict
    ):
        # Compression rotation copies this seed into the continuation row.
        agent._session_init_model_config[TOPIC_SESSION_CONFIG_KEY] = session_override
    agent._topic_segmentation_enabled = enabled
    agent._active_topic_id = None
    if enabled and db is not None and session_id:
        try:
            active = db.get_active_topic(session_id)
            agent._active_topic_id = active["id"] if active else None
        except Exception:
            logger.warning(
                "Could not restore active topic for session=%s; disabling topic segmentation",
                session_id,
                exc_info=True,
            )
            agent._topic_segmentation_enabled = False
    _sync_context_engine_topic(agent)


def refresh_topic_segmentation(agent: Any) -> None:
    """Restore topic state after a reused agent changes physical sessions."""
    initialize_topic_segmentation(
        agent,
        {
            "session": {
                "topic_segmentation": {
                    "enabled": bool(getattr(agent, "_topic_segmentation_default", False))
                }
            }
        },
    )


def _sync_context_engine_topic(agent: Any) -> None:
    """Keep all in-place compaction paths scoped to the selected topic."""
    engine = getattr(agent, "context_compressor", None)
    if engine is not None:
        engine._active_topic_id = (
            getattr(agent, "_active_topic_id", None)
            if getattr(agent, "_topic_segmentation_enabled", False)
            else None
        )


def normalize_topic_title(value: Any) -> str:
    """Return a short stable display title accepted from model/CLI output."""
    text = " ".join(str(value or "").strip().split())
    text = re.sub(r"^[`'\"*_\-]+|[`'\"*_\-]+$", "", text).strip()
    return text[:_MAX_TOPIC_TITLE_CHARS].rstrip()


def _topic_key(value: Any) -> str:
    return "-".join(_TOPIC_WORD_RE.findall(normalize_topic_title(value).lower()))


def parse_topic_signal(content: Any) -> Optional[str]:
    """Parse the final valid ``TOPIC:`` line; malformed/absent output is a no-op."""
    if not isinstance(content, str):
        return None
    match = _TOPIC_LINE_RE.search(content)
    if match is None:
        return None
    title = normalize_topic_title(match.group(1))
    return title or None


def match_existing_topic(title: str, topics: Iterable[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Conservatively fuzzy-match general and hyphen-qualified topic names."""
    wanted = _topic_key(title)
    if not wanted:
        return None
    exact = []
    prefix = []
    wanted_parts = wanted.split("-")
    for topic in topics:
        key = _topic_key(topic.get("title"))
        if key == wanted:
            exact.append(topic)
        elif key and (
            key.split("-")[0] == wanted_parts[0]
            and (key.startswith(wanted + "-") or wanted.startswith(key + "-"))
        ):
            prefix.append(topic)
    candidates = exact or prefix
    return candidates[0] if len(candidates) == 1 else None


def _initial_topic_title(message: Any) -> str:
    if isinstance(message, list):
        message = " ".join(
            str(part.get("text") or "")
            for part in message
            if isinstance(part, dict) and part.get("type") == "text"
        )
    text = normalize_topic_title(message)
    if not text:
        return "session"
    first_line = text.splitlines()[0]
    words = first_line.split()
    return normalize_topic_title(" ".join(words[:6])) or "session"


def prepare_topic_turn(
    agent: Any,
    messages: list[dict[str, Any]],
    current_turn_user_idx: int,
    original_user_message: Any,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Bind this turn to the active topic and load only that topic's model history.

    The durable database is the owner.  If it cannot be read, the supplied
    history is retained (normal chat remains usable) and segmentation is
    disabled for this agent so a partial topic view is never silently claimed.
    """
    prior = list(messages[:current_turn_user_idx])
    if not getattr(agent, "_topic_segmentation_enabled", False):
        return messages, current_turn_user_idx, prior
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if db is None or not session_id:
        agent._topic_segmentation_enabled = False
        agent._active_topic_id = None
        return messages, current_turn_user_idx, prior
    if not (0 <= current_turn_user_idx < len(messages)):
        return messages, current_turn_user_idx, prior

    user_message = messages[current_turn_user_idx]
    try:
        active = db.ensure_session_topic(
            session_id, _initial_topic_title(original_user_message)
        )
        topic_id = int(active["id"])
        history = db.get_messages_as_conversation(
            session_id,
            include_ancestors=True,
            repair_alternation=True,
            include_row_ids=True,
            topic_id=topic_id,
        )
    except Exception:
        logger.warning(
            "Topic history load failed for session=%s; keeping unsegmented history",
            session_id,
            exc_info=True,
        )
        agent._topic_segmentation_enabled = False
        agent._active_topic_id = None
        return messages, current_turn_user_idx, prior

    current_row_id = user_message.get("_row_id") if isinstance(user_message, dict) else None
    if isinstance(current_row_id, int):
        history = [row for row in history if row.get("_row_id") != current_row_id]
    user_message[TOPIC_MESSAGE_FIELD] = topic_id
    rebuilt = [*history, user_message]
    agent._active_topic_id = topic_id
    _sync_context_engine_topic(agent)
    agent._persist_user_message_idx = len(rebuilt) - 1
    return rebuilt, len(rebuilt) - 1, history


def topic_prompt_context(agent: Any) -> str:
    """Build the per-turn topic index/instruction without touching the system prompt."""
    if not getattr(agent, "_topic_segmentation_enabled", False):
        return ""
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if db is None or not session_id:
        return ""
    try:
        topics = db.get_topics(session_id)
    except Exception:
        logger.warning("Could not build session topic index", exc_info=True)
        return ""
    lines = ["[SESSION TOPICS — classify this turn; no extra model call]"]
    for topic in topics[:8]:
        marker = " active" if topic.get("state") == "active" else ""
        lines.append(
            f"- {topic['id']}: {topic['title']} ({topic.get('message_count', 0)} messages{marker})"
        )
    lines.extend(
        (
            "Append exactly one line to the response: TOPIC: <short-name>",
            "Reuse an existing short name for a follow-up; choose a new short name only for a clear subject change.",
        )
    )
    return "\n".join(lines)


def merge_topic_prompt_context(existing: str, agent: Any) -> str:
    block = topic_prompt_context(agent)
    if not block:
        return existing
    return f"{existing.rstrip()}\n\n{block}" if existing and existing.strip() else block


def process_turn_topic(agent: Any, messages: list[dict[str, Any]], final_response: Any) -> None:
    """Apply the response topic to the entire current turn, atomically when rows exist.

    The user row is crash-persisted before the provider call.  A later topic
    transition therefore retags every already-durable row in this turn inside
    the same transaction that activates/creates the topic; unflushed rows carry
    ``_topic_id`` into the normal append-only persistence funnel.
    """
    if not getattr(agent, "_topic_segmentation_enabled", False):
        return
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if db is None or not session_id:
        return
    start = getattr(agent, "_persist_user_message_idx", None)
    if not isinstance(start, int) or not (0 <= start < len(messages)):
        return

    title = parse_topic_signal(final_response)
    try:
        topics = db.get_topics(session_id)
        active = next((topic for topic in topics if topic.get("state") == "active"), None)
        target = match_existing_topic(title, topics) if title else active
        if target is None and not title:
            return
        turn_rows = [row for row in messages[start:] if isinstance(row, dict)]
        row_ids = [
            row["_row_id"]
            for row in turn_rows
            if isinstance(row.get("_row_id"), int) and row["_row_id"] > 0
        ]
        selected = db.activate_topic_for_messages(
            session_id,
            topic_id=int(target["id"]) if target else None,
            title=title if target is None else None,
            message_ids=row_ids,
            turn_lease_holder=getattr(agent, "_active_session_turn_lease_holder", None),
        )
        topic_id = int(selected["id"])
    except Exception:
        logger.warning(
            "Topic transition failed for session=%s; keeping the prior active topic",
            session_id,
            exc_info=True,
        )
        return

    agent._active_topic_id = topic_id
    _sync_context_engine_topic(agent)
    for row in messages[start:]:
        if isinstance(row, dict):
            row[TOPIC_MESSAGE_FIELD] = topic_id
