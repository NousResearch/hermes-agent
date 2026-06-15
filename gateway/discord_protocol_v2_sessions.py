"""Session mapping helpers for Discord Native Multi-Bot Protocol v2.

Protocol v2 maps a Discord topic/work unit to a separate Hermes transcript per
agent.  The authoritative durable mapping lives in ``topic_agent_sessions``;
``SessionStore`` is used only to create/rebind the Hermes transcript entry.
"""

from __future__ import annotations

from dataclasses import dataclass

from gateway.config import Platform
from gateway.discord_protocol_v2_store import DiscordProtocolV2Store
from gateway.session import (
    SessionSource,
    SessionStore,
    build_discord_v2_session_key,
)


@dataclass(frozen=True)
class DiscordV2TopicAgentSession:
    """Resolved Discord protocol v2 topic × agent session mapping."""

    topic_id: str
    agent_id: str
    session_key: str
    session_id: str

    @property
    def hermes_session_id(self) -> str:
        """Alias matching the durable store column name."""
        return self.session_id


def default_discord_v2_session_source(topic_id: str) -> SessionSource:
    """Build a minimal topic-scoped Discord source for v2 session bookkeeping.

    Agent identity intentionally does not appear in this source.  It is part of
    the durable v2 key/mapping, while the topic remains the work unit.
    """

    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=str(topic_id),
        chat_name=f"Discord topic {topic_id}",
        chat_type="thread",
        thread_id=str(topic_id),
    )


def get_or_create_discord_v2_session(
    *,
    protocol_store: DiscordProtocolV2Store,
    session_store: SessionStore,
    topic_id: str,
    agent_id: str,
    source: SessionSource | None = None,
) -> DiscordV2TopicAgentSession:
    """Resolve or create the Hermes transcript for a v2 topic × agent pair.

    The ``topic_agent_sessions`` row is authoritative across restarts.  When it
    already exists, ``SessionStore`` is rebound to that transcript ID even if the
    legacy JSON session index is absent or stale.  When it does not exist, a new
    Hermes transcript session is created through ``SessionStore`` and persisted
    to ``topic_agent_sessions``.
    """

    topic_id = str(topic_id)
    agent_id = str(agent_id)
    session_key = build_discord_v2_session_key(topic_id, agent_id)
    source = source or default_discord_v2_session_source(topic_id)

    existing = protocol_store.get_topic_agent_session(
        topic_id=topic_id,
        agent_id=agent_id,
    )
    if existing is not None:
        entry = session_store.bind_session_key(
            session_key,
            str(existing["hermes_session_id"]),
            source,
        )
        return DiscordV2TopicAgentSession(
            topic_id=topic_id,
            agent_id=agent_id,
            session_key=session_key,
            session_id=entry.session_id,
        )

    entry = session_store.get_or_create_session_for_key(session_key, source)
    protocol_store.upsert_topic_agent_session(
        topic_id=topic_id,
        agent_id=agent_id,
        hermes_session_id=entry.session_id,
        session_key=session_key,
    )
    return DiscordV2TopicAgentSession(
        topic_id=topic_id,
        agent_id=agent_id,
        session_key=session_key,
        session_id=entry.session_id,
    )
