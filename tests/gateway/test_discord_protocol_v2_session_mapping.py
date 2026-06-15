"""Discord protocol v2 topic × agent session mapping tests."""

from __future__ import annotations

import json

from hermes_state import SessionDB

from gateway.config import GatewayConfig, Platform
from gateway.discord_protocol_v2_sessions import get_or_create_discord_v2_session
from gateway.discord_protocol_v2_store import DiscordProtocolV2Store
from gateway.session import (
    SessionSource,
    SessionStore,
    build_discord_v2_session_key,
    build_session_key,
)


def _session_store(tmp_path, monkeypatch) -> SessionStore:
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    sessions_dir = tmp_path / "sessions"
    return SessionStore(
        sessions_dir=sessions_dir,
        config=GatewayConfig(sessions_dir=sessions_dir),
    )


def _seed_identity(store: DiscordProtocolV2Store, agent_id: str) -> None:
    store.upsert_identity(
        agent_id=agent_id,
        hermes_profile="default",
        discord_application_id=f"app-{agent_id}",
        discord_bot_user_id=f"bot-{agent_id}",
        token_secret_ref=f"secret://discord/{agent_id}",
        capabilities=["reply"],
        scopes={"guild_ids": ["guild-1"]},
        enabled=True,
    )


def _seed_topic(store: DiscordProtocolV2Store, topic_id: str) -> None:
    store.upsert_topic(
        topic_id=topic_id,
        guild_id="guild-1",
        channel_id=f"channel-{topic_id}",
        thread_id=f"thread-{topic_id}",
        parent_channel_id="parent-1",
        title=f"Topic {topic_id}",
        state={"phase": "open"},
    )


def _seed_store(store: DiscordProtocolV2Store) -> None:
    for agent_id in ("bohumil", "karel"):
        _seed_identity(store, agent_id)
    for topic_id in ("topic-1", "topic-2"):
        _seed_topic(store, topic_id)


def test_build_discord_v2_session_key_is_topic_agent_scoped() -> None:
    assert (
        build_discord_v2_session_key("topic-1", "bohumil")
        == "discord:v2:topic:topic-1:agent:bohumil"
    )


def test_same_topic_same_agent_reuses_session_after_restart(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "discord-v2.sqlite3"
    sessions_dir = tmp_path / "sessions"
    protocol_store = DiscordProtocolV2Store(db_path)
    _seed_store(protocol_store)

    first = get_or_create_discord_v2_session(
        protocol_store=protocol_store,
        session_store=_session_store(tmp_path, monkeypatch),
        topic_id="topic-1",
        agent_id="bohumil",
    )
    stored = protocol_store.get_topic_agent_session(
        topic_id="topic-1",
        agent_id="bohumil",
    )
    assert stored is not None
    assert stored["hermes_session_id"] == first.session_id
    protocol_store.close()

    # Prove restart reuse is owned by topic_agent_sessions, not only sessions.json.
    (sessions_dir / "sessions.json").unlink()

    restarted_store = DiscordProtocolV2Store(db_path)
    restarted = get_or_create_discord_v2_session(
        protocol_store=restarted_store,
        session_store=_session_store(tmp_path, monkeypatch),
        topic_id="topic-1",
        agent_id="bohumil",
    )

    assert restarted.session_key == "discord:v2:topic:topic-1:agent:bohumil"
    assert restarted.session_id == first.session_id
    assert SessionDB(db_path=tmp_path / "state.db").get_session(first.session_id) is not None

    saved_sessions = json.loads((sessions_dir / "sessions.json").read_text())
    assert saved_sessions[restarted.session_key]["session_id"] == first.session_id


def test_same_topic_different_agents_get_different_sessions(tmp_path, monkeypatch) -> None:
    protocol_store = DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3")
    _seed_store(protocol_store)
    session_store = _session_store(tmp_path, monkeypatch)

    bohumil = get_or_create_discord_v2_session(
        protocol_store=protocol_store,
        session_store=session_store,
        topic_id="topic-1",
        agent_id="bohumil",
    )
    karel = get_or_create_discord_v2_session(
        protocol_store=protocol_store,
        session_store=session_store,
        topic_id="topic-1",
        agent_id="karel",
    )

    assert bohumil.session_key == "discord:v2:topic:topic-1:agent:bohumil"
    assert karel.session_key == "discord:v2:topic:topic-1:agent:karel"
    assert bohumil.session_id != karel.session_id
    assert {
        row["agent_id"]: row["hermes_session_id"]
        for row in protocol_store.list_topic_agent_sessions(topic_id="topic-1")
    } == {"bohumil": bohumil.session_id, "karel": karel.session_id}


def test_different_topics_same_agent_get_different_sessions(tmp_path, monkeypatch) -> None:
    protocol_store = DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3")
    _seed_store(protocol_store)
    session_store = _session_store(tmp_path, monkeypatch)

    topic_1 = get_or_create_discord_v2_session(
        protocol_store=protocol_store,
        session_store=session_store,
        topic_id="topic-1",
        agent_id="bohumil",
    )
    topic_2 = get_or_create_discord_v2_session(
        protocol_store=protocol_store,
        session_store=session_store,
        topic_id="topic-2",
        agent_id="bohumil",
    )

    assert topic_1.session_key == "discord:v2:topic:topic-1:agent:bohumil"
    assert topic_2.session_key == "discord:v2:topic:topic-2:agent:bohumil"
    assert topic_1.session_id != topic_2.session_id
    assert {
        row["topic_id"]: row["hermes_session_id"]
        for row in protocol_store.list_topic_agent_sessions(agent_id="bohumil")
    } == {"topic-1": topic_1.session_id, "topic-2": topic_2.session_id}


def test_legacy_discord_session_key_is_unchanged_when_v2_helper_exists() -> None:
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-1",
        chat_type="thread",
        user_id="human-1",
        thread_id="thread-1",
        guild_id="guild-1",
        parent_chat_id="parent-1",
    )

    assert build_session_key(source) == "agent:main:discord:thread:channel-1:thread-1"
    assert build_session_key(source) != build_discord_v2_session_key("thread-1", "bohumil")
