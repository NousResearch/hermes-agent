"""Passive Telegram ingress must preserve context without acquiring a turn."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import (
    GatewayRunner, _build_gateway_agent_history, _wrap_current_message_with_observed_context,
)
from gateway.session import SessionSource, SessionStore
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_telegram_group_gating import _group_message, _mention_entity


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # The global fixture pins DEFAULT_DB_PATH; restore dynamic per-scope lookup
    # so the real multiplex store follows this test's isolated profile homes.
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    store = SessionStore(home / "sessions", GatewayConfig())
    yield store
    store.close_all_db_handles()


def _adapter(store, *, username="research_bot", bot_id=999, **extra):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={
        "require_mention": True,
        "allow_bots": "mentions",
        "exclusive_bot_mentions": True,
        "observe_unmentioned_group_messages": True,
        "allowed_chats": ["-100"],
        "group_allowed_chats": ["-100"],
        "group_allow_from": ["111", "222"],
        "allowed_topics": [],
        "ignored_threads": [],
        "mention_patterns": [],
        **extra,
    }))
    # Only the Telegram transport and downstream agent callback are fake.
    adapter._bot = SimpleNamespace(
        id=bot_id, username=username, set_my_commands=AsyncMock(),
        send_chat_action=AsyncMock(),
    )
    adapter.set_session_store(store)
    runner = object.__new__(GatewayRunner)
    runner.config = store.config
    adapter.gateway_runner = runner
    adapter.set_message_handler(AsyncMock(return_value=None))
    return adapter


async def _deliver(adapter, message):
    await adapter._handle_text_message(
        SimpleNamespace(update_id=message.message_id, message=message, effective_message=message), None,
    )
    # Drain real batching and base-adapter dispatch, without timing assertions.
    await asyncio.gather(*list(adapter._pending_text_batch_tasks.values()))
    await asyncio.gather(*list(adapter._background_tasks))


@pytest.mark.asyncio
@pytest.mark.parametrize("is_bot", [False, True], ids=["human", "peer-bot"])
@pytest.mark.parametrize("case,observed", [
    ("plain", True), ("reply", True), ("wake", True),
    ("open", True), ("free-chat", True), ("free-topic", True),
    ("disabled", False), ("no-chat-allowlist", False), ("outside-chat", False),
    ("unauthorized", False), ("ignored-topic", False), ("outside-topic", False),
])
async def test_peer_addressing_is_passive_only_inside_observation_gates(session_store, is_bot, case, observed):
    settings = {
        "wake": {"mention_patterns": [r"^wake\b"]},
        "open": {"require_mention": False},
        "free-chat": {"free_response_chats": ["-100"]},
        "free-topic": {"free_response_topics": ["-100:7"]},
        "disabled": {"observe_unmentioned_group_messages": False},
        "no-chat-allowlist": {"group_allowed_chats": []},
        "outside-chat": {"allowed_chats": ["-200"]},
        "unauthorized": {"group_allow_from": ["222"]},
        "ignored-topic": {"ignored_threads": [7]},
        "outside-topic": {"allowed_topics": ["8"]},
    }.get(case, {})
    adapter = _adapter(session_store, **settings)
    text = "wake @other_bot keep this report as context"
    msg = _group_message(text, thread_id=7, reply_to_bot=case == "reply",
                         entities=[_mention_entity(text, "@other_bot")])
    msg.from_user.is_bot = is_bot
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", thread_id="7")
    entry = session_store.get_or_create_session(source)

    await _deliver(adapter, msg)

    adapter._message_handler.assert_not_awaited()
    assert not adapter._pending_text_batches
    rows = session_store.load_transcript(entry.session_id)
    assert [row["content"] for row in rows] == ([f"[Alice Example|111]\n{text}"] if observed else [])
    assert all(row.get("observed") is True for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id", [None, 7], ids=["group", "topic"])
@pytest.mark.parametrize("is_bot", [False, True], ids=["human-trigger", "bot-handoff"])
async def test_multiplex_observations_rejoin_the_owning_turn_without_changing_replay(session_store, thread_id, is_bot):
    session_store.config.multiplex_profiles = True
    profile_home = Path.home() / ".hermes" / "profiles" / "research"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    adapters = {
        "default": _adapter(session_store, username="default_bot", bot_id=998),
        "research": _adapter(session_store),
    }
    entries, prefixes = {}, {}
    for profile, adapter in adapters.items():
        adapter.set_owner_profile(profile)
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="-100", chat_type="group",
            thread_id=str(thread_id) if thread_id else None, profile=profile,
        )
        entries[profile] = entry = session_store.get_or_create_session(source)
        for row in [{"role": "user", "content": f"Earlier request for {profile}"},
                    {"role": "assistant", "content": f"Earlier answer for {profile}"}]:
            session_store.append_to_transcript(entry.session_id, row)
        prefixes[profile], _ = _build_gateway_agent_history(session_store.load_transcript(entry.session_id))

    text = "@third_bot the shared report is ready"
    common = _group_message(text, thread_id=thread_id, entities=[_mention_entity(text, "@third_bot")])
    common.from_user.is_bot = True
    for profile, adapter in adapters.items():
        await _deliver(adapter, common)
        private = _group_message(f"only {profile} received this", thread_id=thread_id)
        private.message_id = 43
        await _deliver(adapter, private)
        elsewhere = _group_message("different topic report", thread_id=8)
        elsewhere.message_id = 46
        await _deliver(adapter, elsewhere)
        adapter._message_handler.assert_not_awaited()

    assert entries["default"].session_id != entries["research"].session_id
    for profile, adapter in adapters.items():
        entry = entries[profile]
        rows = session_store.load_transcript(entry.session_id)
        observed = [row["content"] for row in rows if row.get("observed")]
        observed_text = "\n".join(observed)
        other = "research" if profile == "default" else "default"
        assert f"[Alice Example|111]\n{text}" in observed_text
        assert f"only {profile} received this" in observed_text
        assert f"only {other} received this" not in observed_text
        assert "different topic report" not in observed_text
        # An authorized explicit own mention still triggers, even from a peer bot.
        trigger_text = f"@{adapter._bot.username} what did Alice report?"
        trigger = _group_message(
            trigger_text, thread_id=thread_id, from_user_id=222, from_user_name="Bob",
            entities=[_mention_entity(trigger_text, f"@{adapter._bot.username}")],
        )
        trigger.message_id = 44
        trigger.from_user.is_bot = is_bot
        await _deliver(adapter, trigger)
        adapter._message_handler.assert_awaited_once()
        event = adapter._message_handler.await_args.args[0]
        assert event.source.profile == profile
        assert event.source.user_id is None
        assert session_store.get_or_create_session(event.source).session_id == entry.session_id
        assert adapter._text_batch_key(event) == entry.session_key
        history, context = _build_gateway_agent_history(rows, channel_prompt=event.channel_prompt)
        assert history == prefixes[profile]
        assert context == "\n".join(observed)
        wrapped = _wrap_current_message_with_observed_context(event.text, context)
        assert wrapped.endswith(event.text)
        assert text in wrapped
        assert context not in event.text

        # Later passive input changes only the current-turn wrapper, never replay.
        later = _group_message("later report", thread_id=thread_id)
        later.message_id = 45
        await _deliver(adapter, later)
        later_rows = session_store.load_transcript(entry.session_id)
        later_history, later_context = _build_gateway_agent_history(later_rows, channel_prompt=event.channel_prompt)
        assert later_history == history == prefixes[profile]
        assert "later report" not in wrapped
        assert "later report" in _wrap_current_message_with_observed_context(event.text, later_context)
        adapter._message_handler.assert_awaited_once()

    # The physical profile DBs, not only in-memory keys, must stay isolated.
    import sqlite3
    for profile, home in [("default", Path.home() / ".hermes"), ("research", profile_home)]:
        with sqlite3.connect(home / "state.db") as db:
            stored = [row[0] for row in db.execute("SELECT content FROM messages")]
        other = "research" if profile == "default" else "default"
        assert any(f"only {profile} received this" in content for content in stored), profile
        assert not any(f"only {other} received this" in content for content in stored)
