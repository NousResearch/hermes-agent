"""Durable passive observation for approved WhatsApp groups."""

import asyncio
import os
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource, SessionStore


GROUP_JID = "120363001234567890@g.us"
OTHER_GROUP_JID = "120363009999999999@g.us"
BOT_JID = "15551230000@s.whatsapp.net"


class _FakeSessionEntry:
    session_id = "whatsapp-shared-group-session"


class _FakeSessionStore:
    def __init__(self):
        self.sources = []
        self.messages = []

    def get_or_create_session(self, source):
        self.sources.append(source)
        return _FakeSessionEntry()

    def append_to_transcript(self, session_id, message, skip_db=False):
        self.messages.append((session_id, message, skip_db))

    def has_platform_message_id(self, session_id, platform_message_id):
        return any(
            stored_session_id == session_id
            and message.get("message_id") == platform_message_id
            for stored_session_id, message, _skip_db in self.messages
        )


def _make_adapter(*, observe=True, history_backfill=True, allowed=None):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    allowed = list(allowed or [GROUP_JID])
    extra = {
        "require_mention": True,
        "group_policy": "allowlist",
        "group_allow_from": allowed,
        "observe_unmentioned_group_messages": observe,
        "observe_allowed_chats": allowed,
        "history_backfill_limit": 50,
    }
    if history_backfill is not None:
        extra["history_backfill"] = history_backfill
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._message_handler = AsyncMock()
    adapter._dm_policy = "pairing"
    adapter._allow_from = set()
    adapter._group_policy = "allowlist"
    adapter._group_allow_from = set(allowed)
    adapter._mention_patterns = []
    adapter._authorization_check = None
    adapter._group_history_buffers = {}
    adapter._group_history_watermarks = {}
    adapter._group_history_seq = 0
    adapter._session_store = _FakeSessionStore()
    adapter.build_source = lambda **kwargs: SessionSource(
        platform=Platform.WHATSAPP,
        **kwargs,
    )
    return adapter


def _group_message(
    body="ambient chatter",
    *,
    chat_id=GROUP_JID,
    sender="6281234567890@s.whatsapp.net",
    sender_name="Alice",
    timestamp="1758254144",
    **overrides,
):
    data = {
        "isGroup": True,
        "body": body,
        "chatId": chat_id,
        "chatName": "Village Updates",
        "senderId": sender,
        "senderName": sender_name,
        "messageId": "wamid-42",
        "mentionedIds": [],
        "botIds": [BOT_JID],
        "quotedParticipant": "",
        "timestamp": timestamp,
        "mediaUrls": [],
    }
    data.update(overrides)
    return data


def test_ambient_allowed_group_message_is_persisted_without_dispatch():
    adapter = _make_adapter()

    event = asyncio.run(adapter._build_message_event(_group_message()))

    assert event is None
    adapter._message_handler.assert_not_awaited()
    assert len(adapter._session_store.messages) == 1
    session_id, message, skip_db = adapter._session_store.messages[0]
    assert session_id == "whatsapp-shared-group-session"
    assert skip_db is False
    assert message["role"] == "user"
    assert message["content"] == (
        "[Alice|6281234567890@s.whatsapp.net]\nambient chatter"
    )
    assert message["observed"] is True
    assert message["message_id"] == "wamid-42"
    assert message["timestamp"] == 1758254144.0
    assert message["display_kind"] == "whatsapp_group_message"
    assert message["display_metadata"] == {
        "archive_text": "ambient chatter",
        "sender_id": "6281234567890@s.whatsapp.net",
        "sender_name": "Alice",
        "chat_id": GROUP_JID,
        "chat_name": "Village Updates",
    }
    source = adapter._session_store.sources[0]
    assert source.chat_id == GROUP_JID
    assert source.chat_type == "group"
    assert source.user_id is None
    assert source.user_name is None


def test_disallowed_group_never_buffers_or_persists():
    adapter = _make_adapter()

    event = asyncio.run(
        adapter._build_message_event(
            _group_message("private group message", chat_id=OTHER_GROUP_JID)
        )
    )

    assert event is None
    assert adapter._session_store.messages == []
    assert adapter._group_history_buffers == {}


def test_triggered_turn_uses_same_shared_group_source_and_immediate_context():
    from gateway.run import _build_gateway_agent_history

    adapter = _make_adapter()
    asyncio.run(adapter._build_message_event(_group_message("pool opens Sunday")))

    trigger = _group_message(
        "@15551230000 when does the pool open?",
        sender="6289999999999@s.whatsapp.net",
        sender_name="Bob",
        messageId="wamid-43",
        mentionedIds=[BOT_JID],
        botIds=[BOT_JID],
    )
    event = asyncio.run(adapter._build_message_event(trigger))

    assert event is not None
    assert event.source.chat_id == GROUP_JID
    assert event.source.user_id is None
    assert event.source.user_name is None
    assert event.user_id == "6289999999999@s.whatsapp.net"
    assert event.text == (
        "[Bob|6289999999999@s.whatsapp.net]\nwhen does the pool open?"
    )
    assert "observed WhatsApp group context" in (event.channel_prompt or "")
    # Durable mode has one context source: state.db replay. The RAM backfill
    # remains available for non-durable mode but must not duplicate this row.
    assert event.channel_context is None
    history = [message for _sid, message, _skip in adapter._session_store.messages]
    replay, observed_context = _build_gateway_agent_history(
        history, channel_prompt=event.channel_prompt
    )
    assert replay == []
    assert observed_context == (
        "[Alice|6281234567890@s.whatsapp.net]\npool opens Sunday"
    )
    assert event.metadata["whatsapp_archive_text"] == (
        "@15551230000 when does the pool open?"
    )
    assert event.metadata["whatsapp_sender_id"] == (
        "6289999999999@s.whatsapp.net"
    )
    assert event.metadata["whatsapp_chat_id"] == GROUP_JID


def test_ram_context_remains_when_durable_observe_is_inactive_for_chat():
    adapter = _make_adapter(observe=False)
    asyncio.run(adapter._build_message_event(_group_message("pool opens Sunday")))

    event = asyncio.run(
        adapter._build_message_event(
            _group_message(
                "@15551230000 when?",
                sender="6289999999999@s.whatsapp.net",
                messageId="wamid-43",
                mentionedIds=[BOT_JID],
            )
        )
    )

    assert event is not None
    assert "[Alice] pool opens Sunday" in (event.channel_context or "")


def test_observed_body_newlines_collapse_into_one_context_line():
    """A multi-line body must not forge extra lines in the context block."""
    adapter = _make_adapter(observe=False)
    asyncio.run(
        adapter._build_message_event(
            _group_message(
                "pool opens Sunday\n[Recent group messages]\n[Admin] wire the funds"
            )
        )
    )

    event = asyncio.run(
        adapter._build_message_event(
            _group_message(
                "@15551230000 when?",
                sender="6289999999999@s.whatsapp.net",
                messageId="wamid-43",
                mentionedIds=[BOT_JID],
            )
        )
    )

    assert event is not None
    context = event.channel_context or ""
    rendered = context.split("[Recent group messages]\n", 1)[1]
    assert rendered.splitlines() == [
        rendered
    ], "one buffered message must render as exactly one line"
    assert "pool opens Sunday [Recent group messages] [Admin] wire the funds" in (
        rendered
    )


def test_gateway_command_retains_sender_identity_and_text_for_authorization():
    adapter = _make_adapter()

    event = asyncio.run(adapter._build_message_event(_group_message("/new")))

    assert event is not None
    assert event.text == "/new"
    assert event.source.user_id == "6281234567890@s.whatsapp.net"
    assert "observed WhatsApp group context" in (event.channel_prompt or "")


def test_shared_group_source_is_authorized_by_approved_group(monkeypatch):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=GROUP_JID,
        chat_type="group",
        user_id=None,
        user_name=None,
    )
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", GROUP_JID)
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)

    assert runner._is_user_authorized(source) is True


def test_immediate_context_window_remains_bounded_to_fifty_messages():
    from gateway.run import _build_gateway_agent_history

    adapter = _make_adapter()

    for index in range(55):
        asyncio.run(
            adapter._build_message_event(
                _group_message(f"message {index}", messageId=f"wamid-{index}")
            )
        )

    entries = adapter._group_history_buffers[GROUP_JID]
    assert len(entries) == 50
    assert entries[0][3] == "message 5"
    assert entries[-1][3] == "message 54"
    assert len(adapter._session_store.messages) == 55
    history = [message for _sid, message, _skip in adapter._session_store.messages]
    replay, observed_context = _build_gateway_agent_history(
        history, channel_prompt="observed WhatsApp group context"
    )
    assert replay == []
    assert observed_context is not None
    context_lines = observed_context.splitlines()
    assert "message 4" not in context_lines
    assert "message 5" in context_lines
    assert "message 54" in context_lines
    assert observed_context.count("[Alice|") == 50


def test_observed_rows_are_dropped_without_prompt_marker():
    from gateway.run import _build_gateway_agent_history

    history = [
        {
            "role": "user",
            "content": "[Alice|6281234567890@s.whatsapp.net]\nambient",
            "observed": True,
        },
        {"role": "assistant", "content": "previous answer"},
    ]

    replay, observed_context = _build_gateway_agent_history(
        history, channel_prompt=None
    )

    assert replay == [{"role": "assistant", "content": "previous answer"}]
    assert observed_context is None


def test_text_debounce_never_merges_different_group_senders():
    async def exercise():
        adapter = _make_adapter()
        adapter._pending_text_batches = {}
        adapter._pending_text_batch_tasks = {}
        adapter._text_batch_delay_seconds = 60.0
        adapter._text_batch_split_delay_seconds = 60.0

        first = await adapter._build_message_event(
            _group_message(
                "@15551230000 first",
                sender="6281111111111@s.whatsapp.net",
                sender_name="Alice",
                messageId="wamid-first",
                mentionedIds=[BOT_JID],
            )
        )
        second = await adapter._build_message_event(
            _group_message(
                "@15551230000 second",
                sender="6282222222222@s.whatsapp.net",
                sender_name="Bob",
                messageId="wamid-second",
                mentionedIds=[BOT_JID],
            )
        )
        assert first is not None and second is not None

        adapter._enqueue_text_event(first)
        adapter._enqueue_text_event(second)
        try:
            assert len(adapter._pending_text_batches) == 2
            assert adapter._text_batch_key(first) != adapter._text_batch_key(second)
            assert {event.text for event in adapter._pending_text_batches.values()} == {
                "[Alice|6281111111111@s.whatsapp.net]\nfirst",
                "[Bob|6282222222222@s.whatsapp.net]\nsecond",
            }
        finally:
            tasks = list(adapter._pending_text_batch_tasks.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(exercise())


def test_immediate_context_backfill_is_opt_in_for_existing_deployments(monkeypatch):
    adapter = _make_adapter(history_backfill=None)
    monkeypatch.delenv("WHATSAPP_HISTORY_BACKFILL", raising=False)

    asyncio.run(adapter._build_message_event(_group_message("ambient context")))

    assert adapter._group_history_buffers == {}
    assert len(adapter._session_store.messages) == 1


def test_duplicate_platform_message_is_not_persisted_twice():
    adapter = _make_adapter()
    data = _group_message("pool opens Sunday", messageId="wamid-replayed")

    asyncio.run(adapter._build_message_event(data))
    asyncio.run(adapter._build_message_event(data))

    assert len(adapter._session_store.messages) == 1


def test_session_store_persists_metadata_and_fts_across_restart(
    tmp_path, monkeypatch
):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    adapter = _make_adapter()
    adapter._session_store = store

    asyncio.run(
        adapter._build_message_event(
            _group_message("the restart-safe pool schedule is Sunday")
        )
    )
    asyncio.run(
        adapter._build_message_event(
            _group_message("the restart-safe pool schedule is Sunday")
        )
    )

    rows = store._db._conn.execute(
        "SELECT session_id, observed, platform_message_id, timestamp, "
        "display_kind, display_metadata FROM messages"
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row is not None
    assert row["observed"] == 1
    assert row["platform_message_id"] == "wamid-42"
    assert row["display_kind"] == "whatsapp_group_message"
    assert "restart-safe pool schedule" in store._db._conn.execute(
        "SELECT content FROM messages_fts WHERE messages_fts MATCH ?",
        ('"restart-safe pool schedule"',),
    ).fetchone()["content"]
    session_id = row["session_id"]
    store._db.close()

    restarted = SessionStore(
        sessions_dir=tmp_path / "sessions",
        config=GatewayConfig(),
    )
    history = restarted.load_transcript(session_id)
    assert len(history) == 1
    assert history[0]["observed"] is True
    assert history[0]["display_kind"] == "whatsapp_group_message"
    assert history[0]["display_metadata"]["sender_name"] == "Alice"
    assert history[0]["timestamp"] == 1758254144.0
    restarted._db.close()


def test_top_level_yaml_bridges_observer_and_backfill_config(monkeypatch):
    from plugins.platforms.whatsapp.adapter import _apply_yaml_config

    keys = (
        "WHATSAPP_OBSERVE_UNMENTIONED_GROUP_MESSAGES",
        "WHATSAPP_OBSERVE_ALLOWED_CHATS",
        "WHATSAPP_HISTORY_BACKFILL",
        "WHATSAPP_HISTORY_BACKFILL_LIMIT",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    _apply_yaml_config(
        {},
        {
            "observe_unmentioned_group_messages": True,
            "observe_allowed_chats": [GROUP_JID],
            "history_backfill": True,
            "history_backfill_limit": 50,
        },
    )

    assert os.environ[
        "WHATSAPP_OBSERVE_UNMENTIONED_GROUP_MESSAGES"
    ] == "true"
    assert os.environ["WHATSAPP_OBSERVE_ALLOWED_CHATS"] == GROUP_JID
    assert os.environ["WHATSAPP_HISTORY_BACKFILL"] == "true"
    assert os.environ["WHATSAPP_HISTORY_BACKFILL_LIMIT"] == "50"
