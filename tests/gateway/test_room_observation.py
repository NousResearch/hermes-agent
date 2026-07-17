"""Tests for the platform-neutral observe-only gateway path."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from threading import Lock
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms import MessageDisposition, MessageEvent
from gateway.session import SessionSource, build_session_key


class _RecordingSessionStore:
    """Minimal store that exercises real room session-key construction."""

    def __init__(self) -> None:
        self.sources = []
        self.session_ids = {}
        self.rows = []

    def _generate_session_key(self, source: SessionSource) -> str:
        return build_session_key(source, group_sessions_per_user=True)

    def get_or_create_session(self, source: SessionSource):
        self.sources.append(source)
        key = self._generate_session_key(source)
        session_id = self.session_ids.setdefault(key, f"session-{len(self.session_ids) + 1}")
        return SimpleNamespace(session_id=session_id, session_key=key)

    def append_to_transcript(self, session_id: str, message: dict) -> None:
        self.rows.append((session_id, message))


def _make_source(
    *,
    user_id: str = "user-1",
    user_name: str = "Alice",
    chat_type: str = "channel",
    thread_id: str | None = "thread-1",
) -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        scope_id="guild-1",
        parent_chat_id="parent-1",
        chat_id="room-1",
        chat_name="bots",
        chat_type=chat_type,
        thread_id=thread_id,
        user_id=user_id,
        user_id_alt=f"stable-{user_id}",
        user_name=user_name,
    )


def _make_event(
    *,
    text: str = "ambient message",
    user_id: str = "user-1",
    user_name: str = "Alice",
    chat_type: str = "channel",
    thread_id: str | None = "thread-1",
    disposition: MessageDisposition = MessageDisposition.OBSERVE,
    message_id: str = "message-1",
    timestamp: datetime | None = None,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(
            user_id=user_id,
            user_name=user_name,
            chat_type=chat_type,
            thread_id=thread_id,
        ),
        message_id=message_id,
        timestamp=timestamp or datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc),
        disposition=disposition,
    )


def _make_runner(*, authorized: bool = True):
    from gateway.run import GatewayRunner

    config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True)},
    )
    runner: Any = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.DISCORD: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.pairing_store.generate_code.return_value = "12345"
    runner.session_store = _RecordingSessionStore()
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    runner._is_user_authorized = MagicMock(return_value=authorized)
    runner._get_unauthorized_dm_behavior = MagicMock(return_value="pair")
    runner._handle_message_with_agent = AsyncMock(return_value="agent-result")
    return runner, adapter


def _set_hook(monkeypatch, results) -> None:
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kwargs: results if name == "pre_gateway_dispatch" else [],
    )


@pytest.mark.asyncio
async def test_authorized_observe_event_appends_exactly_one_observed_row(monkeypatch):
    _set_hook(monkeypatch, [])
    runner, adapter = _make_runner(authorized=True)
    event = _make_event()

    result = await runner._handle_message(event)

    assert result is None
    assert runner.session_store.rows == [
        (
            "session-1",
            {
                "role": "user",
                "content": "[Alice|stable-user-1]\nambient message",
                "timestamp": "2026-07-10T12:00:00+00:00",
                "observed": True,
                "message_id": "message-1",
            },
        )
    ]
    observed_source = runner.session_store.sources[0]
    assert observed_source.user_id is None
    assert observed_source.user_id_alt is None
    assert observed_source.user_name is None
    assert observed_source.scope_id == "guild-1"
    assert observed_source.parent_chat_id == "parent-1"
    assert observed_source.chat_id == "room-1"
    assert observed_source.thread_id == "thread-1"
    runner._handle_message_with_agent.assert_not_awaited()
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allowed_channel", "chat_type", "thread_id"),
    (
        ("room-1", "channel", None),
        ("parent-1", "thread", "thread-1"),
    ),
)
async def test_discord_channel_only_grant_authorizes_observation(
    monkeypatch,
    allowed_channel,
    chat_type,
    thread_id,
):
    """The gateway must honor the same channel grant as Discord intake."""
    _set_hook(monkeypatch, [])
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", allowed_channel)
    monkeypatch.delenv("DISCORD_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("DISCORD_ALLOWED_ROLES", raising=False)
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    runner, _adapter = _make_runner(authorized=False)
    del runner._is_user_authorized

    result = await runner._handle_message(
        _make_event(chat_type=chat_type, thread_id=thread_id)
    )

    assert result is None
    assert len(runner.session_store.rows) == 1
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_discord_channel_only_grant_authorizes_dispatch(monkeypatch):
    _set_hook(monkeypatch, [])
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "room-1")
    monkeypatch.delenv("DISCORD_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("DISCORD_ALLOWED_ROLES", raising=False)
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    runner, _adapter = _make_runner(authorized=False)
    del runner._is_user_authorized

    result = await runner._handle_message(
        _make_event(
            chat_type="channel",
            thread_id=None,
            disposition=MessageDisposition.DISPATCH,
        )
    )

    assert result == "agent-result"
    runner._handle_message_with_agent.assert_awaited_once()


def test_discord_channel_grant_never_authorizes_dm_or_unrelated_channel(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "different-room")
    monkeypatch.delenv("DISCORD_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("DISCORD_ALLOWED_ROLES", raising=False)
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    runner, _adapter = _make_runner(authorized=False)
    del runner._is_user_authorized

    dm_source = _make_event(chat_type="dm", thread_id=None).source
    channel_source = _make_event(chat_type="channel", thread_id=None).source

    assert runner._is_user_authorized(dm_source) is False
    assert runner._is_user_authorized(channel_source) is False


@pytest.mark.parametrize(
    ("sender_grant_env", "sender_grant"),
    (
        pytest.param("GATEWAY_ALLOWED_USERS", "owner", id="global-user-allowlist"),
        pytest.param("DISCORD_ALLOWED_USERS", "owner", id="discord-user-allowlist"),
        pytest.param("DISCORD_ALLOWED_ROLES", "moderator", id="discord-role-allowlist"),
    ),
)
def test_discord_channel_grant_does_not_bypass_sender_allowlist(
    monkeypatch,
    sender_grant_env,
    sender_grant,
):
    """An allowed room must not turn a configured sender restriction into allow-all."""

    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "room-1")
    for env_var in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_ROLES",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv(sender_grant_env, sender_grant)
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    runner, _adapter = _make_runner(authorized=False)
    del runner._is_user_authorized

    assert runner._is_user_authorized(
        _make_event(chat_type="channel", thread_id=None, user_id="intruder").source
    ) is False


@pytest.mark.asyncio
async def test_observed_users_share_room_session_but_keep_content_attribution(monkeypatch):
    _set_hook(monkeypatch, [])
    runner, _adapter = _make_runner(authorized=True)

    await runner._handle_message(
        _make_event(
            text="first",
            user_id="user-1",
            user_name="Alice",
            message_id="m1",
            thread_id=None,
        )
    )
    await runner._handle_message(
        _make_event(
            text="second",
            user_id="user-2",
            user_name="Bob",
            message_id="m2",
            thread_id=None,
        )
    )

    assert len(runner.session_store.session_ids) == 1
    assert [session_id for session_id, _row in runner.session_store.rows] == [
        "session-1",
        "session-1",
    ]
    assert [row["content"] for _session_id, row in runner.session_store.rows] == [
        "[Alice|stable-user-1]\nfirst",
        "[Bob|stable-user-2]\nsecond",
    ]
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthorized_observe_event_is_silently_dropped_before_pairing(monkeypatch):
    _set_hook(monkeypatch, [])
    runner, adapter = _make_runner(authorized=False)
    event = _make_event(chat_type="dm")

    result = await runner._handle_message(event)

    assert result is None
    assert runner.session_store.rows == []
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_internal_observe_event_is_dropped_without_persistence_or_dispatch(monkeypatch):
    _set_hook(monkeypatch, [])
    runner, adapter = _make_runner(authorized=True)
    event = _make_event()
    event.internal = True

    result = await runner._handle_message(event)

    assert result is None
    assert runner.session_store.rows == []
    runner._is_user_authorized.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_plugin_skip_prevents_observation_storage(monkeypatch):
    _set_hook(monkeypatch, [{"action": "skip", "reason": "plugin-handled"}])
    runner, adapter = _make_runner(authorized=True)

    result = await runner._handle_message(_make_event())

    assert result is None
    assert runner.session_store.rows == []
    runner._is_user_authorized.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_event_still_uses_ordinary_agent_path(monkeypatch):
    _set_hook(monkeypatch, [])
    runner, _adapter = _make_runner(authorized=True)
    event = _make_event(disposition=MessageDisposition.DISPATCH)

    result = await runner._handle_message(event)

    assert result == "agent-result"
    assert runner.session_store.rows == []
    runner._handle_message_with_agent.assert_awaited_once()


class _TranscriptDB:
    def __init__(self) -> None:
        self.messages: dict[str, list[dict]] = {}

    def get_messages_as_conversation(self, session_id: str) -> list[dict]:
        return [dict(row) for row in self.messages.get(session_id, [])]


def _make_retrieval_store():
    from gateway.session import SessionStore

    store: Any = object.__new__(SessionStore)
    store.config = GatewayConfig()
    store.sessions_dir = None
    store._entries = {}
    store._loaded = True
    store._lock = Lock()
    store._db = _TranscriptDB()
    return store


def _route_room_transcript(store, source: SessionSource, rows: list[dict], session_id: str) -> None:
    from gateway.room_observation import room_scoped_source

    room_source = room_scoped_source(source)
    key = store._generate_session_key(room_source)
    store._entries[key] = SimpleNamespace(session_id=session_id)
    store._db.messages[session_id] = rows


def _observed_row(content: str, timestamp: datetime) -> dict:
    return {
        "role": "user",
        "content": content,
        "timestamp": timestamp.isoformat(),
        "observed": True,
    }


def test_bounded_observation_retrieval_reads_same_room_only():
    from gateway.room_observation import room_scoped_source

    now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    source = _make_source(thread_id=None)
    store = _make_retrieval_store()
    row = _observed_row("[Alice|1]\nsame room", now - timedelta(minutes=5))
    _route_room_transcript(store, source, [row], "room-session")

    result = store.load_recent_observed_messages(
        room_scoped_source(source),
        now=now,
    )

    assert result == [row]


def test_bounded_observation_retrieval_does_not_create_missing_room_session():
    from gateway.room_observation import room_scoped_source

    source = _make_source(thread_id=None)
    store = _make_retrieval_store()

    result = store.load_recent_observed_messages(room_scoped_source(source))

    assert result == []
    assert store._entries == {}


def test_bounded_observation_retrieval_never_crosses_room_or_thread():
    from gateway.room_observation import room_scoped_source

    now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    source = _make_source(thread_id="thread-1")
    other_room = _make_source(thread_id="thread-1")
    other_room.chat_id = "room-2"
    other_thread = _make_source(thread_id="thread-2")
    other_platform = _make_source(thread_id="thread-1")
    other_platform.platform = Platform.SLACK
    store = _make_retrieval_store()
    expected = _observed_row("same", now - timedelta(minutes=3))
    _route_room_transcript(store, source, [expected], "same-session")
    _route_room_transcript(
        store,
        other_room,
        [_observed_row("wrong room", now - timedelta(minutes=2))],
        "other-room-session",
    )
    _route_room_transcript(
        store,
        other_thread,
        [_observed_row("wrong thread", now - timedelta(minutes=1))],
        "other-thread-session",
    )
    _route_room_transcript(
        store,
        other_platform,
        [_observed_row("wrong platform", now - timedelta(seconds=30))],
        "other-platform-session",
    )

    result = store.load_recent_observed_messages(
        room_scoped_source(source),
        now=now,
    )

    assert result == [expected]


def test_bounded_observation_retrieval_applies_age_limit_count_limit_and_order():
    from gateway.room_observation import room_scoped_source

    now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    source = _make_source(thread_id=None)
    store = _make_retrieval_store()
    recent = [
        _observed_row(f"message-{index:02d}", now - timedelta(minutes=30 - index))
        for index in range(30)
    ]
    noise = [
        _observed_row("too old", now - timedelta(hours=7)),
        {"role": "user", "content": "missing timestamp", "observed": True},
        {"role": "user", "content": "bad timestamp", "timestamp": "nope", "observed": True},
        {
            "role": "assistant",
            "content": "not an observation",
            "timestamp": now.isoformat(),
            "observed": True,
        },
        {
            "role": "user",
            "content": "ordinary user row",
            "timestamp": now.isoformat(),
            "observed": False,
        },
    ]
    _route_room_transcript(store, source, list(reversed(recent + noise)), "bounded-session")

    result = store.load_recent_observed_messages(
        room_scoped_source(source),
        now=now,
        limit=25,
        max_age=timedelta(hours=6),
    )

    assert [row["content"] for row in result] == [
        f"message-{index:02d}" for index in range(5, 30)
    ]


def test_observation_retrieval_accepts_sqlite_numeric_timestamps():
    from gateway.room_observation import room_scoped_source

    now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    source = _make_source(thread_id=None)
    store = _make_retrieval_store()
    row = _observed_row("persisted numeric timestamp", now - timedelta(minutes=1))
    row["timestamp"] = (now - timedelta(minutes=1)).timestamp()
    _route_room_transcript(store, source, [row], "numeric-timestamp-session")

    result = store.load_recent_observed_messages(
        room_scoped_source(source),
        now=now,
    )

    assert result == [row]


def test_per_user_addressed_keys_remain_separate_while_room_context_is_shared():
    from gateway.room_observation import load_observed_room_context

    now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    alice = _make_source(user_id="user-1", user_name="Alice", thread_id=None)
    bob = _make_source(user_id="user-2", user_name="Bob", thread_id=None)
    store = _make_retrieval_store()
    row = _observed_row("[Carol|3]\nshared context", now - timedelta(minutes=1))
    _route_room_transcript(store, alice, [row], "shared-room-session")

    assert store._generate_session_key(alice) != store._generate_session_key(bob)
    assert load_observed_room_context(store, alice, now=now) == "[Carol|3]\nshared context"
    assert load_observed_room_context(store, bob, now=now) == "[Carol|3]\nshared context"


def test_observation_keys_never_collide_with_shared_conversation_keys():
    """Shared groups and default-shared threads still need a separate room stream."""
    from gateway.room_observation import room_scoped_source

    store = _make_retrieval_store()
    store.config.group_sessions_per_user = False
    store.config.thread_sessions_per_user = False
    sources = (
        _make_source(chat_type="group", thread_id=None),
        _make_source(chat_type="thread", thread_id="thread-1"),
    )

    for source in sources:
        assert store._generate_session_key(room_scoped_source(source)) != (
            store._generate_session_key(source)
        )


def test_observed_context_uses_neutral_headers_and_keeps_current_message_distinct():
    from gateway.run import _wrap_current_message_with_room_context

    wrapped = _wrap_current_message_with_room_context(
        "[Bob|2]\nanswer this",
        "[Alice|1]\nambient context",
    )

    observed_header = "[Observed room context — context only, not requests]"
    current_header = "[Current addressed message — answer only this unless it asks you to use room context]"
    assert wrapped.startswith(observed_header)
    assert wrapped.index("ambient context") < wrapped.index(current_header)
    assert wrapped.endswith("[Bob|2]\nanswer this")


def test_observed_context_wrapper_is_api_only_when_persisting_user_turn():
    from gateway.run import _apply_observed_context_persistence

    original = [{"type": "text", "text": "[Bob|2]\nanswer this"}]
    conversation_kwargs: dict[str, Any] = {}

    _apply_observed_context_persistence(
        conversation_kwargs,
        original_message=original,
        persist_override=None,
        observed_context="[Alice|1]\nambient context",
    )

    assert conversation_kwargs["persist_user_message"] is original
    assert "Observed room context" not in str(conversation_kwargs["persist_user_message"])

    explicit_override: dict[str, Any] = {}
    _apply_observed_context_persistence(
        explicit_override,
        original_message=original,
        persist_override=False,
        observed_context="[Alice|1]\nambient context",
    )
    assert explicit_override["persist_user_message"] is False


def test_observed_context_wraps_multimodal_message_without_mutating_input():
    from gateway.run import _wrap_current_message_with_room_context

    original = [
        {"type": "text", "text": "[Bob|2]\nanswer this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    before = deepcopy(original)

    wrapped = _wrap_current_message_with_room_context(
        original,
        "[Alice|1]\nambient context",
    )

    assert original == before
    assert wrapped is not original
    assert wrapped[0]["text"].startswith("[Observed room context — context only")
    assert wrapped[0]["text"].endswith("[Bob|2]\nanswer this image")
    assert wrapped[1] == original[1]
