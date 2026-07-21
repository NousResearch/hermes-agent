"""Tests for the gateway /queue command handler (running-agent path).

/queue stores a turn-boundary follow-up in the adapter's pending queue
without interrupting the active run. The queued event must carry the
full payload — media attachments and reply context — not just the text.
Previously the handler rebuilt the event with only text/type/source/
message_id/channel_prompt, silently dropping any photo/document/reply
metadata the user attached to the /queue message.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_runner(session_entry: SessionEntry):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._queued_events = {}
    runner._pending_approvals = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner, adapter


def _session_entry() -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


def _running(runner):
    """Mark the session as having a running agent so /queue hits the
    early-intercept path."""
    sk = build_session_key(_make_source())
    runner._running_agents[sk] = MagicMock()
    return sk


def test_queue_registry_supports_management_without_reserving_prompt_words():
    from hermes_cli.commands import resolve_command

    queue = resolve_command("queue")
    alias = resolve_command("q")
    assert queue is not None and alias is queue
    assert queue.name == "queue"
    assert queue.aliases == ("q",)
    assert queue.args_hint == "[prompt]"
    assert not queue.subcommands

    dequeue = resolve_command("dequeue")
    short = resolve_command("dq")
    assert dequeue is not None and short is dequeue
    assert dequeue.name == "dequeue"
    assert dequeue.aliases == ("dq",)
    assert dequeue.args_hint == "<N|all>"
    assert not dequeue.subcommands


@pytest.mark.asyncio
async def test_queue_text_only_queues_and_does_not_interrupt():
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)
    running_agent = runner._running_agents[sk]

    event = MessageEvent(
        text="/queue do this next",
        message_type=MessageType.COMMAND,
        source=_make_source(),
        message_id="q1",
    )
    result = await runner._handle_message(event)

    assert result is not None and "queued" in result.lower()
    running_agent.interrupt.assert_not_called()
    assert sk in adapter._pending_messages
    queued = adapter._pending_messages[sk]
    assert queued.text == "do this next"
    assert queued.message_type == MessageType.TEXT


@pytest.mark.asyncio
async def test_empty_queue_opens_text_snapshot_and_dequeue_removes_snapshot_position():
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)

    await runner._handle_message(
        MessageEvent(text="/queue first turn", source=_make_source(), message_id="q1")
    )
    await runner._handle_message(
        MessageEvent(text="/queue second turn", source=_make_source(), message_id="q2")
    )

    view = await runner._handle_message(
        MessageEvent(text="/queue", source=_make_source(), message_id="list")
    )
    assert "1. first turn" in view
    assert "2. second turn" in view
    assert "/dequeue 2" in view
    assert "q-" not in view

    removed = await runner._handle_message(
        MessageEvent(text="/dequeue 2", source=_make_source(), message_id="remove")
    )
    assert "removed" in removed.lower()
    assert "1. first turn" in removed
    assert "second turn" not in removed
    assert [event.text for event in runner._snapshot_fifo_events(sk, adapter)] == [
        "first turn"
    ]


@pytest.mark.asyncio
async def test_dequeue_all_only_removes_ids_frozen_in_latest_snapshot():
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)
    for index, text in enumerate(("old one", "old two"), start=1):
        await runner._handle_message(
            MessageEvent(text=f"/queue {text}", source=_make_source(), message_id=f"q{index}")
        )
    await runner._handle_message(
        MessageEvent(text="/queue", source=_make_source(), message_id="list")
    )
    await runner._handle_message(
        MessageEvent(text="/queue new arrival", source=_make_source(), message_id="q3")
    )

    result = await runner._handle_message(
        MessageEvent(text="/dq all", source=_make_source(), message_id="clear")
    )

    assert "removed 2" in result.lower()
    assert [event.text for event in runner._snapshot_fifo_events(sk, adapter)] == [
        "new arrival"
    ]
    assert "1. new arrival" in result


@pytest.mark.asyncio
async def test_dequeue_requires_recent_snapshot_and_never_uses_live_position():
    runner, adapter = _make_runner(_session_entry())
    _running(runner)
    await runner._handle_message(
        MessageEvent(text="/queue only item", source=_make_source(), message_id="q1")
    )

    result = await runner._handle_message(
        MessageEvent(text="/dequeue 1", source=_make_source(), message_id="remove")
    )

    assert "run `/queue`" in result.lower()
    assert len(adapter._pending_messages) == 1


@pytest.mark.asyncio
async def test_dequeue_consumed_snapshot_item_fails_safely_and_refreshes():
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)
    await runner._handle_message(
        MessageEvent(text="/queue soon running", source=_make_source(), message_id="q1")
    )
    await runner._handle_message(
        MessageEvent(text="/queue", source=_make_source(), message_id="list")
    )
    adapter._pending_messages.pop(sk)

    result = await runner._handle_message(
        MessageEvent(text="/dequeue 1", source=_make_source(), message_id="remove")
    )

    assert "already started" in result.lower() or "no longer queued" in result.lower()
    assert "no queued turns" in result.lower()


@pytest.mark.asyncio
async def test_busy_user_turn_is_manageable_but_internal_turn_is_hidden():
    runner, adapter = _make_runner(_session_entry())
    source = _make_source()
    sk = build_session_key(source)
    await runner._queue_or_replace_pending_event(
        sk,
        MessageEvent(text="busy follow-up", source=source, message_id="busy"),
    )
    await runner._enqueue_fifo(
        sk,
        MessageEvent(text="internal completion", source=source, internal=True),
        adapter,
    )

    items = runner._list_manageable_queue_items(sk, adapter=adapter)

    assert [item["preview"] for item in items] == ["busy follow-up"]
    marker = adapter._pending_messages[sk].metadata["_hermes_explicit_queue"]
    assert marker["owner_user_id"] == "u1"
    assert marker["origin"] == "busy"


@pytest.mark.asyncio
async def test_identity_less_busy_user_turn_is_still_session_manageable():
    runner, adapter = _make_runner(_session_entry())
    source = _make_source()
    source.user_id = None
    source.user_id_alt = None
    session_key = build_session_key(source)

    await runner._queue_or_replace_pending_event(
        session_key,
        MessageEvent(text="identity-less follow-up", source=source, message_id="busy"),
    )

    items = runner._list_manageable_queue_items(session_key, adapter=adapter)
    assert [item["preview"] for item in items] == ["identity-less follow-up"]
    marker = adapter._pending_messages[session_key].metadata[
        "_hermes_explicit_queue"
    ]
    assert marker["owner_user_id"] is None
    assert marker["origin"] == "busy"


@pytest.mark.asyncio
async def test_queue_management_uses_stable_alt_user_identity():
    runner, adapter = _make_runner(_session_entry())
    source = _make_source()
    source.user_id = None
    source.user_id_alt = "stable-user"
    session_key = build_session_key(source)

    await runner._queue_or_replace_pending_event(
        session_key,
        MessageEvent(text="alt-owned turn", source=source, message_id="alt-1"),
    )
    rendered = runner._open_text_queue_snapshot(source, session_key, adapter)

    assert "1. alt-owned turn" in rendered
    marker = adapter._pending_messages[session_key].metadata[
        "_hermes_explicit_queue"
    ]
    assert marker["owner_user_id"] == "stable-user"


@pytest.mark.asyncio
async def test_explicit_queue_uses_stable_alt_user_identity_for_management():
    runner, adapter = _make_runner(_session_entry())
    source = _make_source()
    source.user_id = "transient-user"
    source.user_id_alt = "stable-user"
    session_key = build_session_key(source)
    runner._running_agents[session_key] = MagicMock()

    await runner._handle_message(
        MessageEvent(text="/queue alt-owned turn", source=source, message_id="q-alt")
    )
    rendered = await runner._handle_message(
        MessageEvent(text="/queue", source=source, message_id="list-alt")
    )

    assert "1. alt-owned turn" in rendered
    marker = adapter._pending_messages[session_key].metadata[
        "_hermes_explicit_queue"
    ]
    assert marker["owner_user_id"] == "stable-user"


@pytest.mark.asyncio
async def test_group_text_queue_management_is_public_in_source_session():
    runner, adapter = _make_runner(_session_entry())
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="group-1",
        user_name="tester",
        chat_type="group",
        scope_id="workspace-1",
    )
    session_key = build_session_key(source)
    runner._running_agents[session_key] = MagicMock()

    await runner._handle_message(
        MessageEvent(text="/queue shared turn", source=source, message_id="q1")
    )
    listed = await runner._handle_message(
        MessageEvent(text="/queue", source=source, message_id="list")
    )
    removed = await runner._handle_message(
        MessageEvent(text="/dequeue 1", source=source, message_id="remove")
    )

    assert "1. shared turn" in listed
    assert "removed 1" in removed.lower()
    assert "no queued turns" in removed.lower()
    assert runner._snapshot_fifo_events(session_key, adapter) == []


@pytest.mark.asyncio
async def test_group_text_queue_lists_all_users_and_allows_cross_user_removal():
    runner, adapter = _make_runner(_session_entry())
    adapter.send_private_notice = AsyncMock(return_value=SimpleNamespace(success=True))
    alice = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="alice",
        chat_id="group-1",
        user_name="Alice",
        chat_type="group",
        scope_id="workspace-1",
    )
    bob = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="bob",
        chat_id="group-1",
        user_name="Bob",
        chat_type="group",
        scope_id="workspace-1",
    )
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        group_sessions_per_user=False,
    )
    runner.session_store._generate_session_key = MagicMock(
        side_effect=lambda source: build_session_key(
            source, group_sessions_per_user=False
        )
    )
    session_key = runner._session_key_for_source(alice)
    assert runner._session_key_for_source(bob) == session_key
    runner._running_agents[session_key] = MagicMock()

    await runner._handle_message(
        MessageEvent(text="/queue alice turn", source=alice, message_id="q1")
    )
    await runner._handle_message(
        MessageEvent(text="/queue bob turn", source=bob, message_id="q2")
    )
    listed = await runner._handle_message(
        MessageEvent(text="/queue", source=bob, message_id="list")
    )
    removed = await runner._handle_message(
        MessageEvent(text="/dequeue 1", source=bob, message_id="remove")
    )

    assert "1. alice turn" in listed
    assert "2. bob turn" in listed
    assert "removed 1" in removed.lower()
    assert "alice turn" not in removed
    assert "1. bob turn" in removed
    assert [event.text for event in runner._snapshot_fifo_events(session_key, adapter)] == [
        "bob turn"
    ]
    adapter.send_private_notice.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_text_queue_management_does_not_depend_on_private_delivery():
    runner, adapter = _make_runner(_session_entry())
    adapter.send_private_notice = AsyncMock(return_value=SimpleNamespace(success=False))
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="group-1",
        user_name="tester",
        chat_type="group",
        scope_id="workspace-1",
    )
    session_key = build_session_key(source)
    runner._running_agents[session_key] = MagicMock()

    await runner._handle_message(
        MessageEvent(text="/queue shared turn", source=source, message_id="q1")
    )
    listed = await runner._handle_message(
        MessageEvent(text="/queue", source=source, message_id="list")
    )
    removed = await runner._handle_message(
        MessageEvent(text="/dequeue 1", source=source, message_id="remove")
    )

    assert "1. shared turn" in listed
    assert "removed 1" in removed.lower()
    assert runner._snapshot_fifo_events(session_key, adapter) == []
    adapter.send_private_notice.assert_not_awaited()


@pytest.mark.asyncio
async def test_queue_snapshot_cannot_be_managed_from_another_chat_in_same_scope():
    runner, adapter = _make_runner(_session_entry())
    source_a = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="group-a",
        chat_name="Group A",
        user_name="tester",
        chat_type="group",
        scope_id="workspace-1",
    )
    source_b = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="group-b",
        chat_name="Group B",
        user_name="tester",
        chat_type="group",
        scope_id="workspace-1",
    )
    session_a = build_session_key(source_a)
    await runner._queue_or_replace_pending_event(
        session_a,
        MessageEvent(text="only in group a", source=source_a, message_id="q-a"),
    )
    runner._open_text_queue_snapshot(source_a, session_a, adapter)

    result = await runner._handle_text_dequeue(
        MessageEvent(text="/dequeue 1", source=source_b, message_id="remove-b"),
        build_session_key(source_b),
        "1",
        adapter,
    )

    assert "run `/queue`" in result.lower()
    assert [event.text for event in runner._snapshot_fifo_events(session_a, adapter)] == [
        "only in group a"
    ]


@pytest.mark.asyncio
async def test_queue_snapshot_isolated_from_another_scope():
    runner, adapter = _make_runner(_session_entry())
    source_a = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="group-a",
        user_name="tester",
        chat_type="group",
        scope_id="workspace-1",
    )
    source_b = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="group-b",
        user_name="tester",
        chat_type="group",
        scope_id="workspace-2",
    )
    session_a = build_session_key(source_a)
    await runner._queue_or_replace_pending_event(
        session_a,
        MessageEvent(text="only in group a", source=source_a, message_id="q-a"),
    )
    runner._open_text_queue_snapshot(source_a, session_a, adapter)

    result = await runner._handle_text_dequeue(
        MessageEvent(text="/dequeue 1", source=source_b, message_id="remove-b"),
        runner._queue_control_key(source_b) or "",
        "1",
        adapter,
    )

    assert "run `/queue`" in result.lower()
    assert [event.text for event in runner._snapshot_fifo_events(session_a, adapter)] == [
        "only in group a"
    ]


@pytest.mark.asyncio
async def test_queue_preserves_photo_media():
    """A /queue carrying a photo must keep the attachment + type."""
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)

    event = MessageEvent(
        text="/queue look at this",
        message_type=MessageType.PHOTO,
        source=_make_source(),
        message_id="q-photo",
        media_urls=["/tmp/photo-a.jpg"],
        media_types=["image/jpeg"],
    )
    result = await runner._handle_message(event)

    assert result is not None and "queued" in result.lower()
    queued = adapter._pending_messages[sk]
    assert queued.text == "look at this"
    assert queued.message_type == MessageType.PHOTO
    assert queued.media_urls == ["/tmp/photo-a.jpg"]
    assert queued.media_types == ["image/jpeg"]


@pytest.mark.asyncio
async def test_queue_allows_media_without_prompt_text():
    """`/queue` as a bare caption on a document is valid — media-only."""
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)

    event = MessageEvent(
        text="/queue",
        message_type=MessageType.DOCUMENT,
        source=_make_source(),
        message_id="q-doc",
        media_urls=["/tmp/file.pdf"],
        media_types=["application/pdf"],
    )
    result = await runner._handle_message(event)

    assert result is not None and "queued" in result.lower()
    queued = adapter._pending_messages[sk]
    assert queued.text == ""
    assert queued.message_type == MessageType.DOCUMENT
    assert queued.media_urls == ["/tmp/file.pdf"]


@pytest.mark.asyncio
async def test_queue_preserves_reply_context():
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)

    event = MessageEvent(
        text="/queue and this",
        source=_make_source(),
        message_id="q-reply",
        reply_to_message_id="orig-7",
        reply_to_text="the original message",
        reply_to_author_id="a1",
        reply_to_author_name="alice",
    )
    result = await runner._handle_message(event)

    assert result is not None and "queued" in result.lower()
    queued = adapter._pending_messages[sk]
    assert queued.reply_to_message_id == "orig-7"
    assert queued.reply_to_text == "the original message"
    assert queued.reply_to_author_id == "a1"
    assert queued.reply_to_author_name == "alice"


@pytest.mark.asyncio
async def test_queue_copies_complete_event_and_mutable_containers():
    """The queued copy must track MessageEvent's full dataclass surface."""
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)
    timestamp = datetime(2026, 7, 17, 1, 2, 3, tzinfo=timezone.utc)
    source = _make_source()
    raw_message = object()
    metadata = {"existing": {"nested": ["value"]}}
    auto_skill = ["skill-a", "skill-b"]
    media_urls = ["/tmp/photo.jpg"]
    media_types = ["image/jpeg"]
    event = MessageEvent(
        text="/q inspect everything",
        message_type=MessageType.PHOTO,
        source=source,
        raw_message=raw_message,
        message_id="q-all",
        platform_update_id=77,
        media_urls=media_urls,
        media_types=media_types,
        reply_to_message_id="orig",
        reply_to_text="original text",
        reply_to_author_id="author-id",
        reply_to_author_name="author-name",
        reply_to_is_own_message=True,
        auto_skill=auto_skill,
        channel_prompt="channel prompt",
        channel_context="missed context",
        internal=True,
        metadata=metadata,
        timestamp=timestamp,
    )

    result = await runner._handle_message(event)

    assert result is not None and "queued" in result.lower()
    queued = adapter._pending_messages[sk]
    assert queued is not event
    assert queued.text == "inspect everything"
    for field_name in (
        "message_type",
        "source",
        "raw_message",
        "message_id",
        "platform_update_id",
        "reply_to_message_id",
        "reply_to_text",
        "reply_to_author_id",
        "reply_to_author_name",
        "reply_to_is_own_message",
        "channel_prompt",
        "channel_context",
        "internal",
        "timestamp",
    ):
        assert getattr(queued, field_name) == getattr(event, field_name)
    assert queued.media_urls == media_urls and queued.media_urls is not media_urls
    assert queued.media_types == media_types and queued.media_types is not media_types
    assert queued.auto_skill == auto_skill and queued.auto_skill is not auto_skill
    assert queued.metadata is not metadata
    assert queued.metadata["existing"] == metadata["existing"]
    assert queued.metadata["existing"] is not metadata["existing"]
    assert (
        queued.metadata["existing"]["nested"]
        is not metadata["existing"]["nested"]
    )
    marker = queued.metadata["_hermes_explicit_queue"]
    assert marker["origin"] == "explicit"
    assert marker["owner_user_id"] == "u1"
    assert marker["id"].startswith("q-")
    assert marker["created_at"]
    assert event.metadata == metadata


@pytest.mark.asyncio
async def test_repeated_queue_ids_are_unique_within_session():
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)

    await runner._handle_message(
        MessageEvent(text="/queue first", source=_make_source(), message_id="q1")
    )
    await runner._handle_message(
        MessageEvent(text="/q second", source=_make_source(), message_id="q2")
    )

    events = runner._snapshot_fifo_events(sk, adapter)
    ids = [event.metadata["_hermes_explicit_queue"]["id"] for event in events]
    assert len(ids) == len(set(ids)) == 2


@pytest.mark.asyncio
async def test_queue_without_owner_still_submits_and_is_session_manageable():
    runner, adapter = _make_runner(_session_entry())
    source = _make_source()
    source.user_id = None
    sk = build_session_key(source)
    runner._running_agents[sk] = MagicMock()

    result = await runner._handle_message(
        MessageEvent(text="/queue anonymous", source=source, message_id="q-anon")
    )
    listed = await runner._handle_message(
        MessageEvent(text="/queue", source=source, message_id="list-anon")
    )
    removed = await runner._handle_message(
        MessageEvent(text="/dequeue 1", source=source, message_id="remove-anon")
    )

    assert result is not None and "queued" in result.lower()
    assert "1. anonymous" in listed
    assert "removed 1" in removed.lower()
    assert runner._snapshot_fifo_events(sk, adapter) == []


@pytest.mark.parametrize(
    ("command_text", "expected_payload"),
    [
        ("/queue list", "list"),
        ("/queue remove q-example", "remove q-example"),
        ("/queue clear", "clear"),
        ("/q list", "list"),
    ],
)
@pytest.mark.asyncio
async def test_queue_words_never_become_text_subcommands(command_text, expected_payload):
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)

    result = await runner._handle_message(
        MessageEvent(text=command_text, source=_make_source(), message_id="q-word")
    )

    assert result is not None and "queued" in result.lower()
    assert adapter._pending_messages[sk].text == expected_payload


@pytest.mark.asyncio
async def test_native_discord_bare_queue_management_returns_all_user_items_in_session():
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-discord",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="dm",
        total_tokens=0,
    )
    runner, adapter = _make_runner(session_entry)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.DISCORD: adapter}
    sk = build_session_key(source)
    runner._running_agents[sk] = MagicMock()
    explicit = MessageEvent(
        text="first prompt",
        source=source,
        metadata={
            "_hermes_explicit_queue": {
                "id": "q-owned",
                "owner_user_id": "u1",
                "created_at": "2026-07-17T00:00:00+00:00",
                "origin": "explicit",
            }
        },
    )
    other_owner = MessageEvent(
        text="second prompt",
        source=source,
        metadata={
            "_hermes_explicit_queue": {
                "id": "q-other",
                "owner_user_id": "u2",
                "created_at": "2026-07-17T00:00:00+00:00",
                "origin": "explicit",
            }
        },
    )
    ordinary = MessageEvent(text="ordinary", source=source)
    for queued in (explicit, other_owner, ordinary):
        await runner._enqueue_fifo(sk, queued, adapter)

    event = MessageEvent(
        text="/queue",
        source=source,
        message_id="manage",
        metadata={"_hermes_native_discord_queue_management": {"action": "list"}},
    )
    result = await runner._handle_message(event)

    assert result["type"] == "queue_management"
    assert result["action"] == "list"
    assert [item["id"] for item in result["items"]] == ["q-owned", "q-other"]
    assert [item["preview"] for item in result["items"]] == [
        "first prompt",
        "second prompt",
    ]


@pytest.mark.asyncio
async def test_native_queue_management_fails_closed_without_queue_storage():
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )
    runner, _adapter = _make_runner(_session_entry())
    runner.adapters = {}

    result = await runner._handle_message(
        MessageEvent(
            text="/queue",
            source=source,
            metadata={"_hermes_native_discord_queue_management": {"action": "clear"}},
        )
    )

    assert result == {
        "type": "queue_management",
        "action": "clear",
        "ok": False,
        "error": "unavailable",
    }


@pytest.mark.asyncio
async def test_native_discord_bare_queue_management_works_without_active_agent():
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-discord",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="dm",
        total_tokens=0,
    )
    runner, adapter = _make_runner(session_entry)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.DISCORD: adapter}

    result = await runner._handle_message(
        MessageEvent(
            text="/queue",
            source=source,
            metadata={"_hermes_native_discord_queue_management": True},
        )
    )

    assert result["type"] == "queue_management"
    assert result["action"] == "list"
    assert result["ok"] is True
    assert result["session_key"] == build_session_key(source)
    assert result["items"] == []
    assert isinstance(result["snapshot_id"], str) and result["snapshot_id"]


@pytest.mark.asyncio
async def test_native_discord_card_snapshot_is_superseded_by_new_text_view():
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )
    session_key = build_session_key(source)
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-discord",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="dm",
        total_tokens=0,
    )
    runner, adapter = _make_runner(session_entry)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.DISCORD: adapter}
    queued = MessageEvent(
        text="owned",
        source=source,
        metadata={
            "_hermes_explicit_queue": {
                "id": "q-owned",
                "owner_user_id": "u1",
                "created_at": "2026-07-17T00:00:00+00:00",
                "origin": "explicit",
            }
        },
    )
    await runner._enqueue_fifo(session_key, queued, adapter)

    card_view = await runner._handle_message(
        MessageEvent(
            text="/queue",
            source=source,
            metadata={"_hermes_native_discord_queue_management": {"action": "list"}},
        )
    )
    old_snapshot_id = card_view["snapshot_id"]
    runner._open_text_queue_snapshot(source, session_key, adapter)

    result = await runner._handle_message(
        MessageEvent(
            text="/queue",
            source=source,
            metadata={
                "_hermes_native_discord_queue_management": {
                    "action": "remove",
                    "queue_id": "q-owned",
                    "session_key": session_key,
                    "snapshot_id": old_snapshot_id,
                }
            },
        )
    )

    assert result == {
        "type": "queue_management",
        "action": "remove",
        "ok": False,
        "error": "snapshot_stale",
    }
    assert runner._snapshot_fifo_events(session_key, adapter) == [queued]


@pytest.mark.asyncio
async def test_native_discord_remove_and_clear_are_session_scoped_and_structured():
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-discord",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="dm",
        total_tokens=0,
    )
    runner, adapter = _make_runner(session_entry)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.DISCORD: adapter}
    sk = build_session_key(source)
    running_agent = MagicMock()
    runner._running_agents[sk] = running_agent

    def queued(text, queue_id, owner):
        return MessageEvent(
            text=text,
            source=source,
            metadata={
                "_hermes_explicit_queue": {
                    "id": queue_id,
                    "owner_user_id": owner,
                    "created_at": "2026-07-17T00:00:00+00:00",
                    "origin": "explicit",
                }
            },
        )

    owned_one = queued("owned one", "q-owned-1", "u1")
    other = queued("other", "q-other", "u2")
    ordinary = MessageEvent(text="ordinary", source=source)
    owned_two = queued("owned two", "q-owned-2", "u1")
    for item in (owned_one, other, ordinary, owned_two):
        await runner._enqueue_fifo(sk, item, adapter)

    opened_view = await runner._handle_message(
        MessageEvent(
            text="/q",
            source=source,
            metadata={"_hermes_native_discord_queue_management": {"action": "list"}},
        )
    )
    remove_result = await runner._handle_message(
        MessageEvent(
            text="/q",
            source=source,
            metadata={
                "_hermes_native_discord_queue_management": {
                    "action": "remove",
                    "queue_id": "q-owned-1",
                    "session_key": sk,
                    "snapshot_id": opened_view["snapshot_id"],
                }
            },
        )
    )
    other_remove = await runner._handle_message(
        MessageEvent(
            text="/queue",
            source=source,
            metadata={
                "_hermes_native_discord_queue_management": {
                    "action": "remove",
                    "queue_id": "q-other",
                    "session_key": sk,
                    "snapshot_id": remove_result["snapshot_id"],
                }
            },
        )
    )
    clear_result = await runner._handle_message(
        MessageEvent(
            text="/queue",
            source=source,
            metadata={
                "_hermes_native_discord_queue_management": {
                    "action": "clear",
                    "queue_ids": ["q-owned-2"],
                    "session_key": sk,
                    "snapshot_id": other_remove["snapshot_id"],
                }
            },
        )
    )

    assert [item["id"] for item in opened_view["items"]] == [
        "q-owned-1",
        "q-other",
        "q-owned-2",
    ]
    assert remove_result["ok"] is True
    assert remove_result["removed"] is True
    assert remove_result["queue_id"] == "q-owned-1"
    assert remove_result["session_key"] == sk
    assert isinstance(remove_result["snapshot_id"], str)
    assert [item["id"] for item in remove_result["items"]] == [
        "q-other",
        "q-owned-2",
    ]
    assert other_remove["ok"] is True
    assert other_remove["removed"] is True
    assert other_remove["queue_id"] == "q-other"
    assert [item["id"] for item in other_remove["items"]] == ["q-owned-2"]
    assert clear_result["ok"] is True
    assert clear_result["removed_count"] == 1
    assert clear_result["session_key"] == sk
    assert clear_result["items"] == []
    assert isinstance(clear_result["snapshot_id"], str)
    assert runner._snapshot_fifo_events(sk, adapter) == [ordinary]
    running_agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_native_discord_clear_requires_queue_id_snapshot():
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-discord",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="dm",
        total_tokens=0,
    )
    runner, adapter = _make_runner(session_entry)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.DISCORD: adapter}
    session_key = build_session_key(source)
    queued = MessageEvent(
        text="owned",
        source=source,
        metadata={
            "_hermes_explicit_queue": {
                "id": "q-owned",
                "owner_user_id": "u1",
                "created_at": "2026-07-17T00:00:00+00:00",
                "origin": "explicit",
            }
        },
    )
    await runner._enqueue_fifo(session_key, queued, adapter)

    result = await runner._handle_message(
        MessageEvent(
            text="/queue",
            source=source,
            metadata={
                "_hermes_native_discord_queue_management": {
                    "action": "clear",
                    "session_key": session_key,
                }
            },
        )
    )

    assert result == {
        "type": "queue_management",
        "action": "clear",
        "ok": False,
        "error": "invalid_action",
    }
    assert runner._snapshot_fifo_events(session_key, adapter) == [queued]


@pytest.mark.asyncio
async def test_native_discord_manager_rejects_stale_session_before_mutation():
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-discord",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="dm",
        total_tokens=0,
    )
    runner, adapter = _make_runner(session_entry)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.DISCORD: adapter}
    session_key = build_session_key(source)
    queued = MessageEvent(
        text="owned",
        source=source,
        metadata={
            "_hermes_explicit_queue": {
                "id": "q-owned",
                "owner_user_id": "u1",
                "created_at": "2026-07-17T00:00:00+00:00",
                "origin": "explicit",
            }
        },
    )
    await runner._enqueue_fifo(session_key, queued, adapter)

    result = await runner._handle_message(
        MessageEvent(
            text="/q",
            source=source,
            metadata={
                "_hermes_native_discord_queue_management": {
                    "action": "remove",
                    "queue_id": "q-owned",
                    "session_key": "agent:other:discord:dm:other",
                }
            },
        )
    )

    assert result == {
        "type": "queue_management",
        "action": "remove",
        "ok": False,
        "error": "session_changed",
    }
    assert runner._snapshot_fifo_events(session_key, adapter) == [queued]


@pytest.mark.asyncio
async def test_native_discord_remove_requires_view_session_key():
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-discord",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="dm",
        total_tokens=0,
    )
    runner, adapter = _make_runner(session_entry)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.DISCORD: adapter}
    session_key = build_session_key(source)
    queued = MessageEvent(
        text="owned",
        source=source,
        metadata={
            "_hermes_explicit_queue": {
                "id": "q-owned",
                "owner_user_id": "u1",
                "created_at": "2026-07-17T00:00:00+00:00",
                "origin": "explicit",
            }
        },
    )
    await runner._enqueue_fifo(session_key, queued, adapter)

    result = await runner._handle_message(
        MessageEvent(
            text="/q",
            source=source,
            metadata={
                "_hermes_native_discord_queue_management": {
                    "action": "remove",
                    "queue_id": "q-owned",
                }
            },
        )
    )

    assert result == {
        "type": "queue_management",
        "action": "remove",
        "ok": False,
        "error": "invalid_action",
    }
    assert runner._snapshot_fifo_events(session_key, adapter) == [queued]


@pytest.mark.asyncio
async def test_native_management_marker_requires_discord_bare_no_media_event():
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-discord",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="dm",
        total_tokens=0,
    )
    runner, adapter = _make_runner(session_entry)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.DISCORD: adapter}
    sk = build_session_key(source)
    runner._running_agents[sk] = MagicMock()
    marker = {"_hermes_native_discord_queue_management": {"action": "list"}}

    text_result = await runner._handle_message(
        MessageEvent(
            text="/queue actual prompt",
            source=source,
            message_id="text",
            metadata=marker,
        )
    )
    media_result = await runner._handle_message(
        MessageEvent(
            text="/queue",
            message_type=MessageType.DOCUMENT,
            source=source,
            message_id="media",
            media_urls=["/tmp/file.pdf"],
            media_types=["application/pdf"],
            metadata=marker,
        )
    )

    assert isinstance(text_result, str) and "queued" in text_result.lower()
    assert isinstance(media_result, str) and "queued" in media_result.lower()
    events = runner._snapshot_fifo_events(sk, adapter)
    assert [event.text for event in events] == ["actual prompt", ""]


@pytest.mark.asyncio
async def test_plain_gateway_bare_q_opens_text_manager_without_native_marker():
    runner, adapter = _make_runner(_session_entry())
    _running(runner)

    result = await runner._handle_message(
        MessageEvent(text="/q", source=_make_source(), message_id="plain-q")
    )

    assert "No queued turns" in result
    assert "q-" not in result
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_idle_bare_queue_with_media_remains_a_normal_agent_turn():
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-discord",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="dm",
        total_tokens=0,
    )
    runner, _adapter = _make_runner(session_entry)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.DISCORD: _adapter}
    runner._external_drain_active = True
    event = MessageEvent(
        text="/queue",
        message_type=MessageType.DOCUMENT,
        source=source,
        media_urls=["/tmp/file.pdf"],
        media_types=["application/pdf"],
        metadata={"_hermes_native_discord_queue_management": True},
    )

    result = await runner._handle_message(event)

    assert isinstance(result, str) and "draining" in result
    assert event.text == ""


@pytest.mark.asyncio
async def test_queue_no_text_no_media_opens_empty_snapshot():
    runner, adapter = _make_runner(_session_entry())
    _running(runner)

    event = MessageEvent(text="/queue", source=_make_source(), message_id="q-empty")
    result = await runner._handle_message(event)

    assert result is not None and "No queued turns" in result
    assert "expires in 5 minutes" in result
    assert adapter._pending_messages == {}


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
