"""A reaction received during final delivery resumes its session once."""

import asyncio
import inspect
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer
from plugins.platforms.matrix.adapter import MatrixAdapter
from plugins.platforms.matrix.reaction_followups import (
    REGISTRATION_REPLAY_LIMIT, REGISTRATION_REPLAY_SECONDS, REPLY_EXCERPT_CHARS, WATCH_SECONDS,
    ReactionWatchStore,
)


@pytest.fixture
def delivery(tmp_path):
    wall = [1_800_000_000.0]
    monotonic = [100.0]
    timeline = []
    room = "!room:test"
    source = SessionSource(platform=Platform.MATRIX, chat_id=room, chat_name="Project room",
                           chat_type="group", user_id="@alice:test", user_name="Alice",
                           thread_id="$thread", chat_topic="Plans", profile="work")
    path = tmp_path / "watches.sqlite"

    def record(sender, content, *, room_id=room, timestamp=None):
        event = SimpleNamespace(
            room_id=room_id, event_id=f"$event{len(timeline)}", sender=sender,
            content=content, timestamp=wall[0] if timestamp is None else timestamp,
        )
        timeline.append(event)
        return event

    async def request(_method, path, *, query_params):
        if "/context/" in path:
            event_id = unquote(path.rsplit("/", 1)[1])
            index = next(index for index, event in enumerate(timeline) if event.event_id == event_id)
            return {"end": str(index + 1)}
        assert path.endswith("/messages")
        assert query_params["dir"] == "f"
        start = int(query_params["from"])
        chunk = timeline[start:start + int(query_params["limit"])]
        return {"chunk": [{"event_id": event.event_id} for event in chunk],
                **({"end": str(start + len(chunk))} if chunk else {})}

    async def send(_room, _kind, content):
        return record("@hermes:test", content).event_id

    client = SimpleNamespace(send_message_event=AsyncMock(side_effect=send),
                             api=SimpleNamespace(request=AsyncMock(side_effect=request)))

    def make_adapter():
        result = MatrixAdapter(PlatformConfig(enabled=True, extra={"user_id": "@hermes:test"}))
        result._reaction_watch_store = ReactionWatchStore(path, clock=lambda: wall[0])
        result._client = client
        result._is_allowed_matrix_room_event = AsyncMock(return_value=True)
        result._source_session_key = lambda _source: "session"
        result._session_store = SimpleNamespace(peek_session_id=lambda _key: "sid")
        result.platform = Platform.MATRIX
        result.gateway_runner = None
        result._owner_profile = 'work'
        result.set_authorization_check(lambda *_args, **_kwargs: True)
        result._resolve_room_identity = AsyncMock(
            return_value=SimpleNamespace(display_name=source.chat_name, room_topic=source.chat_topic)
        )
        result._get_display_name = AsyncMock(return_value=source.user_name)
        result._message_handler = AsyncMock()
        result.handle_message = AsyncMock()
        return result

    adapter = make_adapter()
    adapter._active_sessions["session"] = asyncio.Event()
    adapter._event_session_key = lambda _event: "session"
    adapter._reactions_enabled = False

    async def configure(emoji_filter=()):
        assert await adapter.configure_reaction_followups(
            "session", True, emoji_filter, room_id=room, requester=source.user_id,
            thread_id=source.thread_id, profile=source.profile, session_id="sid",
        )
        adapter._reaction_followup_actions["session"].pending.clock = lambda: monotonic[0]

    def reaction(target, *, emoji="👍", sender=source.user_id, room_id=room, timestamp=None):
        return record(sender, {"m.relates_to": {
            "rel_type": "m.annotation", "event_id": target, "key": emoji,
        }}, room_id=room_id, timestamp=timestamp)

    async def register(ids, text="Final answer"):
        result = GatewayRunner._run_agent_notify_streamed_final_delivery(
            adapter, source, "session", SimpleNamespace(final_message_ids=ids), text,
        )
        if inspect.isawaitable(result):
            await result

    return SimpleNamespace(
        wall=wall, monotonic=monotonic, timeline=timeline, source=source, path=path,
        client=client, adapter=adapter, make_adapter=make_adapter, configure=configure,
        reaction=reaction, register=register, request=request,
        record=record,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("emoji_filter,emoji", [((), "🦋"), (("👍",), "👍")])
@pytest.mark.parametrize("server_offset", [-120, 0, 120])
async def test_sync_before_watch_registration_resumes_once_on_the_live_adapter(
    delivery, streamed, emoji_filter, emoji, server_offset,
):
    adapter = delivery.adapter
    source = delivery.source
    await delivery.configure(emoji_filter)
    delivered = asyncio.Event()
    register = asyncio.Event()
    target = []
    if streamed:
        sent = await adapter.send(source.chat_id, "Final answer", metadata={"expect_edits": True})
        target.append(sent.message_id)
        old = delivery.reaction(target[0], emoji=emoji, timestamp=delivery.wall[0] + 1000)
        await adapter._on_reaction(old)
        consumer = GatewayStreamConsumer(adapter, source.chat_id)
        consumer._message_id = target[0]
        consumer._last_sent_text = "Final answer"
        consumer._preview_message_ids.add(target[0])
        consumer._segment_preview_message_ids.add(target[0])
        consumer._should_send_fresh_final = lambda: False

        async def deliver():
            assert await consumer._edit_existing("Final answer", finalize=True, is_turn_final=True)
            delivered.set()
            await register.wait()
            await delivery.register(tuple(target))
    else:
        adapter._final_delivery_adapter = lambda _source: adapter
        adapter._record_delivery_obligation = AsyncMock(return_value="obligation")
        adapter._release_turn_marker = AsyncMock()

        async def finalized(_obligation, result, _event, _adapter):
            assert result.success
            target.append(result.message_id)
            delivered.set()
            await register.wait()

        adapter._finalize_delivery_obligation = finalized

        async def deliver():
            result, owner = await adapter.send_final_ledgered(
                MessageEvent(text="Question", source=source), "session", "Final answer", {}, reply_to=None,
            )
            assert result.success and owner is adapter

    task = asyncio.create_task(deliver())
    try:
        await delivered.wait()
        fresh = delivery.reaction(target[0], emoji=emoji, timestamp=delivery.wall[0] + server_offset)
        await adapter._on_reaction(fresh)
        await adapter._on_reaction(fresh)
        adapter.handle_message.assert_not_awaited()
        delivery.wall[0] += 5
        delivery.monotonic[0] += 5
        register.set()
        await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    context = (f"Matrix reaction by {source.user_id}: {emoji} on reply {target[0]} "
               f"(reaction event {fresh.event_id}).")
    assert event == MessageEvent(
        text=context, source=replace(source, message_id=fresh.event_id),
        message_id=fresh.event_id, raw_message=fresh.content,
        reply_to_message_id=target[0], reply_to_text="Final answer", reply_to_is_own_message=True,
        user_id=source.user_id, allow_gateway_control=False, defer_until_idle=True,
        timestamp=event.timestamp,
        metadata={"gateway_session_key": "session", "gateway_session_id": "sid", "gateway_session_strict": True},
    )
    await adapter._on_reaction(fresh)
    restarted = delivery.make_adapter()
    await restarted._on_reaction(fresh)
    restarted.handle_message.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
    assert adapter._followup_store().candidate(source.chat_id, target[0]) is None
    assert adapter._reaction_followup_actions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    ["followup", "approval", "model-picker", "choice-picker", "removed", "filtered"],
)
async def test_queued_reconcile_awaits_replay_with_control_precedence_and_filters(
    delivery, outcome
):
    adapter = delivery.adapter
    source = delivery.source
    await delivery.configure(("👍",))
    sent = await adapter.send(source.chat_id, "Draft", metadata={"expect_edits": True})
    target = sent.message_id
    consumer = SimpleNamespace(message_id=target, final_message_ids=(target,))
    delivered, register = asyncio.Event(), asyncio.Event()
    edit_message = adapter.edit_message

    async def edit(**kwargs):
        result = await edit_message(**kwargs)
        assert result.success
        delivered.set()
        await register.wait()
        return result

    adapter.edit_message = edit
    runner = object.__new__(GatewayRunner)
    runner._send_queued_final_text = AsyncMock()
    task = asyncio.create_task(
        runner._deliver_queued_first_response(
            "Final answer",
            source,
            adapter,
            deliver_media=False,
            stream_consumer=consumer,
            session_key="session",
        )
    )
    try:
        await delivered.wait()
        for ignored in (
            delivery.reaction(target, sender="@bob:test"),
            delivery.reaction(target, room_id="!other:test"),
            delivery.reaction(target, emoji="👎"),
        ):
            await adapter._on_reaction(ignored)
        fresh = delivery.reaction(target, emoji="👎" if outcome == "filtered" else "👍")
        await adapter._on_reaction(fresh)
        await adapter._on_reaction(fresh)
        adapter.handle_message.assert_not_awaited()
        if outcome == "removed":
            await adapter._on_redaction(
                SimpleNamespace(room_id=source.chat_id, redacts=fresh.event_id)
            )
        controls = []

        def control(label):
            async def handle(*_args):
                controls.append(label)
                return outcome == label

            return handle

        adapter._handle_approval_reaction = control("approval")
        adapter._handle_model_picker_reaction = control("model-picker")
        adapter._handle_choice_picker_reaction = control("choice-picker")
        register.set()
        assert await task is True
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert (
        controls
        == {
            "followup": ["approval", "model-picker", "choice-picker"],
            "approval": ["approval"],
            "model-picker": ["approval", "model-picker"],
            "choice-picker": ["approval", "model-picker", "choice-picker"],
            "removed": [],
            "filtered": [],
        }[outcome]
    )
    runner._send_queued_final_text.assert_not_awaited()
    assert adapter._reaction_followup_actions == {}
    if outcome == "followup":
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert (event.message_id, event.reply_to_text, event.source.thread_id) == (
            fresh.event_id,
            "Final answer",
            source.thread_id,
        )
    else:
        adapter.handle_message.assert_not_awaited()
        assert adapter._followup_store().candidate(source.chat_id, target) is not None
        adapter._handle_approval_reaction = AsyncMock(return_value=False)
        adapter._handle_model_picker_reaction = AsyncMock(return_value=False)
        adapter._handle_choice_picker_reaction = AsyncMock(return_value=False)
        await adapter._on_reaction(delivery.reaction(target))
        adapter.handle_message.assert_awaited_once()
    await adapter._on_reaction(fresh)
    await adapter._on_reaction(delivery.reaction(target))
    restarted = delivery.make_adapter()
    await restarted._on_reaction(delivery.reaction(target))
    restarted.handle_message.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
    assert adapter._followup_store().candidate(source.chat_id, target) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", [
    "expired-buffer", "removed", "removed-during-order", "expired-during-order", "cancelled-during-order",
    "failed-final", "empty-final", "preview-only", "unknown-order", "denied", "wrong-profile",
    "new-session", "wrong-session-key", "new-turn", "disabled", "expired-watch", "capacity", "controls",
])
async def test_registration_replay_preserves_eligibility_boundaries(delivery, boundary):
    adapter = delivery.adapter
    source = delivery.source
    await delivery.configure(("👍",))
    sent = await adapter.send(source.chat_id, "Final answer",
                              metadata={"expect_edits": True} if boundary == "preview-only" else None)
    target = sent.message_id
    action = adapter._reaction_followup_actions["session"]
    if boundary == "capacity":
        for _ in range(REGISTRATION_REPLAY_LIMIT):
            await adapter._on_reaction(delivery.reaction("$unrelated"))
        assert len(action.pending.events) == REGISTRATION_REPLAY_LIMIT

    for ignored in (
        delivery.reaction(target, sender="@bob:test"),
        delivery.reaction(target, room_id="!other:test"),
        delivery.reaction(target, emoji="👎"),
    ):
        await adapter._on_reaction(ignored)
    fresh = delivery.reaction(target)
    await adapter._on_reaction(fresh)
    adapter.handle_message.assert_not_awaited()

    async def redact():
        await adapter._on_redaction(SimpleNamespace(room_id=source.chat_id, redacts=fresh.event_id))

    if boundary == "expired-buffer":
        delivery.monotonic[0] += REGISTRATION_REPLAY_SECONDS
    if boundary == "removed":
        await redact()
    if boundary == "denied":
        adapter.set_authorization_check(lambda *_args, **_kwargs: False)
    if boundary == "wrong-profile":
        adapter.platform = Platform.MATRIX
        adapter.gateway_runner = None
        adapter._owner_profile = 'other'
    if boundary == "new-session":
        adapter._session_store.peek_session_id = lambda _key: "new-session"
    if boundary == "wrong-session-key":
        adapter._source_session_key = lambda _source: "other-session"
    if boundary == "new-turn":
        await adapter.on_processing_start(MessageEvent(text="Next", source=source))
    if boundary == "disabled":
        assert await adapter.configure_reaction_followups(
            "session", False, (), room_id=source.chat_id, requester=source.user_id,
            thread_id=source.thread_id, profile=source.profile, session_id="sid",
        )
    if boundary == "unknown-order":
        delivery.client.api.request = AsyncMock(return_value={})
    if boundary == "controls":
        adapter._handle_choice_picker_reaction = AsyncMock(return_value=True)

    in_order = asyncio.Event()
    finish_order = asyncio.Event()

    async def blocked_order(method, path, *, query_params):
        in_order.set()
        await finish_order.wait()
        return await delivery.request(method, path, query_params=query_params)

    during_order = boundary in {
        "removed-during-order", "expired-during-order", "cancelled-during-order", "expired-watch",
    }
    if during_order:
        delivery.client.api.request = AsyncMock(side_effect=blocked_order)

    async def register():
        if boundary == "failed-final":
            adapter._final_delivery_adapter = lambda _source: adapter
            adapter._record_delivery_obligation = AsyncMock(return_value=None)
            adapter._send_with_retry = AsyncMock(return_value=SimpleNamespace(success=False))
            await adapter.send_final_ledgered(
                MessageEvent(text="Question", source=source), "session", "Final answer", {}, reply_to=None,
            )
            return
        await delivery.register((target,), "" if boundary == "empty-final" else "Final answer")

    task = asyncio.create_task(register())
    try:
        if during_order:
            await in_order.wait()
            if boundary == "removed-during-order":
                await redact()
            if boundary == "expired-during-order":
                delivery.monotonic[0] += REGISTRATION_REPLAY_SECONDS
            if boundary == "cancelled-during-order":
                await adapter.on_processing_complete(MessageEvent(text="Question", source=source),
                                                      ProcessingOutcome.CANCELLED)
            if boundary == "expired-watch":
                delivery.wall[0] += WATCH_SECONDS
            finish_order.set()
        await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    adapter.handle_message.assert_not_awaited()
    assert not action.pending.events
    watch = adapter._followup_store().candidate(source.chat_id, target)
    if boundary in {"failed-final", "empty-final", "preview-only", "new-turn", "disabled", "expired-watch"}:
        assert watch is None
        return
    assert watch is not None
    if boundary == "controls":
        adapter._handle_choice_picker_reaction.assert_awaited_once()
        adapter._handle_choice_picker_reaction = AsyncMock(return_value=False)
    delivery.client.api.request = AsyncMock(side_effect=delivery.request)
    adapter.set_authorization_check(lambda *_args, **_kwargs: True)
    adapter.platform = Platform.MATRIX
    adapter.gateway_runner = None
    adapter._owner_profile = 'work'
    adapter._session_store.peek_session_id = lambda _key: "sid"
    adapter._source_session_key = lambda _source: "session"
    await adapter._on_reaction(delivery.reaction(target))
    adapter.handle_message.assert_awaited_once()
    assert adapter._followup_store().candidate(source.chat_id, target) is None


@pytest.mark.asyncio
async def test_replay_failure_preserves_the_successful_final_delivery(delivery):
    adapter = delivery.adapter
    source = delivery.source
    await delivery.configure()
    adapter._final_delivery_adapter = lambda _source: adapter
    adapter._record_delivery_obligation = AsyncMock(return_value="obligation")
    adapter._release_turn_marker = AsyncMock()
    target = []

    async def finalized(_obligation, result, _event, _adapter):
        target.append(result.message_id)
        await adapter._on_reaction(delivery.reaction(result.message_id))
        adapter._handle_choice_picker_reaction = AsyncMock(side_effect=RuntimeError("picker unavailable"))

    adapter._finalize_delivery_obligation = finalized
    result, owner = await adapter.send_final_ledgered(
        MessageEvent(text="Question", source=source), "session", "Final answer", {}, reply_to=None,
    )
    assert result.success and owner is adapter
    adapter.handle_message.assert_not_awaited()
    assert adapter._followup_store().candidate(source.chat_id, target[0]) is not None
    assert adapter._reaction_followup_actions == {}
    assert delivery.client.send_message_event.await_count == 1


@pytest.mark.asyncio
async def test_split_replay_uses_terminal_delivery_and_consumes_all_reply_parts(delivery):
    adapter = delivery.adapter
    source = delivery.source
    await delivery.configure()
    adapter.max_message_length = 500
    sent_ids = []

    async def send(_room, _kind, content):
        event = delivery.record("@hermes:test", content)
        sent_ids.append(event.event_id)
        if len(sent_ids) == 1:
            early = delivery.reaction(event.event_id, timestamp=delivery.wall[0] + 1000)
            await adapter._on_reaction(early)
        return event.event_id

    delivery.client.send_message_event = AsyncMock(side_effect=send)
    final_text = "Head and tail. " * 100
    sent = await adapter.send(source.chat_id, final_text)
    ids = (*sent.continuation_message_ids, sent.message_id)
    assert tuple(sent_ids) == ids and sent.continuation_message_ids
    fresh = delivery.reaction(ids[0])
    await adapter._on_reaction(fresh)
    await adapter._on_reaction(delivery.reaction(ids[-1]))
    await delivery.register(ids, final_text)
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].message_id == fresh.event_id
    assert [adapter._followup_store().candidate(source.chat_id, event_id) for event_id in ids] == [None] * len(ids)
    assert adapter._reaction_followup_actions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["queue", "split-final", "failed-final", "empty-final"]
)
async def test_replacement_transport_preserves_turn_queue_and_final_choice(
    delivery, boundary
):
    original, source = delivery.adapter, delivery.source
    replacement = delivery.make_adapter()
    if boundary != "queue":
        await delivery.configure(("👍",))
        original._final_delivery_adapter = lambda _source: replacement
        original._record_delivery_obligation = AsyncMock(return_value=None)
        replacement.max_message_length = 500
        final = "Delivered final answer. " * 70 if boundary == "split-final" else ""
        if boundary == "failed-final":
            final = "Failed answer"
            delivery.client.send_message_event = AsyncMock(
                side_effect=RuntimeError("send failed")
            )
            replacement._send_with_retry = replacement.send
        result, owner = await original.send_final_ledgered(
            MessageEvent(text="Question", source=source),
            "session",
            final,
            {"thread_id": source.thread_id},
            reply_to=None,
        )
        assert owner is replacement
        ids = (
            (*result.continuation_message_ids, result.message_id)
            if result.message_id
            else ()
        )
        watches = [
            replacement._followup_store().candidate(source.chat_id, event_id)
            for event_id in ids
        ]
        if boundary != "split-final":
            assert all(watch is None for watch in watches)
            assert original._reaction_followup_actions == {}
            return
        assert result.success and result.continuation_message_ids
        assert all(watch is not None for watch in watches)
        import sqlite3
        from contextlib import closing

        with closing(sqlite3.connect(delivery.path)) as db:
            rows = db.execute(
                "SELECT turn_id, profile, session_key, session_id, requester, thread_id, "
                "emoji_json, text_content, delivery_event_id FROM watches ORDER BY event_id"
            ).fetchall()
        assert len({row[0] for row in rows}) == 1
        assert [(row[1:6], json.loads(row[6]), row[7:]) for row in rows] == [
            (
                ("work", "session", "sid", source.user_id, source.thread_id),
                ["👍"],
                (final[:REPLY_EXCERPT_CHARS], ids[-1]),
            )
        ] * len(ids)
        assert original._reaction_followup_actions == {}
        await replacement._on_reaction(delivery.reaction(ids[-1]))
        replacement.handle_message.assert_awaited_once()
        assert all(
            replacement._followup_store().candidate(source.chat_id, event_id) is None
            for event_id in ids
        )
        return

    from unittest.mock import Mock

    older = MessageEvent(text="Earlier reaction", source=source, defer_until_idle=True)
    newer = MessageEvent(text="Newer input", source=source)
    original._pending_messages["session"] = older
    replacement._pending_messages["session"] = newer
    live = [original]
    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda _source: live[0]
    runner._get_proxy_url = lambda: None
    overflow = []
    state = SimpleNamespace(conversation=SimpleNamespace(queued_events=overflow))
    runner._session_state = lambda _key: state
    runner._peek_session_state = lambda _key: state
    runner._strict_session_current = AsyncMock(return_value=True)
    runner._pending_event_audio_paths = Mock(return_value=[])
    runner._run_agent_display_settings = Mock(
        return_value=SimpleNamespace(
            _native_slack_task_cards=False,
            needs_progress_queue=False,
            log_mode_enabled=False,
        )
    )
    turn = SimpleNamespace(
        mute_notification_reply=True,
        stream_consumer_holder=[],
        result_holder=[{"completed": True}],
    )
    runner._run_agent_build_turn_context = Mock(
        return_value=(turn, SimpleNamespace(run_sync=lambda: None), None)
    )
    runner._run_agent_bind_turn_wiring = Mock(return_value=None)
    runner._run_agent_start_turn_worker = Mock(
        return_value=SimpleNamespace(executor_task=None)
    )
    for method in (
        "_run_agent_stream_consumer_task",
        "_run_agent_track_agent",
        "_run_agent_monitor_for_interrupt",
        "_run_agent_finalize_streaming_tts",
        "_run_agent_cleanup_turn_tasks",
        "_run_agent_mark_streamed_delivery",
    ):
        setattr(runner, method, AsyncMock())
    runner._run_agent_evict_on_fallback = Mock()
    runner._run_agent_schedule_bubble_cleanup = Mock()

    async def completed(*_args):
        live[0] = replacement
        return {"completed": True}

    runner._run_agent_await_turn_worker = completed
    consumed = []

    async def next_turn(_turn, owner, text, event, *_args):
        consumed.append((owner, event, text))
        return {"completed": True}

    runner._run_agent_queued_followup = next_turn
    await runner._run_agent_inner(
        "Question", "", [], source, "sid", session_key="session"
    )
    assert consumed == [(replacement, older, older.text)]
    await runner._run_agent_inner(
        "Earlier reaction", "", [], source, "sid", session_key="session"
    )
    assert consumed == [
        (replacement, older, older.text),
        (replacement, newer, newer.text),
    ]
    assert (original._pending_messages, replacement._pending_messages, overflow) == (
        {},
        {},
        [],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("valid_choice", [True, False])
@pytest.mark.parametrize("control", ["approval", "model-picker", "choice-picker"])
async def test_pending_control_on_a_watched_reply_claims_the_reaction(
    delivery, monkeypatch, control, valid_choice
):
    from plugins.platforms.matrix.adapter import _MatrixApprovalPrompt, _MatrixPickerPrompt
    from tools import approval

    adapter = delivery.adapter
    source = delivery.source
    await delivery.configure()
    sent = await adapter.send(source.chat_id, "Final answer")
    target = sent.message_id
    await delivery.register((target,))
    adapter._allowed_user_ids = {source.user_id}
    resolved = []
    monkeypatch.setattr(
        approval, "resolve_gateway_approval",
        lambda session_key, choice: resolved.append((session_key, choice)) or 1,
    )

    async def selected(_room, choice):
        resolved.append(("picker", choice))

    if control == "approval":
        adapter._approval_prompts_by_event[target] = _MatrixApprovalPrompt(
            "session", source.chat_id, target, requester_user_id=source.user_id,
        )
        emoji, expected = "✅", ("session", "once")
    else:
        registry = (adapter._model_picker_prompts_by_event if control == "model-picker"
                    else adapter._choice_picker_prompts_by_event)
        registry[target] = _MatrixPickerPrompt(
            source.chat_id, target, "session", {"1️⃣": "first"}, selected,
            requester_user_id=source.user_id,
        )
        emoji, expected = "1️⃣", ("picker", "first")

    await adapter._on_reaction(delivery.reaction(target, emoji=emoji if valid_choice else "👍"))

    adapter.handle_message.assert_not_awaited()
    assert (
        resolved,
        adapter._followup_store().candidate(source.chat_id, target) is not None,
    ) == ([expected] if valid_choice else [], True)
