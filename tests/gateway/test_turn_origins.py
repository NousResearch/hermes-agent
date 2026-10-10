"""Turn origins: the platform messages a gateway turn was built from, as ``pre_llm_call`` sees them.

The Telegram builder identifies each message (an unedited message has no ``edit_date``); every
boundary that merges, copies or replays events keeps the origins, concatenated and complete only
when every part was identified; the turn captures them when it starts. A turn without origins
keeps today's ``run_conversation`` call shape and runs exactly as before.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import merge_pending_message_event
from gateway.platforms.event import MessageEvent, MessageOrigin, MessageType
from gateway.session import SessionEntry, SessionSource
from tests.gateway.test_active_session_text_merge import _make_adapter as _make_busy_adapter
from tests.gateway.test_telegram_reply_quote import _make_adapter as _make_telegram_adapter
from tests.gateway.test_telegram_reply_quote import _make_message
from tests.gateway.test_telegram_text_batching import _make_adapter as _make_batching_adapter

SOURCE = SessionSource(platform=Platform.TELEGRAM, chat_id="111", chat_type="dm", user_id="u1")


def _origin(message_id: int) -> MessageOrigin:
    return MessageOrigin("111", str(message_id), 500 + message_id)


def _event(text: str, message_id: int, *, message_type=MessageType.TEXT, media=False,
           origins=None, complete=True) -> MessageEvent:
    return MessageEvent(
        text=text, message_type=message_type, source=SOURCE, message_id=str(message_id),
        media_urls=["/tmp/p.jpg"] if media else [], media_types=["image/jpeg"] if media else [],
        source_origins=(_origin(message_id),) if origins is None else origins,
        source_origins_complete=complete)


# --- Builder -------------------------------------------------------------------------------------

def test_an_ordinary_unedited_message_is_one_complete_origin():
    event = _make_telegram_adapter()._build_message_event(_make_message(), MessageType.TEXT, update_id=7001)

    assert event.source_origins == (MessageOrigin("111", "1001", 7001, None),)
    assert event.source_origins_complete is True


def test_an_edited_message_is_a_distinct_revision():
    adapter = _make_telegram_adapter()
    original = adapter._build_message_event(_make_message(), MessageType.TEXT, update_id=7001)
    edited_message = _make_message(text="follow-up, fixed")
    edited_message.edit_date = datetime(2026, 1, 2, tzinfo=timezone.utc)
    edited = adapter._build_message_event(edited_message, MessageType.TEXT, update_id=7002)

    assert edited.source_origins == (MessageOrigin("111", "1001", 7002, 1767312000),)
    assert edited.source_origins_complete is True
    assert edited.source_origins != original.source_origins


def test_a_message_without_its_update_id_is_incomplete():
    event = _make_telegram_adapter()._build_message_event(_make_message(), MessageType.TEXT)

    assert event.source_origins == (MessageOrigin("111", "1001", None, None),)
    assert event.source_origins_complete is False


def test_an_event_built_without_origins_is_incomplete():
    """Every other adapter, synthetic turn and older caller builds events without the fields."""
    event = MessageEvent(text="hello", source=SOURCE)

    assert (event.source_origins, event.source_origins_complete) == ((), False)


# --- Merges --------------------------------------------------------------------------------------

@pytest.mark.parametrize("first, second, merge_text", [
    (dict(text="part one"), dict(text="part two"), True),
    (dict(text="", message_type=MessageType.PHOTO, media=True),
     dict(text="", message_type=MessageType.PHOTO, media=True), False),
    (dict(text="look at this"), dict(text="caption", message_type=MessageType.PHOTO, media=True), False),
], ids=["text", "photo-burst", "text-then-photo"])
def test_a_pending_merge_carries_every_origin(first, second, merge_text):
    pending = {"k": _event(message_id=1, **first)}
    merge_pending_message_event(pending, "k", _event(message_id=2, **second), merge_text=merge_text)

    assert pending["k"].source_origins == (_origin(1), _origin(2))
    assert pending["k"].source_origins_complete is True


def test_identical_texts_from_distinct_messages_keep_distinct_origins():
    pending = {"k": _event("ok", 1)}
    merge_pending_message_event(pending, "k", _event("ok", 2), merge_text=True)

    assert pending["k"].source_origins == (_origin(1), _origin(2))


@pytest.mark.parametrize("missing", [
    dict(origins=(), complete=False),  # a part from a path that does not identify its message
    dict(origins=(MessageOrigin("111", "2", None),), complete=False),  # identified, update id missing
], ids=["no-origin", "no-update-id"])
@pytest.mark.parametrize("missing_first", [False, True], ids=["missing-second", "missing-first"])
def test_a_merge_with_an_unidentified_part_is_incomplete(missing, missing_first):
    identified, unidentified = _event("one", 1), _event("two", 2, **missing)
    first, second = (unidentified, identified) if missing_first else (identified, unidentified)
    expected = first.source_origins + second.source_origins
    pending = {"k": first}
    merge_pending_message_event(pending, "k", second, merge_text=True)

    assert pending["k"].source_origins == expected
    assert pending["k"].source_origins_complete is False


@pytest.mark.asyncio
async def test_a_debounced_busy_text_burst_carries_every_origin():
    adapter = _make_busy_adapter()
    await adapter._queue_text_debounce("k", _event("first", 1))
    await adapter._queue_text_debounce("k", _event("second", 2))
    await adapter._flush_text_debounce_now("k")

    assert adapter._pending_messages["k"].source_origins == (_origin(1), _origin(2))
    assert adapter._pending_messages["k"].source_origins_complete is True


@pytest.mark.asyncio
async def test_a_client_split_text_batch_carries_every_origin():
    adapter = _make_batching_adapter()
    adapter._enqueue_text_event(_event("This is part one of a long", 1))
    adapter._enqueue_text_event(_event("message that was split by Telegram.", 2))
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())

    [dispatched] = [call.args[0] for call in adapter.handle_message.call_args_list]
    assert dispatched.source_origins == (_origin(1), _origin(2))
    assert dispatched.source_origins_complete is True


@pytest.mark.asyncio
async def test_an_album_carries_every_origin():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = _make_batching_adapter()
    with patch.object(TelegramAdapter, "MEDIA_GROUP_WAIT_SECONDS", 0.05):
        await adapter._queue_media_group_event("album", _event("caption", 1, message_type=MessageType.PHOTO, media=True))
        await adapter._queue_media_group_event("album", _event("", 2, message_type=MessageType.PHOTO, media=True))
        await asyncio.gather(*adapter._media_group_tasks.values())

    [dispatched] = [call.args[0] for call in adapter.handle_message.call_args_list]
    assert dispatched.source_origins == (_origin(1), _origin(2))
    assert dispatched.source_origins_complete is True


# --- Copies and replays --------------------------------------------------------------------------

def _busy_runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: adapter
    runner._peek_session_state = lambda key: None
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/queue check the logs", "/steer check the logs"])
async def test_a_queued_command_keeps_its_message_origin(command):
    adapter = SimpleNamespace(_pending_messages={})
    runner = _busy_runner(adapter)
    event = _event(command, 1)
    handler = runner._busy_queue_command if command.startswith("/queue") else runner._busy_steer_command

    await handler(event, "k", SOURCE)

    queued = adapter._pending_messages["k"]
    assert queued.text == "check the logs"
    assert (queued.source_origins, queued.source_origins_complete) == ((_origin(1),), True)


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_event, expected", [
    (_event("the follow-up", 2), ((_origin(2),), True)),
    (None, ((), False)),  # leftover steer text queued without an event
], ids=["queued-message", "event-less"])
async def test_a_queued_follow_up_turn_runs_with_its_own_origins(pending_event, expected):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value={"final_response": "done", "messages": []})
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value="k")
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="the follow-up")
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._delivery_adapter_for = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    turn_ctx = SimpleNamespace(
        source=SOURCE, session_id="sid", session_key="k", run_generation=1, _interrupt_depth=0,
        history=[], _status_thread_metadata=None, context_prompt=None, channel_prompt=None, result_holder=[None])

    await GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=None, pending="the follow-up", pending_event=pending_event,
        response="resp", result={"interrupted": True, "messages": []}, stream_task=None)

    kwargs = runner._run_agent.await_args.kwargs
    assert (kwargs["source_origins"], kwargs["source_origins_complete"]) == expected


@pytest.mark.asyncio
async def test_a_held_message_is_redispatched_with_its_origins():
    adapter = _make_batching_adapter()
    adapter._mark_disconnected()
    adapter._enqueue_text_event(_event("sent during a reconnect", 1))
    adapter._drop_delayed_deliveries = False

    await adapter._redispatch_held_inbound()

    [redispatched] = [call.args[0] for call in adapter.handle_message.call_args_list]
    assert (redispatched.source_origins, redispatched.source_origins_complete) == ((_origin(1),), True)


@pytest.mark.asyncio
@pytest.mark.parametrize("event", [
    _event("sent during startup", 1),
    MessageEvent(text="an event built without origins", source=SOURCE),
], ids=["identified", "without-origins"])
async def test_a_startup_restore_replay_keeps_the_origins_it_had(event):
    from gateway.run import GatewayRunner

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = object.__new__(GatewayRunner)
    runner._intake_adapter_for = lambda source: adapter
    runner._queue_startup_restore_event(event)
    expected = (event.source_origins, event.source_origins_complete)

    assert await runner._drain_startup_restore_queue() == 1

    replayed = adapter.handle_message.await_args.args[0]
    assert (replayed.source_origins, replayed.source_origins_complete) == expected


# --- The turn ------------------------------------------------------------------------------------

class _TurnObserved(BaseException):
    """Stop after the real prologue (``pre_llm_call`` included), before any model request."""


async def _run_gateway_turn(home, monkeypatch, event, callbacks=()):
    """One real gateway turn — runner, ``TurnContext``, ``TurnRunner``, ``AIAgent`` facade and
    prologue — stopped before the model request. Returns ``(run_conversation kwargs, model input,
    persisted user rows)``."""
    from gateway.run import GatewayRunner
    from gateway.turn_context import TurnContext
    from hermes_cli import plugins as plugins_mod
    from hermes_state import SessionDB
    from run_agent import AIAgent

    manager = plugins_mod.PluginManager()
    manager._discovered = True
    manager._hooks["pre_llm_call"] = list(callbacks)
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr("gateway.run._load_gateway_config", dict)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", dict)
    model_inputs = []

    def observe_model(agent, messages):
        # Per-row identity and clock values differ between any two runs; the rest is the request.
        model_inputs.append([{k: v for k, v in m.items() if k not in {"timestamp", "message_uid"}}
                             for m in messages])
        raise _TurnObserved

    monkeypatch.setattr("agent.conversation_loop.begin_iteration", observe_model)

    home.mkdir()
    entry = SessionEntry(session_id="origins", session_key="telegram:origins",
                         created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner._get_proxy_url = lambda: None
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._hmwa_resolve_session = AsyncMock(return_value=(event.source, entry, entry.session_key))
    runner._hmwa_open_session = AsyncMock(return_value=(False, True))
    runner._set_session_env = lambda context: {}
    runner._clear_session_env = lambda tokens: None
    runner._pinned_session_context_prompt = lambda *args, **kwargs: ""
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._mark_durable_active_turn = AsyncMock()
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(_store=runner.session_store, load_transcript=AsyncMock(return_value=[]))
    runner._hmwa_run_session_hygiene = AsyncMock(return_value=[])
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._voice_channel_sidecar_note = lambda *args: None
    runner._consume_pending_native_image_paths = lambda key: []
    runner._adapter_for_source = lambda source: None
    runner._bind_adapter_run_generation = lambda *args: None

    async def propagate_error(exc, *args):
        raise exc

    runner._hmwa_agent_error_reply = propagate_error
    defaults = TurnContext()
    display = SimpleNamespace(**{name: getattr(defaults, name) for name in runner._DISPLAY_TO_TURN_CTX})
    display.platform_key = "telegram"
    display.resolve_display_setting = lambda *args: False
    runner._run_agent_display_settings = lambda source: display

    db = SessionDB(home / "state.db")
    agent = AIAgent(session_db=db, model="test-model", api_key="test-key", base_url="http://127.0.0.1:1/v1",
                    platform="telegram", session_id=entry.session_id, enabled_toolsets=[],
                    quiet_mode=True, skip_memory=True, skip_context_files=True)
    agent.compression_enabled = False
    calls = []
    run_conversation = agent.run_conversation

    def recording_run_conversation(*args, **kwargs):
        calls.append(kwargs)
        return run_conversation(*args, **kwargs)

    agent.run_conversation = recording_run_conversation

    def run_without_delivery(ctx, worker, *args):
        worker._native_image_run_message = lambda: ctx.message
        worker._run_conversation_with_approval(agent, [], None, ctx.persist_user_message, ctx.persist_user_timestamp)

    runner._run_agent_bind_turn_wiring = run_without_delivery
    runner._run_agent = runner._run_agent_inner
    try:
        with pytest.raises(_TurnObserved):
            await runner._handle_message_with_agent(event, event.source, entry.session_key, 1)
        rows = [row["content"] for row in db.get_messages(entry.session_id) if row["role"] == "user"]
        return calls[0], model_inputs[0], rows
    finally:
        agent.close()
        db.close()


@pytest.mark.asyncio
async def test_pre_llm_call_gets_the_frozen_origins_of_a_merged_turn(tmp_path, monkeypatch):
    """Both merged messages reach every callback. One callback cannot change what the next sees,
    and a message merged into the event after the turn started does not reach the running turn."""
    pending = {"k": _event("first", 1)}
    merge_pending_message_event(pending, "k", _event("second", 2), merge_text=True)
    event = pending["k"]
    seen, refused = [], []
    attempts = (
        lambda origins: setattr(origins[0], "message_id", "forged"),
        lambda origins: object.__setattr__(origins[0], "message_id", "forged"),
        lambda origins: origins.__setitem__(0, _origin(9)),
    )

    def tampering(source_origins, source_origins_complete, **kwargs):
        seen.append((source_origins, source_origins_complete))
        event.absorb(_event("arrived later", 3, complete=False))
        for attempt in attempts:
            try:
                attempt(source_origins)
            except (AttributeError, TypeError):
                refused.append(attempt)

    def observer(source_origins, source_origins_complete, **kwargs):
        seen.append((source_origins, source_origins_complete))

    await _run_gateway_turn(tmp_path / "home", monkeypatch, event, [tampering, observer])

    assert refused == list(attempts)
    assert seen == [((_origin(1), _origin(2)), True)] * 2
    assert event.source_origins == (_origin(1), _origin(2), _origin(3))  # the event moved on; the turn did not


@pytest.mark.asyncio
async def test_a_narrow_legacy_callback_is_called_without_the_new_fields(tmp_path, monkeypatch):
    calls = []

    def legacy(session_id, user_message):
        calls.append((session_id, user_message))

    await _run_gateway_turn(tmp_path / "home", monkeypatch, _event("hello", 1), [legacy])

    assert calls == [("origins", "hello")]


@pytest.mark.asyncio
async def test_a_turn_without_origins_runs_exactly_as_before(tmp_path, monkeypatch):
    """Differential: with no plugin consuming them, identified and unidentified turns send the
    model the same request and persist the same row; only the identified one passes the new
    ``run_conversation`` keywords, and a turn without origins hands hooks the safe default."""
    seen = []
    identified = await _run_gateway_turn(tmp_path / "a", monkeypatch, _event("hello", 1))
    plain = await _run_gateway_turn(tmp_path / "b", monkeypatch, MessageEvent(text="hello", source=SOURCE, message_id="1"))
    await _run_gateway_turn(tmp_path / "c", monkeypatch, MessageEvent(text="hello", source=SOURCE, message_id="1"),
                            [lambda source_origins, source_origins_complete, **kw: seen.append((source_origins, source_origins_complete))])

    (identified_kwargs, *identified_turn), (plain_kwargs, *plain_turn) = identified, plain
    assert identified_turn == plain_turn
    assert (identified_kwargs["source_origins"], identified_kwargs["source_origins_complete"]) == ((_origin(1),), True)
    assert identified_kwargs.keys() ^ plain_kwargs.keys() == {"source_origins", "source_origins_complete"}
    assert seen == [((), False)]
