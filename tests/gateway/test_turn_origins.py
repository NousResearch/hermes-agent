"""Turn origins: the platform messages a gateway turn was built from, as ``pre_llm_call`` sees them.

The Telegram builder identifies a message only when its chat, message and update ids are all
present (an unedited message has no ``edit_date``). Every boundary that merges, copies or replays
events keeps the origins, concatenated and complete only when every part was identified; the turn
captures them when it starts, and every ``pre_llm_call`` callback gets its own list of dicts. A path
that does not carry identity (the shutdown spool, a synthetic turn) leaves the safe default.
"""

import asyncio
import copy
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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


def _dict(message_id: int) -> dict:
    """What a ``pre_llm_call`` callback receives for ``_origin(message_id)``."""
    return {"chat_id": "111", "message_id": str(message_id), "update_id": 500 + message_id, "edit_date": None}


def _event(text: str, message_id: int, *, message_type=MessageType.TEXT, media=False,
           origins=None, complete=True, reply_expected=None) -> MessageEvent:
    return MessageEvent(
        text=text, message_type=message_type, source=SOURCE, message_id=str(message_id),
        media_urls=["/tmp/p.jpg"] if media else [], media_types=["image/jpeg"] if media else [],
        source_origins=(_origin(message_id),) if origins is None else origins,
        source_origins_complete=complete, reply_expected=reply_expected)


# --- A real gateway turn -------------------------------------------------------------------------

class _TurnObserved(BaseException):
    """Stop after the real prologue (``pre_llm_call`` included), before any model request."""


@contextmanager
def _gateway_turns(home, monkeypatch, callbacks=()):
    """A real runner wired to a real ``AIAgent``: ``TurnContext`` → ``TurnRunner`` → facade → loop →
    prologue, with ``pre_llm_call`` dispatched by a real ``PluginManager`` holding *callbacks*. Each
    turn stops before the model request."""
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
    runner._hmwa_resolve_session = AsyncMock(side_effect=lambda event, *args, **kwargs: (event.source, entry, entry.session_key))
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
    # Queued follow-up turns: the text and channel inputs are the pending event's own.
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._is_goal_continuation_event = lambda event: False
    runner._session_key_for_source = lambda source: entry.session_key
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(side_effect=lambda event, **kwargs: event.text)
    runner._reply_anchor_for_event = lambda event: None
    runner._pinned_channel_inputs = lambda key, prompt, source, internal: (prompt, source)
    runner._persist_prompt_pins = AsyncMock()
    runner._intake_adapter_for = lambda source: None
    runner._refresh_agent_cache_message_count = AsyncMock()

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
        yield SimpleNamespace(
            runner=runner, entry=entry, calls=calls, model_inputs=model_inputs,
            user_rows=lambda: [row["content"] for row in db.get_messages(entry.session_id) if row["role"] == "user"])
    finally:
        agent.close()
        db.close()


async def _run_turn(turns, event):
    with pytest.raises(_TurnObserved):
        await turns.runner._handle_message_with_agent(event, event.source, turns.entry.session_key, 1)


async def _run_queued_followup(turns, pending_event):
    """The drained pending slot runs as the next turn (``_run_agent_queued_followup``)."""
    from gateway.run import GatewayRunner

    turn_ctx = SimpleNamespace(
        source=SOURCE, session_id=turns.entry.session_id, session_key=turns.entry.session_key,
        run_generation=1, _interrupt_depth=0, history=[], _status_thread_metadata=None,
        context_prompt=None, channel_prompt=None, result_holder=[None])
    with pytest.raises(_TurnObserved):
        await GatewayRunner._run_agent_queued_followup(
            turns.runner, turn_ctx, adapter=None, pending=getattr(pending_event, "text", "leftover steer"),
            pending_event=pending_event, response="", result={"interrupted": True, "messages": []}, stream_task=None)


def _recorder():
    seen = []
    return seen, lambda source_origins, source_origins_complete, **kwargs: seen.append(
        (source_origins, source_origins_complete))


async def _hook_sees(home, monkeypatch, event):
    """``(source_origins, source_origins_complete)`` as a ``pre_llm_call`` callback receives them."""
    seen, record = _recorder()
    with _gateway_turns(home, monkeypatch, [record]) as turns:
        await _run_turn(turns, event)
    [received] = seen
    return received


# --- Builder -------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_ordinary_unedited_message_is_one_complete_origin(tmp_path, monkeypatch):
    event = _make_telegram_adapter()._build_message_event(_make_message(), MessageType.TEXT, update_id=7001)

    assert event.source_origins == (MessageOrigin("111", "1001", 7001, None),)
    assert event.source_origins_complete is True
    assert await _hook_sees(tmp_path / "home", monkeypatch, event) == (
        [{"chat_id": "111", "message_id": "1001", "update_id": 7001, "edit_date": None}], True)


def test_an_edited_message_is_a_distinct_revision():
    adapter = _make_telegram_adapter()
    original = adapter._build_message_event(_make_message(), MessageType.TEXT, update_id=7001)
    edited_message = _make_message(text="follow-up, fixed")
    edited_message.edit_date = datetime(2026, 1, 2, tzinfo=timezone.utc)
    edited = adapter._build_message_event(edited_message, MessageType.TEXT, update_id=7002)

    assert edited.source_origins == (MessageOrigin("111", "1001", 7002, 1767312000),)
    assert edited.source_origins_complete is True
    assert edited.source_origins != original.source_origins


@pytest.mark.parametrize("missing", ["update_id", "chat_id", "message_id"])
def test_a_message_missing_a_raw_id_is_unidentified(missing):
    """No origin is invented from a missing id: no stringified ``None``, the safe default instead."""
    message = _make_message()
    if missing == "chat_id":
        message.chat.id = None
    elif missing == "message_id":
        message.message_id = None
    update_id = None if missing == "update_id" else 7001

    event = _make_telegram_adapter()._build_message_event(message, MessageType.TEXT, update_id=update_id)

    assert (event.source_origins, event.source_origins_complete) == ((), False)


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


@pytest.mark.parametrize("missing_first", [False, True], ids=["missing-second", "missing-first"])
def test_a_merge_with_an_unidentified_part_is_incomplete(missing_first):
    identified, unidentified = _event("one", 1), _event("two", 2, origins=(), complete=False)
    first, second = (unidentified, identified) if missing_first else (identified, unidentified)
    pending = {"k": first}
    merge_pending_message_event(pending, "k", second, merge_text=True)

    assert pending["k"].source_origins == (_origin(1),)
    assert pending["k"].source_origins_complete is False


@pytest.mark.asyncio
async def test_a_debounced_busy_text_burst_reaches_the_hook_with_every_origin(tmp_path, monkeypatch):
    adapter = _make_busy_adapter()
    await adapter._queue_text_debounce("k", _event("first", 1))
    await adapter._queue_text_debounce("k", _event("second", 2))
    await adapter._flush_text_debounce_now("k")

    assert await _hook_sees(tmp_path / "home", monkeypatch, adapter._pending_messages["k"]) == (
        [_dict(1), _dict(2)], True)


@pytest.mark.asyncio
async def test_a_client_split_text_batch_reaches_the_hook_with_every_origin(tmp_path, monkeypatch):
    adapter = _make_batching_adapter()
    adapter._enqueue_text_event(_event("This is part one of a long", 1))
    adapter._enqueue_text_event(_event("message that was split by Telegram.", 2))
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())

    [dispatched] = [call.args[0] for call in adapter.handle_message.call_args_list]
    assert await _hook_sees(tmp_path / "home", monkeypatch, dispatched) == ([_dict(1), _dict(2)], True)


async def _flush_media(adapter, path, first, second):
    """Dispatch two photos through the ordinary album or photo-burst ingress; return the event."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter._media_batch_delay_seconds = 0.05
    with patch.object(TelegramAdapter, "MEDIA_GROUP_WAIT_SECONDS", 0.05):
        for event in (first, second):
            if path == "album":
                await adapter._queue_media_group_event("album", event)
            else:
                adapter._enqueue_photo_event("burst", event)
        await asyncio.gather(*adapter._media_group_tasks.values(), *adapter._pending_photo_batch_tasks.values())
    [dispatched] = [call.args[0] for call in adapter.handle_message.call_args_list]
    return dispatched


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["album", "photo-burst"])
async def test_an_album_reaches_the_hook_with_every_origin(path, tmp_path, monkeypatch):
    photo = dict(message_type=MessageType.PHOTO, media=True)
    dispatched = await _flush_media(_make_batching_adapter(), path, _event("caption", 1, **photo), _event("", 2, **photo))

    assert await _hook_sees(tmp_path / "home", monkeypatch, dispatched) == ([_dict(1), _dict(2)], True)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["album", "photo-burst"])
@pytest.mark.parametrize("first, second", [(False, True), (None, True), (True, False), (None, False)])
async def test_an_album_keeps_the_first_photo_reply_policy(path, first, second):
    """Regression: photo merges have always kept the first photo's ``reply_expected``; origins do
    not change that."""
    photo = dict(message_type=MessageType.PHOTO, media=True)
    dispatched = await _flush_media(_make_batching_adapter(), path, _event("", 1, reply_expected=first, **photo),
                                    _event("", 2, reply_expected=second, **photo))

    assert dispatched.reply_expected is first
    assert dispatched.media_urls == ["/tmp/p.jpg", "/tmp/p.jpg"]


# --- Copies and replays --------------------------------------------------------------------------

def _busy_runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: adapter
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/queue check the logs", "/steer check the logs"])
async def test_a_queued_command_keeps_its_message_origin(command):
    adapter = SimpleNamespace(_pending_messages={})
    runner = _busy_runner(adapter)
    handler = runner._busy_queue_command if command.startswith("/queue") else runner._busy_steer_command

    await handler(_event(command, 1), "k", SOURCE)

    queued = adapter._pending_messages["k"]
    assert queued.text == "check the logs"
    assert (queued.source_origins, queued.source_origins_complete) == ((_origin(1),), True)


@pytest.mark.asyncio
async def test_a_queue_overflow_chain_runs_each_turn_with_its_own_origins(tmp_path, monkeypatch):
    """The first ``/queue`` takes the pending slot, the second the overflow FIFO; draining the slot and
    promoting the overflow head runs each as its own follow-up turn with its own origin."""
    adapter = SimpleNamespace(_pending_messages={})
    busy = _busy_runner(adapter)
    await busy._busy_queue_command(_event("/queue first", 1), "k", SOURCE)
    await busy._busy_queue_command(_event("/queue second", 2), "k", SOURCE)
    assert len(busy._overflow_queue("k")) == 1
    head = adapter._pending_messages.pop("k")
    promoted = busy._promote_queued_event("k", adapter, None)
    seen, record = _recorder()

    with _gateway_turns(tmp_path / "home", monkeypatch, [record]) as turns:
        await _run_queued_followup(turns, head)
        await _run_queued_followup(turns, promoted)

    assert seen == [([_dict(1)], True), ([_dict(2)], True)]
    assert [message[-1]["content"] for message in turns.model_inputs] == ["first", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_event, expected", [
    (None, ([], False)),  # leftover steer text queued without an event
    (_event("one part unidentified", 1, complete=False), ([_dict(1)], False)),
], ids=["event-less", "incomplete"])
async def test_a_follow_up_turn_never_claims_more_than_its_event(pending_event, expected, tmp_path, monkeypatch):
    seen, record = _recorder()
    with _gateway_turns(tmp_path / "home", monkeypatch, [record]) as turns:
        await _run_queued_followup(turns, pending_event)

    assert seen == [expected]


@pytest.mark.asyncio
async def test_a_held_message_is_redispatched_with_its_origins(tmp_path, monkeypatch):
    adapter = _make_batching_adapter()
    adapter._mark_disconnected()
    adapter._enqueue_text_event(_event("sent during a reconnect", 1))
    adapter._drop_delayed_deliveries = False

    await adapter._redispatch_held_inbound()

    [redispatched] = [call.args[0] for call in adapter.handle_message.call_args_list]
    assert await _hook_sees(tmp_path / "home", monkeypatch, redispatched) == ([_dict(1)], True)


@pytest.mark.asyncio
@pytest.mark.parametrize("event, expected", [
    (_event("sent during startup", 1), ([_dict(1)], True)),
    (MessageEvent(text="an event built without origins", source=SOURCE), ([], False)),
], ids=["identified", "without-origins"])
async def test_a_startup_restore_replay_keeps_the_origins_it_had(event, expected, tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = object.__new__(GatewayRunner)
    runner._intake_adapter_for = lambda source: adapter
    runner._queue_startup_restore_event(event)

    assert await runner._drain_startup_restore_queue() == 1

    replayed = adapter.handle_message.await_args.args[0]
    assert await _hook_sees(tmp_path / "home", monkeypatch, replayed) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("flush", ["pending", "overflow"])
async def test_the_shutdown_spool_keeps_text_not_identity(flush, tmp_path, monkeypatch):
    """The spool's payload shape is unchanged and carries no identity: recovery appends the text to
    the transcript, and the turn that resumes after the restart has no origins."""
    from gateway.shutdown_flush import _get_flush_dir, flush_overflow_to_file, flush_pending_to_file, recover_pending_spool

    event = _event("sent before shutdown", 1)
    written = flush_pending_to_file({"k": event}) if flush == "pending" else flush_overflow_to_file({"k": [event]})
    assert written == 1
    [spooled] = _get_flush_dir().glob("*.json")
    assert json.loads(spooled.read_text(encoding="utf-8"))["data"] == {"text": "sent before shutdown"}
    rows = []
    sink = SimpleNamespace(append_message=lambda **row: rows.append(row))

    assert recover_pending_spool(sink, session_resolver=lambda key, not_after=None: ("origins", sink))[0] == 1

    assert [(row["role"], row["content"]) for row in rows] == [("user", "sent before shutdown")]
    resume = MessageEvent(text="", message_type=MessageType.TEXT, source=SOURCE, internal=True)
    assert await _hook_sees(tmp_path / "home", monkeypatch, resume) == ([], False)


# --- The turn ------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_every_callback_gets_its_own_copy_of_the_frozen_origins(tmp_path, monkeypatch):
    """Two consumers each get a new list of new dicts. One mutating its copy, and a message merged
    into the event after the turn started, change neither the turn nor the other consumer."""
    pending = {"k": _event("first", 1)}
    merge_pending_message_event(pending, "k", _event("second", 2), merge_text=True)
    event = pending["k"]
    seen, copies = [], []

    def tampering(source_origins, source_origins_complete, **kwargs):
        copies.append(source_origins)
        seen.append((copy.deepcopy(source_origins), source_origins_complete))
        source_origins[0]["message_id"] = "forged"
        source_origins.append(_dict(9))
        event.absorb(_event("arrived later", 3, complete=False))

    def observer(source_origins, source_origins_complete, **kwargs):
        copies.append(source_origins)
        seen.append((copy.deepcopy(source_origins), source_origins_complete))

    with _gateway_turns(tmp_path / "home", monkeypatch, [tampering, observer]) as turns:
        await _run_turn(turns, event)

    assert seen == [([_dict(1), _dict(2)], True)] * 2
    assert copies[0] is not copies[1] and copies[0][1] is not copies[1][1]
    assert event.source_origins == (_origin(1), _origin(2), _origin(3))  # the event moved on; the turn did not


@pytest.mark.asyncio
async def test_narrow_callbacks_get_exactly_the_fields_they_declare(tmp_path, monkeypatch):
    """A legacy signature is still called without the new fields; a narrow one that declares them
    gets its own copy too."""
    calls = []

    def legacy(session_id, user_message):
        calls.append((session_id, user_message))

    def origins_only(source_origins, source_origins_complete):
        calls.append((source_origins, source_origins_complete))

    with _gateway_turns(tmp_path / "home", monkeypatch, [legacy, origins_only]) as turns:
        await _run_turn(turns, _event("hello", 1))

    assert calls == [("origins", "hello"), ([_dict(1)], True)]


@pytest.mark.asyncio
async def test_origins_change_only_the_hook_inputs(tmp_path, monkeypatch):
    """Within this tree, an identified and an unidentified turn send the model the same messages and
    persist the same user row; only the identified one passes the two ``run_conversation`` keywords,
    and a turn without origins hands hooks the safe default."""
    runs = {}
    for name, event in (("identified", _event("hello", 1)),
                        ("plain", MessageEvent(text="hello", source=SOURCE, message_id="1"))):
        seen, record = _recorder()
        with _gateway_turns(tmp_path / name, monkeypatch, [record]) as turns:
            await _run_turn(turns, event)
            runs[name] = (turns.calls[0], turns.model_inputs[0], turns.user_rows(), seen)

    (identified_kwargs, *identified_turn, identified_seen), (plain_kwargs, *plain_turn, plain_seen) = runs.values()
    assert identified_turn == plain_turn
    assert (identified_kwargs["source_origins"], identified_kwargs["source_origins_complete"]) == ((_origin(1),), True)
    assert identified_kwargs.keys() ^ plain_kwargs.keys() == {"source_origins", "source_origins_complete"}
    assert (identified_seen, plain_seen) == ([([_dict(1)], True)], [([], False)])
