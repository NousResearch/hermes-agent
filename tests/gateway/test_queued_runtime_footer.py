"""Queued replies keep their runtime footer without re-sending a delivered footer."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


SESSION_KEY = "agent:main:discord:dm:room-1"
FIRST_TEXT = "first reply"


class _Transport:
    # A relay-style adapter without the BasePlatformAdapter contract: the queued lane
    # takes the plain send path and this test can inspect the exact final text.
    name = "fake"
    extract_media = staticmethod(BasePlatformAdapter.extract_media)
    extract_images = staticmethod(BasePlatformAdapter.extract_images)

    def __init__(self, *, accepted=True):
        self.accepted = accepted
        self.sent = []
        self._pending_messages = {}

    def get_pending_message(self, session_key):
        return self._pending_messages.pop(session_key, None)

    async def send(self, chat_id, content, **kwargs):
        self.sent.append((chat_id, content, kwargs.get("metadata")))
        return SendResult(
            success=self.accepted,
            message_id="sent",
            error=None if self.accepted else "refused",
        )


def _footer_config(monkeypatch, *, enabled=True, fields=("model", "latency")):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "display": {"runtime_footer": {"enabled": enabled, "fields": list(fields)}}
        },
    )
    monkeypatch.setattr(gateway_run, "_platform_config_key", lambda platform: "discord")
    monkeypatch.setattr(gateway_run, "_terminal_scope_cwd", lambda _cwd: "")
    monkeypatch.setattr(
        gateway_run,
        "_strip_response_attachments_for_direct_send",
        lambda response, _adapter: response,
    )


def _runner(transport):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._delivery_adapter_for = lambda _source: transport
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    return runner


def _turn_context(streamed=False):
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="room-1",
        thread_id="thread-1",
        chat_type="dm",
        parent_chat_id=None,
    )
    consumer = SimpleNamespace(
        final_response_sent=True,
        delivered_final_matches=lambda text: text == FIRST_TEXT,
    )
    turn_ctx = TurnContext(
        source=source,
        session_id="session",
        session_key=SESSION_KEY,
        history=[],
        _interrupt_depth=0,
        context_prompt=None,
        stream_consumer_holder=[consumer if streamed else None],
        mute_notification_reply=False,
        persist_user_display_kind=None,
        reply_expected=True,
        _status_thread_metadata={"thread_id": source.thread_id},
        event_message_id="input-1",
        inbound_message_id="input-1",
        run_generation=1,
    )
    turn_ctx.turn_seconds = 22.0
    turn_ctx.result_holder = [None]
    return source, turn_ctx


def _queued_chain(runner, turn_ctx, transport, response, result, *, followup_result):
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value=followup_result)
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value=SESSION_KEY)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        return_value="follow up"
    )
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._pinned_channel_inputs = MagicMock(
        side_effect=lambda _key, channel_prompt, source, *, internal: (
            channel_prompt,
            source,
        )
    )
    runner._persist_prompt_pins = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._intake_adapter_for = MagicMock(return_value=None)
    # The first response was already delivered above; this call exercises the recursive
    # chain boundary, not the delivery method a second time.
    runner._run_agent_deliver_first_response = AsyncMock()
    pending_event = MessageEvent(
        text="follow up",
        source=turn_ctx.source,
        message_id="input-2",
        reply_expected=True,
    )
    return pending_event, gateway_run.GatewayRunner._run_agent_queued_followup(
        runner,
        turn_ctx,
        transport,
        "follow up",
        pending_event,
        response,
        result,
        None,
    )


def _sent_texts(transport):
    return [content for _chat, content, _metadata in transport.sent]


def _followup_result(*, text="tail reply", model="example/model-two", offset=5):
    return {
        "final_response": text,
        "model": model,
        "messages": [],
        "history_offset": offset,
        "turn_seconds": 33.0,
    }


def _completion_event(source):
    return MessageEvent(
        text="follow up", source=source, message_id="input-2", reply_expected=True
    )


def _completion_call(source, merged, footer):
    event = _completion_event(source)
    response = merged.get("final_response", "tail reply")
    if footer and not merged.get("already_sent"):
        response = f"{response}\n\n{footer}"
    return (
        event,
        source,
        SimpleNamespace(session_id="session"),
        SESSION_KEY,
        1,
        merged,
        [],
        response,
        footer,
        False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "followup_offset", [5, 20], ids=["unchanged-followup-result", "offset-copy"]
)
async def test_queued_chain_delivers_one_configured_footer_and_carries_latency(
    tmp_path,
    monkeypatch,
    followup_offset,
):
    _footer_config(monkeypatch)
    transport = _Transport()
    runner = _runner(transport)
    source, turn_ctx = _turn_context(streamed=False)
    response = {
        "final_response": FIRST_TEXT,
        "model": "example/model-one",
        "failed": False,
    }
    # Production has two distinct dictionaries: run_sync's returned response and
    # TurnContext.result_holder[0].
    result = {
        "final_response": FIRST_TEXT,
        "model": "example/model-one",
        "messages": [],
        "history_offset": 10,
        "failed": False,
    }

    await gateway_run.GatewayRunner._run_agent_deliver_first_response(
        runner,
        turn_ctx,
        transport,
        response,
        result,
        None,
    )
    assert _sent_texts(transport) == [f"{FIRST_TEXT}\n\nmodel-one · 22s"]

    followup_result = {
        "final_response": "tail reply",
        "model": "example/model-two",
        "messages": [],
        "history_offset": followup_offset,
        "turn_seconds": 33.0,
    }
    _pending_event, chain = _queued_chain(
        runner,
        turn_ctx,
        transport,
        response,
        result,
        followup_result=followup_result,
    )
    merged = await chain

    assert merged["history_offset"] == (10 if followup_offset == 20 else 5)
    # A real follow-up result owns its footer and latency; the first delivery never suppresses it.
    assert "runtime_footer_already_delivered" not in merged
    assert "queued_runtime_footer_seconds" not in merged
    terminal_footer = gateway_run.GatewayRunner._hmwa_runtime_footer_line(
        runner,
        merged,
        source,
        33.0,
    )
    assert terminal_footer == "model-two · 33s"
    assert (
        await gateway_run.GatewayRunner._hmwa_deliver_turn_response(
            runner, *_completion_call(source, merged, terminal_footer)
        )
        == "tail reply\n\nmodel-two · 33s"
    )
    assert _sent_texts(transport) == [f"{FIRST_TEXT}\n\nmodel-one · 22s"]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_runtime_footer_toggle_stays_authoritative_for_queued_replies(
    tmp_path,
    monkeypatch,
    enabled,
):
    _footer_config(monkeypatch, enabled=enabled)
    transport = _Transport()
    runner = _runner(transport)
    source, turn_ctx = _turn_context(streamed=True)
    response = {
        "final_response": FIRST_TEXT,
        "model": "example/model-one",
        "failed": False,
    }
    result = {
        "final_response": FIRST_TEXT,
        "model": "example/model-one",
        "messages": [],
        "failed": False,
    }

    await gateway_run.GatewayRunner._run_agent_deliver_first_response(
        runner,
        turn_ctx,
        transport,
        response,
        result,
        None,
    )
    expected = ["model-one · 22s"] if enabled else []
    assert _sent_texts(transport) == expected
    followup_result = {
        "final_response": "tail reply",
        "messages": [],
        "history_offset": 5,
    }
    _pending, chain = _queued_chain(
        runner,
        turn_ctx,
        transport,
        response,
        result,
        followup_result=followup_result,
    )
    merged = await chain

    assert "runtime_footer_already_delivered" not in merged
    await gateway_run.GatewayRunner._hmwa_deliver_turn_response(
        runner, *_completion_call(source, merged, "")
    )
    assert _sent_texts(transport) == expected


@pytest.mark.asyncio
async def test_failed_queued_footer_send_is_not_marked_delivered_and_completion_retries(
    tmp_path,
    monkeypatch,
):
    _footer_config(monkeypatch)
    transport = _Transport(accepted=False)
    runner = _runner(transport)
    source, turn_ctx = _turn_context(streamed=False)
    response = {
        "final_response": FIRST_TEXT,
        "model": "example/model-one",
        "failed": False,
    }
    result = {
        "final_response": FIRST_TEXT,
        "model": "example/model-one",
        "messages": [],
        "failed": False,
    }

    await gateway_run.GatewayRunner._run_agent_deliver_first_response(
        runner,
        turn_ctx,
        transport,
        response,
        result,
        None,
    )
    assert _sent_texts(transport) == [f"{FIRST_TEXT}\n\nmodel-one · 22s"]
    assert "already_sent" not in result
    assert "runtime_footer_already_delivered" not in result

    footer = gateway_run.GatewayRunner._hmwa_runtime_footer_line(
        runner,
        result,
        source,
        22.0,
    )
    assert footer == "model-one · 22s"
    assert (
        await gateway_run.GatewayRunner._hmwa_deliver_turn_response(
            runner, *_completion_call(source, result, footer)
        )
        == f"{FIRST_TEXT}\n\nmodel-one · 22s"
    )
    # The completion method only decides the outbound text; the caller (base delivery)
    # performs the send. The refused queued lane left it as the outstanding fallback.
    assert _sent_texts(transport) == [f"{FIRST_TEXT}\n\nmodel-one · 22s"]


@pytest.mark.asyncio
async def test_failed_trailing_footer_send_retries_through_normal_completion(
    tmp_path,
    monkeypatch,
):
    _footer_config(monkeypatch)
    transport = _Transport(accepted=False)
    runner = _runner(transport)
    source, turn_ctx = _turn_context(streamed=True)
    response = {
        "final_response": FIRST_TEXT,
        "model": "example/model-one",
        "failed": False,
    }
    result = {
        "final_response": FIRST_TEXT,
        "model": "example/model-one",
        "messages": [],
        "failed": False,
    }

    await gateway_run.GatewayRunner._run_agent_deliver_first_response(
        runner,
        turn_ctx,
        transport,
        response,
        result,
        None,
    )
    assert _sent_texts(transport) == ["model-one · 22s"]
    assert result["already_sent"] is True
    assert "runtime_footer_already_delivered" not in result

    transport.accepted = True
    footer = gateway_run.GatewayRunner._hmwa_runtime_footer_line(
        runner,
        result,
        source,
        22.0,
    )
    assert footer == "model-one · 22s"
    assert (
        await gateway_run.GatewayRunner._hmwa_deliver_turn_response(
            runner, *_completion_call(source, result, footer)
        )
        is None
    )
    assert _sent_texts(transport) == ["model-one · 22s", "model-one · 22s"]


@pytest.mark.asyncio
async def test_normal_completed_turn_still_formats_and_returns_the_configured_footer(
    tmp_path,
    monkeypatch,
):
    _footer_config(monkeypatch)
    transport = _Transport()
    runner = _runner(transport)
    source, _turn_ctx = _turn_context(streamed=False)
    result = {"final_response": "done", "model": "example/model-one", "failed": False}

    footer = gateway_run.GatewayRunner._hmwa_runtime_footer_line(
        runner, result, source, 22.0
    )
    assert footer == "model-one · 22s"
    assert (
        await gateway_run.GatewayRunner._hmwa_deliver_turn_response(
            runner, *_completion_call(source, result, footer)
        )
        == "done\n\nmodel-one · 22s"
    )
    assert _sent_texts(transport) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "followup_offset", [5, 20], ids=["unchanged-followup-result", "offset-copy"]
)
@pytest.mark.parametrize(
    "terminal_text",
    ["tail reply", FIRST_TEXT],
    ids=["different-text", "identical-text"],
)
async def test_fresh_terminal_reply_keeps_its_own_footer_even_with_identical_text(
    tmp_path,
    monkeypatch,
    followup_offset,
    terminal_text,
):
    """The first delivery suppresses only its own result object, never a fresh answer."""
    _footer_config(monkeypatch)
    transport = _Transport()
    runner = _runner(transport)
    source, turn_ctx = _turn_context(streamed=False)
    response = {
        "final_response": FIRST_TEXT,
        "model": "example/model-one",
        "failed": False,
    }
    result = {
        "final_response": FIRST_TEXT,
        "model": "example/model-one",
        "messages": [],
        "history_offset": 10,
        "failed": False,
    }
    await gateway_run.GatewayRunner._run_agent_deliver_first_response(
        runner,
        turn_ctx,
        transport,
        response,
        result,
        None,
    )

    followup = _followup_result(
        text=terminal_text, model="example/model-one", offset=followup_offset
    )
    _pending, chain = _queued_chain(
        runner,
        turn_ctx,
        transport,
        response,
        result,
        followup_result=followup,
    )
    merged = await chain

    assert merged is not response
    assert merged is not result
    assert "runtime_footer_already_delivered" not in merged
    assert "queued_runtime_footer_seconds" not in merged
    assert (
        gateway_run.GatewayRunner._hmwa_runtime_footer_line(
            runner,
            merged,
            source,
            33.0,
        )
        == "model-one · 33s"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refusal", ["goal-stale", "prepared-none"], ids=["goal-stale", "prepared-none"]
)
async def test_refused_continuation_returning_the_same_result_keeps_footer_suppressed(
    tmp_path,
    monkeypatch,
    refusal,
):
    _footer_config(monkeypatch)
    transport = _Transport()
    runner = _runner(transport)
    source, turn_ctx = _turn_context(streamed=False)
    response = {
        "final_response": FIRST_TEXT,
        "model": "example/model-one",
        "failed": False,
    }
    result = {
        "final_response": FIRST_TEXT,
        "model": "example/model-one",
        "messages": [],
        "history_offset": 10,
        "failed": False,
    }
    pending_event = _completion_event(source)
    runner._MAX_INTERRUPT_DEPTH = 8
    if refusal == "goal-stale":
        runner._is_goal_continuation_event = MagicMock(return_value=True)
        runner._goal_still_active_for_session = MagicMock(return_value=False)
    else:
        runner._is_goal_continuation_event = MagicMock(return_value=False)
        runner._session_key_for_source = MagicMock(return_value=SESSION_KEY)
        runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
            return_value=None
        )
    returned = await gateway_run.GatewayRunner._run_agent_queued_followup(
        runner,
        turn_ctx,
        transport,
        "follow up",
        pending_event,
        response,
        result,
        None,
    )

    assert returned is result
    assert result["runtime_footer_already_delivered"] is True
    assert (
        gateway_run.GatewayRunner._hmwa_runtime_footer_line(
            runner,
            result,
            source,
            99.0,
        )
        == ""
    )
    # The real queued helper delivers once before refusing the continuation.
    assert _sent_texts(transport) == [
        f"{FIRST_TEXT}\n\nmodel-one · 22s",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("returned", ["finalized", "raw"])
@pytest.mark.parametrize("offset", [5, 20], ids=["unchanged-offset", "offset-copy"])
@pytest.mark.parametrize("terminal_metadata", [False, True])
async def test_returned_delivered_reply_survives_all_result_copies(
    monkeypatch, streamed, returned, offset, terminal_metadata
):
    _footer_config(monkeypatch)
    transport = _Transport()
    runner = _runner(transport)
    source, ctx = _turn_context(streamed=streamed)
    finalized = {
        "final_response": FIRST_TEXT,
        "model": "example/model-one",
        "history_offset": offset,
        "messages": [],
        "failed": False,
        "turn_seconds": 22.0,
    }
    raw = {**finalized, "history_offset": 10}
    followup = finalized if returned == "finalized" else raw
    if returned == "raw":
        finalized["history_offset"] = 10
    if terminal_metadata:
        followup["queued_terminal_inbound_id"] = "deepest-input"
        followup["queued_terminal_notification_category"] = "result"
    await runner._run_agent_deliver_first_response(ctx, transport, finalized, raw, None)
    if returned == "raw":
        # The parent has an earlier persistence boundary than the returned raw result.
        raw["history_offset"] = offset
    parent = {**raw, "history_offset": 10}
    _pending, chain = _queued_chain(
        runner, ctx, transport, finalized, parent, followup_result=followup
    )
    merged = await chain

    assert merged["history_offset"] == min(offset, 10)
    assert merged["queued_terminal_inbound_id"] == (
        "deepest-input" if terminal_metadata else "input-2"
    )
    assert merged["queued_terminal_notification_category"] == "result"
    footer = runner._hmwa_runtime_footer_line(merged, source, 99.0)
    assert footer == ""
    assert merged["already_sent"] is True
    assert merged["media_already_delivered"] is True
    assert await runner._hmwa_deliver_turn_response(
        *_completion_call(source, merged, footer)
    ) is None
    assert _sent_texts(transport) == [
        "model-one · 22s" if streamed else f"{FIRST_TEXT}\n\nmodel-one · 22s"
    ]


def _outer_completion_runner(runner, source):
    """Keep the real outer footer/delivery flow, with persistence and admission isolated."""
    entry = SimpleNamespace(session_id="session")
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._hmwa_resolve_session = AsyncMock(return_value=(source, entry, SESSION_KEY))
    runner._hmwa_prepare_turn = AsyncMock(return_value=(
        runner._PreparedTurn([], "", "hello", "hello", None, None), {}
    ))
    runner._persist_prompt_pins = AsyncMock()
    runner._pinned_channel_inputs = lambda _key, prompt, source, **_kw: (prompt, source)
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._clear_session_env = lambda _tokens: None
    runner._hmwa_stop_typing_for_turn = AsyncMock()
    runner._is_session_run_current = lambda *_args: True
    runner._hmwa_shape_agent_response = AsyncMock(side_effect=lambda result, *_a, **_k: (
        result["final_response"], False, result.get("messages", [])
    ))
    runner._hmwa_post_turn_hooks = AsyncMock()
    runner._hmwa_persist_turn_transcript = AsyncMock()


def _measured_queue_runner(monkeypatch, *, streamed, queued, terminal_seconds=33.0):
    """Fake only worker inputs and unrelated chrome; drain, recursion and completion stay real."""
    from gateway import run_turn

    transport = _Transport()
    runner = _runner(transport)
    source, _ctx = _turn_context()
    _outer_completion_runner(runner, source)
    runner._get_proxy_url = lambda: None
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._draining = False
    runner._overflow_queue = lambda _key: []
    runner._is_goal_continuation_event = lambda _event: False
    runner._session_key_for_source = lambda _source: SESSION_KEY
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="follow up")
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._intake_adapter_for = lambda _source: None
    runner._run_agent = runner._run_agent_inner
    runner._run_agent_display_settings = lambda _source: SimpleNamespace(
        needs_progress_queue=False, log_mode_enabled=False, _native_slack_task_cards=False
    )
    contexts = []

    def build_context(_disp, _agent, **kwargs):
        _source, ctx = _turn_context(streamed=streamed and not contexts)
        ctx.turn_seconds = None
        ctx._interrupt_depth = kwargs["_interrupt_depth"]
        ctx.inbound_message_id = kwargs["inbound_message_id"]
        index = len(contexts)
        raw = {
            "final_response": FIRST_TEXT if index == 0 else "tail reply",
            "model": "example/model-one" if index == 0 else "example/model-two",
            "messages": [], "history_offset": 10 + index * 10, "failed": False,
        }
        ctx.result_holder[0] = raw
        contexts.append(ctx)
        return ctx, SimpleNamespace(run_sync=lambda: dict(raw)), None

    def start_worker(ctx, run_sync):
        future = asyncio.get_running_loop().create_future()
        future.set_result(run_sync())
        assert future.result() is not ctx.result_holder[0]
        return SimpleNamespace(executor_task=future, agent_timeout=None)

    runner._run_agent_build_turn_context = build_context
    runner._run_agent_start_turn_worker = start_worker
    runner._run_agent_bind_turn_wiring = lambda *_a, **_k: None
    runner._run_agent_start_streaming_tts = lambda *_a, **_k: None
    runner._run_agent_schedule_bubble_cleanup = lambda *_a, **_k: None
    runner._release_running_agent_state = lambda *_a, **_k: None
    for name in (
        "_run_agent_stream_consumer_task", "_run_agent_track_agent",
        "_run_agent_monitor_for_interrupt", "_run_agent_notify_long_running",
    ):
        setattr(runner, name, AsyncMock())
    runner._await_stream_task = AsyncMock()
    if queued:
        transport._pending_messages[SESSION_KEY] = _completion_event(source)
    # Rebind this module's time object; never replace the shared asyncio clock.
    times = iter(
        [90.0, 100.0, 122.0, 150.0, 150.0 + terminal_seconds, 200.0]
        if queued else [90.0, 100.0, 122.0, 200.0]
    )
    monkeypatch.setattr(run_turn, "time", SimpleNamespace(
        time=time.time, monotonic=lambda: next(times)
    ))
    return runner, source, transport, contexts


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("footer_mode", ["enabled", "disabled", "platform-enabled", "platform-disabled"])
async def test_measured_reply_latency_reaches_real_queue_and_outer_completion(
    monkeypatch, streamed, queued, footer_mode
):
    _footer_config(monkeypatch, enabled=footer_mode == "enabled")
    enabled = footer_mode in ("enabled", "platform-enabled")
    if footer_mode.startswith("platform-"):
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"display": {
            "runtime_footer": {"enabled": not enabled, "fields": ["model", "latency"]},
            "platforms": {"discord": {"runtime_footer": {"enabled": enabled}}},
        }})
    runner, source, transport, _contexts = _measured_queue_runner(
        monkeypatch, streamed=streamed, queued=queued
    )
    response = await runner._handle_message_with_agent(
        _completion_event(source), source, SESSION_KEY, 1
    )
    if queued:
        assert not transport._pending_messages
        expected = (["model-one · 22s"] if enabled else []) if streamed else [
            f"{FIRST_TEXT}\n\nmodel-one · 22s" if enabled else FIRST_TEXT
        ]
        assert _sent_texts(transport) == expected
        assert response == ("tail reply\n\nmodel-two · 33s" if enabled else "tail reply")
    else:
        if streamed:
            assert response is None
            assert _sent_texts(transport) == (["model-one · 22s"] if enabled else [])
        else:
            assert response == (f"{FIRST_TEXT}\n\nmodel-one · 22s" if enabled else FIRST_TEXT)
            assert _sent_texts(transport) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [None, -1.0, "missing"])
async def test_unavailable_reply_latency_is_omitted_at_both_delivery_boundaries(
    monkeypatch, seconds
):
    from gateway import run_turn

    _footer_config(monkeypatch)
    transport = _Transport()
    runner = _runner(transport)
    source, ctx = _turn_context()
    if seconds == "missing":
        del ctx.turn_seconds
    else:
        ctx.turn_seconds = seconds
    raw = _followup_result(text=FIRST_TEXT, model="example/model-one")
    if seconds == "missing":
        raw.pop("turn_seconds")
    else:
        raw["turn_seconds"] = seconds
    finalized = dict(raw)
    await runner._run_agent_deliver_first_response(ctx, transport, finalized, raw, None)
    assert _sent_texts(transport) == [f"{FIRST_TEXT}\n\nmodel-one"]

    terminal = _followup_result()
    terminal["queued_terminal_inbound_id"] = "input-2"
    if seconds != "missing":
        terminal["turn_seconds"] = seconds
    else:
        terminal.pop("turn_seconds")
    _outer_completion_runner(runner, source)
    runner._run_agent = AsyncMock(return_value=terminal)
    times = iter([90.0, 200.0])
    monkeypatch.setattr(run_turn, "time", SimpleNamespace(time=time.time, monotonic=lambda: next(times)))
    assert await runner._handle_message_with_agent(
        _completion_event(source), source, SESSION_KEY, 1
    ) == "tail reply\n\nmodel-two"


@pytest.mark.asyncio
async def test_negative_measured_duration_does_not_become_zero_latency(monkeypatch):
    _footer_config(monkeypatch)
    runner, source, transport, _contexts = _measured_queue_runner(
        monkeypatch, streamed=False, queued=True, terminal_seconds=-1.0
    )
    assert await runner._handle_message_with_agent(
        _completion_event(source), source, SESSION_KEY, 1
    ) == "tail reply\n\nmodel-two"
    assert _sent_texts(transport) == [f"{FIRST_TEXT}\n\nmodel-one · 22s"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["body", "footer"])
@pytest.mark.parametrize("raises", [False, True], ids=["refused", "exception"])
async def test_failed_sends_only_preserve_confirmed_delivery_receipts(
    monkeypatch, stage, raises
):
    _footer_config(monkeypatch)

    class FailingTransport(_Transport):
        async def send(self, chat_id, content, **kwargs):
            if raises and not self.accepted:
                raise OSError("transport unavailable")
            return await super().send(chat_id, content, **kwargs)

    transport = FailingTransport(accepted=False)
    runner = _runner(transport)
    source, ctx = _turn_context(streamed=stage == "footer")
    raw = _followup_result(text=FIRST_TEXT, model="example/model-one")
    raw["turn_seconds"] = 22.0
    finalized = dict(raw)
    await runner._run_agent_deliver_first_response(ctx, transport, finalized, raw, None)
    for result in (finalized, raw):
        assert not result.get("runtime_footer_already_delivered")
        assert bool(result.get("already_sent")) == (stage == "footer")
        assert bool(result.get("media_already_delivered")) == (stage == "footer")
    transport.accepted = True
    _pending, chain = _queued_chain(
        runner, ctx, transport, finalized, raw, followup_result=finalized
    )
    merged = await chain
    footer = runner._hmwa_runtime_footer_line(merged, source, 99.0)
    assert footer == "model-one · 22s"
    response = await runner._hmwa_deliver_turn_response(
        *_completion_call(source, merged, footer)
    )
    if stage == "footer":
        assert response is None
        assert _sent_texts(transport)[-1] == footer
    else:
        assert response == f"{FIRST_TEXT}\n\n{footer}"
