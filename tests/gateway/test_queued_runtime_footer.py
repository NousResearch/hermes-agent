"""Queued replies keep their runtime footer without re-sending a delivered footer."""

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

    async def send(self, chat_id, content, **kwargs):
        self.sent.append((chat_id, content, kwargs.get("metadata")))
        return SendResult(
            success=self.accepted, message_id="sent",
            error=None if self.accepted else "refused",
        )


def _footer_config(monkeypatch, *, enabled=True, fields=("model", "latency")):
    monkeypatch.setattr(
        gateway_run, "_load_gateway_config",
        lambda: {"display": {"runtime_footer": {"enabled": enabled, "fields": list(fields)}}},
    )
    monkeypatch.setattr(gateway_run, "_platform_config_key", lambda platform: "discord")
    monkeypatch.setattr(gateway_run, "_terminal_scope_cwd", lambda _cwd: "")
    monkeypatch.setattr(
        gateway_run, "_strip_response_attachments_for_direct_send", lambda response, _adapter: response,
    )


def _runner(transport):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._delivery_adapter_for = lambda _source: transport
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    return runner


def _turn_context(streamed=False):
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="room-1", thread_id="thread-1", chat_type="dm",
        parent_chat_id=None,
    )
    consumer = SimpleNamespace(
        final_response_sent=True, delivered_final_matches=lambda text: text == FIRST_TEXT,
    )
    turn_ctx = TurnContext(
        source=source, session_id="session", session_key=SESSION_KEY, history=[],
        _interrupt_depth=0, context_prompt=None,
        stream_consumer_holder=[consumer if streamed else None],
        mute_notification_reply=False, persist_user_display_kind=None, reply_expected=True,
        _status_thread_metadata={"thread_id": source.thread_id},
        event_message_id="input-1", inbound_message_id="input-1", run_generation=1,
        turn_seconds=22.0, runtime_footer_already_delivered=False,
    )
    turn_ctx.result_holder = [None]
    return source, turn_ctx


def _queued_chain(runner, turn_ctx, transport, response, result, *, followup_result):
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value=followup_result)
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value=SESSION_KEY)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="follow up")
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._pinned_channel_inputs = MagicMock(
        side_effect=lambda _key, channel_prompt, source, *, internal: (channel_prompt, source)
    )
    runner._persist_prompt_pins = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._intake_adapter_for = MagicMock(return_value=None)
    # The first response was already delivered above; this call exercises the recursive
    # chain boundary, not the delivery method a second time.
    runner._run_agent_deliver_first_response = AsyncMock()
    pending_event = MessageEvent(
        text="follow up", source=turn_ctx.source, message_id="input-2", reply_expected=True,
    )
    return pending_event, gateway_run.GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, transport, "follow up", pending_event, response, result, None,
    )


def _sent_texts(transport):
    return [content for _chat, content, _metadata in transport.sent]


def _completion_call(source, merged, footer):
    event = MessageEvent(text="follow up", source=source, message_id="input-2", reply_expected=True)
    response = merged.get("final_response", "tail reply")
    if footer and not merged.get("already_sent"):
        response = f"{response}\n\n{footer}"
    return (
        event, source, SimpleNamespace(session_id="session"), SESSION_KEY, 1, merged, [],
        response, footer, False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("followup_offset", [5, 20], ids=["unchanged-followup-result", "offset-copy"])
async def test_queued_chain_delivers_one_configured_footer_and_carries_latency(
    tmp_path, monkeypatch, followup_offset,
):
    _footer_config(monkeypatch)
    transport = _Transport()
    runner = _runner(transport)
    source, turn_ctx = _turn_context(streamed=False)
    response = {"final_response": FIRST_TEXT, "model": "example/model-one", "failed": False}
    # Production has two distinct dictionaries: run_sync's returned response and
    # TurnContext.result_holder[0].
    result = {
        "final_response": FIRST_TEXT, "model": "example/model-one", "messages": [],
        "history_offset": 10, "failed": False,
    }

    await gateway_run.GatewayRunner._run_agent_deliver_first_response(
        runner, turn_ctx, transport, response, result, None,
    )
    assert _sent_texts(transport) == [f"{FIRST_TEXT}\n\nmodel-one · 22s"]

    followup_result = {
        "final_response": "tail reply", "model": "example/model-two", "messages": [],
        "history_offset": followup_offset,
    }
    pending_event, chain = _queued_chain(
        runner, turn_ctx, transport, response, result, followup_result=followup_result,
    )
    merged = await chain

    assert merged["history_offset"] == (10 if followup_offset == 20 else 5)
    assert merged["runtime_footer_already_delivered"] is True
    assert merged["queued_runtime_footer_seconds"] == 22.0
    assert gateway_run.GatewayRunner._hmwa_runtime_footer_line(
        runner, merged, source, 99,
    ) == ""
    assert await gateway_run.GatewayRunner._hmwa_deliver_turn_response(
        runner, *_completion_call(source, merged, "")
    ) == "tail reply"
    assert _sent_texts(transport) == [f"{FIRST_TEXT}\n\nmodel-one · 22s"]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_runtime_footer_toggle_stays_authoritative_for_queued_replies(
    tmp_path, monkeypatch, enabled,
):
    _footer_config(monkeypatch, enabled=enabled)
    transport = _Transport()
    runner = _runner(transport)
    source, turn_ctx = _turn_context(streamed=True)
    response = {"final_response": FIRST_TEXT, "model": "example/model-one", "failed": False}
    result = {
        "final_response": FIRST_TEXT, "model": "example/model-one", "messages": [], "failed": False,
    }

    await gateway_run.GatewayRunner._run_agent_deliver_first_response(
        runner, turn_ctx, transport, response, result, None,
    )
    expected = ["model-one · 22s"] if enabled else []
    assert _sent_texts(transport) == expected
    followup_result = {"final_response": "tail reply", "messages": [], "history_offset": 5}
    _pending, chain = _queued_chain(
        runner, turn_ctx, transport, response, result, followup_result=followup_result,
    )
    merged = await chain

    if enabled:
        assert merged["runtime_footer_already_delivered"] is True
    else:
        assert "runtime_footer_already_delivered" not in merged
    await gateway_run.GatewayRunner._hmwa_deliver_turn_response(
        runner, *_completion_call(source, merged, "")
    )
    assert _sent_texts(transport) == expected


@pytest.mark.asyncio
async def test_failed_queued_footer_send_is_not_marked_delivered_and_completion_retries(
    tmp_path, monkeypatch,
):
    _footer_config(monkeypatch)
    transport = _Transport(accepted=False)
    runner = _runner(transport)
    source, turn_ctx = _turn_context(streamed=False)
    response = {"final_response": FIRST_TEXT, "model": "example/model-one", "failed": False}
    result = {
        "final_response": FIRST_TEXT, "model": "example/model-one", "messages": [], "failed": False,
    }

    await gateway_run.GatewayRunner._run_agent_deliver_first_response(
        runner, turn_ctx, transport, response, result, None,
    )
    assert _sent_texts(transport) == [f"{FIRST_TEXT}\n\nmodel-one · 22s"]
    assert "already_sent" not in result
    assert "runtime_footer_already_delivered" not in result

    footer = gateway_run.GatewayRunner._hmwa_runtime_footer_line(
        runner, result, source, 22.0,
    )
    assert footer == "model-one · 22s"
    assert await gateway_run.GatewayRunner._hmwa_deliver_turn_response(
        runner, *_completion_call(source, result, footer)
    ) == f"{FIRST_TEXT}\n\nmodel-one · 22s"
    # The completion method only decides the outbound text; the caller (base delivery)
    # performs the send. The refused queued lane left it as the outstanding fallback.
    assert _sent_texts(transport) == [f"{FIRST_TEXT}\n\nmodel-one · 22s"]


@pytest.mark.asyncio
async def test_failed_trailing_footer_send_retries_through_normal_completion(
    tmp_path, monkeypatch,
):
    _footer_config(monkeypatch)
    transport = _Transport(accepted=False)
    runner = _runner(transport)
    source, turn_ctx = _turn_context(streamed=True)
    response = {"final_response": FIRST_TEXT, "model": "example/model-one", "failed": False}
    result = {
        "final_response": FIRST_TEXT, "model": "example/model-one", "messages": [], "failed": False,
    }

    await gateway_run.GatewayRunner._run_agent_deliver_first_response(
        runner, turn_ctx, transport, response, result, None,
    )
    assert _sent_texts(transport) == ["model-one · 22s"]
    assert result["already_sent"] is True
    assert "runtime_footer_already_delivered" not in result

    transport.accepted = True
    footer = gateway_run.GatewayRunner._hmwa_runtime_footer_line(
        runner, result, source, 22.0,
    )
    assert footer == "model-one · 22s"
    assert await gateway_run.GatewayRunner._hmwa_deliver_turn_response(
        runner, *_completion_call(source, result, footer)
    ) is None
    assert _sent_texts(transport) == ["model-one · 22s", "model-one · 22s"]


@pytest.mark.asyncio
async def test_normal_completed_turn_still_formats_and_returns_the_configured_footer(
    tmp_path, monkeypatch,
):
    _footer_config(monkeypatch)
    transport = _Transport()
    runner = _runner(transport)
    source, _turn_ctx = _turn_context(streamed=False)
    result = {"final_response": "done", "model": "example/model-one", "failed": False}

    footer = gateway_run.GatewayRunner._hmwa_runtime_footer_line(runner, result, source, 22.0)
    assert footer == "model-one · 22s"
    assert await gateway_run.GatewayRunner._hmwa_deliver_turn_response(
        runner, *_completion_call(source, result, footer)
    ) == "done\n\nmodel-one · 22s"
    assert _sent_texts(transport) == []
