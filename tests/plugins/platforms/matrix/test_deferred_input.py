"""Deferred gateway input keeps its processing and preparation boundaries."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from gateway.platforms.event import MessageType
from gateway.platforms.base import ProcessingOutcome
from plugins.platforms.matrix.adapter import MatrixAdapter
from tests.plugins.platforms.matrix.test_feedback_policy import (
    _SteeringReceiver,
    _busy_runner,
    _intake_adapter,
    _receipt_event,
    _receipt_turn,
    _text_input,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["explicit", "ordinary-steer", "ordinary-interrupt"])
async def test_input_during_opening_delivery_runs_after_the_fifo(
    monkeypatch, tmp_path, route
):
    from threading import Event
    from openai.types.chat import ChatCompletion
    from run_agent import AIAgent

    adapter, receipts, _sender = _intake_adapter(monkeypatch, "after_processing")
    adapter._text_batch_delay_seconds = 0
    adapter._dm_rooms["!room:example.org"] = True
    adapter._reactions_enabled = True
    adapter._schedule_reaction_redaction = MagicMock()
    runner = _busy_runner(
        monkeypatch, adapter, "interrupt" if route.endswith("interrupt") else "steer"
    )
    gates = [(Event(), Event()) for _ in range(3)]
    delivery_started, delivery_release = asyncio.Event(), asyncio.Event()
    background_timer = asyncio.Event()
    requests, sent = [], []

    async def send_event(_room, event_type, content):
        if str(event_type) == "m.room.message" and content.get("body") == "done-1":
            delivery_started.set()
            await delivery_release.wait()
        sent.append((str(event_type), content))
        return f"$sent-{len(sent)}"

    adapter._client = MagicMock()
    adapter._client.send_message_event = AsyncMock(side_effect=send_event)
    monkeypatch.setattr(adapter, "send", MatrixAdapter.send.__get__(adapter))

    def model_response(_agent, request, **_kwargs):
        index = len(requests)
        requests.append(request)
        assert index < len(gates), "Unexpected repeated input"
        gates[index][0].set()
        assert gates[index][1].wait(2), "Model response was not released"
        return ChatCompletion.model_validate({
            "id": f"response-{index}",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": f"done-{index + 1}"},
                }
            ],
        })

    original_sleep = asyncio.sleep

    async def lifecycle_sleep(delay):
        if delay >= 1:
            await background_timer.wait()
            return
        await original_sleep(0)

    monkeypatch.setattr("gateway.run_turn.asyncio.sleep", lifecycle_sleep)
    monkeypatch.setattr(runner, "_persist_active_agents", lambda: None)
    monkeypatch.setattr(AIAgent, "_interruptible_api_call", model_response)
    monkeypatch.setattr("tools.env_probe.get_environment_probe_line", lambda: None)
    (tmp_path / "config.yaml").write_text(
        "model: {provider: custom, default: fake-model, context_length: 128000}\n"
        "auxiliary: {title_generation: {enabled: false}}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "custom",
            "base_url": "http://127.0.0.1:1/v1",
            "api_key": "test-key",
        },
    )
    agent = AIAgent(
        provider="custom",
        base_url="http://127.0.0.1:1/v1",
        api_key="test-key",
        model="fake-model",
        enabled_toolsets=["terminal"],
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
        skip_background_review=True,
    )
    monkeypatch.setattr(agent, "_disable_streaming", True)

    def fresh_agent(turn_runner, *_args):
        monkeypatch.setattr(agent, "session_id", turn_runner._ctx.session_id)
        return agent

    async def copy_inbound(event, _source):
        return replace(event)

    monkeypatch.setattr(
        "gateway.run_turn_runner.TurnRunner._build_fresh_agent", fresh_agent
    )
    monkeypatch.setattr(runner, "_hm_pre_gateway_dispatch_hook", copy_inbound)
    adapter.set_message_handler(runner._handle_message)
    try:
        await _text_input(adapter, "$opening", "opening")
        assert await asyncio.to_thread(gates[0][0].wait, 2)
        await _text_input(adapter, "$queued", "/queue queued")
        gates[0][1].set()
        await asyncio.wait_for(delivery_started.wait(), 2)
        await _text_input(
            adapter, "$late", "/steer late" if route == "explicit" else "late"
        )
        assert receipts.call_args_list == []
        delivery_release.set()
        assert await asyncio.to_thread(gates[1][0].wait, 2)
        assert receipts.call_args_list == [call("!room:example.org", "$opening")]
        gates[1][1].set()
        assert await asyncio.to_thread(gates[2][0].wait, 2)
        assert receipts.call_args_list == [
            call("!room:example.org", "$opening"),
            call("!room:example.org", "$queued"),
        ]
        assert [
            content["m.relates_to"] for kind, content in sent if kind == "m.reaction"
        ] == [
            {"rel_type": "m.annotation", "event_id": event_id, "key": emoji}
            for event_id, emojis in [
                ("$opening", ["👀", "✅"]),
                ("$queued", ["👀", "✅"]),
                ("$late", ["👀"]),
            ]
            for emoji in emojis
        ]
        gates[2][1].set()
        await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 2)
        assert (
            [
                request["messages"][-1]["content"].split("\n\n[System note:", 1)[0]
                for request in requests
            ],
            [
                content["body"]
                for kind, content in sent
                if kind == "m.room.message" and content["body"].startswith("done-")
            ],
            receipts.call_args_list,
        ) == (
            ["opening", "queued", "late"],
            ["done-1", "done-2", "done-3"],
            [
                call("!room:example.org", event_id)
                for event_id in ["$opening", "$queued", "$late"]
            ],
        )
        assert (
            requests[0]["messages"][0]
            == requests[1]["messages"][0]
            == requests[2]["messages"][0]
        )
        for request in requests:
            roles = [
                message["role"]
                for message in request["messages"]
                if message["role"] != "system"
            ]
            assert all(left != right for left, right in zip(roles, roles[1:]))
    finally:
        delivery_release.set()
        for _started, release in gates:
            release.set()
        await adapter.cancel_background_tasks()
        agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("queued", [False, True])
async def test_deferred_voice_keeps_its_preparation_and_inbound_context(
    monkeypatch, echo, queued
):
    from gateway.session_identity import identity_of

    adapter, receipts, sender = _intake_adapter(monkeypatch, "after_processing")
    adapter._dm_rooms["!room:example.org"] = True
    runner = _busy_runner(monkeypatch, adapter, "steer")
    runner.config.stt_enabled = True
    monkeypatch.setattr(runner, "_should_echo_stt_transcripts", lambda: echo)
    stt = MagicMock(return_value={"success": True, "transcript": "spoken correction"})
    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", stt)
    monkeypatch.setattr(
        runner, "_inbound_model_context_length", AsyncMock(return_value=128000)
    )
    receiver = _SteeringReceiver()
    opening, voice = _receipt_event("opening"), _receipt_event("voice.ogg")
    adapter.gateway_runner = runner
    assert adapter._canonicalize(voice.source) is not None
    voice.message_type = MessageType.VOICE
    voice.media_urls, voice.media_types = ["/tmp/voice.ogg"], ["audio/ogg"]
    voice.reply_to_message_id, voice.reply_to_text = "$quote", "quoted context"
    voice.metadata = {"application": "preserved"}
    ctx = _receipt_turn(runner, adapter, opening, receiver)
    key = ctx.session_key
    assert key is not None
    await adapter._run_processing_hook("on_processing_start", opening)
    if queued:
        runner._enqueue_fifo(key, _receipt_event("queued"), adapter)
    await runner._handle_active_session_busy_message(voice, key)
    admitted = receiver._drain_pending_steer()
    assert admitted is not None and admitted.endswith(
        '"spoken correction"\n\nvoice.ogg'
    )
    selected, _text = await runner._run_agent_drain_pending(
        {"final_response": "done", "pending_steer": admitted},
        adapter,
        opening.source,
        key,
        processing_event=opening,
    )
    if queued:
        selected, _text = await runner._run_agent_drain_pending(
            {"final_response": "queued done"},
            adapter,
            opening.source,
            key,
            processing_event=selected,
        )
    assert selected is not None
    prepared = await runner._prepare_profile_scoped_inbound_message_text(
        event=selected,
        source=selected.source,
        history=[],
        session_key=key,
    )
    assert (
        stt.call_args_list,
        sender.await_args_list,
        prepared,
        selected.source is voice.source,
        identity_of(selected.source) is identity_of(voice.source),
        selected.receipt_message_id,
        selected.media_urls,
        selected.media_types,
        selected.message_type,
        selected.reply_to_message_id,
        selected.metadata,
    ) == (
        [call("/tmp/voice.ogg", None, "gateway")],
        [call("!room:example.org", '🎙️ "spoken correction"', metadata=None)]
        if echo
        else [],
        '[Replying to: "quoted context"]\n\n' + admitted,
        True,
        True,
        "$voice.ogg",
        ["/tmp/voice.ogg"],
        ["audio/ogg"],
        MessageType.VOICE,
        "$quote",
        {"application": "preserved"},
    )
    await adapter._run_processing_hook(
        "on_processing_complete", opening, ProcessingOutcome.SUCCESS
    )
    await adapter._run_processing_hook("on_processing_start", selected)
    await adapter._run_processing_hook(
        "on_processing_complete", selected, ProcessingOutcome.SUCCESS
    )
    assert receipts.call_args_list == [
        call("!room:example.org", "$opening"),
        call("!room:example.org", "$voice.ogg"),
    ]
