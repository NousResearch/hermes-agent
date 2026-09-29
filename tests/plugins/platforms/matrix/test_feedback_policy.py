"""Matrix feedback settings belong to the receiving adapter's profile."""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from threading import Lock
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from agent import secret_scope
from agent.interrupt_control import InterruptControlMixin
from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL, _INTERRUPT_REASON_STOP, _profile_runtime_scope
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.platforms.base import ExecApprovalPrompt, SendResult
from gateway.turn_context import TurnContext
from gateway.session import SessionSource
from hermes_cli.config import atomic_config_write
from plugins.platforms.matrix.adapter import MatrixAdapter


def _cooperatively_interrupted_result() -> dict:
    from run_agent import AIAgent

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

    def stop_during_request(_request, **_kwargs):
        agent.interrupt(_INTERRUPT_REASON_STOP)
        raise InterruptedError("Provider request interrupted")

    agent._interruptible_api_call = stop_during_request
    agent._interruptible_streaming_api_call = stop_during_request
    try:
        result = agent.run_conversation("Stop this turn during the provider request")
        assert (result.get("interrupted"), result.get("completed")) == (True, False)
        return result
    finally:
        agent.close()


def _intake_adapter(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> tuple[MatrixAdapter, MagicMock, AsyncMock]:
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            typing_indicator=False,
            extra={
                "read_receipts": mode,
                "reactions": False,
                "auto_thread": False,
                "user_id": "@hermes:example.org",
            },
        )
    )
    adapter._dm_rooms["!room:example.org"] = False
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._download_and_cache_media = AsyncMock(
        side_effect=lambda *args: f"/tmp/{args[1]}.ogg"
    )
    receipts = MagicMock()
    sender = AsyncMock(return_value=SendResult(success=True, message_id="$reply"))
    adapter._background_read_receipt = receipts
    adapter.send = sender
    adapter.stop_typing = AsyncMock()
    return adapter, receipts, sender


async def _text_input(adapter: MatrixAdapter, event_id: str, body: str) -> None:
    await adapter._handle_text_message(
        "!room:example.org",
        "@alice:example.org",
        event_id,
        0,
        {"body": body, "m.mentions": {"user_ids": ["@hermes:example.org"]}},
        {},
    )


def _busy_runner(
    monkeypatch: pytest.MonkeyPatch, adapter: MatrixAdapter, mode: str
) -> GatewayRunner:
    runner = GatewayRunner(config=GatewayConfig())
    runner.adapters[Platform.MATRIX] = adapter
    runner._busy_input_mode = mode
    runner._busy_text_mode = "interrupt"
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["immediate", "after_processing", "disabled"])
@pytest.mark.parametrize("aggregation", ["batch", "queued-photos", "queued-text"])
async def test_aggregated_turn_receipts_cover_the_latest_native_input(
    monkeypatch, mode, aggregation
):
    adapter, receipts, _sender = _intake_adapter(monkeypatch, mode)
    batch_ready, started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sleep = asyncio.sleep

    async def timer(delay):
        if delay == 0:
            await sleep(0)
            return
        await batch_ready.wait()

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", timer)
    seen = []

    async def respond(event):
        seen.append((event.message_id, event.text, list(event.media_urls)))
        started.set()
        await release.wait()
        return "done"

    adapter.set_message_handler(respond)
    room = "!room:example.org"
    try:
        if aggregation == "batch":
            await _text_input(adapter, "$first", "first")
            await _text_input(adapter, "$second", "second")
            batch_ready.set()
            await asyncio.gather(*adapter._pending_text_batch_tasks.values())
        else:
            adapter._text_batch_delay_seconds = 0
            runner = _busy_runner(monkeypatch, adapter, "queue")
            await _text_input(adapter, "$opening", "opening")
            await asyncio.wait_for(started.wait(), 2)
            if aggregation == "queued-photos":
                for event_id, body in (("$first", "first"), ("$second", "second")):
                    await adapter._handle_media_message(
                        room,
                        "@alice:example.org",
                        event_id,
                        0,
                        {
                            "body": body,
                            "msgtype": "m.image",
                            "url": f"mxc://example.org/{event_id}",
                            "m.mentions": {"user_ids": ["@hermes:example.org"]},
                        },
                        {},
                        "m.image",
                    )
            else:
                adapter._busy_text_mode = "queue"
                runner._busy_text_mode = "queue"
                await _text_input(adapter, "$first", "first")
                await _text_input(adapter, "$second", "second")

        await asyncio.wait_for(started.wait(), 2)
        arrival_ids = (
            ["$first", "$second"]
            if aggregation == "batch"
            else ["$opening", "$first", "$second"]
        )
        assert receipts.call_args_list == (
            [call(room, event_id) for event_id in arrival_ids]
            if mode == "immediate"
            else []
        )
        release.set()
        while adapter._background_tasks:
            await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 2)
        photos = (
            ["/tmp/$first.ogg", "/tmp/$second.ogg"]
            if aggregation == "queued-photos"
            else []
        )
        retained_id = "$second" if aggregation == "queued-text" else "$first"
        merged_text = (
            "first\n\nsecond" if aggregation == "queued-photos" else "first\nsecond"
        )
        assert seen == (
            [] if aggregation == "batch" else [("$opening", "opening", [])]
        ) + [
            (retained_id, merged_text, photos),
        ]
        completed_ids = (
            ["$second"] if aggregation == "batch" else ["$opening", "$second"]
        )
        expected_ids = {
            "immediate": arrival_ids,
            "after_processing": completed_ids,
            "disabled": [],
        }[mode]
        assert receipts.call_args_list == [
            call(room, event_id) for event_id in expected_ids
        ]
    finally:
        release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ["startup", "priority"])
@pytest.mark.parametrize("rewritten", [False, True])
async def test_deferred_admission_receives_no_completion_until_replayed(
    monkeypatch, admission, rewritten
):
    adapter, receipts, sender = _intake_adapter(monkeypatch, "after_processing")
    adapter._reactions_enabled = True
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    adapter._schedule_reaction_redaction = MagicMock()
    runner = _busy_runner(monkeypatch, adapter, "queue")
    event = MessageEvent(
        text="deferred",
        message_id="$deferred",
        source=SessionSource(
            platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="dm"
        ),
    )
    key = adapter._event_session_key(event)
    adapter._spawn_drain_task = MagicMock()
    adapter._requeue_backoff_delay = MagicMock(return_value=1)
    queued = []
    if admission == "startup":
        runner._startup_restore_in_progress = True
    else:
        runner._session_state(key).turn.agent = MagicMock()

    async def admit(incoming):
        incoming = replace(incoming) if rewritten else incoming
        if admission == "startup":
            result = await runner._handle_message(incoming)
            queued.extend(runner._startup_restore_queue)
            return result
        result = await runner._hm_handle_running_session_message(
            incoming, incoming.source, key
        )
        queued.extend(adapter._pending_messages.values())
        return result

    adapter.set_message_handler(admit)
    await adapter._process_message_background(event, key)
    assert (queued, receipts.call_args_list, sender.await_args_list) == (
        [event],
        [],
        [],
    )
    adapter._pending_messages.clear()
    adapter.set_message_handler(AsyncMock(return_value="processed"))
    await adapter._process_message_background(event, key)
    assert receipts.call_args_list == [call(event.source.chat_id, event.message_id)]
    assert adapter._send_reaction.await_args_list == [
        call(event.source.chat_id, event.message_id, "👀"),
        call(event.source.chat_id, event.message_id, "✅"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_count", [2, 3])
@pytest.mark.parametrize("stop_kind", ["task-cancel", "cooperative"])
async def test_completed_turns_are_acknowledged_before_a_cancellable_followup(
    monkeypatch, turn_count, stop_kind
):
    interrupted_result = (
        await asyncio.to_thread(_cooperatively_interrupted_result)
        if stop_kind == "cooperative"
        else None
    )
    adapter, receipts, sender = _intake_adapter(monkeypatch, "after_processing")
    adapter._reactions_enabled = True
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    adapter._schedule_reaction_redaction = MagicMock()
    runner = _busy_runner(monkeypatch, adapter, "queue")
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        side_effect=lambda **kwargs: kwargs["event"].text
    )
    runner._refresh_agent_cache_message_count = AsyncMock()
    events = [
        MessageEvent(
            text=f"turn-{index}",
            message_id=f"$turn-{index}",
            source=SessionSource(
                platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="dm"
            ),
        )
        for index in range(turn_count)
    ]
    key = adapter._event_session_key(events[0])
    started, release = asyncio.Event(), asyncio.Event()
    at_last_turn = []

    async def run_turn(current, following):
        result = {"final_response": f"reply-{current.text}", "messages": []}
        ctx = TurnContext(
            source=current.source,
            session_key=key,
            session_id="receipt-chain",
            event_message_id=current.message_id,
            inbound_message_id=current.message_id,
            processing_event=current,
        )
        return await runner._run_agent_queued_followup(
            ctx, adapter, following.text, following, result, result, None
        )

    async def next_turn(**kwargs):
        index = int(kwargs["message"].split("-")[-1])
        if index < turn_count - 1:
            return await run_turn(events[index], events[index + 1])
        at_last_turn.append((
            list(receipts.call_args_list),
            list(sender.await_args_list),
        ))
        started.set()
        await release.wait()
        return interrupted_result

    monkeypatch.setattr(runner, "_run_agent", next_turn)

    async def respond(event):
        result = await run_turn(event, events[1])
        return result.get("final_response")

    adapter.set_message_handler(respond)
    try:
        await adapter.handle_message(events[0])
        await asyncio.wait_for(started.wait(), 2)
        completed = events[:-1]
        expected_receipts = [
            call(event.source.chat_id, event.message_id) for event in completed
        ]
        assert at_last_turn == [
            (
                expected_receipts,
                [
                    call(
                        chat_id=event.source.chat_id,
                        content=f"reply-{event.text}",
                        reply_to=event.message_id,
                        metadata={"notify": True},
                    )
                    for event in completed
                ],
            )
        ]
        if stop_kind == "task-cancel":
            await adapter.cancel_session_processing(key)
        else:
            release.set()
            await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 2)
        assert receipts.call_args_list == expected_receipts
        assert adapter._send_reaction.await_args_list == [
            invocation
            for index, event in enumerate(events)
            for invocation in (
                [
                    call(event.source.chat_id, event.message_id, "👀"),
                    call(event.source.chat_id, event.message_id, "✅"),
                ]
                if index < turn_count - 1
                else [call(event.source.chat_id, event.message_id, "👀")]
            )
        ]
    finally:
        release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["immediate", "after_processing", "disabled"])
@pytest.mark.parametrize("result_kind", ["interrupted", "failed", "handler-error"])
async def test_ordinary_turn_completion_preserves_the_returned_agent_outcome(
    monkeypatch, tmp_path, mode, result_kind
):
    result = (
        await asyncio.to_thread(_cooperatively_interrupted_result)
        if result_kind == "interrupted"
        else {
            "final_response": "Provider unavailable",
            "messages": [],
            "failed": True,
            "completed": False,
        }
    )
    adapter, receipts, _sender = _intake_adapter(monkeypatch, mode)
    adapter._reactions_enabled = True
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    adapter._schedule_reaction_redaction = MagicMock()
    adapter._text_batch_delay_seconds = 0
    runner = _busy_runner(monkeypatch, adapter, "queue")
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs", lambda: {"api_key": "test-key"}
    )
    runner._run_agent = (
        AsyncMock(side_effect=RuntimeError("Provider failed"))
        if result_kind == "handler-error"
        else AsyncMock(return_value=result)
    )
    adapter.set_message_handler(runner._handle_message)
    room = "!room:example.org"
    try:
        await _text_input(adapter, "$opening", "opening")
        await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 2)
        runner._run_agent.assert_awaited_once()
        expected_receipts = (
            [call(room, "$opening")]
            if mode == "immediate"
            or (mode == "after_processing" and result_kind != "interrupted")
            else []
        )
        expected_reactions = [call(room, "$opening", "👀")]
        if result_kind != "interrupted":
            expected_reactions.append(call(room, "$opening", "✅"))
        assert (receipts.call_args_list, adapter._send_reaction.await_args_list) == (
            expected_receipts,
            expected_reactions,
        )
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["immediate", "after_processing", "disabled"])
@pytest.mark.parametrize(
    "command_reply", ["active", None, RuntimeError("status failed")]
)
async def test_inline_command_receipt_is_independent_of_the_active_turn(
    monkeypatch, mode, command_reply
):
    adapter, receipts, _sender = _intake_adapter(monkeypatch, mode)
    adapter._reactions_enabled = True
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    adapter._schedule_reaction_redaction = MagicMock()
    adapter._text_batch_delay_seconds = 0
    _busy_runner(monkeypatch, adapter, "queue")
    started, release = asyncio.Event(), asyncio.Event()

    async def respond(event):
        if event.get_command() == "status":
            if isinstance(command_reply, Exception):
                raise command_reply
            return command_reply
        started.set()
        await release.wait()
        return "done"

    adapter.set_message_handler(respond)
    room = "!room:example.org"
    try:
        await _text_input(adapter, "$opening", "opening")
        await asyncio.wait_for(started.wait(), 2)
        await _text_input(adapter, "$status", "/status")
        assert (receipts.call_args_list, adapter._send_reaction.await_args_list) == (
            {
                "immediate": [call(room, "$opening"), call(room, "$status")],
                "after_processing": [call(room, "$status")],
                "disabled": [],
            }[mode],
            [call(room, "$opening", "👀")],
        )
        release.set()
        await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 2)
        assert (receipts.call_args_list, adapter._send_reaction.await_args_list) == (
            {
                "immediate": [call(room, "$opening"), call(room, "$status")],
                "after_processing": [call(room, "$status"), call(room, "$opening")],
                "disabled": [],
            }[mode],
            [call(room, "$opening", "👀"), call(room, "$opening", "✅")],
        )
    finally:
        release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["immediate", "after_processing", "disabled"])
@pytest.mark.parametrize(
    "route",
    [
        "redirect",
        "steer",
        "claimed-voice",
        "priority-redirect",
        "priority-steer",
        "slash-steer",
    ],
)
@pytest.mark.parametrize("outcome", list(ProcessingOutcome))
@pytest.mark.parametrize("rewritten", [False, True])
async def test_injected_turn_receipts_cover_the_latest_accepted_input(
    monkeypatch, mode, route, outcome, rewritten
):
    adapter, receipts, sender = _intake_adapter(monkeypatch, mode)
    adapter._text_batch_delay_seconds = 0
    runner = _busy_runner(
        monkeypatch, adapter, "interrupt" if "redirect" in route else "steer"
    )
    receiver = MagicMock(_supports_active_turn_redirect=True)
    receiver._active_children = []
    receiver.redirect.return_value = receiver.steer.return_value = True
    runner._transcribe_and_echo_pending_voice = AsyncMock(
        return_value=("voice transcript", ["voice transcript"])
    )
    started, release = asyncio.Event(), asyncio.Event()
    opening = []

    async def rewrite_hook(*args, **kwargs):
        return [{"action": "rewrite", "text": "rewritten opening"}]

    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", rewrite_hook)

    async def respond(event):
        if rewritten:
            event = await runner._hm_pre_gateway_dispatch_hook(event, event.source)
            assert event is not None
        opening.append(event)
        turn = runner._session_state(adapter._event_session_key(event)).turn
        turn.agent, turn.event, turn.processing_event = receiver, event, event
        turn.ctx = TurnContext(
            event_message_id=event.message_id, inbound_message_id=event.message_id
        )
        started.set()
        await release.wait()
        return "done"

    adapter.set_message_handler(respond)
    incoming = []

    async def inject(event, key):
        incoming.append(event)
        if route == "priority-redirect":
            await runner._hm_busy_interrupt(event, event.source, receiver, key)
            return True
        if route == "priority-steer":
            runner._hm_busy_steer(event, receiver, key)
            return True
        if route == "slash-steer":
            event.text = "/steer " + event.text
            await runner._busy_steer_command(event, key, event.source)
            return True
        return await runner._handle_active_session_busy_message(event, key)

    adapter.set_busy_session_handler(inject)
    room = "!room:example.org"
    try:
        await _text_input(adapter, "$opening", "opening")
        await asyncio.wait_for(started.wait(), 2)
        if route == "claimed-voice":
            await adapter._handle_media_message(
                room,
                "@alice:example.org",
                "$voice",
                0,
                {
                    "body": "voice.ogg",
                    "msgtype": "m.audio",
                    "url": "mxc://example.org/voice",
                    "org.matrix.msc3245.voice": {},
                },
                {},
                "m.audio",
            )
            await _text_input(adapter, "$correction", "@hermes:example.org")
        else:
            await _text_input(adapter, "$correction", "correction")
        assert [event.message_id for event in incoming] == (
            ["$voice"] if route == "claimed-voice" else ["$correction"]
        )
        assert adapter._pending_messages == {}
        verb = receiver.redirect if "redirect" in route else receiver.steer
        verb.assert_called_once()
        arrival_ids = (
            ["$opening", "$voice", "$correction"]
            if route == "claimed-voice"
            else ["$opening", "$correction"]
        )
        assert receipts.call_args_list == (
            [call(room, event_id) for event_id in arrival_ids]
            if mode == "immediate"
            else []
        )
        if outcome == ProcessingOutcome.CANCELLED:
            await adapter.cancel_session_processing(
                adapter._event_session_key(opening[0])
            )
        else:
            if outcome == ProcessingOutcome.FAILURE:
                sender.return_value = SendResult(success=False, error="refused")
            release.set()
            await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 2)
        expected_ids = {
            "immediate": arrival_ids,
            "after_processing": []
            if outcome == ProcessingOutcome.CANCELLED
            else ["$correction"],
            "disabled": [],
        }[mode]
        assert receipts.call_args_list == [
            call(room, event_id) for event_id in expected_ids
        ]
        assert [event.message_id for event in opening] == ["$opening"]
    finally:
        release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["immediate", "after_processing", "disabled"])
@pytest.mark.parametrize("command", ["queue", "steer", "steer-fallback", "late-steer"])
@pytest.mark.parametrize("completion", ["success", "task-cancel", "cooperative"])
@pytest.mark.parametrize("delayed_ack", [False, True])
async def test_inline_agent_work_is_acknowledged_only_after_its_processing(
    monkeypatch, mode, command, completion, delayed_ack
):
    adapter, receipts, _sender = _intake_adapter(monkeypatch, mode)
    adapter._text_batch_delay_seconds = 0
    runner = _busy_runner(monkeypatch, adapter, "queue")
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        side_effect=lambda **kwargs: kwargs["event"].text
    )
    runner._refresh_agent_cache_message_count = AsyncMock()
    receiver = MagicMock()
    receiver.steer.return_value = command != "steer-fallback"
    receiver._active_children = []
    started, release, followup_started = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    followup_release = asyncio.Event()
    ack_started, ack_release = asyncio.Event(), asyncio.Event()
    opening = []
    work_task = None

    async def send(chat_id, content, **kwargs):
        if delayed_ack and "queued" in content.lower():
            ack_started.set()
            await ack_release.wait()
        return SendResult(success=True, message_id="$reply")

    _sender.side_effect = send

    async def followup(**kwargs):
        followup_started.set()
        await followup_release.wait()
        return {
            "final_response": "followup done",
            "messages": [],
            "interrupted": completion == "cooperative",
        }

    monkeypatch.setattr(runner, "_run_agent", followup)

    async def respond(event):
        if event.is_command():
            assert runner._session_key_for_source(
                event.source
            ) == adapter._event_session_key(event)
            reply = await runner._handle_message(event)
            assert "queued" in str(reply).lower(), reply
            return reply
        opening.append(event)
        key = adapter._event_session_key(event)
        turn = runner._session_state(key).turn
        turn.agent, turn.event, turn.processing_event = receiver, event, event
        if command == "steer-fallback":
            turn.agent = _AGENT_PENDING_SENTINEL
        ctx = TurnContext(
            source=event.source,
            session_key=key,
            session_id="command-receipt",
            processing_event=event,
            event_message_id=event.message_id,
            inbound_message_id=event.message_id,
        )
        turn.ctx = ctx
        started.set()
        await release.wait()
        if command == "steer":
            if completion == "cooperative":
                event._processing_state.outcome = ProcessingOutcome.CANCELLED
            return "done"
        result = {"final_response": "opening done", "messages": []}
        if command == "late-steer":
            result["pending_steer"] = receiver.steer.call_args.args[0]
        pending, payload = await runner._run_agent_drain_pending(
            result, adapter, event.source, key, processing_event=event
        )
        if pending is None and payload is None:
            return "done"
        result = await runner._run_agent_queued_followup(
            ctx, adapter, payload, pending, result, result, None
        )
        return result["final_response"]

    adapter.set_message_handler(respond)
    room = "!room:example.org"
    try:
        await _text_input(adapter, "$opening", "opening")
        await asyncio.wait_for(started.wait(), 2)
        work_task = asyncio.create_task(
            _text_input(
                adapter,
                "$work",
                "/" + ("queue" if command == "queue" else "steer") + " work",
            )
        )
        if delayed_ack:
            await asyncio.wait_for(ack_started.wait(), 2)
        else:
            await asyncio.wait_for(work_task, 2)
        arrivals = [call(room, "$opening"), call(room, "$work")]
        assert receipts.call_args_list == (arrivals if mode == "immediate" else [])
        if command != "steer":
            release.set()
            await asyncio.wait_for(followup_started.wait(), 2)
            assert (
                receipts.call_args_list
                == {
                    "immediate": arrivals,
                    "after_processing": [call(room, "$opening")],
                    "disabled": [],
                }[mode]
            )
        ack_release.set()
        await asyncio.wait_for(work_task, 2)
        assert (
            receipts.call_args_list
            == {
                "immediate": arrivals,
                "after_processing": [call(room, "$opening")]
                if command != "steer"
                else [],
                "disabled": [],
            }[mode]
        )
        if completion == "task-cancel":
            await adapter.cancel_session_processing(
                adapter._event_session_key(opening[0])
            )
        else:
            release.set()
            followup_release.set()
            await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 2)
        completed_ids = ["$opening"] if command != "steer" else []
        if completion == "success":
            completed_ids.append("$work")
        assert (
            receipts.call_args_list
            == {
                "immediate": arrivals,
                "after_processing": [
                    call(room, event_id) for event_id in completed_ids
                ],
                "disabled": [],
            }[mode]
        )
    finally:
        release.set()
        followup_release.set()
        ack_release.set()
        await adapter.cancel_background_tasks()
        if work_task is not None:
            await asyncio.wait_for(work_task, 2)


@pytest.mark.asyncio
async def test_late_steer_preserves_the_completed_batch_receipt(monkeypatch):
    adapter, receipts, _sender = _intake_adapter(monkeypatch, "after_processing")
    source = SessionSource(
        platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="dm"
    )
    opening = MessageEvent(text="first", message_id="$first", source=source)
    batched = MessageEvent(text="second", message_id="$second", source=source)
    steer = MessageEvent(text="steering", message_id="$steer", source=source)
    opening.absorb_turn_input(batched)
    await adapter._run_processing_hook("on_processing_start", opening)
    opening.absorb_turn_input(steer, input_text=steer.text)
    pending = opening._processing_state.take_pending_input(steer.text)
    await adapter._run_processing_hook(
        "on_processing_complete", opening, ProcessingOutcome.SUCCESS
    )
    assert (pending, opening.message_id, receipts.call_args_list) == (
        steer,
        "$first",
        [call(source.chat_id, batched.message_id)],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route", ["redirect", "steer", "priority-redirect", "priority-steer"]
)
async def test_recursive_turn_corrections_do_not_update_the_completed_ancestor(
    monkeypatch, route
):
    adapter, receipts, _sender = _intake_adapter(monkeypatch, "after_processing")
    adapter._text_batch_delay_seconds = 0
    runner = _busy_runner(
        monkeypatch, adapter, "interrupt" if "redirect" in route else "steer"
    )
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        side_effect=lambda **kwargs: kwargs["event"].text
    )
    runner._refresh_agent_cache_message_count = AsyncMock()
    receiver = MagicMock(_supports_active_turn_redirect=True)
    receiver.redirect.return_value = receiver.steer.return_value = True
    receiver._active_children = []
    started, release = asyncio.Event(), asyncio.Event()
    events = [
        MessageEvent(
            text=text,
            message_id="$" + text,
            source=SessionSource(
                platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="dm"
            ),
        )
        for text in ("opening", "queued", "correction")
    ]
    key = adapter._event_session_key(events[0])

    async def track(event):
        ctx = TurnContext(
            source=event.source,
            session_key=key,
            session_id="recursive-receipt",
            processing_event=event,
            event_message_id=event.message_id,
            inbound_message_id=event.message_id,
            agent_holder=[receiver],
        )
        await runner._run_agent_track_agent(ctx)
        return ctx

    async def followup(**kwargs):
        await track(kwargs["processing_event"])
        started.set()
        await release.wait()
        return {"final_response": "queued done", "messages": []}

    monkeypatch.setattr(runner, "_run_agent", followup)

    async def respond(event):
        runner._session_state(key).turn.event = event
        ctx = await track(event)
        result = {"final_response": "opening done", "messages": []}
        return (
            await runner._run_agent_queued_followup(
                ctx, adapter, events[1].text, events[1], result, result, None
            )
        )["final_response"]

    adapter.set_message_handler(respond)
    try:
        await adapter.handle_message(events[0])
        await asyncio.wait_for(started.wait(), 2)
        assert receipts.call_args_list == [
            call(events[0].source.chat_id, events[0].message_id)
        ]
        if route == "priority-redirect":
            await runner._hm_busy_interrupt(events[2], events[2].source, receiver, key)
        elif route == "priority-steer":
            runner._hm_busy_steer(events[2], receiver, key)
        else:
            await runner._handle_active_session_busy_message(events[2], key)
        release.set()
        await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 2)
        assert receipts.call_args_list == [
            call(event.source.chat_id, event.message_id)
            for event in (events[0], events[2])
        ]
        assert [
            (event.message_id, event._processing_state.receipt_message_id)
            for event in events[:2]
        ] == [("$opening", None), ("$queued", "$correction")]
    finally:
        release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.parametrize("section", ["matrix", "platforms", "gateway"])
@pytest.mark.parametrize(
    "env_override", [False, True, None], ids=["yaml", "override", "env-only"]
)
def test_feedback_config_resolves_for_each_served_profile(
    tmp_path, monkeypatch, section, env_override
):
    homes = [tmp_path / "a", tmp_path / "b"]
    settings = [
        {"read_receipts": "disabled", "reactions": False},
        {"read_receipts": "after_processing", "reactions": True},
    ]
    for home, feedback in zip(homes, settings):
        home.mkdir()
        matrix = {"enabled": True, **feedback}
        if env_override is None:
            matrix.pop("reactions")
        config = {"matrix": matrix}
        if section == "platforms":
            config = {"platforms": config}
        if section == "gateway":
            config = {"gateway": {"platforms": config}}
        atomic_config_write(home / "config.yaml", config)
        reactions = "true" if home == homes[0] else "false"
        (home / ".env").write_text(
            f"MATRIX_REACTIONS={reactions}\n" if env_override is not False else "",
            encoding="utf-8",
        )

    monkeypatch.setenv("MATRIX_REACTIONS", "false")
    monkeypatch.setenv("MATRIX_READ_RECEIPTS", "immediate")
    original_env = {
        key: os.environ.get(key) for key in ("MATRIX_REACTIONS", "MATRIX_READ_RECEIPTS")
    }
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        resolved = []
        for home in (homes[0], homes[1], homes[0]):
            with _profile_runtime_scope(home, hydrate_secrets=False):
                config = load_gateway_config().platforms[Platform.MATRIX]
                adapter = MatrixAdapter(config)
                resolved.append((
                    adapter._read_receipts_mode,
                    adapter._reactions_enabled,
                ))
        expected_reactions = (False, True) if env_override is False else (True, False)
        assert resolved == [
            ("disabled", expected_reactions[0]),
            ("after_processing", expected_reactions[1]),
            ("disabled", expected_reactions[0]),
        ]
        assert {key: os.environ.get(key) for key in original_env} == original_env
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)


def test_receipt_default_does_not_read_an_unreleased_env_setting(monkeypatch):
    monkeypatch.setenv("MATRIX_READ_RECEIPTS", "disabled")
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    adapter = MatrixAdapter(PlatformConfig(enabled=True))
    assert (adapter._read_receipts_mode, adapter._reactions_enabled) == (
        "immediate",
        True,
    )


@pytest.mark.parametrize(
    ("value", "mode"),
    [
        ("  AFTER_PROCESSING ", "after_processing"),
        ("DISABLED", "disabled"),
        (None, "immediate"),
        ("sometimes", "immediate"),
    ],
)
def test_receipt_configuration_normalises_modes(monkeypatch, value, mode):
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    adapter = MatrixAdapter(
        PlatformConfig(enabled=True, extra={"read_receipts": value})
    )
    assert (adapter._read_receipts_mode, adapter._reactions_enabled) == (mode, True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["immediate", "after_processing", "disabled"])
async def test_claimed_voice_receipts_include_the_triggering_mention(monkeypatch, mode):
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "read_receipts": mode,
                "reactions": False,
                "auto_thread": False,
                "user_id": "@hermes:example.org",
            },
        )
    )
    room_id, sender = "!room:example.org", "@alice:example.org"
    adapter._dm_rooms[room_id] = False
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter.handle_message = AsyncMock()
    adapter._background_read_receipt = MagicMock()
    voice = {"msgtype": "m.audio", "body": "voice", "org.matrix.msc3245.voice": {}}
    await adapter._handle_media_message(
        room_id, sender, "$voice", 0, voice, {}, "m.audio"
    )
    adapter.handle_message.assert_not_awaited()
    await adapter._handle_text_message(
        room_id,
        sender,
        "$mention",
        0,
        {
            "body": "@hermes:example.org",
            "m.mentions": {"user_ids": ["@hermes:example.org"]},
        },
        {},
    )
    (invocation,) = adapter.handle_message.await_args_list
    event = invocation.args[0]
    assert event.message_id == "$voice"
    arrival = list(adapter._background_read_receipt.call_args_list)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert (arrival, adapter._background_read_receipt.call_args_list) == {
        "immediate": (
            [call(room_id, "$voice"), call(room_id, "$mention")],
            [call(room_id, "$voice"), call(room_id, "$mention")],
        ),
        "after_processing": ([], [call(room_id, "$mention")]),
        "disabled": ([], []),
    }[mode]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["immediate", "after_processing", "disabled"])
@pytest.mark.parametrize("outcome", list(ProcessingOutcome))
async def test_receipt_timing_follows_processing_outcome(monkeypatch, mode, outcome):
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    adapter = MatrixAdapter(
        PlatformConfig(enabled=True, extra={"read_receipts": mode, "reactions": False, "user_id": "@hermes:example.org"})
    )
    adapter._text_batch_delay_seconds = 0
    adapter._dm_rooms["!room:example.org"] = True
    adapter._read_room_member_profiles = AsyncMock(
        return_value=({adapter._user_id, "@alice:example.org"}, {})
    )
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter.handle_message = AsyncMock()
    adapter._background_read_receipt = MagicMock()
    await adapter._handle_text_message(
        "!room:example.org",
        "@alice:example.org",
        "$message",
        0,
        {"body": "hello"},
        {},
    )
    (invocation,) = adapter.handle_message.await_args_list
    event = invocation.args[0]
    arrival = list(adapter._background_read_receipt.call_args_list)
    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, outcome)
    receipt = [call(event.source.chat_id, event.message_id)]
    assert (arrival, adapter._background_read_receipt.call_args_list) == (
        receipt if mode == "immediate" else [],
        receipt
        if mode == "immediate"
        or (mode == "after_processing" and outcome != ProcessingOutcome.CANCELLED)
        else [],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("outcome", list(ProcessingOutcome))
async def test_lifecycle_toggle_preserves_completion_and_cancellation(
    monkeypatch, enabled, outcome
):
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource

    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True, extra={"read_receipts": "disabled", "reactions": enabled}
        )
    )
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    adapter._schedule_reaction_redaction = MagicMock()
    event = MessageEvent(
        text="hello",
        message_id="$message",
        source=SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:example.org",
            chat_type="dm",
        ),
    )
    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, outcome)
    expected_reactions = []
    expected_redactions = []
    expected_pending = {}
    if enabled:
        expected_reactions.append(call(event.source.chat_id, event.message_id, "👀"))
        if outcome == ProcessingOutcome.CANCELLED:
            expected_pending[(event.source.chat_id, event.message_id)] = "$eyes"
        else:
            emoji = "✅" if outcome == ProcessingOutcome.SUCCESS else "❌"
            expected_reactions.append(
                call(event.source.chat_id, event.message_id, emoji)
            )
            expected_redactions.append(
                call(event.source.chat_id, "$eyes", "processing complete")
            )
    assert (
        adapter._send_reaction.await_args_list,
        adapter._schedule_reaction_redaction.call_args_list,
        adapter._pending_reactions,
    ) == (expected_reactions, expected_redactions, expected_pending)


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["approval", "model", "choice"])
async def test_lifecycle_disabled_keeps_reaction_controls(monkeypatch, control):
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={"reactions": False}))
    adapter._client = MagicMock()
    adapter.send = AsyncMock(
        return_value=SendResult(success=True, message_id="$prompt")
    )
    adapter._send_reaction = AsyncMock(return_value="$control")
    room_id = "!room:example.org"
    if control == "approval":
        await adapter._send_exec_approval_prompt(
            ExecApprovalPrompt(
                session_key="session",
                chat_id=room_id,
                text="Approve command",
                actions=[("Approve", "once", "primary"), ("Deny", "deny", "danger")],
                command="echo hello",
                description="Print hello",
                smart_denied=False,
                request_id="fixture-approval",
            )
        )
        choices = ["✅", "❌"]
    elif control == "model":
        await adapter.send_model_picker(
            room_id,
            [{"slug": "custom", "models": ["test-model"]}],
            "test-model",
            "custom",
            "session",
            AsyncMock(),
        )
        choices = ["1️⃣"]
    else:
        await adapter.send_choice_picker(
            room_id,
            "Choose",
            [{"value": "fast", "label": "Fast"}],
            "session",
            AsyncMock(),
        )
        choices = ["1️⃣"]
    assert adapter._send_reaction.await_args_list == [
        call(room_id, "$prompt", emoji) for emoji in choices
    ]


class _SteeringReceiver(InterruptControlMixin):
    def __init__(self) -> None:
        self._pending_steer = None
        self._pending_steer_lock = Lock()
        self._active_children = []


def _receipt_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="$" + text,
        source=SessionSource(
            platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="dm"
        ),
    )


def _receipt_turn(
    runner: GatewayRunner,
    adapter: MatrixAdapter,
    event: MessageEvent,
    receiver: _SteeringReceiver,
) -> TurnContext:
    key = adapter._event_session_key(event)
    ctx = TurnContext(
        source=event.source,
        session_key=key,
        session_id="steering-receipts",
        event_message_id=event.message_id,
        inbound_message_id=event.message_id,
        processing_event=event,
    )
    turn = runner._session_state(key).turn
    turn.agent, turn.event, turn.processing_event, turn.ctx = (
        receiver,
        event,
        event,
        ctx,
    )
    return ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_count", [0, 1, 2])
@pytest.mark.parametrize("interrupted", [False, True])
async def test_fifo_does_not_acknowledge_or_discard_a_late_steer(
    monkeypatch, queue_count, interrupted
):
    adapter, receipts, _sender = _intake_adapter(monkeypatch, "after_processing")
    adapter._reactions_enabled = True
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    adapter._schedule_reaction_redaction = MagicMock()
    runner = _busy_runner(monkeypatch, adapter, "steer")
    receiver = _SteeringReceiver()
    opening, late = _receipt_event("opening"), _receipt_event("late")
    queued = [_receipt_event(f"queued-{i}") for i in range(queue_count)]
    ctx = _receipt_turn(runner, adapter, opening, receiver)
    key = ctx.session_key
    assert key is not None
    await adapter._run_processing_hook("on_processing_start", opening)
    for event in queued:
        runner._enqueue_fifo(key, event, adapter)
    await runner._handle_active_session_busy_message(late, key)
    result = {
        "final_response": "opening done",
        "messages": [],
        "pending_steer": receiver._drain_pending_steer(),
        "interrupted": interrupted,
        "interrupt_message": "interruption",
    }
    selected, text = await runner._run_agent_drain_pending(
        result, adapter, opening.source, key, processing_event=opening
    )
    await adapter._run_processing_hook(
        "on_processing_complete",
        opening,
        ProcessingOutcome.CANCELLED if interrupted else ProcessingOutcome.SUCCESS,
    )
    selected_expected = (
        queued[0]
        if queued
        else (None if interrupted else replace(late, text=result["pending_steer"]))
    )
    opening_receipts = (
        [] if interrupted else [call(opening.source.chat_id, opening.message_id)]
    )
    assert (selected, text, receipts.call_args_list) == (
        selected_expected,
        (queued[0].text if queued else result["pending_steer"])
        if selected_expected is not None
        else "interruption",
        opening_receipts,
    )
    if selected is None:
        selected, text = await runner._run_agent_drain_pending(
            {"final_response": "interruption done"}, adapter, opening.source, key
        )
    processed = []
    while selected is not None:
        processed.append((selected.message_id, selected.receipt_message_id))
        await adapter._run_processing_hook("on_processing_start", selected)
        await adapter._run_processing_hook(
            "on_processing_complete", selected, ProcessingOutcome.SUCCESS
        )
        selected, text = await runner._run_agent_drain_pending(
            {"final_response": "done"},
            adapter,
            opening.source,
            key,
            processing_event=selected,
        )
    assert (
        processed,
        receipts.call_args_list,
        adapter._send_reaction.await_args_list,
    ) == (
        [(event.message_id, event.message_id) for event in [*queued, late]],
        opening_receipts
        + [call(event.source.chat_id, event.message_id) for event in [*queued, late]],
        [
            call(opening.source.chat_id, opening.message_id, emoji)
            for emoji in (("👀",) if interrupted else ("👀", "✅"))
        ]
        + [
            call(event.source.chat_id, event.message_id, emoji)
            for event in [*queued, late]
            for emoji in ("👀", "✅")
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["success", "task-cancel", "cooperative"])
@pytest.mark.parametrize("rewritten", [False, True])
@pytest.mark.parametrize("echo_consumed", [False, True])
async def test_consumed_correction_receipt_precedes_a_pending_correction(
    monkeypatch, completion, rewritten, echo_consumed
):
    from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results

    adapter, receipts, _sender = _intake_adapter(monkeypatch, "after_processing")
    adapter._text_batch_delay_seconds = 0
    adapter._reactions_enabled = True
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    adapter._schedule_reaction_redaction = MagicMock()
    runner = _busy_runner(monkeypatch, adapter, "steer")
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        side_effect=lambda **kwargs: kwargs["event"].text
    )
    runner._refresh_agent_cache_message_count = AsyncMock()
    receiver = _SteeringReceiver()
    opening, consumed, pending = [
        _receipt_event(text) for text in ("opening", "consumed", "pending")
    ]
    started, release, followup_started, followup_release = [
        asyncio.Event() for _ in range(4)
    ]
    model_messages = [{"role": "tool", "content": "tool finished"}]
    tracked = []

    async def followup(**kwargs):
        tracked.append(kwargs["processing_event"])
        followup_started.set()
        await followup_release.wait()
        return {
            "final_response": "pending done",
            "messages": [],
            "interrupted": completion == "cooperative",
        }

    monkeypatch.setattr(runner, "_run_agent", followup)

    async def respond(event):
        current = replace(event) if rewritten else event
        ctx = _receipt_turn(runner, adapter, current, receiver)
        started.set()
        await release.wait()
        result = {
            "final_response": "consumed done",
            "messages": model_messages,
            "pending_steer": receiver._drain_pending_steer(),
        }
        selected, text = await runner._run_agent_drain_pending(
            result, adapter, current.source, ctx.session_key, processing_event=current
        )
        return (
            await runner._run_agent_queued_followup(
                ctx, adapter, text, selected, result, result, None
            )
        )["final_response"]

    adapter.set_message_handler(respond)
    try:
        await adapter.handle_message(opening)
        await asyncio.wait_for(started.wait(), 2)
        key = adapter._event_session_key(opening)
        await runner._handle_active_session_busy_message(consumed, key)
        consumed_text = receiver._pending_steer
        assert consumed_text is not None
        apply_pending_steer_to_tool_results(receiver, model_messages, 1)
        if echo_consumed:
            pending.text += "\n" + consumed_text
        assert model_messages[-1]["display_kind"] == "steer"
        await runner._handle_active_session_busy_message(pending, key)
        release.set()
        await asyncio.wait_for(followup_started.wait(), 2)
        assert (
            receipts.call_args_list,
            adapter._send_reaction.await_args_list,
            [(event.message_id, event.receipt_message_id) for event in tracked],
            (
                opening.message_id,
                opening.reply_anchor_override,
                opening.ledger_message_id,
            ),
        ) == (
            [call(consumed.source.chat_id, consumed.message_id)],
            [
                call(opening.source.chat_id, opening.message_id, emoji)
                for emoji in ("👀", "✅")
            ]
            + [call(pending.source.chat_id, pending.message_id, "👀")],
            [(pending.message_id, pending.message_id)],
            ("$opening", None, None),
        )
        if completion == "task-cancel":
            await adapter.cancel_session_processing(key)
        else:
            followup_release.set()
            await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 2)
        assert receipts.call_args_list == [
            call(event.source.chat_id, event.message_id)
            for event in (
                [consumed, pending] if completion == "success" else [consumed]
            )
        ]
    finally:
        release.set()
        followup_release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["yes", "no"])
@pytest.mark.parametrize("mode", ["immediate", "after_processing", "disabled"])
@pytest.mark.parametrize("confirmation", ["success", "refused", "error", "cancelled"])
async def test_plaintext_approval_receipt_is_independent_of_the_active_turn(
    monkeypatch, answer, mode, confirmation
):
    from tools.approval import register_gateway_settle, unregister_gateway_notify
    from tools.approval_gateway_wait import _await_gateway_decision

    adapter, receipts, _sender = _intake_adapter(monkeypatch, mode)
    adapter._text_batch_delay_seconds = 0
    adapter._reactions_enabled = True
    adapter._send_reaction = AsyncMock(return_value="$eyes")
    adapter._schedule_reaction_redaction = MagicMock()
    adapter._dm_rooms["!room:example.org"] = True
    adapter._read_room_member_profiles = AsyncMock(
        return_value=({adapter._user_id, "@alice:example.org"}, {})
    )
    runner = _busy_runner(monkeypatch, adapter, "steer")
    opening = _receipt_event("opening")
    key = adapter._event_session_key(opening)
    started, release = asyncio.Event(), asyncio.Event()
    notified = asyncio.Event()
    settled = []
    loop = asyncio.get_running_loop()
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 5)

    def notify(data):
        assert register_gateway_settle(key, data["request_id"], settled.append)
        loop.call_soon_threadsafe(notified.set)

    async def respond(event):
        _receipt_turn(runner, adapter, event, _SteeringReceiver())
        started.set()
        await release.wait()
        return "done"

    adapter.set_message_handler(respond)
    waiter = None
    control_task = None
    delivery_started = asyncio.Event()
    delivery_release = asyncio.Event()

    async def confirmation_send(**kwargs):
        if confirmation == "error":
            raise RuntimeError("Approval confirmation failed")
        if confirmation == "cancelled":
            delivery_started.set()
            await delivery_release.wait()
        return SendResult(
            success=confirmation != "refused",
            message_id="$confirmation",
            error="refused",
        )

    try:
        assert opening.message_id is not None
        await _text_input(adapter, opening.message_id, opening.text)
        await asyncio.wait_for(started.wait(), 2)
        waiter = asyncio.create_task(
            asyncio.to_thread(
                _await_gateway_decision,
                key,
                notify,
                {"command": "receipt-test", "description": "test"},
            )
        )
        await asyncio.wait_for(notified.wait(), 2)
        _sender.side_effect = confirmation_send
        control_task = asyncio.create_task(_text_input(adapter, "$approval", answer))
        if confirmation == "cancelled":
            await asyncio.wait_for(delivery_started.wait(), 2)
            control_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await control_task
        else:
            await asyncio.wait_for(control_task, 2)
        _sender.side_effect = None
        decision = await asyncio.wait_for(waiter, 2)
        assert (
            decision,
            settled,
            receipts.call_args_list,
            adapter._send_reaction.await_args_list,
        ) == (
            {
                "resolved": True,
                "choice": "once" if answer == "yes" else "deny",
                "reason": None,
            },
            ["resolved"],
            {
                "immediate": [
                    call(opening.source.chat_id, "$opening"),
                    call(opening.source.chat_id, "$approval"),
                ],
                "after_processing": []
                if confirmation == "cancelled"
                else [call(opening.source.chat_id, "$approval")],
                "disabled": [],
            }[mode],
            [call(opening.source.chat_id, "$opening", "👀")],
        )
        release.set()
        await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 2)
        assert (
            receipts.call_args_list
            == {
                "immediate": [
                    call(opening.source.chat_id, "$opening"),
                    call(opening.source.chat_id, "$approval"),
                ],
                "after_processing": (
                    [call(opening.source.chat_id, "$approval")]
                    if confirmation != "cancelled"
                    else []
                )
                + [call(opening.source.chat_id, "$opening")],
                "disabled": [],
            }[mode]
        )
        assert (adapter._send_reaction.await_args_list, adapter._pending_messages) == (
            [call(opening.source.chat_id, "$opening", emoji) for emoji in ("👀", "✅")],
            {},
        )
    finally:
        unregister_gateway_notify(key)
        delivery_release.set()
        if control_task is not None and not control_task.done():
            await asyncio.wait_for(control_task, 2)
        release.set()
        await adapter.cancel_background_tasks()
        if waiter is not None:
            await asyncio.wait_for(waiter, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "rewritten", "completion"),
    [
        ("fifo", False, "success"),
        ("fifo", True, "success"),
        ("fifo", True, "failure"),
        ("fifo", True, "cancelled"),
        ("yes", True, "success"),
        ("no", True, "success"),
    ],
)
async def test_recursive_and_inline_completion_uses_the_admitted_event(
    monkeypatch, tmp_path, route, rewritten, completion
):
    import json
    from threading import Event
    from openai.types.chat import ChatCompletion
    from run_agent import AIAgent
    from tools.approval import unregister_gateway_notify

    adapter, receipts, _sender = _intake_adapter(monkeypatch, "after_processing")
    adapter._text_batch_delay_seconds = 0
    adapter._dm_rooms["!room:example.org"] = True
    adapter._reactions_enabled = True
    adapter._schedule_reaction_redaction = MagicMock()
    transport = MagicMock()
    sent_events = []
    final_started, final_release = asyncio.Event(), asyncio.Event()
    final_text = "done-3" if route == "fifo" else "done-2"
    delivered = completion == "success"

    async def send_event(_room, event_type, content):
        if str(event_type) == "m.room.message" and content.get("body", "").endswith(
            final_text
        ):
            final_started.set()
            await final_release.wait()
            if not delivered:
                raise RuntimeError("Final delivery refused")
        sent_events.append((str(event_type), content))
        if "needs your OK" in content.get("body", ""):
            notified.set()
        return f"$sent-{len(sent_events)}"

    transport.send_message_event = AsyncMock(side_effect=send_event)
    adapter._client = transport
    adapter.send = MatrixAdapter.send.__get__(adapter)
    runner = _busy_runner(monkeypatch, adapter, "steer")
    gates = [(Event(), Event()) for _ in range(3)]
    calls = []
    notified = asyncio.Event()
    background_timer = asyncio.Event()

    def model_response(_agent, request, **_kwargs):
        index = len(calls)
        calls.append(request)
        gates[index][0].set()
        assert gates[index][1].wait(2), "Model response was not released"
        if _agent.stream_delta_callback:
            _agent.stream_delta_callback(f"done-{index + 1}")
        message = {"role": "assistant", "content": f"done-{index + 1}"}
        finish_reason = "stop"
        if route != "fifo" and index == 0:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "approval-command",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": json.dumps({
                                "command": "rm -rf /tmp/hermes-receipt-approval-target",
                                "timeout": 10,
                            }),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        return ChatCompletion.model_validate({
            "id": f"response-{index}",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
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
    monkeypatch.setattr(
        runner,
        "_prepare_profile_scoped_inbound_message_text",
        AsyncMock(side_effect=lambda **kwargs: kwargs["event"].text),
    )
    opening = _receipt_event("opening")
    key = adapter._event_session_key(opening)

    if rewritten:

        async def copy_inbound(event, _source):
            return replace(event)

        monkeypatch.setattr(runner, "_hm_pre_gateway_dispatch_hook", copy_inbound)

    adapter.set_message_handler(runner._handle_message)
    monkeypatch.setattr(AIAgent, "_interruptible_api_call", model_response)
    monkeypatch.setattr("tools.env_probe.get_environment_probe_line", lambda: None)
    monkeypatch.setattr(
        "tools.terminal_tool._pre_exec_block", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "tools.terminal_tool._run_foreground",
        lambda *args, **kwargs: json.dumps({"output": "approved", "exit_code": 0}),
    )
    (tmp_path / "config.yaml").write_text(
        "model: {provider: custom, default: fake-model, context_length: 128000}\nauxiliary: {title_generation: {enabled: false}}\napprovals: {mode: manual, timeout: 15}\n"
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
    agent._disable_streaming = True

    def fresh_agent(turn_runner, *_args):
        agent.session_id = turn_runner._ctx.session_id
        return agent

    monkeypatch.setattr(
        "gateway.run_turn_runner.TurnRunner._build_fresh_agent", fresh_agent
    )
    try:
        await _text_input(adapter, "$opening", "opening")
        assert await asyncio.to_thread(gates[0][0].wait, 2)
        if route == "fifo":
            queued, late = _receipt_event("queued"), _receipt_event("late")
            await _text_input(adapter, "$queued", "/queue queued")
            await _text_input(adapter, "$late", "/steer late")
            gates[0][1].set()
            assert await asyncio.to_thread(gates[1][0].wait, 2)
            gates[1][1].set()
            assert await asyncio.to_thread(gates[2][0].wait, 2)
            gates[2][1].set()
            expected = [opening, queued, late]
        else:
            gates[0][1].set()
            await asyncio.wait_for(notified.wait(), 2)
            control = _receipt_event(route)
            await _text_input(adapter, control.message_id, route)
            assert await asyncio.to_thread(gates[1][0].wait, 2)
            gates[1][1].set()
            expected = [opening]
        await asyncio.wait_for(final_started.wait(), 2)
        target_ids = {event.message_id for event in expected}
        assert [
            content["m.relates_to"]
            for event_type, content in sent_events
            if event_type == "m.reaction"
            and content["m.relates_to"]["event_id"] in target_ids
        ] == [
            {"rel_type": "m.annotation", "event_id": event.message_id, "key": emoji}
            for event in expected
            for emoji in (("👀", "✅") if event is not expected[-1] else ("👀",))
        ]
        assert receipts.call_args_list == [
            call(opening.source.chat_id, message_id)
            for message_id in (
                ["$opening", "$queued"] if route == "fifo" else ["$" + route]
            )
        ]
        if completion == "cancelled":
            await adapter.cancel_session_processing(key)
        else:
            final_release.set()
            await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), 2)
        assert receipts.call_args_list == [
            call(opening.source.chat_id, message_id)
            for message_id in (
                (
                    ["$opening", "$queued"]
                    + (["$late"] if completion != "cancelled" else [])
                )
                if route == "fifo"
                else (
                    ["$" + route] + (["$opening"] if completion != "cancelled" else [])
                )
            )
        ]
        assert [
            content["m.relates_to"]
            for event_type, content in sent_events
            if event_type == "m.reaction"
            and content["m.relates_to"]["event_id"]
            in {event.message_id for event in expected}
        ] == [
            {"rel_type": "m.annotation", "event_id": event.message_id, "key": emoji}
            for event in expected
            for emoji in (
                ("👀",)
                if event is expected[-1] and completion == "cancelled"
                else ("👀", "✅" if event is not expected[-1] or delivered else "❌")
            )
        ]
        feedback_order = [
            ("reply", content["body"])
            if event_type == "m.room.message"
            else (
                "reaction",
                content["m.relates_to"]["event_id"],
                content["m.relates_to"]["key"],
            )
            for event_type, content in sent_events
            if (
                event_type == "m.room.message"
                and content.get("body", "").startswith("done-")
            )
            or (
                event_type == "m.reaction"
                and content["m.relates_to"]["event_id"]
                in {event.message_id for event in expected}
            )
        ]
        assert feedback_order == [
            item
            for index, event in enumerate(expected, 1)
            for item in [
                ("reaction", event.message_id, "👀"),
                *(
                    [("reply", f"done-{index}" if route == "fifo" else "done-2")]
                    if event is not expected[-1] or delivered
                    else []
                ),
                *(
                    []
                    if event is expected[-1] and completion == "cancelled"
                    else [
                        (
                            "reaction",
                            event.message_id,
                            "✅" if event is not expected[-1] or delivered else "❌",
                        )
                    ]
                ),
            ]
        ]
        assert len(calls) == (3 if route == "fifo" else 2)
    finally:
        final_release.set()
        for _started, release in gates:
            release.set()
        unregister_gateway_notify(key)
        await adapter.cancel_background_tasks()
        agent.close()
        runner.session_store.close_all_db_handles()
