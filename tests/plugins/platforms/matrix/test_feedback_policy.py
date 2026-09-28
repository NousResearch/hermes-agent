"""Matrix feedback settings belong to the receiving adapter's profile."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from agent import secret_scope
from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.platforms.base import ExecApprovalPrompt, SendResult
from gateway.turn_context import TurnContext
from hermes_cli.config import atomic_config_write
from plugins.platforms.matrix.adapter import MatrixAdapter


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
async def test_injected_turn_receipts_cover_the_latest_accepted_input(
    monkeypatch, mode, route, outcome
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

    async def respond(event):
        opening.append(event)
        turn = runner._session_state(adapter._event_session_key(event)).turn
        turn.agent, turn.event = receiver, event
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
