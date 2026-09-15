"""Reasoning display and auto-TTS use separate, provenance-bound payloads."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools.tts_text_normalize import prepare_spoken_text


def _event(platform: Platform = Platform.DISCORD) -> MessageEvent:
    return MessageEvent(
        text="question",
        source=SessionSource(platform=platform, chat_id="chat", user_id="user"),
    )


def _runner(adapter) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._adapter_for_source = MagicMock(return_value=adapter)
    runner._should_send_voice_reply = MagicMock(return_value=True)
    runner._send_voice_reply = AsyncMock()
    return runner


def test_direct_tts_does_not_infer_gateway_provenance_from_visible_text():
    text = "-# 💭 Reasoning\n-# quoted ordinary text\n\nFinal answer"
    spoken = prepare_spoken_text(text)

    assert "quoted ordinary text" in spoken
    assert "Final answer" in spoken


def test_discord_reasoning_renderer_returns_separate_spoken_payload():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._show_reasoning = True
    source = _event().source

    with (
        patch("gateway.run._load_gateway_config", return_value={}),
        patch("gateway.run._resolve_gateway_display_bool", return_value=True),
        patch(
            "gateway.display_config.resolve_display_setting",
            return_value="subtext",
        ),
    ):
        displayed, spoken = runner._hmwa_prepend_reasoning(
            {"last_reasoning": "hidden plan"}, "Final answer", source, False
        )

    assert "hidden plan" in displayed
    assert displayed.endswith("Final answer")
    assert spoken == "Final answer"


def test_failed_turn_notice_is_carried_into_reasoning_spoken_payload():
    runner = GatewayRunner.__new__(GatewayRunner)
    displayed = "-# 💭 Reasoning\n-# hidden plan\n\nFinal answer"
    notice = runner._PARTIAL_FAILED_TURN_NOTICE

    processed = runner._hmwa_add_failed_turn_notice(displayed, notice)
    spoken = runner._hmwa_carry_spoken_response_suffix(
        "Final answer", displayed, processed
    )

    assert spoken == f"Final answer\n\n{notice}"


def test_compression_reset_notice_is_carried_into_reasoning_spoken_payload():
    runner = GatewayRunner.__new__(GatewayRunner)
    displayed = "-# 💭 Reasoning\n-# hidden plan\n\nFinal answer"
    processed = displayed + (
        "\n\n🔄 Session auto-reset — the conversation exceeded the maximum context "
        "size and could not be compressed further."
    )

    spoken = runner._hmwa_carry_spoken_response_suffix(
        "Final answer", displayed, processed
    )

    assert spoken == (
        "Final answer\n\n🔄 Session auto-reset — the conversation exceeded the maximum "
        "context size and could not be compressed further."
    )


@pytest.mark.asyncio
async def test_whole_file_auto_tts_uses_explicit_final_answer_payload():
    adapter = MagicMock()
    adapter._streaming_tts_turn_completed.return_value = False
    runner = _runner(adapter)
    event = _event()

    displayed = "-# 💭 Reasoning\n-# hidden plan\n\nFinal answer"
    result = await runner._hmwa_deliver_turn_response(
        event,
        event.source,
        SimpleNamespace(session_id="session"),
        "key",
        1,
        {"already_sent": False},
        [],
        displayed,
        "",
        False,
        "Final answer",
    )

    assert result == displayed
    assert event._hermes_spoken_response == "Final answer"
    runner._send_voice_reply.assert_awaited_once_with(event, "Final answer")


@pytest.mark.asyncio
async def test_streaming_tts_success_does_not_send_whole_file_again():
    adapter = MagicMock()
    adapter._streaming_tts_turn_completed.return_value = True
    runner = _runner(adapter)
    event = _event()

    result = await runner._hmwa_deliver_turn_response(
        event,
        event.source,
        SimpleNamespace(session_id="session"),
        "key",
        1,
        {"already_sent": False},
        [],
        "visible reasoning and answer",
        "",
        False,
        "Final answer",
    )

    assert result == "visible reasoning and answer"
    runner._send_voice_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_override_keeps_non_reasoning_response_unchanged():
    adapter = MagicMock()
    adapter._streaming_tts_turn_completed.return_value = False
    runner = _runner(adapter)
    event = _event(Platform.TELEGRAM)

    await runner._hmwa_deliver_turn_response(
        event,
        event.source,
        SimpleNamespace(session_id="session"),
        "key",
        1,
        {"already_sent": False},
        [],
        "Ordinary answer",
        "",
        False,
    )

    assert not hasattr(event, "_hermes_spoken_response")
    runner._send_voice_reply.assert_awaited_once_with(event, "Ordinary answer")
