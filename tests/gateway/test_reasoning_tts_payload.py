"""Reasoning display and speech must use distinct payloads."""

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
    runner._delivery_adapter_for = MagicMock(return_value=adapter)
    runner._should_send_voice_reply = MagicMock(return_value=True)
    runner._send_voice_reply = AsyncMock()
    return runner


def test_direct_tts_does_not_infer_gateway_provenance_from_visible_text():
    text = "-# 💭 Reasoning\n-# quoted ordinary text\n\nFinal answer"

    spoken = prepare_spoken_text(text)

    assert "quoted ordinary text" in spoken
    assert "Final answer" in spoken


def test_discord_reasoning_renderer_keeps_speech_to_final_answer():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._show_reasoning = True
    source = _event().source

    with (
        patch("gateway.run._load_gateway_config", return_value={}),
        patch("gateway.run._resolve_gateway_display_bool", return_value=True),
        patch("gateway.display_config.resolve_display_setting", return_value="subtext"),
    ):
        displayed, spoken = runner._hmwa_prepend_reasoning(
            {"last_reasoning": "private reasoning"}, "Final answer", source, False
        )

    assert "private reasoning" in displayed
    assert displayed.endswith("Final answer")
    assert spoken == "Final answer"


def test_gateway_notices_remain_in_spoken_payload():
    runner = GatewayRunner.__new__(GatewayRunner)
    displayed = "-# 💭 Reasoning\n-# private reasoning\n\nFinal answer"
    notice = "The turn failed; retry after checking completed actions."

    after_failure = runner._hmwa_add_failed_turn_notice(displayed, notice)
    spoken = runner._hmwa_carry_spoken_response_suffix(
        "Final answer", displayed, after_failure
    )
    assert spoken == f"Final answer\n\n{notice}"

    reset = after_failure + "\n\nSession auto-reset"
    assert runner._hmwa_carry_spoken_response_suffix(spoken, after_failure, reset) == (
        f"Final answer\n\n{notice}\n\nSession auto-reset"
    )


@pytest.mark.asyncio
async def test_whole_file_voice_reply_uses_final_answer_not_reasoning():
    adapter = MagicMock()
    adapter._streaming_tts_turn_completed.return_value = False
    runner = _runner(adapter)
    event = _event()
    displayed = "-# 💭 Reasoning\n-# private reasoning\n\nFinal answer"

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
async def test_streaming_success_does_not_send_whole_file_again():
    adapter = MagicMock()
    adapter._streaming_tts_turn_completed.return_value = True
    runner = _runner(adapter)
    event = _event()

    await runner._hmwa_deliver_turn_response(
        event,
        event.source,
        SimpleNamespace(session_id="session"),
        "key",
        1,
        {"already_sent": False},
        [],
        "Visible reasoning and answer",
        "",
        False,
        "Final answer",
    )

    runner._send_voice_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_reasoning_response_keeps_default_voice_payload():
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
