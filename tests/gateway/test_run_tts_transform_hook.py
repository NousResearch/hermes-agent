import asyncio
import json
from unittest.mock import patch


def test_transform_tts_text_uses_last_string_before_synthesis(
    gateway_runner_factory, message_event_factory
):
    runner = gateway_runner_factory()
    event = message_event_factory(text="original")

    with patch(
        "hermes_cli.plugins.invoke_hook",
        return_value=[None, "<filtered>original</filtered>"],
    ):
        with patch("tools.tts_tool.text_to_speech_tool") as tts_mock:
            tts_mock.return_value = json.dumps(
                {"success": True, "file_path": "/tmp/hermes-missing.mp3"}
            )
            asyncio.run(runner._send_voice_reply(event, "original"))

    assert tts_mock.call_args.kwargs["text"] == "<filtered>original</filtered>"


def test_transform_tts_text_none_passes_through(
    gateway_runner_factory, message_event_factory
):
    runner = gateway_runner_factory()
    event = message_event_factory(text="hello")

    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        with patch("tools.tts_tool.text_to_speech_tool") as tts_mock:
            tts_mock.return_value = json.dumps(
                {"success": True, "file_path": "/tmp/hermes-missing.mp3"}
            )
            asyncio.run(runner._send_voice_reply(event, "hello"))

    assert tts_mock.call_args.kwargs["text"] == "hello"


def test_transform_tts_text_exception_falls_back_to_original(
    gateway_runner_factory, message_event_factory, caplog
):
    runner = gateway_runner_factory()
    event = message_event_factory(text="hello")

    with patch(
        "hermes_cli.plugins.invoke_hook",
        side_effect=RuntimeError("plugin TTS filter crashed"),
    ):
        with patch("tools.tts_tool.text_to_speech_tool") as tts_mock:
            tts_mock.return_value = json.dumps(
                {"success": True, "file_path": "/tmp/hermes-missing.mp3"}
            )
            asyncio.run(runner._send_voice_reply(event, "hello"))

    assert tts_mock.call_args.kwargs["text"] == "hello"
    assert "transform_tts_text" in caplog.text
