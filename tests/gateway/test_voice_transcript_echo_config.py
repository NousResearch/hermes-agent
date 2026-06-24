import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_voice_transcript_echo_defaults_to_enabled():
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {"telegram": adapter}
    source = SimpleNamespace(platform="telegram", chat_id="123")

    with patch("gateway.run._load_gateway_config", return_value={}):
        await runner._echo_voice_transcripts(
            source,
            ["hello from voice"],
            metadata={"thread_id": 99},
        )

    adapter.send.assert_awaited_once_with(
        "123",
        '🎙️ "hello from voice"',
        metadata={"thread_id": 99},
    )


@pytest.mark.asyncio
async def test_voice_transcript_echo_can_be_disabled_without_disabling_stt():
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {"telegram": adapter}
    source = SimpleNamespace(platform="telegram", chat_id="123")

    with patch(
        "gateway.run._load_gateway_config",
        return_value={"gateway": {"echo_voice_transcripts": False}},
    ):
        await runner._echo_voice_transcripts(source, ["hidden from chat"])

    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_transcript_echo_accepts_string_false_from_yaml_or_env_bridge():
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {"telegram": adapter}
    source = SimpleNamespace(platform="telegram", chat_id="123")

    with patch(
        "gateway.run._load_gateway_config",
        return_value={"gateway": {"echo_voice_transcripts": "false"}},
    ):
        await runner._echo_voice_transcripts(source, ["hidden from chat"])

    adapter.send.assert_not_awaited()


def test_all_voice_transcript_echo_paths_use_configured_helper():
    source = inspect.getsource(GatewayRunner)

    assert "Voice-interrupt echo failed" not in source
    assert "Voice-drain echo failed" not in source
    assert "_echo_voice_transcripts(\n                                                source," in source
    assert "_echo_voice_transcripts(\n                                    source," in source
    assert source.count("f'🎙️ \"{transcript}\"'") == 1
    assert "f'🎙️ \"{_tx}\"'" not in source
