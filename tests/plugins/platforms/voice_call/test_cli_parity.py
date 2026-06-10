"""CLI parity with OpenClaw's voicecall CLI: conversation default for
`call`, -t/-m short flags, and to_number config fallback."""

import argparse
import asyncio
import json

import pytest

from plugins.platforms.voice_call import cli
from plugins.platforms.voice_call.config import VoiceCallConfig
from plugins.platforms.voice_call.manager import CallManager


def _parse(argv):
    parser = argparse.ArgumentParser()
    cli.register_cli(parser)
    return parser.parse_args(argv)


def test_cli_call_defaults_to_conversation():
    args = _parse(["call", "-t", "+15555550001", "-m", "hello"])
    assert args.mode == "conversation"
    assert args.to == "+15555550001" and args.message == "hello"


def test_cli_call_to_is_optional():
    args = _parse(["call", "-m", "hello"])
    assert args.to is None  # falls back to config to_number at dispatch


def test_cli_call_mode_override():
    args = _parse(["call", "--to", "+15555550001", "--mode", "notify"])
    assert args.mode == "notify"


def test_cli_tail_follow_flags():
    args = _parse(["tail", "-f", "--poll", "0.5", "--lines", "5"])
    assert args.follow is True and args.poll == 0.5 and args.lines == 5


def test_config_to_number_parse_and_validation(monkeypatch):
    monkeypatch.delenv("VOICE_CALL_TO_NUMBER", raising=False)
    cfg = VoiceCallConfig.from_extra({"provider": "mock", "to_number": "+15555550002"})
    assert cfg.to_number == "+15555550002"
    assert cfg.validate() == []
    cfg = VoiceCallConfig.from_extra({"provider": "mock", "to_number": "junk"})
    assert any("to_number" in e for e in cfg.validate())
    monkeypatch.setenv("VOICE_CALL_TO_NUMBER", "+15555550003")
    cfg = VoiceCallConfig.from_extra({"provider": "mock"})
    assert cfg.to_number == "+15555550003"


@pytest.mark.asyncio
async def test_manager_falls_back_to_config_to_number(vc_config, provider, store):
    vc_config.to_number = "+15555550002"
    manager = CallManager(vc_config, provider, store)
    provider.event_sink = manager.process_event
    record = await manager.initiate_call()  # no destination passed
    assert record.to_number == "+15555550002"
    await manager.shutdown()

    vc_config.to_number = None
    manager = CallManager(vc_config, provider, store)
    with pytest.raises(ValueError, match="no destination number"):
        await manager.initiate_call()


@pytest.mark.asyncio
async def test_tool_initiate_uses_default_to_number(tmp_path, make_config):
    from plugins.platforms.voice_call import runtime as runtime_mod
    from plugins.platforms.voice_call.tool import voice_call_handler

    cfg = make_config()
    cfg.serve.port = 0
    cfg.to_number = "+15555550002"
    await runtime_mod.ensure_runtime(cfg, store_dir=tmp_path)
    try:
        result = json.loads(
            await voice_call_handler({"action": "initiate_call", "message": "hi"})
        )
        assert result["success"] is True
        assert result["call"]["peer_number"] == "+15555550002"
    finally:
        await runtime_mod.stop_runtime()


@pytest.mark.asyncio
async def test_slash_call_defaults_to_conversation(tmp_path, make_config):
    from plugins.platforms.voice_call import runtime as runtime_mod
    from plugins.platforms.voice_call.tool import slash_handler

    cfg = make_config()
    cfg.serve.port = 0
    cfg.outbound.default_mode = "notify"  # slash should override to conversation
    runtime = await runtime_mod.ensure_runtime(cfg, store_dir=tmp_path)
    try:
        out = await slash_handler("call --to +15555550001")
        call_id = out.rsplit(" ", 1)[-1]
        record = runtime.manager.get_call(call_id)
        assert record is not None and record.mode == "conversation"
    finally:
        await runtime_mod.stop_runtime()
