"""The auto-TTS voice reply must never delay the TEXT reply (bounded synthesis).

Regression for the 2026-09-22 email outage: ``_hmwa_deliver_turn_response`` awaited the voice
synthesis before handing the text to the adapter, and the OpenAI TTS client's own default
(``timeout=600`` with ``max_retries=2``) let one slow synthesis hold the answer for 1800 s on every
platform (qwen3-tts had restarted on CPU at 800-1000 s per request). The await is now bounded by
``voice.auto_tts_timeout_s`` and the text is sent without audio when the budget elapses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_voice import (
    _AUTO_TTS_TIMEOUT_DEFAULT_S,
    _AUTO_TTS_TIMEOUT_MAX_S,
    _AUTO_TTS_TIMEOUT_MIN_S,
    _auto_tts_timeout_s,
)
from gateway.session import SessionSource


# ---------------------------------------------------------------- budget resolution

class TestAutoTtsTimeoutResolution:

    @pytest.mark.parametrize("raw", [None, "nonsense", {}, True])
    def test_unusable_config_falls_back_to_the_default(self, raw):
        """Unset/garbage must mean the DEFAULT bound, never "wait forever" (that is the bug)."""
        with patch("hermes_cli.config.load_config",
                   return_value={"voice": {"auto_tts_timeout_s": raw}}):
            assert _auto_tts_timeout_s() == _AUTO_TTS_TIMEOUT_DEFAULT_S

    def test_configured_value_is_used(self):
        with patch("hermes_cli.config.load_config",
                   return_value={"voice": {"auto_tts_timeout_s": 12.5}}):
            assert _auto_tts_timeout_s() == 12.5

    def test_value_is_clamped_and_zero_falls_back(self):
        with patch("hermes_cli.config.load_config",
                   return_value={"voice": {"auto_tts_timeout_s": 0.01}}):
            assert _auto_tts_timeout_s() == _AUTO_TTS_TIMEOUT_MIN_S
        with patch("hermes_cli.config.load_config",
                   return_value={"voice": {"auto_tts_timeout_s": 100000}}):
            assert _auto_tts_timeout_s() == _AUTO_TTS_TIMEOUT_MAX_S
        with patch("hermes_cli.config.load_config",
                   return_value={"voice": {"auto_tts_timeout_s": 0}}):
            assert _auto_tts_timeout_s() == _AUTO_TTS_TIMEOUT_DEFAULT_S

    def test_config_read_failure_still_bounds(self):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            assert _auto_tts_timeout_s() == _AUTO_TTS_TIMEOUT_DEFAULT_S


# ---------------------------------------------------------------- the bound itself

class TestDeliverTurnResponseBoundsAutoTts:

    @pytest.mark.asyncio
    async def test_hung_voice_reply_still_returns_the_text(self, caplog):
        """A synthesis that never returns must not hold the turn: the text comes straight back."""
        runner = _make_runner()
        runner._auto_tts_budget_s = lambda: 0.05
        runner._auto_tts_wait_s = lambda: 0.1
        runner._send_voice_reply = AsyncMock(side_effect=_never_returns)
        event = _make_event()
        started = asyncio.get_running_loop().time()

        with caplog.at_level(logging.WARNING, logger="gateway.run"):
            result = await asyncio.wait_for(_deliver(runner, event), timeout=5)

        assert result == "the answer"
        assert asyncio.get_running_loop().time() - started < 3
        assert any("exceeded its" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_voice_reply_receives_the_budget_as_request_timeout(self):
        """The budget is also handed to the TTS call, so the request (not just the await) is bound."""
        runner = _make_runner()
        runner._auto_tts_budget_s = lambda: 7.5
        runner._auto_tts_wait_s = lambda: 12.5
        runner._send_voice_reply = AsyncMock()

        assert await _deliver(runner, _make_event()) == "the answer"
        runner._send_voice_reply.assert_awaited_once()
        assert runner._send_voice_reply.await_args.kwargs["timeout_s"] == 7.5


# ---------------------------------------------------------------- _send_voice_reply plumbing

class TestSendVoiceReplyTimeoutPlumbing:

    @pytest.mark.asyncio
    async def test_request_timeout_reaches_the_tts_tool(self):
        runner = _make_runner()
        adapter = _make_voice_adapter()
        runner.adapters[Platform.TELEGRAM] = adapter
        seen = {}

        def fake_tts(*, text, output_path, request_timeout_s=None):
            seen["timeout"] = request_timeout_s
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"audio")
            return json.dumps({"success": True, "file_path": output_path})

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=fake_tts):
            await runner._send_voice_reply(_make_event(), "hello", timeout_s=9.0)

        assert seen["timeout"] == 9.0

    @pytest.mark.asyncio
    async def test_cancelled_synthesis_does_not_unlink_the_file_being_written(self):
        """The worker thread cannot be cancelled, so its output must survive the abandoned await."""
        runner = _make_runner()
        adapter = _make_voice_adapter()
        runner.adapters[Platform.TELEGRAM] = adapter
        written = threading.Event()
        release = threading.Event()
        paths = []

        def fake_tts(*, text, output_path, request_timeout_s=None):
            paths.append(output_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"audio")
            written.set()
            release.wait(10)  # first write is on disk; the "request" is still in flight
            return json.dumps({"success": True, "file_path": output_path})

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=fake_tts):
            task = asyncio.ensure_future(runner._send_voice_reply(_make_event(), "hello", timeout_s=1.0))
            assert await asyncio.to_thread(written.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            try:
                assert paths and Path(paths[0]).exists()
            finally:
                release.set()  # let the worker finish before the loop's executor shuts down
                await asyncio.sleep(0.1)
        for path in paths:  # tidy the tmp file this test intentionally left behind
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------- helpers

async def _never_returns(*_args, **_kwargs):
    await asyncio.sleep(300)


async def _deliver(runner, event):
    """Run the real delivery decision with voice replies forced on."""
    return await runner._hmwa_deliver_turn_response(
        event, event.source, MagicMock(), "session-key", 1, {}, [], "the answer", None, False)


def _make_runner() -> GatewayRunner:
    with patch("gateway.run.GatewayRunner._load_voice_modes", return_value={}):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._voice_mode = {}
        runner.adapters = {}
        runner._should_send_voice_reply = lambda *_a, **_k: True
        runner._delivery_adapter_for = lambda source: None
    return runner


def _make_voice_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.platform = Platform.TELEGRAM
    adapter.send_voice = AsyncMock()
    return adapter


def _make_event() -> MessageEvent:
    return MessageEvent(
        text="trigger",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1",
                             user_name="User"),
        message_type=MessageType.TEXT,
        message_id="456",
    )
