"""Crash-left redelivery hides the /loop completion marker like every other final-send lane."""

from __future__ import annotations

import time
from types import SimpleNamespace

from gateway.config import Platform
from gateway.run import GatewayRunner


def _reply(content: str):
    runner = object.__new__(GatewayRunner)
    origin = SimpleNamespace(platform=Platform.TELEGRAM, chat_id="1", thread_id=None)
    history = [
        {"role": "user", "content": "check CI", "timestamp": time.time()},
        {"role": "assistant", "content": content, "timestamp": time.time() + 1},
    ]
    return runner._crash_left_reply(history, time.time() - 5, origin)


def test_text_reply_is_redelivered():
    assert _reply("Done.") == "Done."


def test_redelivered_reply_hides_loop_complete_marker():
    assert _reply("CI is green.\nLOOP_COMPLETE") == "CI is green."


def test_bare_loop_complete_reply_redelivers_nothing():
    assert _reply("LOOP_COMPLETE") is None


def test_marker_before_provider_terminal_token_is_still_hidden():
    assert _reply("CI is green.\nLOOP_COMPLETE<|eos|>") == "CI is green."
