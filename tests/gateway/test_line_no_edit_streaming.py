"""LINE must never enter edit-based streaming — it has no message-edit API.

Symptom reported over a real LINE deployment: a single agent reply arrives
as two bubbles, split mid-word, with a stray replacement-character artifact
right at the cut (e.g. "テスト受信しました。正常■" / "に動作しています...").

Root cause: ``gateway/run.py`` ``_build_stream_consumer_config`` decides
whether to skip streaming for a platform via
``getattr(adapter, "SUPPORTS_MESSAGE_EDITING", True)`` — defaulting to
``True`` when the adapter doesn't declare it. ``LineAdapter`` never declared
it, so the gateway's streaming consumer ran edit-based streaming against a
platform that can only ever send NEW messages (LINE's Messaging API has no
edit-message endpoint): it sends a partial bubble it can never update, then
a second bubble with the rest of the turn — the exact "duplicate messages
(partial + final)" failure mode this same function's docstring/comment
already names for QQ/WeChat-style platforms.

These tests exercise the real ``_build_stream_consumer_config`` gate (it
doesn't touch ``self``, so it's called directly) against a real
``LineAdapter`` instance.
"""
from __future__ import annotations

import pytest

from gateway.config import PlatformConfig, Platform, StreamingConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_line_adapter():
    from plugins.platforms.line.adapter import LineAdapter

    return LineAdapter(PlatformConfig(enabled=True))


def test_line_adapter_declares_no_message_editing():
    """Direct regression pin on the class attribute itself."""
    adapter = _make_line_adapter()
    assert adapter.SUPPORTS_MESSAGE_EDITING is False


def test_line_skips_streaming_on_the_in_process_agent_path():
    """``on_missing_cursor="raise"`` is what the in-process agent-run path
    uses (see ``gateway/run.py`` call site) — the caller's ``except`` then
    skips streaming for this turn entirely, so the full reply goes out as a
    single normal ``send()`` once generation finishes. No edit-based
    consumer ever runs, so there is no partial bubble to split."""
    adapter = _make_line_adapter()
    source = SessionSource(platform=Platform.LINE, chat_id="U-test")
    scfg = StreamingConfig()

    with pytest.raises(RuntimeError, match="skip streaming for non-editable platform"):
        GatewayRunner._build_stream_consumer_config(
            None, source, scfg, adapter, on_missing_cursor="raise",
        )


def test_line_gets_a_cursorless_fallback_on_the_proxy_path():
    """``on_missing_cursor="fallback"`` (the remote-proxy agent path) must
    not raise for a non-editable adapter — it streams anyway with an empty
    cursor (``_effective_cursor == ""``), matching the pre-existing fallback
    semantics documented on ``_build_stream_consumer_config``."""
    adapter = _make_line_adapter()
    source = SessionSource(platform=Platform.LINE, chat_id="U-test")
    scfg = StreamingConfig()

    consumer_cfg, _pause_hook = GatewayRunner._build_stream_consumer_config(
        None, source, scfg, adapter, on_missing_cursor="fallback",
    )
    assert consumer_cfg.cursor == ""
