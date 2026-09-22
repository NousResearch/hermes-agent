"""Tests for the post_compaction plugin hook (#118382).

The post_compaction hook fires once after a successful context compaction,
so plugins/providers can react (re-inject context, update indices, etc.).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor


class _RecordingCompressor(ContextCompressor):
    """A compressor whose _generate_summary returns a fixed string."""

    def __init__(self) -> None:
        super().__init__(
            model="test-model", protect_first_n=2, protect_last_n=2,
            quiet_mode=True, config_context_length=40_960,
        )
        self.summary_calls: list = []

    def _generate_summary(self, turns_to_summarize, focus_topic=None, memory_context="", bypass_cooldown=False):
        self.summary_calls.append((focus_topic, memory_context, {}))
        return "## Goal\nTest summary for hook."


def _messages(n: int = 14) -> list[dict]:
    """Build a small synthetic conversation."""
    return [{"role": "system", "content": "system"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i} " + "context " * 50}
        for i in range(n)
    ]


class TestPostCompactionHook:
    """post_compaction hook fires once per successful compaction, observer-only."""

    def test_hook_fired_when_has_hook_true(self) -> None:
        compressor = _RecordingCompressor()
        compressor.bind_session_state(session_id="sess-1")

        messages = _messages()
        before_tokens = len(str(messages))  # rough estimate, exact value not checked
        after_tokens_before = len(str(messages))

        with patch("hermes_cli.lifecycle.has_hook", return_value=True) as mock_has, \
             patch("hermes_cli.lifecycle.invoke_hook") as mock_invoke:
            compressed = compressor.compress(messages, current_tokens=30_000)

        mock_has.assert_called_once_with("post_compaction")
        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args.kwargs

        assert call_kwargs["session_id"] == "sess-1"
        assert call_kwargs["before_tokens"] > 0
        assert call_kwargs["after_tokens"] > 0
        assert call_kwargs["messages_removed"] > 0
        assert call_kwargs["reason"] == "compaction"

    def test_hook_not_fired_when_has_hook_false(self) -> None:
        compressor = _RecordingCompressor()
        compressor.bind_session_state(session_id="sess-2")

        messages = _messages()

        with patch("hermes_cli.lifecycle.has_hook", return_value=False) as mock_has, \
             patch("hermes_cli.lifecycle.invoke_hook") as mock_invoke:
            compressed = compressor.compress(messages, current_tokens=30_000)

        mock_has.assert_called_once_with("post_compaction")
        mock_invoke.assert_not_called()

    def test_hook_not_fired_on_structural_no_op(self) -> None:
        """Compaction that does nothing (too few messages) does NOT fire the hook."""
        compressor = _RecordingCompressor()
        compressor.bind_session_state(session_id="sess-3")

        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}]

        with patch("hermes_cli.lifecycle.has_hook", return_value=True) as mock_has, \
             patch("hermes_cli.lifecycle.invoke_hook") as mock_invoke:
            result = compressor.compress(messages, current_tokens=100)

        # Hook is not fired when no compaction made progress
        mock_has.assert_not_called()
        mock_invoke.assert_not_called()

    def test_hook_payload_values_correct(self) -> None:
        compressor = _RecordingCompressor()
        compressor.bind_session_state(session_id="sess-4")

        messages = _messages()

        with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
             patch("hermes_cli.lifecycle.invoke_hook") as mock_invoke:
            compressed = compressor.compress(messages, current_tokens=30_000)

        kwargs = mock_invoke.call_args.kwargs
        assert kwargs["session_id"] == "sess-4"
        assert kwargs["before_tokens"] > kwargs["after_tokens"]  # compression shrinks
        assert kwargs["messages_removed"] == len(messages) - len(compressed)
        assert kwargs["reason"] == "compaction"

    def test_hook_dispatch_failure_does_not_raise(self) -> None:
        """Hook dispatch failure is caught (non-fatal debug log)."""
        compressor = _RecordingCompressor()
        compressor.bind_session_state(session_id="sess-5")

        messages = _messages()

        with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
             patch("hermes_cli.lifecycle.invoke_hook", side_effect=RuntimeError("boom")):
            # Should not raise
            compressed = compressor.compress(messages, current_tokens=30_000)

        assert len(compressed) < len(messages)  # compression still succeeded

    def test_hook_called_once_per_compress(self) -> None:
        compressor = _RecordingCompressor()
        compressor.bind_session_state(session_id="sess-6")

        messages = _messages()

        with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
             patch("hermes_cli.lifecycle.invoke_hook") as mock_invoke:
            compressed = compressor.compress(messages, current_tokens=30_000)

        mock_invoke.assert_called_once()


class TestPostCompactionHookRegistration:
    """post_compaction appears in VALID_HOOKS."""

    def test_post_compaction_in_valid_hooks(self) -> None:
        from hermes_cli.plugins import VALID_HOOKS

        assert "post_compaction" in VALID_HOOKS
