"""Regression: compression boundary must forward platform to context engine.

Issue #27633 -- when Hermes rotates session_id during context compression,
the on_session_start call at the compression boundary omitted
platform, causing plugin engines (e.g. hermes-lcm) that track
per-session source lineage to reset _session_platform to '' to
'unknown'.  All messages ingested after the compression boundary were
attributed to source='unknown' instead of the actual platform.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest


def _make_agent(platform: str = "telegram") -> MagicMock:
    """Build a mock agent with enough attributes for compress_context."""
    agent = MagicMock()
    agent.platform = platform
    agent.session_id = "old_session_123"
    agent.model = "test-model"
    agent.compression_enabled = True
    agent._compression_feasibility_checked = True
    agent._cached_system_prompt = "You are Hermes."
    agent._gateway_session_key = "gw-key-001"
    agent._session_todos = []
    agent._session_todos_dirty = False
    agent._last_compression_lock_error_sid = None
    agent._last_compression_lock_warning_sid = None
    agent._last_compression_summary_warning = None
    agent._last_aux_fallback_warning_key = None
    agent._compression_lock = None
    agent._compression_lock_holder = None
    agent._last_flushed_db_idx = 0
    agent._memory_manager = None
    agent._todo_store = MagicMock()
    agent._todo_store.format_for_injection.return_value = ""
    agent.log_prefix = ""

    # context_compressor
    agent.context_compressor.context_length = 128_000
    agent.context_compressor.has_content_to_compress.return_value = True
    agent.context_compressor._last_compress_aborted = False
    agent.context_compressor._last_summary_error = None
    agent.context_compressor._last_aux_model_failure_model = None
    agent.context_compressor._last_aux_model_failure_error = None
    agent.context_compressor.compression_count = 1
    agent.context_compressor.compress.return_value = (
        [{"role": "user", "content": "compressed"}],
        "summary text",
    )

    # session_db
    agent._session_db.get_session_title.return_value = "Test Session"
    agent._session_db.try_acquire_compression_lock.return_value = True

    # system prompt
    agent._build_system_prompt.return_value = "system: test"
    agent._invalidate_system_prompt = MagicMock()

    return agent


class TestCompressionBoundaryPlatform:
    """compress_context must pass platform to on_session_start."""

    def test_platform_forwarded_to_on_session_start(self):
        """The compression boundary on_session_start must include platform."""
        agent = _make_agent(platform="discord")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        from agent.conversation_compression import compress_context

        compress_context(agent, messages, "system msg")

        # Find the on_session_start call with boundary_reason="compression"
        compressor = agent.context_compressor
        start_calls = [
            c for c in compressor.on_session_start.call_args_list
            if "boundary_reason" in c.kwargs
            and c.kwargs["boundary_reason"] == "compression"
        ]

        assert len(start_calls) == 1, (
            f"Expected exactly 1 compression-boundary on_session_start call, "
            f"got {len(start_calls)}: {compressor.on_session_start.call_args_list}"
        )

        call_kwargs = start_calls[0].kwargs
        assert call_kwargs.get("platform") == "discord", (
            f"platform not forwarded to on_session_start. "
            f"Got kwargs: {call_kwargs}"
        )

    def test_platform_defaults_to_cli_when_none(self):
        """When agent.platform is None/empty, platform should default to 'cli'."""
        agent = _make_agent(platform="")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        from agent.conversation_compression import compress_context

        compress_context(agent, messages, "system msg")

        compressor = agent.context_compressor
        start_calls = [
            c for c in compressor.on_session_start.call_args_list
            if "boundary_reason" in c.kwargs
            and c.kwargs["boundary_reason"] == "compression"
        ]

        assert len(start_calls) == 1
        assert start_calls[0].kwargs.get("platform") == "cli"

    def test_platform_telegram_preserved(self):
        """Verify 'telegram' platform survives compression boundary."""
        agent = _make_agent(platform="telegram")
        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "reply"},
        ]

        from agent.conversation_compression import compress_context

        compress_context(agent, messages, "system msg")

        compressor = agent.context_compressor
        start_calls = [
            c for c in compressor.on_session_start.call_args_list
            if "boundary_reason" in c.kwargs
            and c.kwargs["boundary_reason"] == "compression"
        ]

        assert len(start_calls) == 1
        assert start_calls[0].kwargs.get("platform") == "telegram"
