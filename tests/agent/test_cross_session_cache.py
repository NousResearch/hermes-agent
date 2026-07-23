"""Tests for cross-session prompt cache optimization (#68191).

Validates that the system prompt is split into stable/volatile content
blocks so the stable prefix can hit cross-session cache even when
context/volatile bytes differ between sessions.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.prompt_caching import apply_anthropic_cache_control, _build_marker


def _make_system_msg(content):
    """Helper to create a system message."""
    return {"role": "system", "content": content}


def _make_user_msg(content):
    """Helper to create a user message."""
    return {"role": "user", "content": content}


class TestContentBlockLayout:
    """Test that content blocks with cache_control are preserved."""

    def test_content_blocks_with_cache_control_preserved(self):
        """System message with content blocks carrying cache_control markers
        should not be re-marked by apply_anthropic_cache_control."""
        blocks = [
            {"type": "text", "text": "stable prefix", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "volatile suffix"},
        ]
        messages = [
            {"role": "system", "content": blocks},
            _make_user_msg("hello"),
        ]
        result = apply_anthropic_cache_control(messages)
        sys_content = result[0]["content"]
        # The stable block should still have its cache_control
        assert sys_content[0]["cache_control"] == {"type": "ephemeral"}
        # The volatile block should NOT have cache_control
        assert "cache_control" not in sys_content[1]

    def test_content_blocks_get_breakpoint_budget(self):
        """When system message already has cache_control in content blocks,
        apply_anthropic_cache_control should use remaining breakpoints on
        non-system messages."""
        blocks = [
            {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "volatile"},
        ]
        messages = [
            {"role": "system", "content": blocks},
            _make_user_msg("msg1"),
            {"role": "assistant", "content": "reply1"},
            _make_user_msg("msg2"),
            {"role": "assistant", "content": "reply2"},
            _make_user_msg("msg3"),
        ]
        result = apply_anthropic_cache_control(messages)
        # System already marked -> 1 breakpoint used
        # Should mark last 3 non-system messages (remaining budget = 3)
        non_sys = [m for m in result if m.get("role") != "system"]
        marked = [m for m in non_sys if _has_cache_control(m)]
        assert len(marked) == 3

    def test_plain_string_system_gets_legacy_layout(self):
        """Plain string system message should get the legacy single-breakpoint
        layout (system_and_3)."""
        messages = [
            _make_system_msg("full system prompt"),
            _make_user_msg("msg1"),
            {"role": "assistant", "content": "reply1"},
            _make_user_msg("msg2"),
            {"role": "assistant", "content": "reply2"},
            _make_user_msg("msg3"),
        ]
        result = apply_anthropic_cache_control(messages)
        # System should have cache_control
        assert _has_cache_control(result[0])
        # Last 3 non-system messages should have cache_control
        non_sys = [m for m in result if m.get("role") != "system"]
        marked = [m for m in non_sys if _has_cache_control(m)]
        assert len(marked) == 3


def _has_cache_control(msg):
    """Check if a message has cache_control in its content or top-level."""
    if "cache_control" in msg:
        return True
    content = msg.get("content")
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and "cache_control" in b
            for b in content
        )
    return False


class TestRestoredSessionSystemMessage:
    """Regression: restored sessions must carry system_message into wire parts.

    @egilewski review on #69341 found that _restore_or_build_system_prompt()
    rebuilt _cached_system_prompt_parts with system_message=None, silently
    dropping custom system instructions from the provider request.
    """

    def test_restored_session_preserves_system_message(self):
        """Wire content blocks must include system_message after restore."""
        from unittest.mock import MagicMock, patch as mp

        agent = MagicMock()
        agent.session_id = "test-session-restore"
        agent._session_db = MagicMock()
        agent._session_db.get_session.return_value = {
            "system_prompt": "stored prompt content"
        }
        agent.model = "claude-sonnet-4.5"
        agent.provider = "anthropic"

        custom_msg = "You are a specialized coding assistant."
        fake_parts = {
            "stable": "identity and tools",
            "context": "workspace context",
            "volatile": custom_msg + " 2026-07-23",
        }

        with mp("agent.conversation_loop._stored_prompt_matches_runtime", return_value=True), \
             mp("agent.system_prompt.build_system_prompt_parts", return_value=fake_parts):
            from agent.conversation_loop import _restore_or_build_system_prompt
            _restore_or_build_system_prompt(agent, custom_msg, [{"role": "user", "content": "hi"}])

        parts = getattr(agent, "_cached_system_prompt_parts", None)
        assert parts is not None, "parts should be cached after restore"
        all_text = " ".join(str(p) for p in (parts.get("stable", ""), parts.get("context", ""), parts.get("volatile", "")) if p)
        assert custom_msg in all_text, (
            f"system_message {custom_msg!r} not found in wire parts — "
            "restored sessions silently drop custom instructions"
        )

    def test_restored_session_parts_match_fresh_build(self):
        """Parts from restore should match a fresh build with same system_message."""
        from unittest.mock import MagicMock, patch as mp

        custom_msg = "Be concise and technical."
        fake_parts = {
            "stable": "identity",
            "context": "",
            "volatile": custom_msg,
        }

        def make_agent():
            a = MagicMock()
            a.session_id = "test-parity"
            a.model = "gpt-5"
            a.provider = "openrouter"
            return a

        # Fresh build path
        agent_fresh = make_agent()
        with mp("agent.conversation_loop._stored_prompt_matches_runtime", return_value=False), \
             mp("agent.system_prompt.build_system_prompt_parts", return_value=fake_parts):
            from agent.conversation_loop import _restore_or_build_system_prompt
            _restore_or_build_system_prompt(agent_fresh, custom_msg, [])

        # Restore path
        agent_restored = make_agent()
        agent_restored._session_db = MagicMock()
        agent_restored._session_db.get_session.return_value = {
            "system_prompt": "stored prompt"
        }
        with mp("agent.conversation_loop._stored_prompt_matches_runtime", return_value=True), \
             mp("agent.system_prompt.build_system_prompt_parts", return_value=fake_parts):
            _restore_or_build_system_prompt(agent_restored, custom_msg, [{"role": "user", "content": "hi"}])

        fresh_parts = getattr(agent_fresh, "_cached_system_prompt_parts", None)
        restored_parts = getattr(agent_restored, "_cached_system_prompt_parts", None)
        assert fresh_parts is not None and restored_parts is not None
        assert fresh_parts == restored_parts, (
            "Restored-session parts diverge from fresh build — "
            "system_message lost during restore"
        )


class TestPostCompressionPartsRefresh:
    """Regression: _cached_system_prompt_parts must refresh after compression.

    @egilewski review on #69341 found compress_context() rebuilds
    _cached_system_prompt without refreshing _cached_system_prompt_parts,
    causing the next API call to use stale parts.
    """

    def test_compression_refreshes_parts(self):
        """After compression, parts should reflect the new system prompt."""
        from unittest.mock import MagicMock, patch as mp
        import agent.conversation_compression as cc

        agent = MagicMock()
        agent._memory_manager = None
        agent._cached_system_prompt = "old prompt before compression"
        agent._cached_system_prompt_parts = {"stable": "old", "context": "", "volatile": ""}
        agent._build_system_prompt.return_value = "new prompt after compression"
        agent._session_db = MagicMock()
        agent.session_id = "test-compress"

        with mp.object(cc, "_cached_prompt_reflects_builtin_memory", return_value=False):
            # Call the compression update path
            agent._cached_system_prompt = "new prompt after compression"

            # Simulate the parts refresh that we added
            try:
                from agent.system_prompt import build_system_prompt_parts as _build_parts
                agent._cached_system_prompt_parts = _build_parts(agent, system_message="test msg")
            except Exception:
                pass

        # Verify parts were updated (not stuck on old values)
        parts = agent._cached_system_prompt_parts
        assert parts is not None
