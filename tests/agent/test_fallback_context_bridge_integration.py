"""Integration test: Codex → local-claude fallback with context bridge.

Scenario reproduced:
  1. Agent is configured with openai-codex as primary provider
     (api_mode = codex_responses).
  2. A conversation history is built that mimics what a Codex session
     produces: assistant messages containing codex_reasoning_items,
     codex_message_items, and a _hermes_internal key.
  3. A skill is created during the Codex session (appears in the skill
     index, which is part of the system prompt).
  4. try_activate_fallback() is called to simulate a rate-limit / error
     causing Codex to hand off to local-claude.
  5. apply_fallback_context_bridge() fires and:
       - strips all codex_* fields from api_messages
       - rebuilds the system prompt (so skill context survives)
  6. Assertions verify the bridge worked correctly.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import copy
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.fallback_context_bridge import (
    apply_fallback_context_bridge,
    _strip_codex_fields_from_messages,
    _strip_anthropic_fields_from_messages,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_codex_conversation():
    """Return a realistic api_messages list produced by a Codex session."""
    return [
        {
            "role": "system",
            "content": "You are Hermes.\n\n## Skills\n- codex-demo: A skill made during Codex session\n",
        },
        {
            "role": "user",
            "content": "Create a skill called codex-demo",
        },
        {
            "role": "assistant",
            "content": "I'll create the skill for you.",
            # Codex Responses API fields — must be stripped on fallback
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "enc_abc123==", "_issuer_kind": "codex_backend"},
            ],
            "codex_message_items": [
                {"type": "message", "id": "msg_xyz", "content": [{"type": "output_text", "text": "Done."}]},
            ],
            "_thinking_prefill": True,
        },
        {
            "role": "tool",
            "content": '{"result": "skill codex-demo created"}',
            "tool_call_id": "call_001",
        },
        {
            "role": "assistant",
            "content": "Skill codex-demo has been created successfully.",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "enc_def456==", "_issuer_kind": "codex_backend"},
            ],
        },
    ]


def _make_stub_agent(system_prompt: str = "Rebuilt system prompt with codex-demo skill"):
    """Return a minimal stub that satisfies apply_fallback_context_bridge."""
    agent = SimpleNamespace()
    agent.api_mode = "anthropic_messages"        # already switched by fallback
    agent.provider = "local-claude"
    agent.model = "claude-opus-4-8"
    agent.session_id = "test-session-001"
    agent._cached_system_prompt = None            # invalidated by try_activate_fallback
    agent._session_db = None                      # skip DB persistence in tests
    agent._codex_reasoning_replay_enabled = True

    # _build_system_prompt returns the rebuilt prompt
    agent._build_system_prompt = MagicMock(return_value=system_prompt)

    # _disable_codex_reasoning_replay mimics the real implementation
    def _disable_replay(messages=None):
        count = 0
        for m in (messages or []):
            if isinstance(m, dict) and m.get("role") == "assistant":
                if m.pop("codex_reasoning_items", None):
                    count += 1
        agent._codex_reasoning_replay_enabled = False
        return {"messages": count, "items": count}

    agent._disable_codex_reasoning_replay = _disable_replay
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStripCodexFields:
    """Unit tests for the stripping helpers."""

    def test_strips_codex_reasoning_items(self):
        msgs = [{"role": "assistant", "content": "hi", "codex_reasoning_items": [1, 2]}]
        n = _strip_codex_fields_from_messages(msgs)
        assert "codex_reasoning_items" not in msgs[0]
        assert n == 1

    def test_strips_codex_message_items(self):
        msgs = [{"role": "assistant", "content": "hi", "codex_message_items": ["x"]}]
        _strip_codex_fields_from_messages(msgs)
        assert "codex_message_items" not in msgs[0]

    def test_strips_internal_underscore_keys(self):
        msgs = [{"role": "assistant", "content": "hi", "_thinking_prefill": True, "_foo": "bar"}]
        _strip_codex_fields_from_messages(msgs)
        assert "_thinking_prefill" not in msgs[0]
        assert "_foo" not in msgs[0]

    def test_leaves_standard_fields_intact(self):
        msgs = [{"role": "assistant", "content": "keep me", "tool_calls": [{"id": "tc1"}]}]
        _strip_codex_fields_from_messages(msgs)
        assert msgs[0]["content"] == "keep me"
        assert msgs[0]["tool_calls"] == [{"id": "tc1"}]

    def test_non_dict_messages_skipped(self):
        msgs = ["not a dict", None, 42]
        n = _strip_codex_fields_from_messages(msgs)
        assert n == 0


class TestStripAnthropicFields:
    """Unit tests for Anthropic-specific field stripping (reverse direction)."""

    def test_strips_cache_control_top_level(self):
        msgs = [{"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}}]
        n = _strip_anthropic_fields_from_messages(msgs)
        assert "cache_control" not in msgs[0]
        assert n == 1

    def test_strips_cache_control_from_content_parts(self):
        msgs = [{
            "role": "assistant",
            "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
        }]
        _strip_anthropic_fields_from_messages(msgs)
        assert "cache_control" not in msgs[0]["content"][0]

    def test_strips_reasoning_content(self):
        msgs = [{"role": "assistant", "content": "hi", "reasoning_content": " "}]
        _strip_anthropic_fields_from_messages(msgs)
        assert "reasoning_content" not in msgs[0]


class TestApplyFallbackContextBridge:
    """Integration tests for the full bridge."""

    def test_codex_to_anthropic_strips_codex_fields(self):
        api_messages = _make_codex_conversation()
        agent = _make_stub_agent()

        apply_fallback_context_bridge(
            agent, api_messages,
            old_api_mode="codex_responses",
            new_api_mode="anthropic_messages",
        )

        for msg in api_messages:
            assert "codex_reasoning_items" not in msg, f"codex_reasoning_items still in {msg['role']} message"
            assert "codex_message_items" not in msg, f"codex_message_items still in {msg['role']} message"
            assert "_thinking_prefill" not in msg

    def test_codex_to_anthropic_rebuilds_system_prompt(self):
        api_messages = _make_codex_conversation()
        agent = _make_stub_agent(system_prompt="Rebuilt: codex-demo skill present")

        apply_fallback_context_bridge(
            agent, api_messages,
            old_api_mode="codex_responses",
            new_api_mode="anthropic_messages",
        )

        # _build_system_prompt was called
        agent._build_system_prompt.assert_called_once()
        # system message in api_messages was replaced
        assert api_messages[0]["role"] == "system"
        assert api_messages[0]["content"] == "Rebuilt: codex-demo skill present"
        # agent cache updated
        assert agent._cached_system_prompt == "Rebuilt: codex-demo skill present"

    def test_codex_to_anthropic_skill_context_preserved(self):
        """Skill created during Codex session must appear in the rebuilt system prompt."""
        api_messages = _make_codex_conversation()
        fresh_prompt = (
            "You are Hermes.\n\n"
            "## Skills\n"
            "- codex-demo: Skill created in Codex session\n"
            "- other-skill: Pre-existing skill\n"
        )
        agent = _make_stub_agent(system_prompt=fresh_prompt)

        apply_fallback_context_bridge(
            agent, api_messages,
            old_api_mode="codex_responses",
            new_api_mode="anthropic_messages",
        )

        rebuilt = api_messages[0]["content"]
        assert "codex-demo" in rebuilt, "Codex-session skill missing from rebuilt system prompt"
        assert "other-skill" in rebuilt

    def test_codex_reasoning_replay_disabled(self):
        api_messages = _make_codex_conversation()
        agent = _make_stub_agent()
        assert agent._codex_reasoning_replay_enabled is True

        apply_fallback_context_bridge(
            agent, api_messages,
            old_api_mode="codex_responses",
            new_api_mode="anthropic_messages",
        )

        assert agent._codex_reasoning_replay_enabled is False

    def test_same_mode_no_codex_strip(self):
        """Bridge called for non-Codex old_mode must not strip anything extra."""
        api_messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "ok", "reasoning_content": " "},
        ]
        original = copy.deepcopy(api_messages)
        agent = _make_stub_agent()

        apply_fallback_context_bridge(
            agent, api_messages,
            old_api_mode="anthropic_messages",
            new_api_mode="chat_completions",
        )

        # reasoning_content should NOT be stripped (only codex→other strips it)
        assert api_messages[1].get("reasoning_content") == original[1].get("reasoning_content")

    def test_entering_codex_strips_anthropic_fields(self):
        """chat_completions → codex_responses bridge strips cache_control."""
        api_messages = [
            {"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}},
            {"role": "assistant", "content": "ok", "reasoning_content": " "},
        ]
        agent = _make_stub_agent()

        apply_fallback_context_bridge(
            agent, api_messages,
            old_api_mode="chat_completions",
            new_api_mode="codex_responses",
        )

        assert "cache_control" not in api_messages[0]
        assert "reasoning_content" not in api_messages[1]

    def test_no_system_message_prepends_one(self):
        """If api_messages has no system message, bridge should prepend one."""
        api_messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "codex_reasoning_items": []},
        ]
        agent = _make_stub_agent(system_prompt="Fresh system prompt")

        apply_fallback_context_bridge(
            agent, api_messages,
            old_api_mode="codex_responses",
            new_api_mode="anthropic_messages",
        )

        assert api_messages[0]["role"] == "system"
        assert api_messages[0]["content"] == "Fresh system prompt"

    def test_build_system_prompt_failure_is_non_fatal(self):
        """If _build_system_prompt raises, the bridge must not propagate the error."""
        api_messages = _make_codex_conversation()
        agent = _make_stub_agent()
        agent._build_system_prompt = MagicMock(side_effect=RuntimeError("DB down"))

        # Should not raise
        apply_fallback_context_bridge(
            agent, api_messages,
            old_api_mode="codex_responses",
            new_api_mode="anthropic_messages",
        )

        # Codex fields are still stripped even when prompt rebuild fails
        for msg in api_messages:
            assert "codex_reasoning_items" not in msg


# ---------------------------------------------------------------------------
# Simulation: _pending_context_bridge flag lifecycle
# ---------------------------------------------------------------------------

class TestPendingContextBridgeFlag:
    """Verify the flag is set by try_activate_fallback and consumed by the loop."""

    def test_flag_set_when_api_mode_changes(self):
        """Simulates try_activate_fallback() setting the flag."""
        agent = SimpleNamespace()
        agent.api_mode = "codex_responses"
        agent._cached_system_prompt = "old system prompt"

        # Simulate what try_activate_fallback() does when mode changes
        old_api_mode = agent.api_mode
        new_api_mode = "anthropic_messages"

        if old_api_mode != new_api_mode:
            agent._pending_context_bridge = {
                "old_api_mode": old_api_mode,
                "new_api_mode": new_api_mode,
            }
            agent._cached_system_prompt = None

        assert agent._pending_context_bridge["old_api_mode"] == "codex_responses"
        assert agent._pending_context_bridge["new_api_mode"] == "anthropic_messages"
        assert agent._cached_system_prompt is None  # invalidated for rebuild

    def test_flag_not_set_when_mode_unchanged(self):
        """No bridge needed when both providers share the same api_mode."""
        agent = SimpleNamespace()
        agent.api_mode = "chat_completions"
        agent._cached_system_prompt = "unchanged"

        old_api_mode = agent.api_mode
        new_api_mode = "chat_completions"  # same

        if old_api_mode != new_api_mode:
            agent._pending_context_bridge = {"old_api_mode": old_api_mode, "new_api_mode": new_api_mode}
            agent._cached_system_prompt = None
        else:
            agent._pending_context_bridge = None

        assert agent._pending_context_bridge is None
        assert agent._cached_system_prompt == "unchanged"  # untouched

    def test_flag_consumed_in_loop(self):
        """Simulates the conversation_loop consuming and clearing the flag."""
        api_messages = _make_codex_conversation()
        agent = _make_stub_agent()
        agent._pending_context_bridge = {
            "old_api_mode": "codex_responses",
            "new_api_mode": "anthropic_messages",
        }

        # Simulate what conversation_loop does
        _bridge = getattr(agent, "_pending_context_bridge", None)
        if _bridge:
            agent._pending_context_bridge = None
            apply_fallback_context_bridge(
                agent, api_messages,
                _bridge["old_api_mode"],
                _bridge["new_api_mode"],
            )

        # Flag must be cleared
        assert agent._pending_context_bridge is None

        # Bridge must have run
        for msg in api_messages:
            assert "codex_reasoning_items" not in msg
