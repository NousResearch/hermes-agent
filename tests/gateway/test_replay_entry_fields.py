"""Tests for ``gateway.run._build_replay_entry``.

The gateway rebuilds ``agent_history`` from the persisted transcript on every
turn (unlike the CLI, which keeps the live in-memory message list).  When a
pure-text assistant turn (no ``tool_calls``) is replayed, the simple-text
branch in ``run_sync`` used to whitelist only three reasoning fields:
``reasoning``, ``reasoning_details``, ``codex_reasoning_items``.

That whitelist predated three fields the DB now persists:
``reasoning_content``, ``codex_message_items``, and ``finish_reason``.  The
unrecovered drop of ``codex_message_items`` in particular kills prefix-cache
hits for OpenAI Codex Responses API users — OpenAI's docs require the
``phase`` field be replayed on every assistant message.

These tests pin the expanded whitelist so it doesn't regress.
"""
from __future__ import annotations


from gateway.run import _ASSISTANT_REPLAY_FIELDS, _build_replay_entry


class TestBuildReplayEntry:
    def test_user_message_has_only_role_and_content(self):
        entry = _build_replay_entry(
            "user",
            "hello",
            {"role": "user", "content": "hello", "reasoning": "leak", "extra": "drop"},
        )
        assert entry == {"role": "user", "content": "hello"}

    def test_tool_message_has_only_role_and_content(self):
        # Tool messages aren't routed through this helper in production
        # (they take the rich-passthrough branch), but the helper itself
        # must not leak reasoning fields onto non-assistant roles even if
        # someone calls it incorrectly.
        entry = _build_replay_entry(
            "tool",
            "result",
            {"role": "tool", "content": "result", "reasoning": "leak"},
        )
        assert entry == {"role": "tool", "content": "result"}


    def test_assistant_drops_falsy_reasoning(self):
        """Empty/None reasoning fields stay dropped (matching PR #2974
        behaviour) — empty strings/lists for these fields carry no info."""
        msg = {
            "role": "assistant",
            "content": "answer",
            "reasoning": "",
            "reasoning_details": [],
            "codex_reasoning_items": [],
            "codex_message_items": [],
            "finish_reason": "",
        }
        entry = _build_replay_entry("assistant", "answer", msg)
        assert entry == {"role": "assistant", "content": "answer"}


    def test_assistant_preserves_all_six_fields_together(self):
        details = [{"type": "reasoning.summary", "summary": "s"}]
        codex_items = [{"type": "reasoning", "encrypted_content": "b"}]
        msg_items = [
            {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "x"}],
            }
        ]
        msg = {
            "role": "assistant",
            "content": "answer",
            "reasoning": "thinking",
            "reasoning_content": "structured",
            "reasoning_details": details,
            "codex_reasoning_items": codex_items,
            "codex_message_items": msg_items,
            "finish_reason": "stop",
        }
        entry = _build_replay_entry("assistant", "answer", msg)
        assert entry["reasoning"] == "thinking"
        assert entry["reasoning_content"] == "structured"
        assert entry["reasoning_details"] == details
        assert entry["codex_reasoning_items"] == codex_items
        assert entry["codex_message_items"] == msg_items
        assert entry["finish_reason"] == "stop"


    def test_replay_fields_constant_is_stable(self):
        """Pin the whitelist explicitly so accidental renames are caught."""
        assert _ASSISTANT_REPLAY_FIELDS == (
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
            "codex_message_items",
            "finish_reason",
        )

    def test_unrelated_keys_are_ignored(self):
        """Random keys on the message must not leak into the replay entry."""
        msg = {
            "role": "assistant",
            "content": "answer",
            "timestamp": 12345.6,
            "internal_marker": "should not flow",
            "tool_call_id": "should not be set on simple-text branch",
        }
        entry = _build_replay_entry("assistant", "answer", msg)
        assert "timestamp" not in entry
        assert "internal_marker" not in entry
        assert "tool_call_id" not in entry


class TestReplayEntryApiContentSidecar:
    """The api_content sidecar (persist-what-you-send) must survive the
    gateway's transcript→agent_history rebuild, or the whole prompt-cache
    fix is inert on gateway platforms — but only when this pipeline did not
    rewrite the content (a rewrite means different bytes must replay)."""

    def test_user_forwards_api_content_when_content_unchanged(self):
        msg = {"role": "user", "content": "hi", "api_content": "hi\n\nCTX"}
        entry = _build_replay_entry("user", "hi", msg)
        assert entry["api_content"] == "hi\n\nCTX"
        assert entry["content"] == "hi"

    def test_assistant_forwards_api_content_when_content_unchanged(self):
        msg = {"role": "assistant", "content": "a", "api_content": "a <memory-context>"}
        entry = _build_replay_entry("assistant", "a", msg)
        assert entry["api_content"] == "a <memory-context>"


class TestGatewayHistoryBuildForwardsSidecar:
    def test_end_to_end_history_build_keeps_sidecar(self):
        from gateway.run import _build_gateway_agent_history

        history = [
            {"role": "user", "content": "hi", "api_content": "hi\n\nCTX", "timestamp": 123.0},
            {"role": "assistant", "content": "hello"},
        ]
        agent_history, _obs = _build_gateway_agent_history(history)
        assert agent_history[0]["api_content"] == "hi\n\nCTX"

