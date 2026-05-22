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

import pytest

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

    def test_assistant_minimal_has_only_role_and_content(self):
        entry = _build_replay_entry(
            "assistant",
            "ok",
            {"role": "assistant", "content": "ok"},
        )
        assert entry == {"role": "assistant", "content": "ok"}

    def test_assistant_preserves_reasoning(self):
        msg = {
            "role": "assistant",
            "content": "answer",
            "reasoning": "I think therefore I am.",
        }
        entry = _build_replay_entry("assistant", "answer", msg)
        assert entry["reasoning"] == "I think therefore I am."

    def test_assistant_preserves_reasoning_content(self):
        """reasoning_content was silently dropped before this fix.

        Required for DeepSeek/Kimi/Moonshot thinking-mode echo so the
        provider receives back what it sent.
        """
        msg = {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "structured CoT",
        }
        entry = _build_replay_entry("assistant", "answer", msg)
        assert entry["reasoning_content"] == "structured CoT"

    def test_assistant_preserves_reasoning_details(self):
        details = [
            {
                "type": "reasoning.summary",
                "format": "text",
                "summary": "thought hard",
            },
            {
                "type": "reasoning.encrypted",
                "data": "opaque_blob",
                "signature": "sig123",
            },
        ]
        msg = {
            "role": "assistant",
            "content": "answer",
            "reasoning_details": details,
        }
        entry = _build_replay_entry("assistant", "answer", msg)
        assert entry["reasoning_details"] == details

    def test_assistant_preserves_codex_reasoning_items(self):
        items = [{"type": "reasoning", "encrypted_content": "blob"}]
        msg = {
            "role": "assistant",
            "content": "answer",
            "codex_reasoning_items": items,
        }
        entry = _build_replay_entry("assistant", "answer", msg)
        assert entry["codex_reasoning_items"] == items

    def test_assistant_preserves_codex_message_items(self):
        """codex_message_items was silently dropped before this fix.

        OpenAI docs: 'preserve and resend phase on all assistant messages
        — dropping it can degrade performance.'  Required for prefix
        cache hits on the Codex Responses API.
        """
        items = [
            {
                "type": "message",
                "role": "assistant",
                "id": "msg_123",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "Done"}],
            }
        ]
        msg = {
            "role": "assistant",
            "content": "Done",
            "codex_message_items": items,
        }
        entry = _build_replay_entry("assistant", "Done", msg)
        assert entry["codex_message_items"] == items

    def test_assistant_preserves_finish_reason(self):
        """finish_reason was silently dropped before this fix.

        Cheap to keep; lets transcripts replay byte-identically across
        CLI and gateway.
        """
        msg = {
            "role": "assistant",
            "content": "answer",
            "finish_reason": "stop",
        }
        entry = _build_replay_entry("assistant", "answer", msg)
        assert entry["finish_reason"] == "stop"

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

    def test_assistant_preserves_empty_reasoning_content(self):
        """Empty reasoning_content is a meaningful sentinel.

        DeepSeek V4 Pro thinking mode rejects bare missing reasoning_content
        with HTTP 400.  ``_copy_reasoning_content_for_api`` upgrades the
        empty string to a single space at API-send time, but only if the
        empty string actually reached it.  Dropping it here would 400 the
        next turn for affected providers.
        """
        msg = {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "",
        }
        entry = _build_replay_entry("assistant", "answer", msg)
        assert "reasoning_content" in entry
        assert entry["reasoning_content"] == ""

    def test_assistant_drops_none_reasoning_content(self):
        """None reasoning_content is just an absent field; drop it."""
        msg = {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": None,
        }
        entry = _build_replay_entry("assistant", "answer", msg)
        assert "reasoning_content" not in entry

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

    def test_assistant_does_not_invent_keys(self):
        """The helper only copies over fields that are explicitly present."""
        msg = {"role": "assistant", "content": "answer", "reasoning": "r"}
        entry = _build_replay_entry("assistant", "answer", msg)
        # reasoning_details/etc. weren't in msg, so they shouldn't be in entry
        for absent in (
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
            "codex_message_items",
            "finish_reason",
        ):
            assert absent not in entry

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
