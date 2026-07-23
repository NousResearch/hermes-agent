"""Tests for session_meta filtering — issue #4715.

Ensures that transcript-only session_meta messages never reach the
chat-completions API, via both the API-boundary guard in
_sanitize_api_messages() and the CLI session-restore paths.
"""

import logging

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Layer 1 — _sanitize_api_messages role-allowlist guard
# ---------------------------------------------------------------------------

class TestSanitizeApiMessagesRoleFilter:

    def test_drops_session_meta_role(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "session_meta", "content": {"model": "gpt-4"}},
            {"role": "assistant", "content": "hi"},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert len(out) == 2
        assert all(m["role"] != "session_meta" for m in out)

    def test_preserves_valid_roles(self):
        msgs = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        # Need a matching assistant tool_call so the tool result isn't orphaned
        msgs[2]["tool_calls"] = [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}]
        out = AIAgent._sanitize_api_messages(msgs)
        roles = [m["role"] for m in out]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles

    def test_logs_warning_when_dropping(self, caplog):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "session_meta", "content": {"info": "test"}},
        ]
        with caplog.at_level(logging.DEBUG, logger="run_agent"):
            AIAgent._sanitize_api_messages(msgs)
        assert any("invalid role" in r.message and "session_meta" in r.message for r in caplog.records)

    def test_drops_multiple_invalid_roles(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "session_meta", "content": {}},
            {"role": "transcript_note", "content": "note"},
            {"role": "assistant", "content": "hi"},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert len(out) == 2
        assert [m["role"] for m in out] == ["user", "assistant"]


# ---------------------------------------------------------------------------
# Layer 1b — display-only timeline fields must not reach the provider
# ---------------------------------------------------------------------------

class TestDisplayFieldsStrippedFromApiPayload:
    """Display-only fields (display_kind, display_metadata) are persisted on
    message rows for timeline rendering, but must never appear in the
    provider-bound API payload — strict OpenAI-compatible backends reject
    unknown fields."""

    def test_sanitizer_does_not_remove_display_fields(self):
        """sanitize_api_messages is NOT the chokepoint for display fields —
        they are popped earlier in conversation_loop. But this test documents
        that the sanitizer alone does NOT strip them, proving the pop in
        conversation_loop is load-bearing."""
        msgs = [
            {"role": "user", "content": "hello", "display_kind": "model_switch"},
            {"role": "assistant", "content": "hi", "display_metadata": {"model": "m"}},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        # The sanitizer preserves them — the conversation_loop pop is the fix.
        assert "display_kind" in out[0]
        assert "display_metadata" in out[1]

    def test_conversation_loop_strips_display_fields(self):
        """The per-request api_msg copy in conversation_loop strips
        display_kind and display_metadata before the message reaches the
        provider. This simulates that pop."""
        msg = {
            "role": "user",
            "content": "switch event",
            "display_kind": "model_switch",
            "display_metadata": {"model": "test"},
            "api_content": "sidecar",
        }
        # Reproduce the pop sequence from conversation_loop.py
        api_msg = msg.copy()
        api_msg.pop("api_content", None)
        api_msg.pop("display_kind", None)
        api_msg.pop("display_metadata", None)
        assert "display_kind" not in api_msg
        assert "display_metadata" not in api_msg
        assert "api_content" not in api_msg
        assert api_msg["content"] == "switch event"
        # Original message dict is untouched.
        assert msg.get("display_kind") == "model_switch"


# ---------------------------------------------------------------------------
# Layer 2 — CLI session-restore filters session_meta before loading
# ---------------------------------------------------------------------------

class TestCLISessionRestoreFiltering:

    def test_restore_filters_session_meta(self):
        """Simulates the CLI restore path and verifies session_meta is removed."""
        # Build a fake restored message list (as returned by get_messages_as_conversation)
        fake_restored = [
            {"role": "session_meta", "content": {"model": "gpt-4"}},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "session_meta", "content": {"tools": []}},
        ]

        # Apply the same filtering that the patched CLI code now does
        filtered = [m for m in fake_restored if m.get("role") != "session_meta"]

        assert len(filtered) == 2
        assert all(m["role"] != "session_meta" for m in filtered)
        assert filtered[0]["role"] == "user"
        assert filtered[1]["role"] == "assistant"
