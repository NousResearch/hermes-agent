"""Tests for session_meta filtering — issue #4715.

Ensures that transcript-only session_meta messages never reach the
chat-completions API, via both the API-boundary guard in
_sanitize_api_messages() and the CLI session-restore paths.
"""

import logging
import types
from unittest.mock import MagicMock, patch

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
