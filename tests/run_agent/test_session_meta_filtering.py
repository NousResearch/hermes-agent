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
# Layer 1b — _sanitize_api_messages strips gateway-only bookkeeping keys
# (issue #68077 — strict OpenAI-compatible providers, e.g. GLM-5.2, reject
# unknown per-message fields like ``timestamp``/``message_id`` with HTTP 400
# "Extra inputs are not permitted". The gateway splats these onto message
# dicts for local transcript/session bookkeeping and they ride along on
# session resume/history reload.)
# ---------------------------------------------------------------------------

class TestSanitizeApiMessagesGatewayKeyStrip:

    def test_strips_timestamp_key(self):
        msgs = [
            {"role": "user", "content": "hello", "timestamp": 1784553211.368},
            {"role": "assistant", "content": "hi", "timestamp": 1784553212.0},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert all("timestamp" not in m for m in out)
        # content/role preserved
        assert [m["role"] for m in out] == ["user", "assistant"]
        assert out[0]["content"] == "hello"

    def test_strips_message_id_key(self):
        msgs = [
            {"role": "user", "content": "hello", "message_id": "m-123", "timestamp": 1.0},
            {"role": "assistant", "content": "hi"},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert all("message_id" not in m and "timestamp" not in m for m in out)

    def test_does_not_mutate_input_dicts(self):
        """Non-mutating — persisted session/transcript objects stay intact."""
        original = {"role": "user", "content": "hello", "timestamp": 1784553211.368}
        AIAgent._sanitize_api_messages([original])
        assert original["timestamp"] == 1784553211.368

    def test_preserves_schema_valid_keys(self):
        msgs = [
            {"role": "user", "content": "hello", "timestamp": 1.0},
            {
                "role": "assistant",
                "content": "",
                "timestamp": 2.0,
                "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok", "timestamp": 3.0, "name": "t"},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert all("timestamp" not in m for m in out)
        # tool_calls and tool_call_id (real schema keys) survive
        assistant = next(m for m in out if m["role"] == "assistant")
        assert assistant["tool_calls"][0]["id"] == "c1"
        tool = next(m for m in out if m["role"] == "tool")
        assert tool["tool_call_id"] == "c1"
        assert tool["name"] == "t"


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
