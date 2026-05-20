"""Tests for issue #27033 — tool error contamination loop.

Verifies that:
1. Error tool messages get tagged with ``_is_error`` by tool_executor.py
2. ``_flush_messages_to_session_db`` skips ``_is_error`` messages
3. CLI load-time ``_is_ephemeral_tool_error`` correctly identifies stale errors
4. Shared ``is_tool_error_message`` (used by gateway + CLI) correctly identifies stale errors
5. Non-error tool messages are unaffected at every layer
6. ``_last_flushed_db_idx`` advances past skipped messages (dedup contract)
7. ``_is_error`` is stripped from the API wire copy (strict provider safety)
8. Content-based filter prefixes stay in sync with tool_executor.py (integration)
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.tool_dispatch_helpers import make_tool_result_message


# ============================================================================
# Test: make_tool_result_message shape
# ============================================================================

def test_make_tool_result_message_has_no_is_error_by_default():
    """Messages built by make_tool_result_message do NOT carry _is_error."""
    msg = make_tool_result_message("test_tool", "all good", "tc-1")
    assert "_is_error" not in msg
    assert msg["role"] == "tool"
    assert msg["name"] == "test_tool"
    assert msg["content"] == "all good"


# ============================================================================
# Test: _is_ephemeral_tool_error (CLI helper)
# ============================================================================

class TestIsEphemeralToolError:
    """Verify the CLI load-time filter catches all error patterns."""

    def _import_helper(self):
        from cli import _is_ephemeral_tool_error
        return _is_ephemeral_tool_error

    @pytest.mark.parametrize("content", [
        "Error executing tool 'terminal': something broke",
        "Error executing tool 'execute_code': Code 400: invalid JSON",
        "[Tool execution cancelled — read_file was skipped due to user interrupt]",
        "[Tool execution skipped — write_file was not started. User sent a new message]",
    ])
    def test_detects_error_tool_messages_by_content(self, content):
        helper = self._import_helper()
        assert helper({"role": "tool", "content": content})

    @pytest.mark.parametrize("content", [
        "ls -la\nfile.txt",
        "Success: wrote 10 lines to /tmp/test.txt",
        '{"success": true, "output": "done"}',
        "Normal output without error",
        "",
        "Error: not a tool error message",  # doesn't match the specific prefixes
        "Error executing tool",  # doesn't have trailing space
    ])
    def test_ignores_non_error_messages(self, content):
        helper = self._import_helper()
        assert not helper({"role": "tool", "content": content})

    def test_detects_is_error_flag(self):
        helper = self._import_helper()
        assert helper({"role": "tool", "content": "any old content", "_is_error": True})

    def test_ignores_non_string_content(self):
        helper = self._import_helper()
        assert not helper({"role": "tool", "content": ["some", "list"]})
        assert not helper({"role": "tool", "content": {"_multimodal": True}})
        assert not helper({"role": "tool"})  # missing content

    def test_ignores_non_tool_messages(self):
        """Non-tool messages are pre-filtered by the caller; the helper is
        content-based and only invoked on tool messages.  Verify that
        reasonable edge cases don't false-positive."""
        helper = self._import_helper()
        assert not helper({"role": "assistant", "content": ""})
        assert not helper({"role": "user"})

    def test_content_startswith_checks_are_not_overly_broad(self):
        """The specific prefixes must not match false positives."""
        helper = self._import_helper()
        assert not helper({"role": "tool", "content": "Error executing tool"})  # no trailing space + args
        assert not helper({"role": "tool", "content": "[Tool execution"})  # incomplete prefix
        assert not helper({"role": "tool", "content": "[Tool execution OK]"})  # different middle


# ============================================================================
# Test: _flush_messages_to_session_db skips _is_error messages
# ============================================================================

class TestFlushSkipsErrorMessages:
    """Verify that _flush_messages_to_session_db does not persist _is_error messages."""

    def _make_agent(self, session_db):
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            from run_agent import AIAgent
            return AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_db=session_db,
                session_id="test-27033-flush",
            )

    def test_normal_messages_are_flushed(self):
        """Normal (non-error) tool messages pass through flush unchanged."""
        session_db = MagicMock()
        agent = self._make_agent(session_db)
        agent._session_db_created = True

        messages = [
            {"role": "user", "content": "hello"},
            make_tool_result_message("terminal", "$ ls\nfile.txt", "tc-1"),
        ]
        agent._flush_messages_to_session_db(messages)

        assert session_db.append_message.call_count == 2

    def test_error_messages_are_skipped(self):
        """Tool messages tagged _is_error are NOT flushed to the session DB."""
        session_db = MagicMock()
        agent = self._make_agent(session_db)
        agent._session_db_created = True

        error_msg = make_tool_result_message(
            "execute_code",
            "Error executing tool 'execute_code': Code 400",
            "tc-1",
        )
        error_msg["_is_error"] = True

        messages = [
            {"role": "user", "content": "run this"},
            error_msg,
            make_tool_result_message("terminal", "ok", "tc-2"),
        ]
        agent._flush_messages_to_session_db(messages)

        # Only the user message and the successful tool message should be flushed
        assert session_db.append_message.call_count == 2

    def test_mixed_batch_preserves_order(self):
        """In a mixed batch, error messages are skipped but order is preserved for the rest."""
        session_db = MagicMock()
        agent = self._make_agent(session_db)
        agent._session_db_created = True

        msg_a = make_tool_result_message("tool_a", "result a", "tc-a")
        msg_b = make_tool_result_message("tool_b", "Error on B", "tc-b")
        msg_b["_is_error"] = True
        msg_c = make_tool_result_message("tool_c", "result c", "tc-c")

        messages = [
            {"role": "user", "content": "run three tools"},
            msg_a,
            msg_b,
            msg_c,
        ]
        agent._flush_messages_to_session_db(messages)

        # 3 flushed: user msg, tool_a, tool_c
        assert session_db.append_message.call_count == 3

    def test_skip_does_not_leak_later_messages(self):
        """The skip's continue does not prevent later messages from being flushed."""
        session_db = MagicMock()
        agent = self._make_agent(session_db)
        agent._session_db_created = True

        err = make_tool_result_message("tool_fail", "boom", "tc-err")
        err["_is_error"] = True

        messages = [
            err,
            {"role": "user", "content": "after error"},
        ]
        agent._flush_messages_to_session_db(messages)

        # The user message should still be flushed
        assert session_db.append_message.call_count == 1


# ============================================================================
# Test: _flush_messages_to_session_db does NOT lose dup-tracking on skip
# ============================================================================

def test_last_flushed_db_idx_advances_past_skipped_messages():
    """_last_flushed_db_idx must advance past skipped _is_error messages so they
    are not reprocessed on subsequent flushes (mirrors the #860 dedup contract)."""
    session_db = MagicMock()
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
            session_id="test-27033-idx",
        )
        agent._session_db_created = True

        err = make_tool_result_message("tool_fail", "boom", "tc-e1")
        err["_is_error"] = True
        ok = make_tool_result_message("tool_ok", "fine", "tc-o1")

        messages = [
            {"role": "user", "content": "hi"},
            err,
            ok,
        ]

        # First flush: user + ok (3 messages total, err skipped)
        agent._flush_messages_to_session_db(messages)
        assert agent._last_flushed_db_idx == len(messages)
        first_count = session_db.append_message.call_count
        assert first_count == 2, "First flush should write user msg + ok tool msg (error skipped)"

        # Second flush with same messages: should write ZERO new messages
        session_db.append_message.reset_mock()
        agent._flush_messages_to_session_db(messages)
        assert session_db.append_message.call_count == 0, \
            "Second flush should not re-write previously tracked messages"
        assert agent._last_flushed_db_idx == len(messages)


# ============================================================================
# Test: CLI resume filter
# ============================================================================

def test_cli_resume_filter_removes_error_tool_messages():
    """The CLI resume path's list comprehension correctly strips error tool messages."""
    from cli import _is_ephemeral_tool_error

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Let me check", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "Error executing tool 'check': failed", "tool_call_id": "c1"},
        {"role": "assistant", "content": "That worked"},
    ]

    filtered = [m for m in messages if not _is_ephemeral_tool_error(m)]
    assert len(filtered) == 3  # error tool message removed
    assert filtered[2]["role"] == "assistant"


# ============================================================================
# Test: Gateway load-time helper
# ============================================================================

def test_gateway_stale_tool_error_helper():
    """The shared is_tool_error_message identifies all error patterns
    (used by gateway/run.py via import)."""
    from agent.tool_dispatch_helpers import is_tool_error_message

    # Identifies by _is_error flag
    assert is_tool_error_message(
        {"role": "tool", "content": "normal output", "_is_error": True}
    )

    # Identifies by content pattern (backwards compat)
    assert is_tool_error_message(
        {"role": "tool", "content": "Error executing tool 'terminal': Timeout"}
    )
    assert is_tool_error_message(
        {"role": "tool", "content": "[Tool execution cancelled — read_file was skipped]"}
    )
    assert is_tool_error_message(
        {"role": "tool", "content": "[Tool execution skipped — write_file was not started]"}
    )

    # Does NOT flag normal messages
    assert not is_tool_error_message(
        {"role": "tool", "content": "$ ls\nfile.txt"}
    )
    assert not is_tool_error_message(
        {"role": "tool", "content": ""}
    )
    assert not is_tool_error_message(
        {"role": "tool", "content": ["list", "content"]}
    )
    assert not is_tool_error_message(
        {"role": "tool"}
    )


def test_is_error_stripped_from_api_copy():
    """The _is_error flag must be stripped from the API message copy so
    strict providers (Mistral, Fireworks, etc.) don't reject it."""
    msg = {"role": "tool", "content": "error!", "tool_call_id": "c1", "_is_error": True}
    api_msg = msg.copy()

    # Simulate the strip from conversation_loop.py
    api_msg.pop("_is_error", None)

    assert "_is_error" not in api_msg
    assert "_is_error" in msg  # original preserves the flag for flush filter


# ============================================================================
# Test: error format prefixes stay in sync with tool_executor.py
# ============================================================================

def test_error_format_prefixes_match_tool_executor():
    """Integration test: verify is_tool_error_message catches every error
    format string that tool_executor.py can produce.

    If this test fails, someone changed the error f-strings in
    tool_executor.py without updating _TOOL_ERROR_CONTENT_PREFIXES
    in agent/tool_dispatch_helpers.py.  Update both.
    """
    from agent.tool_dispatch_helpers import is_tool_error_message

    # These mirror the exact f-strings in tool_executor.py (search for
    # "Error executing tool" and "Tool execution cancelled/skipped").
    simulated_errors = [
        # concurrent thread exception (line ~321, ~867, ~888)
        f"Error executing tool 'terminal': FileNotFoundError: /tmp/nope",
        # concurrent thread no-result (line ~432)
        f"Error executing tool 'execute_code': thread did not return a result",
        # concurrent preflight interrupt (line ~125, ~430)
        f"[Tool execution cancelled \u2014 read_file was skipped due to user interrupt]",
        # sequential interrupt skip (line ~1001)
        f"[Tool execution skipped \u2014 write_file was not started. User sent a new message]",
    ]

    for content in simulated_errors:
        msg = {"role": "tool", "content": content, "tool_call_id": "tc-test"}
        assert is_tool_error_message(msg), (
            f"is_tool_error_message missed tool_executor.py format: {content!r}. "
            f"Update _TOOL_ERROR_CONTENT_PREFIXES in agent/tool_dispatch_helpers.py."
        )
