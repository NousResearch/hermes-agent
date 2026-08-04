"""Tests for stale tool-call marker session repair (hermes_state, #78148).

Before the root-cause fix in ``agent.conversation_loop``, a local tool-call
template could emit a bare bracketed marker (e.g. "[memory]") as assistant
content alongside a real tool call. The loop cached that marker as a
fallback and, when the following turn came back empty, replayed it as the
"final response" — persisting it into the session as if the model had
actually answered.

``_strip_stale_tool_call_markers`` is the load-on-read defense-in-depth
that clears any such stray marker content from sessions written before the
fix, so resuming a polluted session doesn't re-teach the model to keep
emitting the marker. Unaffected sessions pass through unchanged.
"""

from hermes_state import (
    _is_stale_tool_call_marker_message,
    _strip_stale_tool_call_markers,
)


class TestIsStaleToolCallMarkerMessage:
    def test_matches_bare_marker_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "[memory]",
            "tool_calls": [{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}],
        }
        assert _is_stale_tool_call_marker_message(msg) is True

    def test_matches_dotted_marker(self):
        msg = {
            "role": "assistant",
            "content": "[foo.bar]",
            "tool_calls": [{"id": "1", "function": {"name": "foo.bar", "arguments": "{}"}}],
        }
        assert _is_stale_tool_call_marker_message(msg) is True

    def test_ignores_marker_without_tool_calls(self):
        # A genuine final response of "[memory]" with no tool call is not
        # the contamination signature — leave it alone.
        msg = {"role": "assistant", "content": "[memory]"}
        assert _is_stale_tool_call_marker_message(msg) is False

    def test_ignores_real_content_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "I'll check that for you.",
            "tool_calls": [{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}],
        }
        assert _is_stale_tool_call_marker_message(msg) is False

    def test_ignores_user_role(self):
        msg = {
            "role": "user",
            "content": "[memory]",
            "tool_calls": [{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}],
        }
        assert _is_stale_tool_call_marker_message(msg) is False


class TestStripStaleToolCallMarkers:
    def test_clears_contaminated_content_keeps_tool_calls(self):
        messages = [
            {"role": "user", "content": "do the full task"},
            {
                "role": "assistant",
                "content": "[memory]",
                "tool_calls": [{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "1"},
        ]
        out = _strip_stale_tool_call_markers(messages)
        assert out[1]["content"] == ""
        # Tool call itself must survive — provider tool_call/result pairing.
        assert out[1]["tool_calls"] == [{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}]

    def test_unaffected_session_passes_through_unchanged(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "It's sunny."},
        ]
        out = _strip_stale_tool_call_markers(messages)
        assert out == messages


class TestGetMessagesAsConversationStripsStaleMarkers:
    """The load-on-read wiring: get_messages_as_conversation must actually
    call _strip_stale_tool_call_markers, so a session polluted with a stale
    "[memory]" marker resumes clean end-to-end (not just the pure helper in
    isolation)."""

    def test_polluted_session_resumes_without_marker(self):
        import tempfile
        from pathlib import Path
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                db.create_session(session_id="s1", source="cli")
                db.append_message("s1", role="user", content="do the full task")
                # Stray contamination written by an older build (pre-#78148 fix).
                db.append_message(
                    "s1", role="assistant", content="[memory]",
                    tool_calls=[{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}],
                )
                db.append_message("s1", role="tool", content="ok", tool_call_id="1")
                db.append_message("s1", role="assistant", content="Here is the result.")

                conv = db.get_messages_as_conversation("s1")
                contents = [m.get("content") for m in conv if m.get("role") == "assistant"]

                assert "[memory]" not in contents
                assert "Here is the result." in contents
            finally:
                db.close()

    def test_clean_session_resumes_unaffected(self):
        import tempfile
        from pathlib import Path
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                db.create_session(session_id="s1", source="cli")
                db.append_message("s1", role="user", content="What's the weather?")
                db.append_message("s1", role="assistant", content="It's sunny.")

                conv = db.get_messages_as_conversation("s1")
                contents = [m.get("content") for m in conv]

                assert contents == ["What's the weather?", "It's sunny."]
            finally:
                db.close()
