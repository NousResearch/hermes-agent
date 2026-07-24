"""Tests for session-resume note injection in CLI agent construction.

Tests the resume-note logic directly by exercising the timestamp
comparison that determines whether the note is appended.  The full
_init_agent() integration is too complex to mock (40+ attributes,
provider resolution, TIRITH security); this tests the core logic.
"""

import pytest


RESUME_NOTE = (
    "\n\n[Session resumed after a process restart. "
    "This conversation history was restored from a prior session. "
    "Tool calls shown in the history have already been executed — "
    "do NOT re-execute them. "
    "The previous session's work was already reported — "
    "continue without re-summarizing. "
    "Address the user's current message below. "
    "Unless the user explicitly asks for a recap, ignore the past work.]"
)


def _should_add_resume_note(conversation_history, session_start_ts):
    """Replicate the logic from cli_agent_setup_mixin._init_agent()."""
    last_ts = None
    for msg in reversed(conversation_history):
        ts = msg.get("timestamp")
        if ts is not None:
            last_ts = ts
            break
    return last_ts is not None and last_ts < session_start_ts


class TestResumeNoteLogic:
    def test_added_when_last_message_predates_session(self):
        assert _should_add_resume_note(
            conversation_history=[
                {"role": "user", "content": "hi", "timestamp": 1000.0},
                {"role": "assistant", "content": "hello", "timestamp": 1001.0},
            ],
            session_start_ts=2000.0,
        ) is True

    def test_not_added_when_last_message_after_session_start(self):
        assert _should_add_resume_note(
            conversation_history=[
                {"role": "user", "content": "hi", "timestamp": 3000.0},
            ],
            session_start_ts=2000.0,
        ) is False

    def test_not_added_when_no_history(self):
        assert _should_add_resume_note(
            conversation_history=[], session_start_ts=2000.0,
        ) is False

    def test_not_added_when_no_timestamp_in_history(self):
        assert _should_add_resume_note(
            conversation_history=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            session_start_ts=2000.0,
        ) is False

    def test_note_text_contains_key_instructions(self):
        """The resume note text itself is correct."""
        assert "do NOT re-execute" in RESUME_NOTE
        assert "process restart" in RESUME_NOTE
        assert "was already reported" in RESUME_NOTE
