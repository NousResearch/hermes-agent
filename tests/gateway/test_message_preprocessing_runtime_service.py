"""DEAD path: not imported by gateway/run.py — contract-only unit tests.

See gateway/RUNTIME_SERVICES.md. Marked dead_runtime_service so suites can
optionally filter with ``-m "not dead_runtime_service"``; default still runs.
"""
import pytest

pytestmark = pytest.mark.dead_runtime_service

from gateway.message_preprocessing_runtime_service import (
    is_shared_thread_session,
    message_history_contains_reply_snippet,
    prepend_reply_context_if_missing,
    prepend_shared_thread_sender,
)
from gateway.config import Platform
from gateway.session import SessionSource


def test_prepend_reply_context_if_missing_adds_reply_note_when_history_lacks_quote():
    text = prepend_reply_context_if_missing(
        message_text="new message",
        reply_to_text="older context",
        reply_to_message_id="42",
        history=[{"role": "assistant", "content": "different"}],
    )

    assert text == '[Replying to: "older context"]\n\nnew message'


def test_prepend_reply_context_if_missing_skips_when_history_already_has_quote():
    text = prepend_reply_context_if_missing(
        message_text="new message",
        reply_to_text="older context",
        reply_to_message_id="42",
        history=[{"role": "assistant", "content": "older context is already here"}],
    )

    assert text == "new message"


def test_prepend_shared_thread_sender_only_prefixes_non_dm_shared_threads():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="group",
        thread_id="topic-1",
        user_name="Alice",
    )

    assert is_shared_thread_session(source=source, thread_sessions_per_user=False) is True
    assert (
        prepend_shared_thread_sender(
            message_text="hello",
            user_name=source.user_name,
            shared_thread=True,
        )
        == "[Alice] hello"
    )


def test_message_history_contains_reply_snippet_ignores_empty_probe():
    assert message_history_contains_reply_snippet([], reply_snippet="") is False
