"""Tests for thread-parallel interrupt keys and parent-context bootstrap."""

from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _discord_thread_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-99",
        chat_type="thread",
        user_id="u1",
        user_name="tester",
        thread_id="thread-99",
    )


def test_interrupt_session_key_includes_thread_id():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="group-1",
        chat_type="group",
        thread_id="topic-7",
    )

    key = BasePlatformAdapter._interrupt_session_key(source)

    assert key == "group-1:thread:topic-7"


def test_derive_parent_thread_source_for_discord_thread_parent_id():
    source = _discord_thread_source()
    event = MessageEvent(
        text="hi",
        source=source,
        raw_message=SimpleNamespace(channel=SimpleNamespace(parent_id="chan-1")),
    )

    parent = GatewayRunner._derive_parent_thread_source(source, event)

    assert parent is not None
    assert parent.chat_id == "chan-1"
    assert parent.chat_type == "group"
    assert parent.thread_id is None


def test_build_thread_bootstrap_context_uses_parent_recent_turns():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(session_lifecycle=SimpleNamespace(isolate_threads=True))

    source = _discord_thread_source()
    event = MessageEvent(
        text="new thread question",
        source=source,
        raw_message=SimpleNamespace(channel=SimpleNamespace(parent_id="chan-1")),
    )

    parent_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan-1",
        chat_type="group",
        user_id="u1",
        user_name="tester",
    )
    parent_key = build_session_key(parent_source, isolate_threads=True)

    parent_history = [
        {"role": "user", "content": "Need a deploy plan"},
        {"role": "assistant", "content": "Sure. Step 1: run tests"},
        {"role": "user", "content": "Also include rollback details"},
        {"role": "assistant", "content": "Got it, adding rollback checklist."},
    ]

    runner.session_store = SimpleNamespace(
        _entries={parent_key: SimpleNamespace(session_id="sid-parent")},
        load_transcript=lambda _sid: parent_history,
    )

    note = runner._build_thread_bootstrap_context(source, event, history=[])

    assert "New thread detected" in note
    assert "User: Need a deploy plan" in note
    assert "Assistant: Sure. Step 1: run tests" in note


def test_build_thread_bootstrap_context_skips_when_thread_already_has_history():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(session_lifecycle=SimpleNamespace(isolate_threads=True))
    runner.session_store = SimpleNamespace(_entries={}, load_transcript=lambda _sid: [])

    source = _discord_thread_source()
    event = MessageEvent(
        text="follow-up",
        source=source,
        raw_message=SimpleNamespace(channel=SimpleNamespace(parent_id="chan-1")),
    )

    note = runner._build_thread_bootstrap_context(source, event, history=[{"role": "user", "content": "already"}])

    assert note == ""
