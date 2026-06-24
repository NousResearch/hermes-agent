"""Tests for native /new same-thread rebase handoff behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="guild-thread",
        chat_type="thread",
        user_id="user-1",
        user_name="Dionysus",
        thread_id="thread-1",
    )


def _event(text: str = "/new") -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_source(),
        message_id="msg-1",
    )


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._pending_same_thread_rebase_context = {}
    return runner


class _FakeRebaseModule:
    @staticmethod
    def build_rebase_dry_run_capsule(**kwargs):
        assert kwargs["dry_run"] is False
        assert kwargs["trigger_text"] == ""
        return {
            "capsule": {"latest_user_request": "keep going"},
            "json_path": "/tmp/rebase.json",
            "markdown_path": "/tmp/rebase.md",
        }

    @staticmethod
    def capsule_to_handoff_context(capsule, *, json_path: str = "", markdown_path: str = "") -> str:
        assert capsule == {"latest_user_request": "keep going"}
        assert json_path == "/tmp/rebase.json"
        assert markdown_path == "/tmp/rebase.md"
        return "[Hermes same-thread session rebase capsule]\nkeep going"


def test_remember_native_new_rebase_handoff_stores_one_shot_capsule(monkeypatch):
    runner = _runner()
    event = _event()
    session_key = build_session_key(event.source)
    monkeypatch.setattr(
        runner,
        "_load_same_thread_rebase_module",
        lambda: _FakeRebaseModule,
    )

    runner._remember_native_new_rebase_handoff(
        event=event,
        session_key=session_key,
        old_session_id="old-session-id",
    )

    pending = runner._pending_same_thread_rebase_context[session_key]
    assert pending["handoff_context"].startswith("[Hermes same-thread session rebase capsule]")
    assert pending["old_session_id"] == "old-session-id"
    assert pending["json_path"] == "/tmp/rebase.json"
    assert pending["markdown_path"] == "/tmp/rebase.md"
    assert pending["created_at"].endswith("Z")


def test_apply_pending_native_new_rebase_context_prepends_and_consumes():
    runner = _runner()
    event = _event("continue")
    event.channel_context = "existing context"
    session_key = build_session_key(event.source)
    runner._pending_same_thread_rebase_context[session_key] = {
        "handoff_context": "[Hermes same-thread session rebase capsule]\nkeep going",
        "old_session_id": "old-session-id",
        "json_path": "/tmp/rebase.json",
    }

    applied = runner._apply_pending_native_new_rebase_context(event, session_key)

    assert applied is True
    assert event.channel_context == (
        "[Hermes same-thread session rebase capsule]\nkeep going\n\nexisting context"
    )
    assert session_key not in runner._pending_same_thread_rebase_context


def test_apply_pending_native_new_rebase_context_is_one_shot():
    runner = _runner()
    event = _event("continue")
    session_key = build_session_key(event.source)
    runner._pending_same_thread_rebase_context[session_key] = {
        "handoff_context": "capsule",
    }

    assert runner._apply_pending_native_new_rebase_context(event, session_key) is True
    second_event = _event("again")
    assert runner._apply_pending_native_new_rebase_context(second_event, session_key) is False
    assert getattr(second_event, "channel_context", None) in (None, "")


def test_remember_native_new_rebase_handoff_is_fail_soft(monkeypatch):
    runner = _runner()
    event = _event()
    session_key = build_session_key(event.source)

    def _boom():
        raise RuntimeError("helper unavailable")

    monkeypatch.setattr(runner, "_load_same_thread_rebase_module", _boom)

    runner._remember_native_new_rebase_handoff(
        event=event,
        session_key=session_key,
        old_session_id="old-session-id",
    )

    assert runner._pending_same_thread_rebase_context == {}
