import asyncio
import time

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import kanban_db as kb


def _event(text="accept", *, reply_to="msg-1", user_id="user-1", chat_id="chat-1", thread_id=None):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            user_id=user_id,
            user_name="Gibs",
            thread_id=thread_id,
        ),
        message_id="incoming-1",
        reply_to_message_id=reply_to,
    )


def _runner():
    return GatewayRunner.__new__(GatewayRunner)


def _seed_target(*, payload=None, user_id="user-1", expires_at=None):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="rough intake", assignee="triage", triage=True)
        kb.add_reply_target(
            conn,
            platform="telegram",
            chat_id="chat-1",
            message_id="msg-1",
            task_id=tid,
            proposal_id="p_20260523_001",
            user_id=user_id,
            payload=payload or {
                "title": "specified intake",
                "body": "Full task body from the proposal.",
                "assignee": "worker",
            },
            expires_at=expires_at,
        )
        return tid
    finally:
        conn.close()


def _target_for(tid):
    conn = kb.connect()
    try:
        return kb.list_reply_targets(conn, task_id=tid)[0]
    finally:
        conn.close()


def test_direct_reply_accept_applies_triage_proposal(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "accept.db"))
    kb.init_db()
    tid = _seed_target()

    ack = asyncio.run(_runner()._maybe_handle_kanban_reply_target(_event("accept")))

    assert ack == "Accepted — I applied that Kanban proposal."
    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task.title == "specified intake"
        assert task.body == "Full task body from the proposal."
        assert task.assignee == "worker"
        assert task.status == "ready"
        assert kb.list_reply_targets(conn, task_id=tid)[0]["status"] == "accepted"
        comments = [c.body for c in kb.list_comments(conn, tid)]
        assert any("Accepted triage proposal p_20260523_001" in body for body in comments)
    finally:
        conn.close()


def test_direct_reply_reject_marks_proposal_and_comments(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "reject.db"))
    kb.init_db()
    tid = _seed_target()

    ack = asyncio.run(_runner()._maybe_handle_kanban_reply_target(_event("reject: not enough detail")))

    assert ack == "Noted — I rejected that Kanban proposal and saved your feedback."
    conn = kb.connect()
    try:
        assert kb.get_task(conn, tid).status == "triage"
        assert kb.list_reply_targets(conn, task_id=tid)[0]["status"] == "rejected"
        comments = [c.body for c in kb.list_comments(conn, tid)]
        assert any("not enough detail" in body for body in comments)
    finally:
        conn.close()


def test_direct_reply_unknown_anchor_falls_through(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "unknown.db"))
    kb.init_db()

    assert asyncio.run(_runner()._maybe_handle_kanban_reply_target(_event("accept"))) is None


def test_direct_reply_invalid_action_sends_help_without_updating(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "invalid.db"))
    kb.init_db()
    tid = _seed_target()

    ack = asyncio.run(_runner()._maybe_handle_kanban_reply_target(_event("maybe")))

    assert "Reply with `accept`" in ack
    assert _target_for(tid)["status"] == "active"


def test_direct_reply_unauthorized_user_fails_safely(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "unauthorized.db"))
    kb.init_db()
    tid = _seed_target(user_id="owner")

    ack = asyncio.run(_runner()._maybe_handle_kanban_reply_target(_event("accept", user_id="intruder")))

    assert "different user" in ack
    assert _target_for(tid)["status"] == "active"


def test_direct_reply_expired_target_is_not_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "expired.db"))
    kb.init_db()
    tid = _seed_target(expires_at=int(time.time()) - 1)

    ack = asyncio.run(_runner()._maybe_handle_kanban_reply_target(_event("accept")))

    assert "expired" in ack
    conn = kb.connect()
    try:
        assert kb.get_task(conn, tid).status == "triage"
        assert kb.list_reply_targets(conn, task_id=tid)[0]["status"] == "expired"
    finally:
        conn.close()


def test_direct_reply_accepts_explicit_proposal_id_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "proposal-prefix.db"))
    kb.init_db()
    tid = _seed_target()

    ack = asyncio.run(_runner()._maybe_handle_kanban_reply_target(_event("p_20260523_001 accept")))

    assert ack == "Accepted — I applied that Kanban proposal."
    assert _target_for(tid)["status"] == "accepted"
