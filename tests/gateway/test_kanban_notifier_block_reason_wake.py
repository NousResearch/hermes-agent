"""Block reason in the wake synth for blocked / block_loop_detected events.

Non-push (api_server) subscriptions receive ONLY the synthetic wake turn: the
text ping carrying ``⏸ … blocked: <reason>`` is skipped for them. Without the
reason in the synth the woken session learns that the task stopped but not
why. The reason is redacted with the same helper used for review feedback.
"""

import asyncio

from gateway.config import Platform
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn

from tests.gateway.test_kanban_notifier_apiserver_wake import (
    ApiServerLikeAdapter,
    _make_runner,
    _run_one_notifier_tick,
)


def _blocked_sub(reason, *, block_twice=False):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="needs a decision", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="api_server", chat_id="bot-chat")
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
        if block_twice:
            for _ in range(2):
                kb.unblock_task(conn, tid)
                kb.block_task(conn, tid, reason=reason, kind="needs_input")
        return tid
    finally:
        conn.close()


def _completed_sub():
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="finished", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="api_server", chat_id="bot-chat")
        kb.complete_task(conn, tid, summary="all done")
        return tid
    finally:
        conn.close()


def _wake_texts(tmp_path, monkeypatch, make):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "wake.db"))
    kb.init_db()
    make()
    posts = []

    async def fake_self_post(adapter, *, text, session_id):
        posts.append(text)

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)
    runner = _make_runner({Platform.API_SERVER: ApiServerLikeAdapter()})
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    return posts


def test_blocked_wake_carries_block_reason(tmp_path, monkeypatch):
    posts = _wake_texts(
        tmp_path, monkeypatch,
        lambda: _blocked_sub("Should the Queue page show a deferral hint, or stay API-only?"),
    )
    assert len(posts) == 1
    assert "Block reason: Should the Queue page show a deferral hint, or stay API-only?" in posts[0]


def test_blocked_wake_reason_is_redacted_and_bounded(tmp_path, monkeypatch):
    secret = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
    reason = f"token {secret} at /Users/someone/private/notes.txt " + "x" * 400
    posts = _wake_texts(tmp_path, monkeypatch, lambda: _blocked_sub(reason))
    assert len(posts) == 1
    line = next(ln for ln in posts[0].splitlines() if ln.startswith("Block reason: "))
    assert secret not in line
    assert "/Users/someone" not in line and "[local path]" in line
    assert len(line) <= len("Block reason: ") + 160
    assert line.endswith("…")


def test_block_loop_detected_wake_carries_block_reason(tmp_path, monkeypatch):
    posts = _wake_texts(
        tmp_path, monkeypatch,
        lambda: _blocked_sub("waiting on the operator to pick a mode", block_twice=True),
    )
    assert posts, "expected at least one wake"
    assert any("Block reason: waiting on the operator to pick a mode" in p for p in posts)


def test_non_block_wake_has_no_block_reason_line(tmp_path, monkeypatch):
    posts = _wake_texts(tmp_path, monkeypatch, _completed_sub)
    assert len(posts) == 1
    assert "Block reason:" not in posts[0]


def test_blocked_without_reason_adds_no_line(tmp_path, monkeypatch):
    posts = _wake_texts(tmp_path, monkeypatch, lambda: _blocked_sub(None))
    assert len(posts) == 1
    assert "Block reason:" not in posts[0]
