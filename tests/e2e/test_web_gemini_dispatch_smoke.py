"""End-to-end smoke: web_gemini-scope task through the dispatch lane + executor.

Three layers, everything real except the browser session itself:

1. Dispatch: a real ``dispatch_once`` tick routes an explicitly web_gemini-scoped
   task through the dedicated lane (never the profile-worker path, never the
   default-assignee fallback) and hands it to the spawn hook.
2. Executor: the worker protocol against a stubbed Gemini web session — a real
   prompt is built, the stub "session" echoes a protocol-compliant JSON
   response, and the real parser/ledger/executor apply a comment through
   ``kb.add_comment`` on a real board. Replay is refused (no duplicate comment).
3. Guard: a ``comment_only`` task rejects a terminal action before any side
   effect.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_dispatch import dispatch_once
from hermes_cli.web_gemini import (
    PROTOCOL_VERSION,
    ActionBinding,
    ActionLedger,
    build_prompt,
    parse_response,
)
from hermes_cli.web_gemini_worker import ProtocolError, execute_actions


@pytest.fixture()
def sandbox(monkeypatch, tmp_path):
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home / ".hermes"))
    return {"home": home, "ledger": tmp_path / "ledger" / "actions.sqlite3"}


def _make_web_gemini_task(
    conn: sqlite3.Connection, *, assignee: str = "web-helper", **extra
) -> str:
    return kb.create_task(
        conn,
        title="smoke web-gemini card",
        body="summarize the release notes",
        assignee=assignee,
        execution_scope="web_gemini",
        **extra,
    )


def _dispatch_lane_spawn(conn: sqlite3.Connection, sandbox):
    captured: dict = {}

    def spawn(task, workspace, board=None):
        captured["task"] = task
        captured["workspace"] = workspace
        captured["board"] = board
        return 4242  # web_gemini workers are conversations, not subprocesses

    result = dispatch_once(conn, spawn_fn=spawn, max_spawn=1)
    return result, captured


def _stub_session_response(binding: ActionBinding, prompt: str, actions) -> str:
    """The stubbed Gemini web session: echo the binding exactly, as the
    protocol demands. ``prompt`` is consumed (the real session sees it)."""
    del prompt
    return json.dumps(
        {
            "protocol": PROTOCOL_VERSION,
            "board_id": binding.board_id,
            "task_id": binding.task_id,
            "conversation_id": binding.conversation_id,
            "conversation_url": binding.conversation_url,
            "prompt_hash": binding.prompt_hash,
            "nonce": binding.nonce,
            "actions": list(actions),
        },
        separators=(",", ":"),
    )


def _run_one_turn(
    conn: sqlite3.Connection, sandbox, task_id: str, actions,
    *, nonce: str = "nonce-smoke-1", pre_record_intent: bool = False,
):
    """One full worker turn: prompt -> stubbed session -> parse -> execute.

    ``apply_once`` records the intent itself; ``pre_record_intent`` simulates a
    crash after the intent was durably recorded but before the side effect
    applied (the recovery-refusal case).
    """
    ledger = ActionLedger(sandbox["ledger"])
    prompt, prompt_hash = build_prompt(
        board_id="default",
        task_id=task_id,
        conversation_id="conv-smoke-1",
        nonce=nonce,
        task_title="smoke web-gemini card",
        task_body="summarize the release notes",
    )
    binding = ActionBinding(
        task_id=task_id,
        board_id="default",
        conversation_id="conv-smoke-1",
        conversation_url="https://gemini.example/app/conv-smoke-1",
        prompt_hash=prompt_hash,
        nonce=nonce,
    )
    if pre_record_intent:
        ledger.record_intent(binding)
    response = _stub_session_response(binding, prompt, actions)
    parsed = parse_response(response, expected=binding)
    applied = execute_actions(conn, ledger, parsed, author="web_gemini")
    return ledger, parsed, applied


def test_dispatch_lane_routes_scope_task_and_executor_applies_comment(sandbox):
    conn = kb.connect()
    try:
        task_id = _make_web_gemini_task(conn)
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()

        result, captured = _dispatch_lane_spawn(conn, sandbox)
        assert [tid for tid, _a, _w in result.spawned] == [task_id]
        task = captured["task"]
        assert task.execution_scope == "web_gemini"
        row = conn.execute(
            "SELECT status, worker_pid FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        assert row["status"] == "running"
        assert row["worker_pid"] == 4242

        # Unassigned web_gemini cards are a routing gap for a human: the lane
        # must skip them, never default-assign. No spawn cap here — a capped
        # tick legitimately breaks out of the lane loop once its budget is
        # spent, and t1 is still running (occupying one in-progress slot).
        unassigned = kb.create_task(
            conn, title="no assignee", execution_scope="web_gemini"
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (unassigned,))
        conn.commit()

        def _must_not_spawn(*args, **kwargs):
            raise AssertionError("unassigned scope card must not spawn")

        result2 = dispatch_once(
            conn,
            spawn_fn=_must_not_spawn,
            default_assignee="default",
        )
        assert result2.spawned == []
        assert unassigned in result2.skipped_unassigned

        # One worker turn against the stubbed Gemini session: a real prompt,
        # a protocol-compliant stub response, a real comment on the board.
        ledger, parsed, applied = _run_one_turn(
            conn, sandbox, task_id,
            [{"index": 0, "kind": "comment", "body": "notes summarized"}],
        )
        assert all(r.status == "finalized" and not r.replayed for r in applied)
        comments = kb.list_comments(conn, task_id)
        assert [c.body for c in comments] == ["notes summarized"]

        # Replaying the finalized action replays its receipt — never a
        # duplicate external write.
        _, _, replayed = _run_one_turn(
            conn, sandbox, task_id,
            [{"index": 0, "kind": "comment", "body": "notes summarized"}],
        )
        assert all(r.status == "finalized" and r.replayed for r in replayed)
        assert len(kb.list_comments(conn, task_id)) == 1

        # A crash between durable intent and side effect leaves the action
        # pending: the resumed turn refuses rather than risk duplication.
        with pytest.raises(ProtocolError, match="refusing replay"):
            _run_one_turn(
                conn, sandbox, task_id,
                [{"index": 0, "kind": "comment", "body": "notes summarized"}],
                nonce="nonce-smoke-2",
                pre_record_intent=True,
            )
        assert len(kb.list_comments(conn, task_id)) == 1
    finally:
        conn.close()


def test_comment_only_task_rejects_terminal_action_before_side_effects(sandbox):
    conn = kb.connect()
    try:
        task_id = _make_web_gemini_task(
            conn, web_gemini_action_mode="comment_only"
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()

        with pytest.raises(ProtocolError, match="comment_only"):
            _run_one_turn(
                conn, sandbox, task_id,
                [{"index": 0, "kind": "complete", "result": "done"}],
            )

        # Nothing happened: no completion, no comment, and the card was never
        # claimed by the executor (this test never dispatched a spawn).
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.result is None
        assert kb.list_comments(conn, task_id) == []

        # A single comment action still works under comment_only.
        _run_one_turn(
            conn, sandbox, task_id,
            [{"index": 0, "kind": "comment", "body": "comment-only turn"}],
        )
        assert [c.body for c in kb.list_comments(conn, task_id)] == [
            "comment-only turn"
        ]
    finally:
        conn.close()
