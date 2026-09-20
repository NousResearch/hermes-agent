from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.web_gemini import ActionBinding, ActionLedger, ProtocolError, parse_response
from hermes_cli.web_gemini_worker import (
    WEB_GEMINI_SCOPE,
    build_owner_env,
    execute_actions,
    validate_execution_scope,
    _trim_gemini_chrome,
)


def test_execution_scope_is_explicit_and_not_a_profile_alias():
    assert validate_execution_scope(WEB_GEMINI_SCOPE) == WEB_GEMINI_SCOPE
    for value in (None, "", "default", "coder", "openrouter", "gemini"):
        with pytest.raises(ProtocolError):
            validate_execution_scope(value)


def test_execution_scope_persists_and_rejects_unknown_values(tmp_path: Path):
    conn = kb.connect(db_path=tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(
            conn,
            title="scope fixture",
            execution_scope="web_gemini",
            web_gemini_action_mode="comment_only",
            initial_status="blocked",
            block_kind="capability",
            block_reason="fixture",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.execution_scope == "web_gemini"
        assert task.web_gemini_action_mode == "comment_only"
        with pytest.raises(ValueError, match="comment_only"):
            kb.create_task(
                conn,
                title="bad mode",
                execution_scope="hermes",
                web_gemini_action_mode="comment_only",
                initial_status="blocked",
                block_kind="capability",
                block_reason="fixture",
            )
        with pytest.raises(ValueError, match="execution_scope"):
            kb.create_task(
                conn,
                title="bad scope",
                execution_scope="default",
                initial_status="blocked",
                block_kind="capability",
                block_reason="fixture",
            )
    finally:
        conn.close()


def test_dispatch_routes_explicit_web_gemini_scope_without_profile_alias(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    conn = kb.connect(db_path=db_path)
    calls = []
    try:
        task_id = kb.create_task(
            conn,
            title="dispatch scope fixture",
            assignee="web_gemini",
            execution_scope="web_gemini",
            initial_status="running",
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))

        def spawn(task, workspace, *, board=None):
            calls.append((task.id, task.execution_scope, task.assignee, board))
            return 9911

        result = kb.dispatch_once(
            conn,
            spawn_fn=spawn,
            board="fixture-board",
            max_spawn=1,
        )

        assert result.spawned
        assert calls == [(task_id, "web_gemini", "web_gemini", "fixture-board")]
    finally:
        conn.close()


def test_dispatch_never_default_assigns_unassigned_web_gemini_scope(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    conn = kb.connect(db_path=db_path)
    try:
        task_id = kb.create_task(
            conn,
            title="unassigned browser fixture",
            execution_scope="web_gemini",
            initial_status="running",
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *args, **kwargs: pytest.fail("must not spawn"),
            board="fixture-board",
            default_assignee="default",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.assignee is None
        assert task_id in result.skipped_unassigned
    finally:
        conn.close()


def test_rendered_gemini_footer_is_trimmed_but_arbitrary_text_is_not():
    payload = '{"protocol":"probe.v1","value":"OK"}'
    footer = " Flash Your n-gineers.com chats aren’t used to improve our models. Gemini is AI and can make mistakes. Your privacy & Gemini Opens in a new window Check your internet connection and try again"
    assert _trim_gemini_chrome(payload + footer) == payload
    replied_footer = " Flash Your n-gineers.com chats aren’t used to improve our models. Gemini is AI and can make mistakes. Your privacy & Gemini Opens in a new window Gemini replied"
    assert _trim_gemini_chrome(payload + replied_footer) == payload
    assert _trim_gemini_chrome(payload + " unexpected model prose") == payload + " unexpected model prose"
def test_owner_environment_scrubs_api_and_fallback_routes():
    env = build_owner_env(
        {
            "PATH": "safe",
            "OPENROUTER_API_KEY": "[REDACTED]",
            "GOOGLE_API_KEY": "[REDACTED]",
            "GEMINI_API_KEY": "[REDACTED]",
            "OPENAI_API_KEY": "[REDACTED]",
            "HERMES_MODEL": "default",
            "HERMES_PROVIDER": "openrouter",
        },
        profile_dir=Path("C:/isolated/web-gemini/profile"),
    )

    assert env["WEB_GEMINI_SCOPE"] == WEB_GEMINI_SCOPE
    for key in (
        "OPENROUTER_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "HERMES_MODEL",
        "HERMES_PROVIDER",
    ):
        assert key not in env


def test_execute_comment_action_is_bound_and_idempotent(tmp_path: Path):
    conn = kb.connect(db_path=tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(
            conn,
            title="worker fixture",
            body="comment",
            initial_status="blocked",
            block_kind="capability",
            block_reason="fixture",
        )
        binding = ActionBinding(
            task_id=task_id,
            board_id="fixture-board",
            conversation_id="fixture-conversation",
            conversation_url="https://gemini.google.com/app/fixture-conversation",
            prompt_hash="c" * 64,
            nonce="WORKER-NONCE",
            action_index=0,
        )
        parsed = parse_response(
            json.dumps(
                {
                    "protocol": "web_gemini.v1",
                    "board_id": binding.board_id,
                    "task_id": binding.task_id,
                    "conversation_id": binding.conversation_id,
                    "conversation_url": binding.conversation_url,
                    "prompt_hash": binding.prompt_hash,
                    "nonce": binding.nonce,
                    "actions": [{"index": 0, "kind": "comment", "body": "WORKER_OK"}],
                }
            ),
            expected=binding,
        )
        ledger = ActionLedger(tmp_path / "ledger.sqlite3")

        first = execute_actions(
            conn,
            ledger,
            parsed,
            author="web_gemini",
            comment_only=True,
        )
        second = execute_actions(
            conn,
            ledger,
            parsed,
            author="web_gemini",
            comment_only=True,
        )

        comments = kb.list_comments(conn, task_id)
        assert first[0].replayed is False
        assert second[0].replayed is True
        assert len(comments) == 1
        assert comments[0].body == "WORKER_OK"
    finally:
        conn.close()


def test_comment_only_gate_rejects_terminal_or_multiple_actions(tmp_path: Path):
    binding = ActionBinding(
        task_id="t",
        board_id="b",
        conversation_id="c",
        conversation_url="https://gemini.google.com/app/c",
        prompt_hash="d" * 64,
        nonce="N",
    )
    for actions in (
        [{"index": 0, "kind": "complete", "result": "done"}],
        [
            {"index": 0, "kind": "comment", "body": "one"},
            {"index": 1, "kind": "comment", "body": "two"},
        ],
    ):
        parsed = parse_response(
            json.dumps(
                {
                    "protocol": "web_gemini.v1",
                    "board_id": binding.board_id,
                    "task_id": binding.task_id,
                    "conversation_id": binding.conversation_id,
                    "conversation_url": binding.conversation_url,
                    "prompt_hash": binding.prompt_hash,
                    "nonce": binding.nonce,
                    "actions": actions,
                }
            ),
            expected=binding,
        )
        conn = kb.connect(db_path=tmp_path / f"{len(actions)}.db")
        try:
            with pytest.raises(ProtocolError):
                execute_actions(
                    conn,
                    ActionLedger(tmp_path / f"{len(actions)}.ledger.sqlite3"),
                    parsed,
                    author="web_gemini",
                    comment_only=True,
                )
        finally:
            conn.close()
