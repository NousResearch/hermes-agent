"""Canonical Kanban receipts retain the shared one-shot outcome semantics."""
import json
import os
from types import SimpleNamespace

import pytest

from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing


def _bound_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    path = kb.kanban_db_path()
    with connect_closing(path) as conn:
        task_id = kb.create_task(conn, title="private fixture", assignee="default")
        task = kb.claim_task(conn, task_id, claimer="private-host:owner")
        assert task is not None
        context = {"db": str(path), "task_id": task_id,
                   "run_id": task.current_run_id, "claim_lock": task.claim_lock}
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "worker_bound",
                             {"pid": os.getpid(), "claim_lock": task.claim_lock},
                             run_id=task.current_run_id)
    return path, context


@pytest.mark.parametrize(
    "result, expected",
    [
        ({"completed": True}, 0),
        ({"failed": True}, 1),
        ({"partial": True}, 1),
        ({"completed": False}, 1),
        ({"interrupted": True}, 130),
        (None, 1),
        *[({"failed": True, "failure_reason": reason}, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
          for reason in ("rate_limit", "upstream_rate_limit", "billing", "overloaded", "server_error", "timeout")],
        *[({"failed": True, "failure_reason": reason}, kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE)
          for reason in ("auth", "auth_permanent", "model_not_found", "ssl_cert_verification", "upstream_blocked")],
    ],
)
def test_managed_result_persists_exact_worker_exit(tmp_path, monkeypatch, result, expected):
    path, context = _bound_claim(tmp_path, monkeypatch)
    monkeypatch.setattr(session_kanban, "_run_task_turns", lambda *_args: result)
    frame = {"policy": {"kanban_json": json.dumps(context)}}

    assert session_kanban.run_worker_turns(SimpleNamespace(), frame, []) is result
    assert session_kanban.worker_exit_code(path, context) == expected
    assert session_kanban.worker_exit_code(path, context | {"claim_lock": "foreign"}) == 1


def _credential_failures():
    from hermes_cli.auth import AuthError
    from hermes_cli.auth_constants import CODEX_RATE_LIMITED_CODE
    wrapped = RuntimeError("agent init failed")
    wrapped.__cause__ = AuthError("refresh revoked", provider="nous", relogin_required=True)
    return [
        (AuthError("quota", provider="openai-codex", code=CODEX_RATE_LIMITED_CODE), kb.KANBAN_RATE_LIMIT_EXIT_CODE),
        (AuthError("relogin", provider="nous", relogin_required=True), kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE),
        (wrapped, kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE),
        (RuntimeError("unknown startup failure"), 1),
    ]


@pytest.mark.parametrize("error, expected", _credential_failures())
def test_managed_start_failure_books_the_one_shot_credential_exit(tmp_path, monkeypatch, error, expected):
    """Credentials that fail before any turn (agent construction) keep the one-shot CLI's mapping:
    a quota wall is EX_TEMPFAIL, an explicit re-login is EX_CONFIG, anything else stays 1."""
    path, context = _bound_claim(tmp_path, monkeypatch)
    frame = {"policy": {"kanban_json": json.dumps(context)}}

    session_kanban.record_start_failure(frame, error)
    assert session_kanban.worker_exit_code(path, context) == expected
