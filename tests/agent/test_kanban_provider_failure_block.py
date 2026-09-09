"""Kanban worker must block on a terminal provider failure, not exit 0.

When every model/credential is exhausted, ``run_conversation`` returns
``failed=True`` from ``_failed_turn_result`` — an early return that never
reaches ``finalize_turn``. Dispatcher-spawned workers use ``chat -q``
and currently ignore that result, so the process exits 0 and the
dispatcher records a protocol violation instead of the last provider
error.

``maybe_block_kanban_on_provider_failure`` is the single hook both the
quiet and human-facing one-shot paths call so the card is blocked with
the last error text.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import FailoverReason
from agent.turn_recovery import _failed_turn_result, max_retries_exhausted_result
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claimed(conn, *, assignee="engineer"):
    task_id = kb.create_task(conn, title="work", assignee=assignee)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
    task = kb.claim_task(conn, task_id, claimer=assignee)
    assert task is not None and task.current_run_id is not None
    return task_id, task.current_run_id


def test_failed_turn_result_blocks_kanban_task_with_last_error(kanban_home, monkeypatch):
    """Provider-wall result must close the run as blocked with the last error."""
    from agent.kanban_stop import maybe_block_kanban_on_provider_failure

    with kbc.connect_closing() as conn:
        task_id, run_id = _claimed(conn)

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    last_error = (
        "Error code: 429 - Token Plan usage limit reached: "
        "Upgrade your Token Plan or purchase Credits for more usage."
    )
    result = _failed_turn_result(
        last_error, messages=[], api_call_count=3, error=last_error,
    )
    result["failure_reason"] = "rate_limit"

    maybe_block_kanban_on_provider_failure(result)

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        run = kb.get_run(conn, run_id)
        events = kb.list_events(conn, task_id)
        assert task.status == "blocked"
        assert task.block_kind == "transient"
        assert run.outcome == "blocked"
        assert run.ended_at is not None
        combined = " ".join(
            filter(None, [run.summary, run.error, *(str(e.payload) for e in events)])
        )
        assert last_error in combined


def test_failed_turn_result_is_noop_outside_kanban_worker(kanban_home, monkeypatch):
    from agent.kanban_stop import maybe_block_kanban_on_provider_failure

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    result = _failed_turn_result("boom", messages=[], api_call_count=1, error="boom")
    maybe_block_kanban_on_provider_failure(result)  # must not raise


def test_successful_turn_does_not_block(kanban_home, monkeypatch):
    from agent.kanban_stop import maybe_block_kanban_on_provider_failure

    with kbc.connect_closing() as conn:
        task_id, run_id = _claimed(conn)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    maybe_block_kanban_on_provider_failure(
        {"final_response": "ok", "completed": True, "failed": False}
    )

    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "running"
        assert kb.get_run(conn, run_id).ended_at is None


def test_max_retries_exhausted_result_blocks_with_classified_error(
    kanban_home, monkeypatch,
):
    from agent.kanban_stop import maybe_block_kanban_on_provider_failure

    with kbc.connect_closing() as conn:
        task_id, run_id = _claimed(conn)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    last_error = "personal-team-blocked: all model calls failed"
    classified = SimpleNamespace(
        reason=FailoverReason.rate_limit,
        retryable=True,
        billing_unverified=False,
    )
    agent = SimpleNamespace(
        log_prefix="",
        _flush_status_buffer=lambda: None,
        _summarize_api_error=lambda _e: last_error,
        _emit_status=lambda *_a, **_k: None,
        _dump_api_request_debug=lambda *_a, **_k: None,
        _persist_session=lambda *_a, **_k: None,
        _safe_print=lambda *_a, **_k: None,
    )
    with patch("agent.turn_recovery._vlines"):
        result = max_retries_exhausted_result(
            agent,
            Exception(last_error),
            classified,
            max_retries=8,
            is_rate_limited=True,
            error_msg=last_error,
            api_kwargs=None,
            api_messages=[],
            messages=[],
            conversation_history=[],
            api_call_count=8,
            approx_tokens=0,
            provider="xai",
            base_url="https://api.x.ai",
            model="grok-4",
        )

    maybe_block_kanban_on_provider_failure(result)

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        run = kb.get_run(conn, run_id)
        events = kb.list_events(conn, task_id)
        assert task.status == "blocked"
        assert run.outcome == "blocked"
        combined = " ".join(
            filter(None, [run.summary, run.error, *(str(e.payload) for e in events)])
        )
        assert last_error in combined


def test_received_provider_response_skips_block(kanban_home, monkeypatch):
    from agent.kanban_stop import maybe_block_kanban_on_provider_failure

    with kbc.connect_closing() as conn:
        task_id, run_id = _claimed(conn)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)

    last_error = "Error code: 429 - usage limit reached"
    result = _failed_turn_result(last_error, messages=[], api_call_count=1, error=last_error)
    result["failure_reason"] = "rate_limit"
    maybe_block_kanban_on_provider_failure(result, received_provider_response=True)

    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "running"
        assert kb.get_run(conn, run_id).ended_at is None
