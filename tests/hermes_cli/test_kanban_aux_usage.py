"""Kanban specify/decompose run headless (no agent turn), so no ambient accounting context
exists and the aux-usage chokepoint silently dropped their ``session_model_usage`` rows —
``kanban_decomposer`` was the most frequent aux lane in an auto-decompose install yet never
showed up in per-task cost views, while turn-bound tasks (title_generation, approval,
compression) did (#118595). ``_call_aux`` now binds a stable hidden ops session for these
calls; usage rows land with the right ``task`` and accounting trouble never blocks the call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import kanban_decompose as decompose
from hermes_cli import kanban_specify as specify


def _mk_response(model="aux-model", prompt=100, completion=20):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    )


def _chokepoint_call_llm(seen_tasks: list):
    """Stand in for ``agent.auxiliary_client.call_llm`` and replay its usage chokepoint
    (``record_aux_usage`` at the response-validation point) so the test exercises the
    real binding → recording → row-landing chain."""
    def call_llm(**kwargs):
        from agent.aux_accounting import record_aux_usage
        seen_tasks.append(kwargs.get("task"))
        response = _mk_response()
        record_aux_usage(response, kwargs.get("task"))
        return response
    return call_llm


def _usage_rows(db):
    with db._lock:
        rows = db._conn.execute(
            "SELECT session_id, task, model, input_tokens, output_tokens FROM session_model_usage"
        ).fetchall()
    return [dict(r) for r in rows]


@pytest.mark.parametrize("caller,expected_task", [
    (specify._call_aux, "triage_specifier"),
    (decompose._call_aux, "kanban_decomposer"),
])
def test_headless_kanban_aux_call_records_task_usage_row(caller, expected_task, monkeypatch, tmp_path):
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    seen_tasks: list = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("agent.auxiliary_client.call_llm", _chokepoint_call_llm(seen_tasks))
        reply, reason = caller(
            "specify", "t_123", aux_task=expected_task, system="s", user="u", max_tokens=10, timeout=5)
    assert (reply, reason) == ("ok", "")
    assert seen_tasks == [expected_task]

    from agent.aux_accounting import accounting_context_active
    assert not accounting_context_active()  # nothing leaks past the call

    db = SessionDB()  # same HERMES_HOME → the db the headless binder opened
    try:
        rows = _usage_rows(db)
        assert rows == [{
            "session_id": specify._OPS_ACCOUNTING_SESSION_ID,
            "task": expected_task,
            "model": "aux-model",
            "input_tokens": 100,
            "output_tokens": 20,
        }]
        with db._lock:
            ops = db._conn.execute(
                "SELECT source, hidden FROM sessions WHERE id = ?",
                (specify._OPS_ACCOUNTING_SESSION_ID,),
            ).fetchone()
        assert ops is not None
        assert ops["source"] == "kanban"  # already a state-owned lifecycle source
        assert ops["hidden"] == 1  # must not appear in the user's session list
    finally:
        db.close()


def test_in_turn_caller_keeps_its_session(tmp_path):
    """A bound ambient context (agent turn / title thread) must win; no ops session row."""
    from agent.aux_accounting import accounting_context_active, reset_accounting_context, set_accounting_context
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s1", source="cli")
        seen_tasks: list = []
        token = set_accounting_context(db, "s1")
        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr("agent.auxiliary_client.call_llm", _chokepoint_call_llm(seen_tasks))
                reply, reason = specify._call_aux(
                    "specify", "t_123", aux_task="triage_specifier", system="s", user="u",
                    max_tokens=10, timeout=5)
        finally:
            reset_accounting_context(token)
        assert (reply, reason) == ("ok", "")
        assert not accounting_context_active()

        rows = _usage_rows(db)
        assert [r["session_id"] for r in rows] == ["s1"]
        assert rows[0]["task"] == "triage_specifier"
        with db._lock:
            assert db._conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (specify._OPS_ACCOUNTING_SESSION_ID,)
            ).fetchone() is None
    finally:
        db.close()


def test_accounting_failure_never_blocks_the_call(monkeypatch, tmp_path):
    """A broken state db (open failure) degrades to an unrecorded call, never an error."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _boom(*args, **kwargs):
        raise RuntimeError("state db unavailable")

    seen_tasks: list = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("hermes_state.SessionDB", _boom)
        mp.setattr("agent.auxiliary_client.call_llm", _chokepoint_call_llm(seen_tasks))
        reply, reason = specify._call_aux(
            "specify", "t_123", aux_task="triage_specifier", system="s", user="u", max_tokens=10, timeout=5)
    assert (reply, reason) == ("ok", "")
