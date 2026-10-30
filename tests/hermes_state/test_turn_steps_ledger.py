"""Behaviour contracts for the ``turn_steps`` ledger (scheduled-tasks design, phase P0).

The ledger is written in the same transaction as the usage delta it describes, so tokens and
cost reconcile exactly with ``session_model_usage``; ``step_seq`` is dense per session and only
LLM steps tick it. Aggregate rows (background review, named agent runs) collapse N calls into
one row with ``usage_status='aggregated'``.
"""

import sqlite3

import pytest

from hermes_state import SessionDB


def _llm_step(**overrides):
    step = {
        "turn_id": "turn-a",
        "turn_seq": 1,
        "step_index": 1,
        "iteration_uid": "turn-a:it:000000000001",
        "role": "assistant",
        "step_kind": "llm",
        "usage_status": "ok",
        "api_calls": 1,
        "status": "ok",
    }
    step.update(overrides)
    return step


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    database.create_session("s", "cli", model="main-model")
    try:
        yield database
    finally:
        database.close()


def test_step_row_commits_with_the_usage_delta_and_advances_the_clocks(db):
    db.update_token_counts(
        "s", input_tokens=10, output_tokens=5, cache_read_tokens=3, model="main-model",
        billing_provider="prov", api_call_count=1, estimated_cost_usd=0.25,
        step=_llm_step(step_index=1, tool_call_count=1, tool_names=["bash"], total_tokens=15),
    )

    rows = db.turn_steps_for_session("s")
    assert len(rows) == 1
    row = rows[0]
    assert (row["step_seq"], row["turn_seq"], row["role"], row["step_kind"]) == (1, 1, "assistant", "llm")
    assert (row["input_tokens"], row["output_tokens"], row["cache_read_tokens"]) == (10, 5, 3)
    assert row["total_tokens"] == 15
    assert row["tool_names"] == '["bash"]'
    assert row["estimated_cost_usd"] == pytest.approx(0.25)
    assert row["usage_status"] == "ok"
    session = db.get_session("s")
    assert (session["step_seq"], session["turn_seq"], session["api_call_count"]) == (1, 1, 1)

    # Invariant 1: the ledger reconciles exactly with the aggregate it mirrors.
    with db._read_ctx() as conn:
        sums = conn.execute(
            "SELECT model, billing_provider, task, SUM(input_tokens), SUM(output_tokens),"
            " SUM(cache_read_tokens), SUM(estimated_cost_usd) FROM turn_steps GROUP BY 1, 2, 3"
        ).fetchone()
        usage = conn.execute(
            "SELECT model, billing_provider, task, input_tokens, output_tokens, cache_read_tokens,"
            " estimated_cost_usd FROM session_model_usage"
        ).fetchone()
    assert tuple(sums) == tuple(usage)


def test_two_responses_in_one_iteration_get_distinct_step_seqs(db):
    for index in (1, 2):
        db.update_token_counts(
            "s", input_tokens=index, model="main-model", billing_provider="prov", api_call_count=1,
            step=_llm_step(step_index=index, iteration_uid="turn-a:it:shared"),
        )
    rows = db.turn_steps_for_session("s")
    assert [r["step_seq"] for r in rows] == [2, 1]  # newest first
    assert {r["iteration_uid"] for r in rows} == {"turn-a:it:shared"}
    # Identity is step_seq; step_index repeats/skips after refunds and is never a dedupe key.
    db.update_token_counts(
        "s", input_tokens=1, model="main-model", billing_provider="prov", api_call_count=1,
        step=_llm_step(step_index=1, iteration_uid="turn-a:it:rerun"),
    )
    assert [r["step_seq"] for r in db.turn_steps_for_session("s")] == [3, 2, 1]


def test_auxiliary_rows_never_tick_the_step_clock(db):
    db.record_auxiliary_usage("s", "vision", model="aux-model", billing_provider="auxp",
                              input_tokens=7, output_tokens=3, api_call_count=1)
    db.record_auxiliary_usage("s", "background_review", model="aux-model", billing_provider="auxp",
                              input_tokens=70, output_tokens=30, estimated_cost_usd=0.5, api_call_count=7)
    db.record_auxiliary_usage("s", "agent_run", model="aux-model", billing_provider="auxp",
                              input_tokens=4, output_tokens=2, api_call_count=3)

    rows = {r["task"]: r for r in db.turn_steps_for_session("s")}
    assert set(rows) == {"vision", "background_review", "agent_run"}
    for row in rows.values():
        assert (row["role"], row["step_kind"], row["step_seq"], row["turn_id"]) == ("aux", "aux", None, None)
    assert (rows["vision"]["usage_status"], rows["vision"]["api_calls"]) == ("ok", 1)
    assert (rows["background_review"]["usage_status"], rows["background_review"]["api_calls"]) == ("aggregated", 7)
    assert db.get_session("s")["step_seq"] == 0

    # Invariant 3: aux api_calls reconcile with the per-task usage rows.
    with db._read_ctx() as conn:
        ledger = conn.execute(
            "SELECT COALESCE(SUM(api_calls), 0) FROM turn_steps WHERE step_kind='aux'"
        ).fetchone()[0]
        usage = conn.execute(
            "SELECT COALESCE(SUM(api_call_count), 0) FROM session_model_usage WHERE task != ''"
        ).fetchone()[0]
    assert ledger == usage == 11


def test_usage_less_response_writes_missing_row_without_bumping_api_calls(db):
    db.queue_token_counts("s", step=_llm_step(step_index=1, usage_status="missing"))
    db.queue_token_counts(
        "s", input_tokens=4, model="main-model", billing_provider="prov", api_call_count=1,
        step=_llm_step(step_index=2),
    )
    assert db.flush_token_counts() is True

    rows = db.turn_steps_for_session("s")
    assert [(r["step_seq"], r["usage_status"]) for r in rows] == [(2, "ok"), (1, "missing")]
    assert db.get_session("s")["api_call_count"] == 1
    # Invariant 2: LLM api_calls == sessions.api_call_count + count(missing).
    with db._read_ctx() as conn:
        calls = conn.execute(
            "SELECT SUM(api_calls) FROM turn_steps WHERE step_kind='llm' AND turn_id IS NOT NULL"
        ).fetchone()[0]
        missing = conn.execute(
            "SELECT COUNT(*) FROM turn_steps WHERE usage_status='missing'"
        ).fetchone()[0]
    assert calls == db.get_session("s")["api_call_count"] + missing


def test_absolute_writes_carry_no_step(db):
    db.update_token_counts(
        "s", input_tokens=99, model="main-model", api_call_count=1, absolute=True, step=_llm_step(),
    )
    assert db.turn_steps_for_session("s") == []
    assert db.get_session("s")["step_seq"] == 0


def test_managed_worker_receipt_accepts_a_step_and_rejects_unknown_fields(db):
    import hermes_state_runtime as rt

    epoch = rt.begin_runtime_epoch(db, instance_id="owner")
    scope = dict(epoch=epoch, execution_id="w", session_id="s", generation=0)
    rt.register_worker_execution(db, **scope, kind="compute", adoption_secret="private")
    step = _llm_step(step_index=1, total_tokens=12)
    rt.mutate_worker_execution(
        db, **scope, sequence=1, operation="usage.main",
        payload={"input_tokens": 9, "output_tokens": 3, "api_call_count": 1,
                 "model": "main-model", "billing_provider": "prov", "step": step},
    )
    rt.mutate_worker_execution(
        db, **scope, sequence=2, operation="usage.auxiliary",
        payload={"task": "compression", "model": "aux-model", "input_tokens": 2, "api_call_count": 2},
    )
    rows = {r["step_kind"]: r for r in db.turn_steps_for_session("s")}
    assert rows["llm"]["step_seq"] == 1
    assert rows["llm"]["input_tokens"] == 9
    assert rows["aux"]["usage_status"] == "aggregated"
    assert rows["aux"]["api_calls"] == 2

    with pytest.raises(rt.RuntimeStoreError, match="invalid_params"):
        rt.mutate_worker_execution(
            db, **scope, sequence=3, operation="usage.main",
            payload={"input_tokens": 1, "api_call_count": 1, "step": {"turn_id": "x", "bogus": 1}},
        )


def test_queue_after_close_with_step_lands_the_ledger_row(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    database.create_session("s", "cli", model="main-model")
    database.close()  # writer stopped: the fallback path applies the delta inline
    database.queue_token_counts(
        "s", input_tokens=4, model="main-model", billing_provider="prov", api_call_count=1,
        step=_llm_step(),
    )
    [row] = database.turn_steps_for_session("s")
    assert (row["step_seq"], row["input_tokens"], row["usage_status"]) == (1, 4, "ok")
    database.close()


def test_reconcile_adds_clocks_and_ledger_to_a_legacy_store(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        legacy = sqlite3.connect(":memory:")
        legacy.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        db._reconcile_columns(legacy.cursor())
        columns = {row[1] for row in legacy.execute("PRAGMA table_info(sessions)")}
        assert {"turn_seq", "step_seq"} <= columns
        legacy.close()
    finally:
        db.close()
