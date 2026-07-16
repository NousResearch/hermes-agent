"""Tests for auxiliary usage accounting (issue #23270).

Auxiliary LLM calls (vision, compression, title_generation, ...) record
their token usage into session_model_usage with a ``task`` dimension via
the ambient accounting context (agent/aux_accounting.py), making aux model
spend visible in analytics.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _mk_response(model="aux-model", prompt=100, completion=20):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    )


def _usage_rows(db, session_id):
    with db._lock:
        rows = db._conn.execute(
            "SELECT * FROM session_model_usage WHERE session_id = ? ORDER BY task",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


class TestRecordAuxiliaryUsage:
    def test_records_task_row(self, db):
        db.create_session("s1", source="cli")
        db.record_auxiliary_usage(
            "s1", "vision", model="gemini-3-flash",
            billing_provider="gemini", input_tokens=500, output_tokens=50,
        )
        rows = _usage_rows(db, "s1")
        assert len(rows) == 1
        r = rows[0]
        assert r["task"] == "vision"
        assert r["model"] == "gemini-3-flash"
        assert r["billing_provider"] == "gemini"
        assert r["input_tokens"] == 500
        assert r["output_tokens"] == 50
        assert r["api_call_count"] == 1

    def test_accumulates_same_task_and_model(self, db):
        db.create_session("s1", source="cli")
        for _ in range(3):
            db.record_auxiliary_usage(
                "s1", "compression", model="glm-5", input_tokens=1000, output_tokens=100,
            )
        rows = _usage_rows(db, "s1")
        assert len(rows) == 1
        assert rows[0]["input_tokens"] == 3000
        assert rows[0]["api_call_count"] == 3

    def test_task_rows_do_not_touch_session_counters(self, db):
        """Aux usage must NOT increment sessions.input_tokens — the gateway
        overwrites those with absolute main-loop totals."""
        db.create_session("s1", source="cli")
        db.record_auxiliary_usage("s1", "vision", model="m", input_tokens=999)
        sess = db.get_session("s1")
        assert (sess.get("input_tokens") or 0) == 0

    def test_task_row_does_not_inherit_session_route(self, db):
        """An aux call on a different provider must not borrow the session's
        main-loop model/provider."""
        db.create_session("s1", source="cli", model="anthropic/claude-opus-4.6")
        db.update_token_counts(
            "s1", input_tokens=10, model="anthropic/claude-opus-4.6",
            billing_provider="anthropic", api_call_count=1,
        )
        db.record_auxiliary_usage("s1", "vision", input_tokens=5)  # no model given
        rows = {r["task"]: r for r in _usage_rows(db, "s1")}
        assert rows["vision"]["model"] == "unknown"
        assert rows["vision"]["billing_provider"] == ""
        # main-loop row unaffected
        assert rows[""]["model"] == "anthropic/claude-opus-4.6"

    def test_main_loop_and_aux_rows_coexist(self, db):
        db.create_session("s1", source="cli")
        db.update_token_counts(
            "s1", input_tokens=100, output_tokens=10,
            model="main-model", billing_provider="nous", api_call_count=1,
        )
        db.record_auxiliary_usage(
            "s1", "title_generation", model="main-model",
            billing_provider="nous", input_tokens=40, output_tokens=8,
        )
        rows = _usage_rows(db, "s1")
        tasks = sorted(r["task"] for r in rows)
        assert tasks == ["", "title_generation"]

    def test_noop_without_session_or_task(self, db):
        db.record_auxiliary_usage("", "vision", input_tokens=5)
        db.create_session("s1", source="cli")
        db.record_auxiliary_usage("s1", "", input_tokens=5)
        assert _usage_rows(db, "s1") == []

    def test_creates_session_row_if_missing(self, db):
        """FK safety: recording against a not-yet-created session must not fail."""
        db.record_auxiliary_usage("ghost", "vision", model="m", input_tokens=5)
        rows = _usage_rows(db, "ghost")
        assert len(rows) == 1


class TestSchemaMigrationV22:
    def test_v21_db_migrates_with_existing_rows(self, tmp_path):
        """A legacy DB with pre-task rows migrates: rows preserved, task=''."""
        import sqlite3 as _sq
        db = SessionDB(tmp_path / "state.db")
        db.create_session("legacy", source="cli")
        db.update_token_counts(
            "legacy", input_tokens=42, model="old-model",
            billing_provider="openrouter", api_call_count=1,
        )
        db.close()

        # Rebuild the legacy (v21) table shape: task column absent.
        conn = _sq.connect(tmp_path / "state.db")
        conn.executescript("""
            CREATE TABLE smu_old AS SELECT session_id, model, billing_provider,
                billing_base_url, billing_mode, api_call_count, input_tokens,
                output_tokens, cache_read_tokens, cache_write_tokens,
                reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                cost_status, cost_source, first_seen, last_seen
                FROM session_model_usage;
            DROP TABLE session_model_usage;
            CREATE TABLE session_model_usage (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                billing_provider TEXT NOT NULL DEFAULT '',
                billing_base_url TEXT NOT NULL DEFAULT '',
                billing_mode TEXT NOT NULL DEFAULT '',
                api_call_count INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd REAL NOT NULL DEFAULT 0,
                actual_cost_usd REAL NOT NULL DEFAULT 0,
                cost_status TEXT,
                cost_source TEXT,
                first_seen REAL,
                last_seen REAL,
                PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode)
            );
            INSERT INTO session_model_usage SELECT * FROM smu_old;
            DROP TABLE smu_old;
            UPDATE schema_version SET version = 21;
        """)
        conn.commit()
        conn.close()

        # Reopen: v22 migration rebuilds the table with task in the PK.
        db2 = SessionDB(tmp_path / "state.db")
        with db2._lock:
            pk_cols = [
                r[1] for r in db2._conn.execute(
                    "SELECT * FROM pragma_table_info('session_model_usage') WHERE pk > 0"
                ).fetchall()
            ]
            row = db2._conn.execute(
                "SELECT task, input_tokens FROM session_model_usage WHERE session_id = 'legacy'"
            ).fetchone()
        assert "task" in pk_cols
        assert row is not None
        assert row[0] == ""  # legacy rows are main-loop accounting
        assert row[1] == 42
        # And the new dimension works post-migration.
        db2.record_auxiliary_usage("legacy", "vision", model="v", input_tokens=1)
        db2.close()


class TestAmbientAccountingContext:
    def test_record_aux_usage_writes_through_context(self, db):
        from agent.aux_accounting import (
            record_aux_usage,
            reset_accounting_context,
            set_accounting_context,
        )

        db.create_session("s1", source="cli")
        token = set_accounting_context(db, "s1")
        try:
            record_aux_usage(_mk_response(model="aux-m"), "vision", provider="gemini")
        finally:
            reset_accounting_context(token)
        rows = _usage_rows(db, "s1")
        assert len(rows) == 1
        assert rows[0]["task"] == "vision"
        assert rows[0]["model"] == "aux-m"
        assert rows[0]["input_tokens"] == 100
        assert rows[0]["output_tokens"] == 20

    def test_noop_outside_context(self, db):
        from agent.aux_accounting import record_aux_usage

        db.create_session("s1", source="cli")
        record_aux_usage(_mk_response(), "vision")
        assert _usage_rows(db, "s1") == []

    def test_moa_tasks_excluded(self, db):
        """MoA advisor usage is already folded into the main-loop delta by
        conversation_loop — recording it here would double-count."""
        from agent.aux_accounting import (
            record_aux_usage,
            reset_accounting_context,
            set_accounting_context,
        )

        db.create_session("s1", source="cli")
        token = set_accounting_context(db, "s1")
        try:
            record_aux_usage(_mk_response(), "moa_reference")
            record_aux_usage(_mk_response(), "moa_aggregator")
        finally:
            reset_accounting_context(token)
        assert _usage_rows(db, "s1") == []

    def test_no_usage_object_is_noop(self, db):
        from agent.aux_accounting import (
            record_aux_usage,
            reset_accounting_context,
            set_accounting_context,
        )

        db.create_session("s1", source="cli")
        resp = SimpleNamespace(model="m", choices=[])
        token = set_accounting_context(db, "s1")
        try:
            record_aux_usage(resp, "vision")
        finally:
            reset_accounting_context(token)
        assert _usage_rows(db, "s1") == []

    def test_recording_failure_never_raises(self, db):
        from agent.aux_accounting import (
            record_aux_usage,
            reset_accounting_context,
            set_accounting_context,
        )

        class ExplodingDB:
            def record_auxiliary_usage(self, *a, **kw):
                raise RuntimeError("disk full")

        token = set_accounting_context(ExplodingDB(), "s1")
        try:
            record_aux_usage(_mk_response(), "vision")  # must not raise
        finally:
            reset_accounting_context(token)

    def test_validate_llm_response_records(self, db):
        """The aux client's validation chokepoint feeds the recorder."""
        from agent.aux_accounting import (
            reset_accounting_context,
            set_accounting_context,
        )
        from agent.auxiliary_client import _validate_llm_response

        db.create_session("s1", source="cli")
        token = set_accounting_context(db, "s1")
        try:
            out = _validate_llm_response(_mk_response(), "web_extract", provider="openrouter")
        finally:
            reset_accounting_context(token)
        assert out is not None
        rows = _usage_rows(db, "s1")
        assert len(rows) == 1
        assert rows[0]["task"] == "web_extract"
        assert rows[0]["billing_provider"] == "openrouter"

    def test_context_isolated_between_copied_contexts(self, db):
        import contextvars

        from agent.aux_accounting import get_accounting_context, set_accounting_context

        def _set_and_get(sid):
            set_accounting_context(db, sid)
            return get_accounting_context()[1]

        a = contextvars.copy_context().run(_set_and_get, "agent-a")
        b = contextvars.copy_context().run(_set_and_get, "agent-b")
        assert (a, b) == ("agent-a", "agent-b")
        assert get_accounting_context() is None


class TestAnalyticsAuxRows:
    def test_aux_usage_rows_and_merge(self, db):
        from hermes_cli.web_server import (
            _aux_task_summary,
            _aux_usage_rows,
            _merge_aux_into_by_model,
        )

        db.create_session("s1", source="cli")
        db.update_token_counts(
            "s1", input_tokens=1000, output_tokens=100,
            model="main-model", billing_provider="nous", api_call_count=1,
        )
        db.record_auxiliary_usage(
            "s1", "vision", model="vision-model",
            billing_provider="gemini", input_tokens=300, output_tokens=30,
        )
        db.record_auxiliary_usage(
            "s1", "compression", model="main-model",
            billing_provider="nous", input_tokens=200, output_tokens=20,
        )

        aux = _aux_usage_rows(db, cutoff=0)
        assert {r["task"] for r in aux} == {"vision", "compression"}

        by_model = [{
            "model": "main-model", "input_tokens": 1000, "output_tokens": 100,
            "estimated_cost": 0, "sessions": 1, "api_calls": 1,
        }]
        merged = _merge_aux_into_by_model(by_model, aux)
        by_name = {r["model"]: r for r in merged}
        # vision-only model surfaces as its own entry
        assert "vision-model" in by_name
        assert by_name["vision-model"]["input_tokens"] == 300
        # compression folded into the main model's totals
        assert by_name["main-model"]["input_tokens"] == 1200
        assert by_name["main-model"]["api_calls"] == 2

        tasks = _aux_task_summary(aux)
        assert {t["task"] for t in tasks} == {"vision", "compression"}
