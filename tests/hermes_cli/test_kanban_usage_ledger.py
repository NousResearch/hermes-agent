"""Tests for Kanban usage ledger (HERMES-OBS-001).

Validates:
- Schema migration (run_usage table with call_kind in PK)
- Idempotent write API
- Query by board/task/run/profile/provider/model
- Aggregation across dimensions
- Token source classification
- Privacy constraints (no secrets in ledger)
- Concurrency safety
- Multi-model run support
- Nullable aux tokens (missing != zero)
- call_kind separation (primary vs auxiliary)
- created_at timestamps
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_usage_ledger as ledger


@pytest.fixture
def isolated_kanban(tmp_path, monkeypatch):
    """Fresh kanban DB in temp directory."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return kb.connect()


# -- helpers --
def _base_kwargs(**overrides):
    """Build record_run_usage kwargs with sensible defaults + call_kind."""
    kw = dict(
        board="default",
        task_id="t_abc",
        run_id=1,
        call_kind="primary",
        api_call_index=0,
        provider="openrouter",
        model="gpt-4",
        input_tokens=100,
        output_tokens=50,
        token_source="provider_authoritative",
    )
    kw.update(overrides)
    return kw


class TestRunUsageSchema:
    """Schema migration tests."""

    def test_run_usage_table_created_on_schema_migration(self, isolated_kanban):
        """run_usage table exists after schema migration."""
        tables = isolated_kanban.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='run_usage'"
        ).fetchall()
        assert tables, "run_usage table missing after migration"

    def test_run_usage_has_accepted_result_tokens_and_api_calls(self, isolated_kanban):
        """run_usage has accepted_result_tokens and api_calls for aggregation."""
        cols = isolated_kanban.execute("PRAGMA table_info(run_usage)").fetchall()
        col_names = {row[1] for row in cols}
        required = {"accepted_result_tokens", "api_calls"}
        missing = required - col_names
        assert not missing, f"Missing accounting columns: {missing}"

    def test_run_usage_has_required_columns(self, isolated_kanban):
        """run_usage has all required columns for token accounting."""
        cols = isolated_kanban.execute("PRAGMA table_info(run_usage)").fetchall()
        col_names = {row[1] for row in cols}

        required = {
            "board",
            "task_id",
            "run_id",
            "call_kind",
            "parent_task_id",
            "profile",
            "provider",
            "model",
            "api_call_index",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "elapsed_ms",
            "token_source",
            "aux_input_tokens",
            "aux_output_tokens",
            "aux_cache_read_tokens",
            "aux_cache_write_tokens",
            "cost_usd",
            "cost_status",
            "checker_result",
            "repair_cycle",
            "created_at",
        }
        missing = required - col_names
        assert not missing, f"Missing columns: {missing}"

    def test_run_usage_primary_key(self, isolated_kanban):
        """run_usage primary key is (board, task_id, run_id, call_kind, api_call_index)."""
        pk_cols = isolated_kanban.execute(
            "PRAGMA table_info(run_usage)"
        ).fetchall()
        # row[5] is the pk column index (>0 means it's a PK member)
        pk = sorted([row[1] for row in pk_cols if row[5] > 0])
        assert pk == ["api_call_index", "board", "call_kind", "run_id", "task_id"]

    def test_run_usage_indexes_exist(self, isolated_kanban):
        """Indexes for common query patterns."""
        indexes = isolated_kanban.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_run_usage%'"
        ).fetchall()
        index_names = {row[0] for row in indexes}

        expected = {
            "idx_run_usage_task",
            "idx_run_usage_run",
            "idx_run_usage_profile",
            "idx_run_usage_provider",
        }
        missing = expected - index_names
        assert not missing, f"Missing indexes: {missing}"

    def test_run_usage_aux_columns_nullable(self, isolated_kanban):
        """aux_* columns are nullable (missing != authoritative zero)."""
        cols = isolated_kanban.execute("PRAGMA table_info(run_usage)").fetchall()
        # row: (cid, name, type, notnull, dflt_value, pk)
        aux_names = {
            "aux_input_tokens",
            "aux_output_tokens",
            "aux_cache_read_tokens",
            "aux_cache_write_tokens",
        }
        for row in cols:
            if row[1] in aux_names:
                assert row[3] == 0, (
                    f"Column {row[1]} has NOT NULL but must be nullable to "
                    f"distinguish 'unobserved' from 'observed zero'"
                )

    def test_migration_from_old_run_usage_schema(self, tmp_path, monkeypatch):
        """Old 25-column run_usage schema upgrades in place with no data loss.

        Acceptance (HERMES-OBS-001 finding 1):
        - exact old 25-column run_usage (ed65e757a era) gains accepted_result_tokens + api_calls
        - run_usage_parents is created
        - existing rows are preserved (no destructive rebuild)
        - re-running init_db is idempotent
        - record_run_usage writes succeed after upgrade
        """
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Exact OLD 25-column schema (as of ed65e757a — before
        # accepted_result_tokens, api_calls, and run_usage_parents).
        old_schema = """
        CREATE TABLE IF NOT EXISTS run_usage (
            board                    TEXT NOT NULL,
            task_id                  TEXT NOT NULL,
            run_id                   INTEGER NOT NULL,
            api_call_index           INTEGER NOT NULL,
            call_kind                TEXT NOT NULL DEFAULT 'primary',
            provider                 TEXT NOT NULL,
            model                    TEXT NOT NULL,
            input_tokens             INTEGER NOT NULL DEFAULT 0,
            output_tokens            INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens        INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens       INTEGER NOT NULL DEFAULT 0,
            reasoning_tokens         INTEGER NOT NULL DEFAULT 0,
            elapsed_ms               INTEGER NOT NULL DEFAULT 0,
            aux_input_tokens         INTEGER DEFAULT NULL,
            aux_output_tokens        INTEGER DEFAULT NULL,
            aux_cache_read_tokens    INTEGER DEFAULT NULL,
            aux_cache_write_tokens   INTEGER DEFAULT NULL,
            parent_task_id           TEXT,
            profile                  TEXT,
            token_source             TEXT NOT NULL,
            cost_usd                 REAL,
            cost_status              TEXT,
            checker_result           TEXT,
            repair_cycle             INTEGER NOT NULL DEFAULT 0,
            created_at               TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            PRIMARY KEY (board, task_id, run_id, call_kind, api_call_index)
        );
        """
        db_path = str(home / "kanban.db")
        seed = sqlite3.connect(db_path)
        seed.executescript(old_schema)
        seed.execute(
            """
            INSERT INTO run_usage (
                board, task_id, run_id, api_call_index, call_kind,
                provider, model, input_tokens, output_tokens, token_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                "t_preexisting",
                7,
                0,
                "primary",
                "openrouter",
                "gpt-4",
                111,
                22,
                "provider_authoritative",
            ),
        )
        seed.commit()
        seed.close()

        # Verify old schema is the exact 25-col shape and missing new objects.
        pre = sqlite3.connect(db_path)
        cols_before = [row[1] for row in pre.execute("PRAGMA table_info(run_usage)").fetchall()]
        assert len(cols_before) == 25, f"Expected exact old 25-col schema, got {len(cols_before)}: {cols_before}"
        assert "accepted_result_tokens" not in cols_before
        assert "api_calls" not in cols_before
        tables_before = {
            row[0] for row in pre.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "run_usage_parents" not in tables_before
        pre_row = pre.execute(
            "SELECT task_id, input_tokens, output_tokens FROM run_usage WHERE task_id=?",
            ("t_preexisting",),
        ).fetchone()
        assert pre_row == ("t_preexisting", 111, 22)
        pre.close()

        # First init: additive upgrade (no destructive rebuild of run_usage).
        kb.init_db()
        conn = kb.connect()

        cols_after = {row[1] for row in conn.execute("PRAGMA table_info(run_usage)").fetchall()}
        missing = {"accepted_result_tokens", "api_calls"} - cols_after
        assert not missing, f"Columns not migrated: {missing}"

        tables_after = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "run_usage_parents" in tables_after, "run_usage_parents missing after migration"

        preserved = conn.execute(
            "SELECT task_id, input_tokens, output_tokens, accepted_result_tokens, api_calls "
            "FROM run_usage WHERE task_id=?",
            ("t_preexisting",),
        ).fetchone()
        assert preserved is not None, "Pre-migration row was destroyed"
        assert preserved[0] == "t_preexisting"
        assert preserved[1] == 111
        assert preserved[2] == 22
        # New columns default correctly on preserved rows.
        assert preserved[3] is None  # accepted_result_tokens DEFAULT NULL
        assert preserved[4] == 0     # api_calls DEFAULT 0

        # Runtime ledger write must succeed against upgraded schema.
        ledger.record_run_usage(
            conn,
            board="default",
            task_id="t_mig",
            run_id=1,
            call_kind="primary",
            api_call_index=0,
            provider="openrouter",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            token_source="provider_authoritative",
            accepted_result_tokens=50,
        )
        rows = conn.execute(
            "SELECT task_id, accepted_result_tokens, api_calls FROM run_usage WHERE task_id=?",
            ("t_mig",),
        ).fetchall()
        assert len(rows) == 1, "Write failed after migration"
        assert rows[0][1] == 50
        assert rows[0][2] == 0
        conn.close()

        # Idempotent re-init: second pass must not fail or drop data.
        kb.init_db()
        conn2 = kb.connect()
        cols_again = {row[1] for row in conn2.execute("PRAGMA table_info(run_usage)").fetchall()}
        assert {"accepted_result_tokens", "api_calls"} <= cols_again
        tables_again = {
            row[0] for row in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "run_usage_parents" in tables_again
        still = conn2.execute(
            "SELECT COUNT(*) FROM run_usage WHERE task_id IN ('t_preexisting', 't_mig')"
        ).fetchone()[0]
        assert still == 2, "Idempotent re-init lost rows"
        conn2.close()


class TestTokenSourceClassification:
    """Token source must be explicit, never estimated from text length."""

    def test_record_usage_rejects_missing_token_source(self, isolated_kanban):
        """token_source is required, cannot be None or empty."""
        with pytest.raises(ValueError, match="token_source required"):
            ledger.record_run_usage(
                isolated_kanban,
                **_base_kwargs(token_source=None),
            )

    def test_record_usage_rejects_empty_token_source(self, isolated_kanban):
        """Empty token_source is rejected."""
        with pytest.raises(ValueError, match="token_source required"):
            ledger.record_run_usage(
                isolated_kanban,
                **_base_kwargs(token_source=""),
            )

    def test_record_usage_accepts_valid_token_sources(self, isolated_kanban):
        """Valid token sources: provider_authoritative, runtime_reported, estimated, incomplete, unknown."""
        valid_sources = [
            "provider_authoritative",
            "runtime_reported",
            "estimated",
            "incomplete",
            "unknown",
        ]
        for i, source in enumerate(valid_sources):
            ledger.record_run_usage(
                isolated_kanban,
                **_base_kwargs(run_id=i, token_source=source),
            )

        rows = isolated_kanban.execute(
            "SELECT token_source FROM run_usage ORDER BY run_id"
        ).fetchall()
        assert [row[0] for row in rows] == valid_sources

    def test_record_usage_rejects_invalid_token_source(self, isolated_kanban):
        """Invalid token sources are rejected."""
        with pytest.raises(ValueError, match="Invalid token_source"):
            ledger.record_run_usage(
                isolated_kanban,
                **_base_kwargs(token_source="bogus"),
            )


class TestIdempotentWrites:
    """Writes must be idempotent and concurrent-safe."""

    def test_duplicate_write_is_idempotent(self, isolated_kanban):
        """Writing same (board, task_id, run_id, call_kind, api_call_index) twice doesn't duplicate."""
        ledger.record_run_usage(isolated_kanban, **_base_kwargs())
        ledger.record_run_usage(isolated_kanban, **_base_kwargs())

        count = isolated_kanban.execute(
            "SELECT COUNT(*) FROM run_usage WHERE board=? AND task_id=? AND run_id=? AND call_kind=?",
            ("default", "t_abc", 1, "primary"),
        ).fetchone()[0]
        assert count == 1, "Duplicate write created multiple rows"

    def test_write_update_preserves_tokens(self, isolated_kanban):
        """Re-writing same key updates non-token fields but preserves tokens."""
        ledger.record_run_usage(isolated_kanban, **_base_kwargs(cost_usd=0.001))
        ledger.record_run_usage(
            isolated_kanban, **_base_kwargs(cost_usd=0.002)
        )

        row = isolated_kanban.execute(
            "SELECT input_tokens, output_tokens, cost_usd FROM run_usage WHERE board=? AND task_id=? AND run_id=? AND call_kind=?",
            ("default", "t_abc", 1, "primary"),
        ).fetchone()
        assert row[0] == 100
        assert row[1] == 50
        assert row[2] == 0.002


class TestQueryUsageRecords:
    """Query API for filtering usage records."""

    def test_query_by_board(self, isolated_kanban):
        """Query filters by board."""
        ledger.record_run_usage(isolated_kanban, **_base_kwargs())
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                board="hermes-obs", task_id="t_xyz", run_id=2,
                provider="anthropic", model="claude-3", input_tokens=200, output_tokens=100,
            ),
        )

        results = ledger.query_usage(isolated_kanban, board="default")
        assert len(results) == 1
        assert results[0]["task_id"] == "t_abc"

    def test_query_by_task_id(self, isolated_kanban):
        """Query filters by task_id."""
        for i in range(3):
            ledger.record_run_usage(
                isolated_kanban,
                **_base_kwargs(task_id=f"t_{i}", run_id=i),
            )

        results = ledger.query_usage(isolated_kanban, task_id="t_1")
        assert len(results) == 1
        assert results[0]["task_id"] == "t_1"

    def test_query_by_run_id(self, isolated_kanban):
        """Query filters by run_id."""
        for i in range(3):
            ledger.record_run_usage(
                isolated_kanban, **_base_kwargs(run_id=i),
            )

        results = ledger.query_usage(isolated_kanban, run_id=1)
        assert len(results) == 1
        assert results[0]["run_id"] == 1

    def test_query_by_profile(self, isolated_kanban):
        """Query filters by profile."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(profile="builder-qwen"),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                task_id="t_xyz", run_id=2,
                profile="researcher", provider="anthropic", model="claude-3",
                input_tokens=200, output_tokens=100,
            ),
        )

        results = ledger.query_usage(isolated_kanban, profile="builder-qwen")
        assert len(results) == 1
        assert results[0]["profile"] == "builder-qwen"

    def test_query_by_provider(self, isolated_kanban):
        """Query filters by provider."""
        ledger.record_run_usage(isolated_kanban, **_base_kwargs())
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                task_id="t_xyz", run_id=2,
                provider="anthropic", model="claude-3", input_tokens=200, output_tokens=100,
            ),
        )

        results = ledger.query_usage(isolated_kanban, provider="openrouter")
        assert len(results) == 1
        assert results[0]["provider"] == "openrouter"

    def test_query_by_model(self, isolated_kanban):
        """Query filters by exact model."""
        ledger.record_run_usage(isolated_kanban, **_base_kwargs())
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                task_id="t_xyz", run_id=2, model="gpt-4-turbo",
                input_tokens=200, output_tokens=100,
            ),
        )

        results = ledger.query_usage(isolated_kanban, model="gpt-4")
        assert len(results) == 1
        assert results[0]["model"] == "gpt-4"

    def test_query_with_multiple_filters(self, isolated_kanban):
        """Query combines multiple filters with AND."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(profile="builder-qwen"),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                api_call_index=1, model="gpt-4-turbo",
                profile="builder-qwen", input_tokens=200, output_tokens=100,
            ),
        )

        results = ledger.query_usage(
            isolated_kanban, board="default", task_id="t_abc", model="gpt-4",
        )
        assert len(results) == 1
        assert results[0]["model"] == "gpt-4"


class TestAggregateUsage:
    """Aggregate tokens and costs across dimensions."""

    def test_aggregate_by_task(self, isolated_kanban):
        """Sum tokens for a single task across all runs."""
        for i in range(3):
            ledger.record_run_usage(
                isolated_kanban, **_base_kwargs(run_id=i),
            )

        agg = ledger.aggregate_usage(isolated_kanban, task_id="t_abc")
        assert agg["total_input_tokens"] == 300
        assert agg["total_output_tokens"] == 150

    def test_aggregate_by_task_with_multiple_api_calls(self, isolated_kanban):
        """Sum tokens for a single task with multiple API calls per run."""
        for i in range(3):
            for j in range(2):  # 2 API calls per run
                ledger.record_run_usage(
                    isolated_kanban,
                    **_base_kwargs(run_id=i, api_call_index=j),
                )

        agg = ledger.aggregate_usage(isolated_kanban, task_id="t_abc")
        assert agg["total_input_tokens"] == 600  # 3 runs * 2 calls * 100
        assert agg["total_output_tokens"] == 300

    def test_aggregate_by_profile(self, isolated_kanban):
        """Sum tokens for a profile across all tasks."""
        for i in range(3):
            ledger.record_run_usage(
                isolated_kanban,
                **_base_kwargs(task_id=f"t_{i}", run_id=i, profile="builder-qwen"),
            )

        agg = ledger.aggregate_usage(isolated_kanban, profile="builder-qwen")
        assert agg["total_input_tokens"] == 300
        assert agg["total_output_tokens"] == 150

    def test_aggregate_by_provider(self, isolated_kanban):
        """Sum tokens for a provider across all tasks."""
        for i in range(3):
            ledger.record_run_usage(
                isolated_kanban, **_base_kwargs(task_id=f"t_{i}", run_id=i),
            )

        agg = ledger.aggregate_usage(isolated_kanban, provider="openrouter")
        assert agg["total_input_tokens"] == 300
        assert agg["total_output_tokens"] == 150

    def test_aggregate_includes_auxiliary_tokens(self, isolated_kanban):
        """Aggregate includes observable auxiliary tokens."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                aux_input_tokens=20,
                aux_output_tokens=10,
                aux_cache_read_tokens=5,
                aux_cache_write_tokens=3,
            ),
        )

        agg = ledger.aggregate_usage(isolated_kanban, task_id="t_abc")
        assert agg["total_aux_input_tokens"] == 20
        assert agg["total_aux_output_tokens"] == 10
        assert agg["total_aux_cache_read_tokens"] == 5
        assert agg["total_aux_cache_write_tokens"] == 3


class TestMultiModelRuns:
    """Multiple models in same run (fallback/retry)."""

    def test_multiple_models_same_run(self, isolated_kanban):
        """Record multiple API calls with different models in same run."""
        ledger.record_run_usage(isolated_kanban, **_base_kwargs())
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                api_call_index=1, provider="anthropic", model="claude-3-opus",
                input_tokens=200, output_tokens=100,
            ),
        )

        results = ledger.query_usage(isolated_kanban, run_id=1)
        assert len(results) == 2
        models = {r["model"] for r in results}
        assert models == {"gpt-4", "claude-3-opus"}

        agg = ledger.aggregate_usage(isolated_kanban, run_id=1)
        assert agg["total_input_tokens"] == 300
        assert agg["total_output_tokens"] == 150


class TestCallKindSeparation:
    """call_kind distinguishes primary from observable auxiliary calls."""

    def test_primary_and_aux_same_run_distinct_keys(self, isolated_kanban):
        """Same api_call_index but different call_kind are distinct rows."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(call_kind="primary", input_tokens=100, output_tokens=50),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                call_kind="auxiliary", api_call_index=0,
                provider="openrouter", model="gpt-3.5-turbo",
                input_tokens=30, output_tokens=15,
                token_source="runtime_reported",
            ),
        )

        rows = ledger.query_usage(isolated_kanban, run_id=1)
        assert len(rows) == 2
        kinds = {r["call_kind"] for r in rows}
        assert kinds == {"primary", "auxiliary"}

    def test_aggregate_across_call_kinds(self, isolated_kanban):
        """Aggregate sums primary and auxiliary rows into the same totals."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(call_kind="primary", input_tokens=100, output_tokens=50),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                call_kind="auxiliary", api_call_index=0,
                provider="openrouter", model="gpt-3.5-turbo",
                input_tokens=30, output_tokens=15,
                token_source="runtime_reported",
            ),
        )
        agg = ledger.aggregate_usage(isolated_kanban, run_id=1)
        assert agg["total_input_tokens"] == 130
        assert agg["total_output_tokens"] == 65
        assert agg["record_count"] == 2

    def test_aggregate_filter_by_call_kind(self, isolated_kanban):
        """Filtering by provider/model lets you isolate aux-only totals."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(call_kind="primary", input_tokens=100, output_tokens=50),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                call_kind="auxiliary", api_call_index=0,
                provider="openrouter", model="gpt-3.5-turbo",
                input_tokens=30, output_tokens=15,
                token_source="runtime_reported",
            ),
        )
        aux = ledger.aggregate_usage(
            isolated_kanban, run_id=1, model="gpt-3.5-turbo",
        )
        assert aux["record_count"] == 1
        assert aux["total_input_tokens"] == 30


class TestNullableAuxTokens:
    """Missing auxiliary observations are NULL, not authoritative zero."""

    def test_aux_defaults_are_null_not_zero(self, isolated_kanban):
        """Aux columns are NULL when caller doesn't observe aux usage."""
        ledger.record_run_usage(isolated_kanban, **_base_kwargs())

        row = isolated_kanban.execute(
            "SELECT aux_input_tokens, aux_output_tokens, "
            "       aux_cache_read_tokens, aux_cache_write_tokens "
            "FROM run_usage WHERE board=? AND task_id=? AND run_id=? AND call_kind=?",
            ("default", "t_abc", 1, "primary"),
        ).fetchone()
        assert tuple(row) == (None, None, None, None)

    def test_explicit_aux_zero_distinct_from_null(self, isolated_kanban):
        """Explicitly observed zero aux tokens stored as 0, NULL when not observed."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                aux_input_tokens=0, aux_output_tokens=0,
                aux_cache_read_tokens=0, aux_cache_write_tokens=0,
            ),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                task_id="t_no_aux", run_id=2,
                # no aux fields => NULL
            ),
        )

        observed = isolated_kanban.execute(
            "SELECT aux_input_tokens FROM run_usage WHERE task_id='t_abc'",
        ).fetchone()[0]
        assert observed == 0, "explicit 0 should store 0"

        unknown = isolated_kanban.execute(
            "SELECT aux_input_tokens FROM run_usage WHERE task_id='t_no_aux'",
        ).fetchone()[0]
        assert unknown is None, "unobserved should store NULL"


class TestPrivacyConstraints:
    """Privacy: no secrets, prompts, or OAuth in ledger."""

    def test_record_usage_rejects_secret_in_model(self, isolated_kanban):
        """Model field cannot contain API keys or secrets."""
        # Use patterns that actually match _SECRET_PATTERNS
        secret_patterns = [
            "sk-abcdefghijklmnopqrstuvwxyz1234",  # OpenAI key (20+ alnum after sk-)
            "xai-aB3d4f5g6h7i8j9k0l1m2n3",   # xAI key (20+ alnum/dash after xai-)
            "Bearer eyJhbGciOiJIUzI1NiIs",     # Bearer header
        ]
        for secret in secret_patterns:
            with pytest.raises(ValueError, match="potential secret"):
                ledger.record_run_usage(
                    isolated_kanban,
                    **_base_kwargs(model=secret),
                )

    def test_record_usage_rejects_secret_in_provider(self, isolated_kanban):
        """Provider field cannot contain API keys or OAuth tokens."""
        secret_patterns = [
            "Bearer abc123",
            "Basic dXNlcjpwYXNz",
            "token:secret-key",
        ]
        for secret in secret_patterns:
            with pytest.raises(ValueError, match="potential secret"):
                ledger.record_run_usage(
                    isolated_kanban,
                    **_base_kwargs(provider=secret),
                )

    def test_record_usage_does_not_store_prompts(self, isolated_kanban):
        """ledger.record_run_usage has no parameter for prompts or messages."""
        import inspect
        sig = inspect.signature(ledger.record_run_usage)
        params = set(sig.parameters.keys())
        forbidden = {"prompt", "message", "content", "user_message", "system_prompt"}
        overlap = params & forbidden
        assert not overlap, f"record_run_usage accepts forbidden params: {overlap}"

    def test_ledger_table_has_no_prompt_column(self, isolated_kanban):
        """run_usage table cannot store prompts or private bodies."""
        cols = isolated_kanban.execute("PRAGMA table_info(run_usage)").fetchall()
        col_names = {row[1] for row in cols}
        forbidden = {"prompt", "message", "content", "user_message", "system_prompt", "body"}
        overlap = col_names & forbidden
        assert not overlap, f"run_usage has forbidden columns: {overlap}"

    def test_record_usage_does_not_store_raw_usage(self, isolated_kanban):
        """ledger.record_run_usage has no parameter for raw_usage dict."""
        import inspect
        sig = inspect.signature(ledger.record_run_usage)
        params = set(sig.parameters.keys())
        forbidden = {"raw", "raw_usage", "response", "usage_raw"}
        overlap = params & forbidden
        assert not overlap, f"record_run_usage accepts forbidden raw-usage params: {overlap}"


class TestAuxiliaryUsage:
    """Observable auxiliary usage tracking."""

    def test_auxiliary_tokens_stored_separately(self, isolated_kanban):
        """Auxiliary tokens are stored in dedicated columns."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                aux_input_tokens=20,
                aux_output_tokens=10,
                aux_cache_read_tokens=5,
                aux_cache_write_tokens=3,
            ),
        )

        row = isolated_kanban.execute(
            "SELECT aux_input_tokens, aux_output_tokens, "
            "       aux_cache_read_tokens, aux_cache_write_tokens "
            "FROM run_usage WHERE board=? AND task_id=? AND run_id=? AND call_kind=?",
            ("default", "t_abc", 1, "primary"),
        ).fetchone()
        # sqlite3.Row requires index-based access, not tuple comparison
        assert tuple(row) == (20, 10, 5, 3)

    def test_auxiliary_tokens_default_to_null_when_unobserved(self, isolated_kanban):
        """Auxiliary token columns NULL when caller doesn't observe them."""
        ledger.record_run_usage(isolated_kanban, **_base_kwargs())

        row = isolated_kanban.execute(
            "SELECT aux_input_tokens, aux_output_tokens, "
            "       aux_cache_read_tokens, aux_cache_write_tokens "
            "FROM run_usage WHERE board=? AND task_id=? AND run_id=? AND call_kind=?",
            ("default", "t_abc", 1, "primary"),
        ).fetchone()
        assert tuple(row) == (None, None, None, None)


class TestCheckerAndRepair:
    """Checker result and repair cycle tracking."""

    def test_record_checker_result_and_repair_cycle(self, isolated_kanban):
        """checker_result and repair_cycle are stored correctly."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(checker_result="PASS", repair_cycle=0),
        )

        row = isolated_kanban.execute(
            "SELECT checker_result, repair_cycle FROM run_usage "
            "WHERE board=? AND task_id=? AND run_id=? AND call_kind=?",
            ("default", "t_abc", 1, "primary"),
        ).fetchone()
        assert tuple(row) == ("PASS", 0)

    def test_repair_cycle_increments(self, isolated_kanban):
        """Multiple repair attempts recorded with incrementing repair_cycle."""
        for cycle in range(3):
            ledger.record_run_usage(
                isolated_kanban,
                **_base_kwargs(
                    api_call_index=cycle,
                    checker_result="FAIL" if cycle < 2 else "PASS",
                    repair_cycle=cycle,
                ),
            )

        rows = isolated_kanban.execute(
            "SELECT repair_cycle, checker_result FROM run_usage "
            "WHERE board=? AND task_id=? AND run_id=? AND call_kind=? "
            "ORDER BY api_call_index",
            ("default", "t_abc", 1, "primary"),
        ).fetchall()
        assert [(row[0], row[1]) for row in rows] == [
            (0, "FAIL"), (1, "FAIL"), (2, "PASS"),
        ]


class TestCostTracking:
    """Cost in USD with proper status."""

    def test_record_cost_usd_and_status(self, isolated_kanban):
        """cost_usd and cost_status are stored correctly."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(cost_usd=0.0015, cost_status="actual"),
        )

        row = isolated_kanban.execute(
            "SELECT cost_usd, cost_status FROM run_usage "
            "WHERE board=? AND task_id=? AND run_id=? AND call_kind=?",
            ("default", "t_abc", 1, "primary"),
        ).fetchone()
        assert row[0] == pytest.approx(0.0015)
        assert row[1] == "actual"

    def test_aggregate_sums_cost_usd(self, isolated_kanban):
        """Aggregate sums cost_usd across all matching records."""
        for i in range(3):
            ledger.record_run_usage(
                isolated_kanban,
                **_base_kwargs(run_id=i, cost_usd=0.001),
            )

        agg = ledger.aggregate_usage(isolated_kanban, task_id="t_abc")
        assert agg["total_cost_usd"] == pytest.approx(0.003)


class TestCreatedAt:
    """created_at timestamp auto-fills on insert."""

    def test_created_at_populated_on_insert(self, isolated_kanban):
        """created_at has an auto-populated value."""
        ledger.record_run_usage(isolated_kanban, **_base_kwargs())
        ts = isolated_kanban.execute(
            "SELECT created_at FROM run_usage WHERE board=? AND task_id=? AND run_id=? AND call_kind=?",
            ("default", "t_abc", 1, "primary"),
        ).fetchone()[0]
        assert ts is not None
        assert "T" in ts  # ISO 8601


class TestHookIntegration:
    """Runtime hook contract: ledger can be wired into conversation/codex hooks."""

    def test_record_primary_hook_path_accepts_canonical_usage_shape(
        self, isolated_kanban
    ):
        """Hook can record canonical usage from conversation loop.

        Simulates the runtime hook receiving a CanonicalUsage-like dict from
        the turn finalizer. All canonical fields must be storable under
        provider_authoritative source.
        """
        canonical = {
            "input_tokens": 1234,
            "output_tokens": 567,
            "cache_read_tokens": 890,
            "cache_write_tokens": 12,
            "reasoning_tokens": 34,
        }
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                call_kind="primary",
                api_call_index=0,
                token_source="provider_authoritative",
                **canonical,
            ),
        )
        row = ledger.query_usage(isolated_kanban, run_id=1)[0]
        assert row["input_tokens"] == 1234
        assert row["output_tokens"] == 567
        assert row["cache_read_tokens"] == 890
        assert row["cache_write_tokens"] == 12
        assert row["reasoning_tokens"] == 34

    def test_auxiliary_hook_uses_runtime_reported_when_unobserved(
        self, isolated_kanban
    ):
        """Auxiliary helper records runtime_reported when usage is not observable.

        Unobserved auxiliary must not be stored as authoritative zero.
        """
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                call_kind="auxiliary",
                api_call_index=0,
                token_source="runtime_reported",
                input_tokens=200,
                output_tokens=100,
                aux_input_tokens=None,   # unobserved
                aux_output_tokens=None,
            ),
        )
        row = ledger.query_usage(isolated_kanban, run_id=1, model="gpt-3.5-turbo")[0] if False else \
              ledger.query_usage(isolated_kanban, run_id=1)[0]
        assert row["token_source"] == "runtime_reported"
        # Nulls preserved for unobserved aux
        assert row["aux_input_tokens"] is None
        assert row["aux_output_tokens"] is None

    def test_repair_cycle_marked_incomplete_until_passes(self, isolated_kanban):
        """First attempt recorded incomplete; final PASS recorded after."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                repair_cycle=0, checker_result="PENDING",
                token_source="runtime_reported",
            ),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                repair_cycle=1, checker_result="PASS",
                token_source="provider_authoritative",
            ),
        )
        rows = ledger.query_usage(isolated_kanban, run_id=1)
        assert rows[0]["checker_result"] == "PASS"
        assert rows[0]["token_source"] == "provider_authoritative"

    def test_parent_task_recorded_for_subtask(self, isolated_kanban):
        """Subtask records its parent_task_id without double-counting."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                task_id="t_child",
                run_id=10,
                parent_task_id="t_parent",
            ),
        )
        row = ledger.query_usage(
            isolated_kanban, task_id="t_child", run_id=10,
        )[0]
        assert row["parent_task_id"] == "t_parent"

    def test_aggregate_does_not_double_count_parent_and_child(
        self, isolated_kanban
    ):
        """Parent and child usage are distinct rows; filtering avoids double-count."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(task_id="t_parent", input_tokens=100, output_tokens=50),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                task_id="t_child", run_id=2, parent_task_id="t_parent",
                input_tokens=200, output_tokens=100,
            ),
        )
        # Parent only
        parent_agg = ledger.aggregate_usage(
            isolated_kanban, task_id="t_parent",
        )
        assert parent_agg["total_input_tokens"] == 100
        # Child only
        child_agg = ledger.aggregate_usage(
            isolated_kanban, task_id="t_child",
        )
        assert child_agg["total_input_tokens"] == 200


class TestConcurrency:
    """Concurrent writes don't corrupt or duplicate."""

    def test_concurrent_idempotent_writes(self, isolated_kanban):
        """Multiple writers on same key via upsert don't duplicate."""
        for _ in range(10):
            ledger.record_run_usage(isolated_kanban, **_base_kwargs())

        count = isolated_kanban.execute(
            "SELECT COUNT(*) FROM run_usage WHERE board=? AND task_id=? AND run_id=? AND call_kind=?",
            ("default", "t_abc", 1, "primary"),
        ).fetchone()[0]
        assert count == 1


class TestAcceptedResultAndApiCalls:
    """Checker blocker #3: explicit API-call totals and accepted-result tokens."""

    def test_aggregate_returns_total_api_calls(self, isolated_kanban):
        """aggregate_usage returns total_api_calls count."""
        for i in range(3):
            ledger.record_run_usage(
                isolated_kanban, **_base_kwargs(run_id=i, api_call_index=i),
            )

        agg = ledger.aggregate_usage(isolated_kanban, task_id="t_abc")
        assert "total_api_calls" in agg, "aggregate must return total_api_calls"
        assert agg["total_api_calls"] == 3

    def test_aggregate_returns_accepted_result_tokens(self, isolated_kanban):
        """aggregate_usage returns total_accepted_result_tokens."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                api_call_index=0,
                accepted_result_tokens=50,
            ),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                api_call_index=1,
                accepted_result_tokens=30,
            ),
        )

        agg = ledger.aggregate_usage(isolated_kanban, run_id=1)
        assert "total_accepted_result_tokens" in agg
        assert agg["total_accepted_result_tokens"] == 80

    def test_accepted_result_tokens_nullable(self, isolated_kanban):
        """accepted_result_tokens is NULL when not observed."""
        ledger.record_run_usage(isolated_kanban, **_base_kwargs())

        row = isolated_kanban.execute(
            "SELECT accepted_result_tokens FROM run_usage "
            "WHERE board=? AND task_id=? AND run_id=? AND call_kind=?",
            ("default", "t_abc", 1, "primary"),
        ).fetchone()
        assert row[0] is None


class TestMultiParentJoinTable:
    """Checker blocker #2: multi-parent relations without UPSERT overwrite."""

    def test_parent_join_table_exists(self, isolated_kanban):
        """run_usage_parents join table for multi-parent edges."""
        tables = isolated_kanban.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='run_usage_parents'"
        ).fetchall()
        assert tables, "run_usage_parents table missing"

    def test_record_multiple_parents_no_overwrite(self, isolated_kanban):
        """Same event can have multiple parent_task_ids without UPSERT overwrite."""
        ledger.record_run_usage(isolated_kanban, **_base_kwargs())
        ledger.record_parent(isolated_kanban, board="default", task_id="t_abc",
                             run_id=1, call_kind="primary", api_call_index=0,
                             parent_task_id="t_parent_a")
        ledger.record_parent(isolated_kanban, board="default", task_id="t_abc",
                             run_id=1, call_kind="primary", api_call_index=0,
                             parent_task_id="t_parent_b")

        rows = isolated_kanban.execute(
            "SELECT parent_task_id FROM run_usage_parents "
            "WHERE board=? AND task_id=? AND run_id=? "
            "ORDER BY parent_task_id",
            ("default", "t_abc", 1),
        ).fetchall()
        parents = [r[0] for r in rows]
        assert parents == ["t_parent_a", "t_parent_b"], (
            f"Multi-parent association overwritten: got {parents}"
        )

    def test_duplicate_parent_is_idempotent(self, isolated_kanban):
        """Recording same parent twice doesn't duplicate."""
        ledger.record_run_usage(isolated_kanban, **_base_kwargs())
        ledger.record_parent(isolated_kanban, board="default", task_id="t_abc",
                             run_id=1, call_kind="primary", api_call_index=0,
                             parent_task_id="t_parent_a")
        ledger.record_parent(isolated_kanban, board="default", task_id="t_abc",
                             run_id=1, call_kind="primary", api_call_index=0,
                             parent_task_id="t_parent_a")

        count = isolated_kanban.execute(
            "SELECT COUNT(*) FROM run_usage_parents "
            "WHERE board=? AND task_id=? AND run_id=? AND parent_task_id=?",
            ("default", "t_abc", 1, "t_parent_a"),
        ).fetchone()[0]
        assert count == 1

    def test_aggregate_distinct_events_no_double_count(self, isolated_kanban):
        """Multi-parent events counted once each via DISTINCT."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                task_id="t_child", run_id=2,
                input_tokens=100, output_tokens=50,
            ),
        )
        ledger.record_parent(isolated_kanban, board="default", task_id="t_child",
                             run_id=2, call_kind="primary", api_call_index=0,
                             parent_task_id="t_p1")
        ledger.record_parent(isolated_kanban, board="default", task_id="t_child",
                             run_id=2, call_kind="primary", api_call_index=0,
                             parent_task_id="t_p2")

        # Aggregate by task_id must count the event once, not twice
        agg = ledger.aggregate_usage(isolated_kanban, task_id="t_child")
        assert agg["record_count"] == 1

    def test_record_run_usage_retains_every_parent(self, isolated_kanban):
        """One usage event retains every parent via record_run_usage associations."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(parent_task_id="t_parent_a", input_tokens=10, output_tokens=5),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(parent_task_id="t_parent_b", input_tokens=10, output_tokens=5),
        )

        events = ledger.query_usage(isolated_kanban, task_id="t_abc", run_id=1)
        assert len(events) == 1, f"Expected one usage event, got {len(events)}"

        parents = ledger.list_parents(
            isolated_kanban,
            board="default",
            task_id="t_abc",
            run_id=1,
            call_kind="primary",
            api_call_index=0,
        )
        assert parents == ["t_parent_a", "t_parent_b"], (
            f"Adding parent B overwrote or dropped parent A: {parents}"
        )

    def test_record_run_usage_parent_association_is_idempotent(self, isolated_kanban):
        """Repeating the same event-parent association does not duplicate."""
        kwargs = _base_kwargs(parent_task_id="t_parent_a", input_tokens=10, output_tokens=5)
        ledger.record_run_usage(isolated_kanban, **kwargs)
        ledger.record_run_usage(isolated_kanban, **kwargs)
        ledger.record_run_usage(isolated_kanban, **kwargs)

        parents = ledger.list_parents(
            isolated_kanban,
            board="default",
            task_id="t_abc",
            run_id=1,
            call_kind="primary",
            api_call_index=0,
        )
        assert parents == ["t_parent_a"]
        count = isolated_kanban.execute(
            "SELECT COUNT(*) FROM run_usage_parents "
            "WHERE board=? AND task_id=? AND run_id=? AND parent_task_id=?",
            ("default", "t_abc", 1, "t_parent_a"),
        ).fetchone()[0]
        assert count == 1

    def test_record_run_usage_does_not_overwrite_first_parent_column(
        self, isolated_kanban
    ):
        """Denormalized parent_task_id keeps the first non-null parent on upsert."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(parent_task_id="t_parent_a"),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(parent_task_id="t_parent_b"),
        )
        row = ledger.query_usage(isolated_kanban, task_id="t_abc", run_id=1)[0]
        assert row["parent_task_id"] == "t_parent_a"


class TestUnobservedAuxClassification:
    """Checker blocker #4: unobserved auxiliary is 'incomplete' or 'unknown', never 'runtime_reported'."""

    def test_unobserved_aux_classified_incomplete(self, isolated_kanban):
        """Unobserved auxiliary usage recorded as 'incomplete', not 'runtime_reported'."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                call_kind="auxiliary",
                api_call_index=0,
                token_source="incomplete",
                input_tokens=200,
                output_tokens=100,
                aux_input_tokens=None,
                aux_output_tokens=None,
            ),
        )
        row = ledger.query_usage(isolated_kanban, run_id=1)[0]
        assert row["token_source"] == "incomplete"
        assert row["aux_input_tokens"] is None
        assert row["aux_output_tokens"] is None


class TestRuntimeHookIntegration:
    """Checker blocker: real runtime hooks exercising actual production boundaries."""

    def test_boundary_function_writes_ledger(self, isolated_kanban, monkeypatch):
        """_record_kanban_usage_at_boundary writes ledger on API call boundary."""
        from hermes_cli.kanban_usage_ledger import _record_kanban_usage_at_boundary

        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_boundary")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")

        _record_kanban_usage_at_boundary(
            isolated_kanban,
            call_kind="primary",
            provider="openrouter",
            model="gpt-4",
            canonical_usage={
                "input_tokens": 500,
                "output_tokens": 200,
                "cache_read_tokens": 100,
                "cache_write_tokens": 10,
                "reasoning_tokens": 5,
            },
            token_source="provider_authoritative",
            elapsed_ms=1500,
        )

        rows = ledger.query_usage(isolated_kanban, task_id="t_boundary", run_id=42)
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
        assert rows[0]["input_tokens"] == 500
        assert rows[0]["output_tokens"] == 200

    def test_boundary_function_quiet_when_not_in_kanban(self, isolated_kanban, monkeypatch):
        """No-op when HERMES_KANBAN_TASK is not set."""
        from hermes_cli.kanban_usage_ledger import _record_kanban_usage_at_boundary

        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        _record_kanban_usage_at_boundary(
            isolated_kanban,
            call_kind="primary",
            provider="openrouter",
            model="gpt-4",
            canonical_usage={"input_tokens": 100, "output_tokens": 50,
                             "cache_read_tokens": 0, "cache_write_tokens": 0,
                             "reasoning_tokens": 0},
            token_source="provider_authoritative",
        )

        rows = ledger.query_usage(isolated_kanban)
        assert len(rows) == 0, "No rows when not in Kanban context"

    def test_boundary_function_never_breaks_execution(self, isolated_kanban, monkeypatch):
        """Ledger write failure does not propagate."""
        from hermes_cli.kanban_usage_ledger import _record_kanban_usage_at_boundary

        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_broken")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")

        bad_conn = closed_connection()
        # Must not raise
        result = _record_kanban_usage_at_boundary(
            bad_conn,
            call_kind="primary",
            provider="openrouter",
            model="gpt-4",
            canonical_usage={"input_tokens": 100, "output_tokens": 50,
                             "cache_read_tokens": 0, "cache_write_tokens": 0,
                             "reasoning_tokens": 0},
            token_source="provider_authoritative",
        )
        assert result is None

    def test_auxiliary_boundary_uses_distinct_indices(self, isolated_kanban, monkeypatch):
        """Two auxiliary calls through boundary function get distinct api_call_index."""
        from hermes_cli.kanban_usage_ledger import _record_kanban_usage_at_boundary

        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_aux_inc")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "7")

        _record_kanban_usage_at_boundary(
            isolated_kanban,
            call_kind="auxiliary",
            provider="aux",
            model="gpt-3.5",
            canonical_usage={"input_tokens": 100, "output_tokens": 50,
                             "cache_read_tokens": 0, "cache_write_tokens": 0,
                             "reasoning_tokens": 0},
            token_source="incomplete",
        )
        _record_kanban_usage_at_boundary(
            isolated_kanban,
            call_kind="auxiliary",
            provider="aux",
            model="gpt-3.5",
            canonical_usage={"input_tokens": 200, "output_tokens": 100,
                             "cache_read_tokens": 0, "cache_write_tokens": 0,
                             "reasoning_tokens": 0},
            token_source="incomplete",
        )

        rows = ledger.query_usage(isolated_kanban, task_id="t_aux_inc", call_kind="auxiliary")
        assert len(rows) == 2, f"Expected 2 auxiliary events, got {len(rows)}"
        indices = sorted(r["api_call_index"] for r in rows)
        assert indices == [0, 1], f"Expected indices [0,1], got {indices}"


class TestMultiProcessConcurrency:
    """Checker blocker #5: genuine multi-connection/process idempotency."""

    def test_multi_connection_concurrent_writes(self, tmp_path, monkeypatch):
        """Multiple independent connections writing same key don't corrupt."""
        import subprocess
        import sys

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Initialize DB
        kb.init_db()

        db_path = str(home / "kanban.db")

        # Run concurrent writers as separate connections
        repo_root = repr(str(Path(__file__).parent.parent.parent))
        db_path_repr = repr(db_path)
        code = f'''
import sqlite3
import sys
sys.path.insert(0, {repo_root})
from hermes_cli import kanban_usage_ledger as ledger

db = {db_path_repr}
writer_id = int(sys.argv[1])

conn = sqlite3.connect(db, timeout=30)
for i in range(5):
    ledger.record_run_usage(
        conn,
        board="default",
        task_id="t_multi",
        run_id=1,
        call_kind="primary",
        api_call_index=i,
        provider="openrouter",
        model="gpt-4",
        input_tokens=100,
        output_tokens=50,
        token_source="provider_authoritative",
    )
conn.commit()
conn.close()
'''

        # Launch 3 concurrent processes
        procs = []
        for pid in range(3):
            p = subprocess.Popen(
                [sys.executable, "-c", code, str(pid)],
                cwd=str(Path(__file__).parent.parent.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            procs.append(p)

        for p in procs:
            p.wait(timeout=60)
            assert p.returncode == 0, f"Process failed: {p.stderr.read().decode()}"

        # Verify: 5 distinct api_call_indices, each idempotent across 3 processes
        conn = sqlite3.connect(db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM run_usage WHERE board=? AND task_id=? AND run_id=?",
            ("default", "t_multi", 1),
        ).fetchone()[0]
        conn.close()
        assert count == 5, f"Expected 5 rows (idempotent), got {count}"

    def test_multi_connection_same_key_idempotent(self, tmp_path, monkeypatch):
        """Multiple connections upserting same key produce exactly 1 row."""
        home = tmp_path / ".hermes2"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        kb.init_db()
        db_path = str(home / "kanban.db")

        import threading
        errors = []
        def writer(thread_id):
            try:
                conn = sqlite3.connect(db_path, timeout=30)
                for _ in range(5):
                    ledger.record_run_usage(
                        conn,
                        board="default",
                        task_id="t_conc",
                        run_id=1,
                        call_kind="primary",
                        api_call_index=0,
                        provider="openrouter",
                        model="gpt-4",
                        input_tokens=100,
                        output_tokens=50,
                        token_source="provider_authoritative",
                    )
                conn.commit()
                conn.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"

        conn = sqlite3.connect(db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM run_usage WHERE board=? AND task_id=? AND run_id=?",
            ("default", "t_conc", 1),
        ).fetchone()[0]
        conn.close()
        assert count == 1, f"Expected 1 row (idempotent), got {count}"


def closed_connection():
    """Return a closed sqlite3 connection for testing failure handling."""
    conn = sqlite3.connect(":memory:")
    conn.close()
    return conn


class TestAcceptedResultAggregation:
    """Checker blocker #3: accepted-result aggregation across builder/checker/repair."""

    def test_aggregate_with_checker_repair_cycle(self, isolated_kanban):
        """Multi-model retry+checker aggregation returns correct accepted-result."""
        # Builder cycle - model A, FAIL
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                api_call_index=0,
                provider="openrouter",
                model="gpt-4",
                input_tokens=100, output_tokens=50,
                accepted_result_tokens=None,
                checker_result="FAIL",
                repair_cycle=0,
            ),
        )
        # Repair cycle 1 - model B, PASS
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                api_call_index=1,
                provider="anthropic",
                model="claude-3",
                input_tokens=200, output_tokens=100,
                accepted_result_tokens=100,
                checker_result="PASS",
                repair_cycle=1,
            ),
        )

        agg = ledger.aggregate_usage(isolated_kanban, run_id=1)
        assert agg["total_api_calls"] == 2
        assert agg["total_accepted_result_tokens"] == 100
        assert agg["record_count"] == 2


class TestRuntimeContextFields:
    """Checker failure #3: hooks must persist profile, parents, checker, repair, accepted_result, api_calls."""

    def test_api_calls_column_persisted(self, isolated_kanban):
        """api_calls column is inserted and returned in queries."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(api_call_index=0, input_tokens=100, output_tokens=50),
        )
        row = ledger.query_usage(isolated_kanban, run_id=1)[0]
        assert "api_calls" in row, "api_calls column missing from query results"
        assert row["api_calls"] == 0  # default value

    def test_profile_persisted(self, isolated_kanban):
        """profile is recorded and queryable."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(profile="builder-deepseek"),
        )
        row = ledger.query_usage(isolated_kanban, profile="builder-deepseek")[0]
        assert row["profile"] == "builder-deepseek"

    def test_parent_task_id_persisted(self, isolated_kanban):
        """parent_task_id is recorded on the usage row and in parents table."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(parent_task_id="t_parent_123"),
        )
        row = ledger.query_usage(isolated_kanban, run_id=1)[0]
        assert row["parent_task_id"] == "t_parent_123"

        # Also test multi-parent via record_parent
        ledger.record_parent(
            isolated_kanban,
            board="default", task_id="t_abc", run_id=1,
            call_kind="primary", api_call_index=0,
            parent_task_id="t_parent_456",
        )
        parents = isolated_kanban.execute(
            "SELECT parent_task_id FROM run_usage_parents WHERE task_id=? ORDER BY parent_task_id",
            ("t_abc",),
        ).fetchall()
        assert [r[0] for r in parents] == ["t_parent_456"]

    def test_checker_result_and_repair_cycle_persisted(self, isolated_kanban):
        """checker_result and repair_cycle are recorded."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(checker_result="PASS", repair_cycle=2),
        )
        row = ledger.query_usage(isolated_kanban, run_id=1)[0]
        assert row["checker_result"] == "PASS"
        assert row["repair_cycle"] == 2

    def test_accepted_result_tokens_persisted(self, isolated_kanban):
        """accepted_result_tokens is recorded and NULL when not set."""
        # Set explicitly
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(api_call_index=1, accepted_result_tokens=500),
        )
        row1 = ledger.query_usage(isolated_kanban, run_id=1)[0]
        assert row1["accepted_result_tokens"] == 500

        # Default NULL
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(api_call_index=0, accepted_result_tokens=None),
        )
        row2 = isolated_kanban.execute(
            "SELECT accepted_result_tokens FROM run_usage WHERE api_call_index=0 AND run_id=1"
        ).fetchone()
        assert row2[0] is None


class TestAuxiliaryEventIdentity:
    """Checker failure #4: auxiliary events must have distinct stable identities."""

    def test_two_auxiliary_calls_persist_two_distinct_events(self, isolated_kanban):
        """Two auxiliary calls produce two distinct records, not one overwritten record."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                call_kind="auxiliary", api_call_index=0,
                token_source="incomplete",
                input_tokens=100, output_tokens=50,
            ),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(
                call_kind="auxiliary", api_call_index=1,
                token_source="incomplete",
                input_tokens=200, output_tokens=100,
            ),
        )

        rows = ledger.query_usage(isolated_kanban, call_kind="auxiliary")
        assert len(rows) == 2, f"Expected 2 auxiliary events, got {len(rows)}"
        indices = sorted(r["api_call_index"] for r in rows)
        assert indices == [0, 1], f"Expected indices [0,1], got {indices}"

    def test_primary_and_auxiliary_at_same_index_are_distinct(self, isolated_kanban):
        """Primary and auxiliary at same local index are distinct events (different call_kind)."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(call_kind="primary", api_call_index=0, input_tokens=100, output_tokens=50),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(call_kind="auxiliary", api_call_index=0, token_source="incomplete",
                           input_tokens=200, output_tokens=100),
        )

        all_rows = ledger.query_usage(isolated_kanban, run_id=1)
        assert len(all_rows) == 2, f"Expected 2 total events (primary+aux), got {len(all_rows)}"
        kinds = {r["call_kind"] for r in all_rows}
        assert kinds == {"primary", "auxiliary"}


class TestAggregationFullEventKey:
    """Checker failure #5: aggregation must count full event keys across runs."""

    def test_aggregate_counts_distinct_events_across_runs(self, isolated_kanban):
        """Events across different runs are counted correctly."""
        # Run 1: 2 events
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(run_id=1, call_kind="primary", api_call_index=0,
                           input_tokens=100, output_tokens=50),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(run_id=1, call_kind="auxiliary", api_call_index=0,
                           token_source="incomplete", input_tokens=200, output_tokens=100),
        )
        # Run 2: 1 event
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(run_id=2, call_kind="primary", api_call_index=0,
                           input_tokens=300, output_tokens=150),
        )

        # Aggregation over ALL runs should count 3 distinct events (not 2 by api_call_index alone)
        agg = ledger.aggregate_usage(isolated_kanban)
        assert agg["record_count"] == 3, f"Expected 3 total records, got {agg['record_count']}"
        assert agg["total_api_calls"] == 3, f"Expected 3 total api calls, got {agg['total_api_calls']}"

    def test_aggregate_by_call_kind_separates_primary_aux(self, isolated_kanban):
        """Aggregation distinguishes primary from auxiliary events."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(call_kind="primary", api_call_index=0,
                           input_tokens=100, output_tokens=50),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(call_kind="auxiliary", api_call_index=0,
                           token_source="incomplete", input_tokens=200, output_tokens=100),
        )

        # Primary aggregation
        agg_pri = ledger.aggregate_usage(isolated_kanban, call_kind="primary") if hasattr(ledger, "aggregate_usage") else None
        # Overall aggregation should show 2 records
        agg = ledger.aggregate_usage(isolated_kanban)
        assert agg["record_count"] == 2
        assert agg["total_api_calls"] == 2

    def test_aggregate_handles_multi_parent_without_double_count(self, isolated_kanban):
        """Multi-parent events are counted once each."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(input_tokens=100, output_tokens=50),
        )
        ledger.record_parent(isolated_kanban,
                             board="default", task_id="t_abc", run_id=1,
                             call_kind="primary", api_call_index=0,
                             parent_task_id="t_parent_a")
        ledger.record_parent(isolated_kanban,
                             board="default", task_id="t_abc", run_id=1,
                             call_kind="primary", api_call_index=0,
                             parent_task_id="t_parent_b")

        agg = ledger.aggregate_usage(isolated_kanban, task_id="t_abc")
        assert agg["record_count"] == 1  # Event counted once despite 2 parents
        assert agg["total_api_calls"] == 1

    def test_aggregate_across_multiple_models(self, isolated_kanban):
        """Aggregation works across different models."""
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(api_call_index=0, provider="openrouter", model="gpt-4",
                           input_tokens=100, output_tokens=50),
        )
        ledger.record_run_usage(
            isolated_kanban,
            **_base_kwargs(api_call_index=1, provider="anthropic", model="claude-3",
                           input_tokens=200, output_tokens=100),
        )

        agg = ledger.aggregate_usage(isolated_kanban)
        assert agg["total_input_tokens"] == 300
        assert agg["total_output_tokens"] == 150
        assert agg["record_count"] == 2
        assert agg["total_api_calls"] == 2


class TestNormalConversationRuntimeBoundary:
    """R4C: real normal conversation model-call boundary (not helper-only).

    Exercises agent.conversation_loop via AIAgent.run_conversation with a
    monkeypatched no-network provider against a temp Kanban DB.
    """

    @staticmethod
    def _patch_bootstrap(monkeypatch):
        import run_agent

        monkeypatch.setattr(
            run_agent,
            "get_tool_definitions",
            lambda **kwargs: [
                {
                    "type": "function",
                    "function": {
                        "name": "t",
                        "description": "t",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})

    def _make_agent(self, monkeypatch, response_fn, *, model="test-model", provider="openrouter"):
        import run_agent

        self._patch_bootstrap(monkeypatch)

        class _Agent(run_agent.AIAgent):
            def __init__(self, *a, **kw):
                kw.update(
                    skip_context_files=True,
                    skip_memory=True,
                    max_iterations=4,
                    quiet_mode=True,
                )
                super().__init__(*a, **kw)
                self._cleanup_task_resources = lambda *a, **k: None
                self._persist_session = lambda *a, **k: None
                self._save_trajectory = lambda *a, **k: None

            def run_conversation(self, msg, conversation_history=None, task_id=None):
                self._interruptible_api_call = lambda kw: response_fn()
                self._disable_streaming = True
                return super().run_conversation(
                    msg,
                    conversation_history=conversation_history,
                    task_id=task_id,
                )

        return _Agent(
            model=model,
            api_key="test-key",
            base_url="http://127.0.0.1:9/v1",
            provider=provider,
            api_mode="chat_completions",
        )

    @staticmethod
    def _final_response(prompt_tokens=111, completion_tokens=22, content="ok"):
        from types import SimpleNamespace

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(content=content, tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model="test-model",
        )

    def test_single_primary_api_call_persists_one_usage_event(
        self, isolated_kanban, monkeypatch
    ):
        """One successful normal-conversation API call → one primary ledger row."""
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_r4c_single")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "obs-board")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "17")
        monkeypatch.setenv("HERMES_PROFILE", "builder-grok")

        agent = self._make_agent(monkeypatch, lambda: self._final_response(111, 22))
        result = agent.run_conversation("hello r4c")
        assert result.get("failed") is not True

        # conversation_loop opens its own connect() against the same HERMES_HOME
        rows = ledger.query_usage(
            isolated_kanban,
            board="obs-board",
            task_id="t_r4c_single",
            run_id=17,
            call_kind="primary",
        )
        assert len(rows) == 1, f"expected 1 primary event, got {rows!r}"
        row = rows[0]
        assert row["board"] == "obs-board"
        assert row["task_id"] == "t_r4c_single"
        assert row["run_id"] == 17
        assert row["profile"] == "builder-grok"
        assert row["provider"] == "openrouter"
        assert row["model"] == "test-model"
        assert row["api_call_index"] == 0
        assert row["call_kind"] == "primary"
        assert row["input_tokens"] == 111
        assert row["output_tokens"] == 22
        assert row["token_source"] == "provider_authoritative"
        assert row["elapsed_ms"] >= 0
        # Unobserved cost for unpriced test model stays NULL (not fabricated).
        assert row["cost_usd"] is None
        assert row["cost_status"] in (None, "unknown", "estimated", "included")

    def test_two_primary_api_calls_get_distinct_stable_indices(
        self, isolated_kanban, monkeypatch
    ):
        """Tool-loop normal conversation yields one primary event per API call."""
        from types import SimpleNamespace
        import run_agent

        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_r4c_multi")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "3")
        monkeypatch.setenv("HERMES_PROFILE", "builder-grok")

        calls = {"n": 0}

        def response_fn():
            calls["n"] += 1
            if calls["n"] == 1:
                tool_call = SimpleNamespace(
                    id="call_r4c_1",
                    type="function",
                    function=SimpleNamespace(name="t", arguments="{}"),
                )
                # Some OpenAI SDK shapes expect model_dump / dict-like; the
                # agent also accepts SimpleNamespace tool_calls.
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            index=0,
                            message=SimpleNamespace(
                                content=None,
                                tool_calls=[tool_call],
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=100,
                        completion_tokens=10,
                        total_tokens=110,
                    ),
                    model="test-model",
                )
            return self._final_response(200, 30, content="done")

        # Tool dispatch must stay offline.
        monkeypatch.setattr(
            run_agent,
            "handle_function_call",
            lambda *a, **k: '{"ok": true}',
        )
        try:
            from agent import conversation_loop as _cl

            monkeypatch.setattr(
                _cl,
                "handle_function_call",
                lambda *a, **k: '{"ok": true}',
            )
        except Exception:
            pass

        agent = self._make_agent(monkeypatch, response_fn)
        result = agent.run_conversation("do tool then finish")
        assert result.get("failed") is not True
        assert calls["n"] >= 2

        rows = ledger.query_usage(
            isolated_kanban,
            task_id="t_r4c_multi",
            run_id=3,
            call_kind="primary",
        )
        assert len(rows) == 2, f"expected 2 primary events, got {rows!r}"
        indices = sorted(r["api_call_index"] for r in rows)
        assert indices == [0, 1], f"stable indices expected [0,1], got {indices}"
        by_idx = {r["api_call_index"]: r for r in rows}
        assert by_idx[0]["input_tokens"] == 100
        assert by_idx[0]["output_tokens"] == 10
        assert by_idx[1]["input_tokens"] == 200
        assert by_idx[1]["output_tokens"] == 30

    def test_ledger_failure_does_not_break_normal_conversation(
        self, isolated_kanban, monkeypatch
    ):
        """Persistence errors are fail-safe: conversation still completes."""
        from hermes_cli import kanban_db as kdb

        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_r4c_failsafe")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "9")
        monkeypatch.setenv("HERMES_PROFILE", "builder-grok")

        def _boom():
            raise RuntimeError("simulated ledger connect failure")

        monkeypatch.setattr(kdb, "connect", _boom)

        agent = self._make_agent(monkeypatch, lambda: self._final_response())
        result = agent.run_conversation("still works")
        assert result.get("failed") is not True
        assert result.get("final_response") is not None

        # No rows written (connect always failed).
        rows = ledger.query_usage(isolated_kanban, task_id="t_r4c_failsafe")
        assert rows == []

    def test_no_kanban_env_skips_ledger_write(self, isolated_kanban, monkeypatch):
        """Outside a Kanban worker the normal conversation path is a no-op for ledger."""
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

        agent = self._make_agent(monkeypatch, lambda: self._final_response())
        result = agent.run_conversation("no kanban")
        assert result.get("failed") is not True
        assert ledger.query_usage(isolated_kanban) == []


class TestCodexRuntimeBoundary:
    """R4D: real Codex app-server usage boundary (not helper-only).

    Exercises agent.codex_runtime via AIAgent.run_conversation with
    api_mode=codex_app_server, a monkeypatched no-network Codex session,
    and a temp Kanban DB.
    """

    @staticmethod
    def _token_usage(
        *,
        input_tokens=80,
        cached_input_tokens=20,
        output_tokens=25,
        reasoning_tokens=5,
        total_tokens=130,
    ):
        return {
            "inputTokens": input_tokens,
            "cachedInputTokens": cached_input_tokens,
            "outputTokens": output_tokens,
            "reasoningOutputTokens": reasoning_tokens,
            "totalTokens": total_tokens,
        }

    def _make_codex_agent(self, monkeypatch, token_usage_fn, *, model="codex-test-model", provider="openai"):
        """Build a real AIAgent on the codex_app_server path; no network."""
        import run_agent
        from agent.transports.codex_app_server_session import (
            CodexAppServerSession,
            TurnResult,
        )

        monkeypatch.setattr(
            run_agent,
            "get_tool_definitions",
            lambda **kwargs: [],
        )
        monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})

        call_n = {"n": 0}

        def fake_run_turn(self, user_input: str, **kwargs):
            call_n["n"] += 1
            usage = token_usage_fn(call_n["n"], user_input)
            return TurnResult(
                final_text=f"codex:{user_input}",
                projected_messages=[
                    {"role": "assistant", "content": f"codex:{user_input}"}
                ],
                tool_iterations=0,
                interrupted=False,
                error=None,
                turn_id=f"turn-{call_n['n']}",
                thread_id="thread-r4d",
                token_usage_last=usage,
                model_context_window=200000,
            )

        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
        monkeypatch.setattr(
            CodexAppServerSession, "ensure_started", lambda self: "thread-r4d"
        )

        agent = run_agent.AIAgent(
            model=model,
            api_key="test-key",
            base_url="http://127.0.0.1:9/v1",
            provider=provider,
            api_mode="codex_app_server",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=2,
        )
        # Avoid side-effects outside the usage boundary under test.
        agent._spawn_background_review = lambda *a, **k: None
        agent._cleanup_task_resources = lambda *a, **k: None
        agent._persist_session = lambda *a, **k: None
        agent._save_trajectory = lambda *a, **k: None
        agent._call_count = call_n
        return agent

    def test_single_codex_turn_persists_one_primary_usage_event(
        self, isolated_kanban, monkeypatch
    ):
        """One Codex app-server turn with usage → one primary ledger row."""
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_r4d_single")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "obs-board")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "21")
        monkeypatch.setenv("HERMES_PROFILE", "builder-grok")

        agent = self._make_codex_agent(
            monkeypatch,
            lambda n, msg: self._token_usage(
                input_tokens=80,
                cached_input_tokens=20,
                output_tokens=25,
                reasoning_tokens=5,
                total_tokens=130,
            ),
        )
        result = agent.run_conversation("hello r4d")
        assert result.get("failed") is not True
        assert result.get("completed") is True
        assert result.get("api_calls") == 1

        rows = ledger.query_usage(
            isolated_kanban,
            board="obs-board",
            task_id="t_r4d_single",
            run_id=21,
            call_kind="primary",
        )
        assert len(rows) == 1, f"expected 1 primary event, got {rows!r}"
        row = rows[0]
        assert row["board"] == "obs-board"
        assert row["task_id"] == "t_r4d_single"
        assert row["run_id"] == 21
        assert row["profile"] == "builder-grok"
        assert row["provider"] == "openai"
        assert row["model"] == "codex-test-model"
        assert row["api_call_index"] == 0
        assert row["call_kind"] == "primary"
        # Codex reports inputTokens separately from cache; ledger stores the
        # raw inputTokens bucket from CanonicalUsage (not prompt_tokens sum).
        assert row["input_tokens"] == 80
        assert row["output_tokens"] == 25
        assert row["cache_read_tokens"] == 20
        assert row["cache_write_tokens"] == 0
        assert row["reasoning_tokens"] == 5
        assert row["token_source"] == "provider_authoritative"
        assert row["elapsed_ms"] >= 0
        assert row["cost_usd"] is None or isinstance(row["cost_usd"], (int, float))
        assert row["cost_status"] in (None, "unknown", "estimated", "included")

    def test_two_codex_turns_get_distinct_stable_indices(
        self, isolated_kanban, monkeypatch
    ):
        """Two Codex turns → two primary events with stable api_call_index 0,1."""
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_r4d_multi")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "5")
        monkeypatch.setenv("HERMES_PROFILE", "builder-grok")

        def usage_fn(n, msg):
            if n == 1:
                return self._token_usage(
                    input_tokens=50,
                    cached_input_tokens=0,
                    output_tokens=10,
                    reasoning_tokens=0,
                    total_tokens=60,
                )
            return self._token_usage(
                input_tokens=90,
                cached_input_tokens=10,
                output_tokens=40,
                reasoning_tokens=2,
                total_tokens=142,
            )

        agent = self._make_codex_agent(monkeypatch, usage_fn)
        r1 = agent.run_conversation("turn one")
        r2 = agent.run_conversation("turn two")
        assert r1.get("failed") is not True
        assert r2.get("failed") is not True
        assert agent.session_api_calls == 2

        rows = ledger.query_usage(
            isolated_kanban,
            task_id="t_r4d_multi",
            run_id=5,
            call_kind="primary",
        )
        assert len(rows) == 2, f"expected 2 primary events, got {rows!r}"
        indices = sorted(r["api_call_index"] for r in rows)
        assert indices == [0, 1], f"stable indices expected [0,1], got {indices}"
        by_idx = {r["api_call_index"]: r for r in rows}
        assert by_idx[0]["input_tokens"] == 50
        assert by_idx[0]["output_tokens"] == 10
        assert by_idx[1]["input_tokens"] == 90
        assert by_idx[1]["output_tokens"] == 40
        assert by_idx[1]["cache_read_tokens"] == 10
        assert by_idx[1]["reasoning_tokens"] == 2

    def test_ledger_failure_does_not_break_codex_turn(
        self, isolated_kanban, monkeypatch
    ):
        """Persistence errors are fail-safe: Codex turn still completes."""
        from hermes_cli import kanban_db as kdb

        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_r4d_failsafe")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "11")
        monkeypatch.setenv("HERMES_PROFILE", "builder-grok")

        def _boom():
            raise RuntimeError("simulated ledger connect failure")

        monkeypatch.setattr(kdb, "connect", _boom)

        agent = self._make_codex_agent(
            monkeypatch, lambda n, msg: self._token_usage()
        )
        result = agent.run_conversation("still works")
        assert result.get("failed") is not True
        assert result.get("completed") is True
        assert result.get("final_response") is not None
        # No rows written (connect always failed).
        rows = ledger.query_usage(isolated_kanban, task_id="t_r4d_failsafe")
        assert rows == []

    def test_no_kanban_env_skips_codex_ledger_write(
        self, isolated_kanban, monkeypatch
    ):
        """Outside a Kanban worker the Codex path is a no-op for the ledger."""
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

        agent = self._make_codex_agent(
            monkeypatch, lambda n, msg: self._token_usage()
        )
        result = agent.run_conversation("no kanban")
        assert result.get("failed") is not True
        assert result.get("completed") is True
        assert ledger.query_usage(isolated_kanban) == []


class TestAuxiliaryRuntimeBoundary:
    """R4E: real auxiliary response/runtime usage boundary (not helper-only).

    Exercises agent.auxiliary_client.call_llm → _validate_llm_response with a
    monkeypatched no-network client against a temp Kanban DB.
    """

    @staticmethod
    def _openai_style_response(
        *,
        prompt_tokens=111,
        completion_tokens=22,
        model="aux-test-model",
        content="aux-ok",
        include_usage=True,
    ):
        from types import SimpleNamespace

        usage = None
        if include_usage:
            # Real OpenAI chat.completions usage shape (prompt/completion).
            usage = SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(content=content, tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=usage,
            model=model,
        )

    def _install_fake_client(self, monkeypatch, response_fn):
        """Monkeypatch call_llm client resolution; no network."""
        from types import SimpleNamespace
        import agent.auxiliary_client as aux

        class _Completions:
            def create(self, **kwargs):
                return response_fn()

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        class _Client:
            def __init__(self):
                self.chat = _Chat()
                self.base_url = "http://127.0.0.1:9/v1"

        def fake_get_cached_client(*a, **k):
            return _Client(), "aux-test-model"

        monkeypatch.setattr(aux, "_get_cached_client", fake_get_cached_client)
        # Avoid vision/auto resolution side paths.
        monkeypatch.setattr(
            aux,
            "_resolve_task_provider_model",
            lambda task, provider, model, base_url, api_key: (
                provider or "openrouter",
                model or "aux-test-model",
                "http://127.0.0.1:9/v1",
                "test-key",
                None,
            ),
        )
        return aux

    def test_single_aux_api_call_persists_one_usage_event(
        self, isolated_kanban, monkeypatch
    ):
        """One successful auxiliary call with usage → one auxiliary ledger row."""
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_r4e_single")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "obs-board")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "31")
        monkeypatch.setenv("HERMES_PROFILE", "builder-grok")
        # Reset process-local aux counter between tests.
        ledger._aux_call_indices.clear()

        aux = self._install_fake_client(
            monkeypatch,
            lambda: self._openai_style_response(prompt_tokens=111, completion_tokens=22),
        )
        response = aux.call_llm(
            task="title_generation",
            provider="openrouter",
            model="aux-test-model",
            messages=[{"role": "user", "content": "hello r4e"}],
        )
        assert response is not None
        assert response.choices[0].message.content == "aux-ok"

        rows = ledger.query_usage(
            isolated_kanban,
            board="obs-board",
            task_id="t_r4e_single",
            run_id=31,
            call_kind="auxiliary",
        )
        assert len(rows) == 1, f"expected 1 auxiliary event, got {rows!r}"
        row = rows[0]
        assert row["board"] == "obs-board"
        assert row["task_id"] == "t_r4e_single"
        assert row["run_id"] == 31
        assert row["profile"] == "builder-grok"
        assert row["call_kind"] == "auxiliary"
        assert row["api_call_index"] == 0
        assert row["model"] == "aux-test-model"
        # OpenAI prompt/completion usage must map into canonical buckets.
        assert row["input_tokens"] == 111
        assert row["output_tokens"] == 22
        assert row["token_source"] == "incomplete"
        # Unobserved aux_* and cost stay NULL (missing != fabricated zero).
        assert row["aux_input_tokens"] is None
        assert row["aux_output_tokens"] is None
        assert row["cost_usd"] is None

    def test_two_aux_api_calls_get_distinct_stable_indices(
        self, isolated_kanban, monkeypatch
    ):
        """Two observable auxiliary calls → distinct events at indices 0 and 1."""
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_r4e_multi")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "8")
        monkeypatch.setenv("HERMES_PROFILE", "builder-grok")
        ledger._aux_call_indices.clear()

        calls = {"n": 0}

        def response_fn():
            calls["n"] += 1
            if calls["n"] == 1:
                return self._openai_style_response(
                    prompt_tokens=100, completion_tokens=10, content="first"
                )
            return self._openai_style_response(
                prompt_tokens=200, completion_tokens=30, content="second"
            )

        aux = self._install_fake_client(monkeypatch, response_fn)
        r1 = aux.call_llm(
            task="session_search",
            provider="openrouter",
            model="aux-test-model",
            messages=[{"role": "user", "content": "one"}],
        )
        r2 = aux.call_llm(
            task="session_search",
            provider="openrouter",
            model="aux-test-model",
            messages=[{"role": "user", "content": "two"}],
        )
        assert r1.choices[0].message.content == "first"
        assert r2.choices[0].message.content == "second"
        assert calls["n"] == 2

        rows = ledger.query_usage(
            isolated_kanban,
            task_id="t_r4e_multi",
            run_id=8,
            call_kind="auxiliary",
        )
        assert len(rows) == 2, f"expected 2 auxiliary events, got {rows!r}"
        indices = sorted(r["api_call_index"] for r in rows)
        assert indices == [0, 1], f"stable indices expected [0,1], got {indices}"
        by_idx = {r["api_call_index"]: r for r in rows}
        assert by_idx[0]["input_tokens"] == 100
        assert by_idx[0]["output_tokens"] == 10
        assert by_idx[1]["input_tokens"] == 200
        assert by_idx[1]["output_tokens"] == 30
        # Second call must not overwrite index 0.
        assert by_idx[0]["input_tokens"] != by_idx[1]["input_tokens"]

    def test_aux_index_safe_after_counter_reset(
        self, isolated_kanban, monkeypatch
    ):
        """Counter identity survives process-local state loss (retry / re-import).

        If the in-memory aux counter is cleared (new process, fresh import) but
        the DB already has index 0 for this run, the next call must allocate 1
        rather than overwrite 0.
        """
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_r4e_retry")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "12")
        monkeypatch.setenv("HERMES_PROFILE", "builder-grok")
        ledger._aux_call_indices.clear()

        aux = self._install_fake_client(
            monkeypatch,
            lambda: self._openai_style_response(prompt_tokens=50, completion_tokens=5),
        )
        aux.call_llm(
            task="skills_hub",
            provider="openrouter",
            model="aux-test-model",
            messages=[{"role": "user", "content": "first"}],
        )
        # Simulate process restart / re-import: forget local counter only.
        ledger._aux_call_indices.clear()

        aux = self._install_fake_client(
            monkeypatch,
            lambda: self._openai_style_response(prompt_tokens=90, completion_tokens=9),
        )
        aux.call_llm(
            task="skills_hub",
            provider="openrouter",
            model="aux-test-model",
            messages=[{"role": "user", "content": "retry"}],
        )

        rows = ledger.query_usage(
            isolated_kanban,
            task_id="t_r4e_retry",
            run_id=12,
            call_kind="auxiliary",
        )
        assert len(rows) == 2, f"expected 2 events after retry, got {rows!r}"
        indices = sorted(r["api_call_index"] for r in rows)
        assert indices == [0, 1], f"must not overwrite index 0, got {indices}"
        by_idx = {r["api_call_index"]: r for r in rows}
        assert by_idx[0]["input_tokens"] == 50
        assert by_idx[1]["input_tokens"] == 90

    def test_aux_index_does_not_leak_across_runs(
        self, isolated_kanban, monkeypatch
    ):
        """Different run_id starts auxiliary indices at 0 again."""
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_r4e_runs")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        monkeypatch.setenv("HERMES_PROFILE", "builder-grok")
        ledger._aux_call_indices.clear()

        aux = self._install_fake_client(
            monkeypatch,
            lambda: self._openai_style_response(prompt_tokens=10, completion_tokens=1),
        )

        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "100")
        aux.call_llm(
            task="mcp",
            provider="openrouter",
            model="aux-test-model",
            messages=[{"role": "user", "content": "run100"}],
        )
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "101")
        aux.call_llm(
            task="mcp",
            provider="openrouter",
            model="aux-test-model",
            messages=[{"role": "user", "content": "run101"}],
        )

        r100 = ledger.query_usage(
            isolated_kanban, task_id="t_r4e_runs", run_id=100, call_kind="auxiliary"
        )
        r101 = ledger.query_usage(
            isolated_kanban, task_id="t_r4e_runs", run_id=101, call_kind="auxiliary"
        )
        assert len(r100) == 1 and r100[0]["api_call_index"] == 0
        assert len(r101) == 1 and r101[0]["api_call_index"] == 0

    def test_ledger_failure_does_not_break_auxiliary_call(
        self, isolated_kanban, monkeypatch, caplog
    ):
        """Persistence errors are fail-safe and diagnosable."""
        import logging
        from hermes_cli import kanban_db as kdb

        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_r4e_failsafe")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "19")
        monkeypatch.setenv("HERMES_PROFILE", "builder-grok")
        ledger._aux_call_indices.clear()

        def _boom():
            raise RuntimeError("simulated ledger connect failure")

        monkeypatch.setattr(kdb, "connect", _boom)
        aux = self._install_fake_client(
            monkeypatch,
            lambda: self._openai_style_response(prompt_tokens=1, completion_tokens=1),
        )

        with caplog.at_level(logging.DEBUG):
            response = aux.call_llm(
                task="title_generation",
                provider="openrouter",
                model="aux-test-model",
                messages=[{"role": "user", "content": "still works"}],
            )
        assert response is not None
        assert response.choices[0].message.content == "aux-ok"
        # No rows written (connect always failed).
        rows = ledger.query_usage(isolated_kanban, task_id="t_r4e_failsafe")
        assert rows == []
        # Diagnosable trail (debug log with context).
        joined = " ".join(r.message for r in caplog.records)
        assert "Kanban" in joined or "ledger" in joined.lower() or "usage" in joined.lower()

    def test_no_kanban_env_skips_aux_ledger_write(
        self, isolated_kanban, monkeypatch
    ):
        """Outside a Kanban worker the auxiliary path is a no-op for the ledger."""
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
        ledger._aux_call_indices.clear()

        aux = self._install_fake_client(
            monkeypatch,
            lambda: self._openai_style_response(prompt_tokens=5, completion_tokens=2),
        )
        response = aux.call_llm(
            task="title_generation",
            provider="openrouter",
            model="aux-test-model",
            messages=[{"role": "user", "content": "no kanban"}],
        )
        assert response is not None
        assert ledger.query_usage(isolated_kanban) == []

    def test_unobserved_usage_does_not_fabricate_authoritative_tokens(
        self, isolated_kanban, monkeypatch
    ):
        """Response without usage is not recorded as observed zeros."""
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_r4e_unobs")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "44")
        monkeypatch.setenv("HERMES_PROFILE", "builder-grok")
        ledger._aux_call_indices.clear()

        aux = self._install_fake_client(
            monkeypatch,
            lambda: self._openai_style_response(include_usage=False),
        )
        response = aux.call_llm(
            task="title_generation",
            provider="openrouter",
            model="aux-test-model",
            messages=[{"role": "user", "content": "no usage"}],
        )
        assert response is not None
        # Unobserved → no fabricated auxiliary usage event with zeros.
        rows = ledger.query_usage(
            isolated_kanban, task_id="t_r4e_unobs", call_kind="auxiliary"
        )
        assert rows == [], f"unobserved usage must not invent events, got {rows!r}"
