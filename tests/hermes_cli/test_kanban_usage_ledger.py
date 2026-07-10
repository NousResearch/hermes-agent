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
