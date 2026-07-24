"""Tests for the cost_status sticky priority ladder (issue #67764).

Replaces the previous "most-recent-call-wins" semantics at four layers:

1. SQL COALESCE in ``hermes_state.py`` UPDATE ``sessions`` (absolute + incremental paths)
2. SQL COALESCE in ``hermes_state.py`` INSERT/ON CONFLICT on ``session_model_usage``
3. In-memory assignments in ``agent/conversation_loop.py:2321`` and
   ``agent/codex_runtime.py:150``
4. /insights aggregator per-model dict loop in ``agent/insights.py:583``

The new semantics: take the maximum-rank contribution across all sources.
Ranking (high → low confidence): ``actual`` (3) > ``included`` (2) >
``estimated`` (1) > ``unknown`` (0). Once any call has reported
``"actual"``, the aggregated value stays ``"actual"`` forever.

This file exercises the Python-level helper directly and the SQL-level
aggregation via a real ``SessionDB`` against a temp SQLite file.
"""
import pytest

from hermes_state import SessionDB
from agent.usage_pricing import (
    CostStatus,
    sticky_cost_status,
    _COST_STATUS_PRIORITY,
)


# ── Python helper unit tests ──────────────────────────────────────────────────


class TestStickyHelperRankMapping:
    """The priority mapping constants must match the brief."""

    def test_priority_order_actual_highest(self):
        assert _COST_STATUS_PRIORITY["actual"] == 3

    def test_priority_order_included_above_estimated(self):
        # Confidence rank: included ($0 definitive) beats estimated ($X.X approximate).
        assert _COST_STATUS_PRIORITY["included"] > _COST_STATUS_PRIORITY["estimated"]

    def test_priority_order_estimated_above_unknown(self):
        assert _COST_STATUS_PRIORITY["estimated"] > _COST_STATUS_PRIORITY["unknown"]

    def test_all_three_explicit_cost_statuses_have_a_rank(self):
        for status in ("actual", "included", "estimated", "unknown"):
            assert status in _COST_STATUS_PRIORITY
            assert _COST_STATUS_PRIORITY[status] >= 0


class TestStickyHelperBehavior:
    """``sticky_cost_status(current, new)`` must promote to the higher rank,
    never demote."""

    def test_actual_is_sticky_downgrade(self):
        """Once we've seen ``actual``, an ``estimated`` call can't downgrade us."""
        assert sticky_cost_status("actual", "estimated") == "actual"

    def test_actual_is_sticky_unknown(self):
        assert sticky_cost_status("actual", "unknown") == "actual"

    def test_included_is_sticky_estimated(self):
        # included ($0 definitive) beats estimated ($X.X approximate).
        assert sticky_cost_status("included", "estimated") == "included"

    def test_estimated_is_sticky_unknown(self):
        assert sticky_cost_status("estimated", "unknown") == "estimated"

    def test_new_actual_promotes_estimated(self):
        """A new ``actual`` call promotes an existing ``estimated`` aggregate."""
        assert sticky_cost_status("estimated", "actual") == "actual"

    def test_new_included_promotes_estimated(self):
        assert sticky_cost_status("estimated", "included") == "included"

    def test_new_included_promotes_unknown(self):
        assert sticky_cost_status("unknown", "included") == "included"

    def test_equal_higher_keeps_current(self):
        """When both ranks are equal, the current value is preserved (so the
        helper is *idempotent* on the same value pair)."""
        assert sticky_cost_status("estimated", "estimated") == "estimated"
        assert sticky_cost_status("unknown", "unknown") == "unknown"

    def test_none_and_empty_are_rank_zero(self):
        """``None`` and ``""`` are treated as below ``"unknown"``."""
        assert sticky_cost_status(None, "estimated") == "estimated"
        assert sticky_cost_status("estimated", None) == "estimated"
        assert sticky_cost_status("", "actual") == "actual"
        assert sticky_cost_status("actual", "") == "actual"

    def test_both_none_returns_estimated(self):
        """No data on either side → sane default."""
        assert sticky_cost_status(None, None) == "estimated"

    def test_unknown_literal_value_falls_back_safely(self):
        """A typo / non-Literal value must not crash; treated as rank 0."""
        assert sticky_cost_status("bogus", "actual") == "actual"
        # When neither side is recognized, fall back to "estimated".
        assert sticky_cost_status("bogus", "also_bogus") == "estimated"


class TestStickyHelperReturnType:
    """Return type is always a recognized Literal — never None, never a typo."""

    def test_return_type_is_always_literal(self):
        cases = [
            ("actual", "estimated"),
            ("estimated", "actual"),
            (None, "included"),
            ("unknown", None),
            ("", ""),
            ("typo_here", "also_typo"),
        ]
        for current, new in cases:
            result = sticky_cost_status(current, new)
            assert result in ("actual", "estimated", "included", "unknown"), (
                f"sticky_cost_status({current!r}, {new!r}) returned {result!r}"
            )


# ── SQL aggregation tests against real SessionDB ─────────────────────────────


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed_calls(db, session_id, status_sequence):
    """Drive ``update_token_counts`` through a sequence of (status) calls
    on the same main-loop per-model row for ``session_id``.

    Each call uses default token counts but a specific ``cost_status`` so we
    can verify that the per-model row reflects the highest-rank status.
    """
    db.create_session(session_id, source="cli")
    for status in status_sequence:
        db.update_token_counts(
            session_id,
            input_tokens=100,
            output_tokens=10,
            model="m",
            billing_provider="anthropic",
            billing_base_url="",
            billing_mode="",
            api_call_count=1,
            estimated_cost_usd=0.001,
            cost_status=status,
            cost_source="provider_models_api",
        )


def _row_cost_status(db, session_id, task, model):
    with db._lock:
        cursor = db._conn.execute(
            "SELECT cost_status FROM session_model_usage "
            "WHERE session_id = ? AND task = ? AND model = ?",
            (session_id, task, model),
        )
        row = cursor.fetchone()
    return row["cost_status"] if row else None


def _sessions_row_cost_status(db, session_id):
    with db._lock:
        cursor = db._conn.execute(
            "SELECT cost_status FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
    return row["cost_status"] if row else None


class TestSqlSessionModelUsagePriorityLadder:
    """Per-model rows: the last-iterated status wins for that row (since
    each model-row is the accumulation of one model's calls), but the row's
    status is sticky once any of its UPSERTs reported ``actual``."""

    def test_actual_preserved_across_subsequent_estimated_calls(self, db):
        """Send actual, then estimated, then estimated — the row stays actual."""
        _seed_calls(db, "s1", ["actual", "estimated", "estimated"])
        assert _row_cost_status(db, "s1", "", "m") == "actual"

    def test_included_preserved_across_subsequent_estimated(self, db):
        _seed_calls(db, "s1", ["included", "estimated"])
        assert _row_cost_status(db, "s1", "", "m") == "included"

    def test_estimated_preserved_across_subsequent_unknown(self, db):
        """``estimated`` outranks ``unknown`` per the priority ladder
        (rank 1 vs rank 0), so a later ``unknown`` call does not demote the
        per-model row. This is max-rank semantics across the whole ladder;
        ``unknown`` is the floor but never actively overwrites ``estimated``.
        """
        _seed_calls(db, "s1", ["estimated", "unknown"])
        assert _row_cost_status(db, "s1", "", "m") == "estimated"

    def test_actual_promotes_existing_estimated(self, db):
        """Estimated first, then actual — the row promotes to actual."""
        _seed_calls(db, "s1", ["estimated", "actual"])
        assert _row_cost_status(db, "s1", "", "m") == "actual"


class TestSqlSessionsSummaryPriorityLadder:
    """The ``sessions`` aggregate row's cost_status mirrors the same ladder —
    the SQL at ``update_token_counts`` applies the same sticky-up rule."""

    def test_sessions_status_sticky_after_actual_call(self, db):
        """After an ``actual`` call, even an ``estimated`` caller doesn't
        downgrade the sessions row's status."""
        # Note: update_token_counts also pushes per-model rows; we're testing
        # the sessions aggregate here.
        db.create_session("s1", source="cli")
        db.update_token_counts(
            "s1",
            input_tokens=100, output_tokens=10,
            model="anthropic/claude-opus-4-8",
            billing_provider="anthropic",
            api_call_count=1,
            estimated_cost_usd=0.10,
            cost_status="actual",
        )
        # Now an "estimated" call (e.g. fallback to estimation). The aggregate
        # should NOT downgrade from actual to estimated.
        db.update_token_counts(
            "s1",
            input_tokens=100, output_tokens=10,
            model="anthropic/claude-haiku-4-5",
            billing_provider="anthropic",
            api_call_count=1,
            estimated_cost_usd=0.05,
            cost_status="estimated",
        )
        assert _sessions_row_cost_status(db, "s1") == "actual"

    def test_sessions_status_promotes_to_actual(self, db):
        """A first call estimated, a second call actual — the aggregate
        promotes to actual."""
        db.create_session("s1", source="cli")
        db.update_token_counts(
            "s1",
            input_tokens=100, output_tokens=10,
            model="m",
            billing_provider="anthropic",
            api_call_count=1,
            estimated_cost_usd=0.10,
            cost_status="estimated",
        )
        db.update_token_counts(
            "s1",
            input_tokens=100, output_tokens=10,
            model="m",
            billing_provider="anthropic",
            api_call_count=1,
            estimated_cost_usd=0.05,
            cost_status="actual",
        )
        assert _sessions_row_cost_status(db, "s1") == "actual"

    def test_sessions_status_with_no_cost_status_input(self, db):
        """When ``cost_status`` is None, the value falls through to 'estimated'."""
        db.create_session("s1", source="cli")
        db.update_token_counts(
            "s1",
            input_tokens=100, output_tokens=10,
            model="m",
            billing_provider="anthropic",
            api_call_count=1,
            estimated_cost_usd=0.10,
            cost_status=None,
        )
        assert _sessions_row_cost_status(db, "s1") == "estimated"

    def test_sessions_status_with_unknown_literal_input(self, db):
        """When ``cost_status`` is an unrecognized literal (e.g. a typo),
        the implementation falls back to 'estimated' rather than passing the
        bad value through to the SQL string. SQL-injection safety check."""
        db.create_session("s1", source="cli")
        # This should NOT raise — the allow-list sanitizer catches the typo.
        db.update_token_counts(
            "s1",
            input_tokens=100, output_tokens=10,
            model="m",
            billing_provider="anthropic",
            api_call_count=1,
            estimated_cost_usd=0.10,
            cost_status="'; DROP TABLE sessions; --",
        )
        assert _sessions_row_cost_status(db, "s1") == "estimated"
        # Sessions table still exists (the malicious string was rejected).
        assert _sessions_row_cost_status(db, "s1") == "estimated"


# ── In-memory aggregator test (insights) ──────────────────────────────────────


class TestInsightsPerModelStickyAggregation:
    """Reproduces the per-model dict loop logic in ``agent/insights.py:583``
    to verify the sticky aggregation is correct."""

    def test_actual_promotes_in_per_model_dict(self):
        """When ``_accumulate`` processes an estimated row, then an actual
        row, the per-model dict should end up at ``actual``."""

        def accumulate_per_model(calls, model):
            """Simplified version of the per-model aggregation, for testing."""
            d = {"cost_status": "unknown"}
            for status in calls:
                # This mimics agent/insights.py:583 with our sticky helper.
                d["cost_status"] = sticky_cost_status(d.get("cost_status"), status)
            return d["cost_status"]

        result = accumulate_per_model(
            ["estimated", "unknown", "actual"], model="m"
        )
        assert result == "actual"

    def test_included_beats_estimated_in_per_model_dict(self):
        def accumulate_per_model(calls, model):
            d = {"cost_status": "unknown"}
            for status in calls:
                d["cost_status"] = sticky_cost_status(d.get("cost_status"), status)
            return d["cost_status"]

        result = accumulate_per_model(["estimated", "included"], model="m")
        assert result == "included"


# ── Dispatch shape tests (in-memory agent attributes) ────────────────────────


class TestAgentAttributeDispatch:
    """The two in-memory assignment sites must call sticky_cost_status."""

    def test_conversation_loop_uses_sticky_helper(self):
        """Inspect the AST at ``agent/conversation_loop.py`` around the
        in-memory assignment to confirm it dispatches via sticky_cost_status,
        not direct ``=``."""
        import inspect
        from agent import conversation_loop
        src = inspect.getsource(conversation_loop)
        # Find lines around the cost_status assignment to check.
        for i, line in enumerate(src.splitlines()):
            if "agent.session_cost_status = cost_result.status" in line:
                # This pattern was the old behavior; the new code should
                # never have direct assignment.
                assert False, f"Direct assignment at conversation_loop:{i+1}: {line}"
            if "agent.session_cost_status = sticky_cost_status(" in line:
                # The new dispatch is present.
                return
        # If neither pattern found, that's also a problem.
        assert False, "Couldn't find either dispatch shape in conversation_loop"

    def test_codex_runtime_uses_sticky_helper(self):
        import inspect
        from agent import codex_runtime
        src = inspect.getsource(codex_runtime)
        for i, line in enumerate(src.splitlines()):
            if "agent.session_cost_status = cost_result.status" in line:
                assert False, f"Direct assignment at codex_runtime:{i+1}: {line}"
            if "agent.session_cost_status = sticky_cost_status(" in line:
                return
        assert False, "Couldn't find either dispatch shape in codex_runtime"


# ── End-to-end: the brief's reproduction scenarios ───────────────────────────


class TestEndToEndBriefScenarios:
    """Reproduces the failure-mode scenarios from the issue body."""

    def test_scenario_a_downgrade_actual_to_estimated_blocked(self):
        """Scenario A from the brief: 100 actual calls followed by 1
        estimated call. Without sticky behavior the row would downgrade."""
        d = "unknown"  # starting state
        # 100 actual calls.
        for _ in range(100):
            d = sticky_cost_status(d, "actual")
        assert d == "actual"
        # 1 estimated call — must NOT downgrade.
        d = sticky_cost_status(d, "estimated")
        assert d == "actual", "actual must be sticky against estimated"

    def test_scenario_b_upgrade_estimated_to_actual(self):
        """Scenario B from the brief: 100 estimated calls then 1 actual.
        Without sticky behavior the row stays estimated; with it, promotes."""
        d = "unknown"
        for _ in range(100):
            d = sticky_cost_status(d, "estimated")
        assert d == "estimated"
        d = sticky_cost_status(d, "actual")
        assert d == "actual", "new actual must promote existing estimated"
