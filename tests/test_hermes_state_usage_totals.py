"""Sidebar usage total must include subagent sessions and aux usage (#126979).

``usage_totals`` feeds the Desktop sidebar total. Subagent (delegate child)
sessions carry independent counters, and auxiliary calls (title generation,
compression, ...) live only in ``session_model_usage`` with ``task != ''`` —
both are real spend and must be counted, while the main-loop ``task = ''``
per-model rows (which mirror the ``sessions`` counters) must never double
count.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


class TestUsageTotalsIncludesChildrenAndAux:
    def test_child_and_aux_tokens_and_cost_counted(self, db):
        db.create_session("root", "cli")
        db.append_message("root", role="user", content="hi")
        db.update_token_counts(
            "root", input_tokens=100, output_tokens=50,
            estimated_cost_usd=1.0,
        )
        db.create_session("child", "cli", parent_session_id="root")
        db.append_message("child", role="user", content="hi")
        db.update_token_counts(
            "child", input_tokens=200, output_tokens=100,
            estimated_cost_usd=2.0,
        )
        db.record_auxiliary_usage(
            "root", "title_generation",
            input_tokens=10, output_tokens=5,
            estimated_cost_usd=0.5,
        )

        totals = db.usage_totals()

        # main-loop rows (100+50, 200+100) + aux row (10+5); the
        # task='' per-model rows written by update_token_counts must
        # NOT be double-counted.
        assert totals["tokens"] == 465
        assert totals["cost_usd"] == pytest.approx(3.5)

    def test_archived_sessions_and_their_aux_excluded(self, db):
        db.create_session("live", "cli")
        db.append_message("live", role="user", content="hi")
        db.update_token_counts("live", input_tokens=100, output_tokens=0)
        db.create_session("gone", "cli")
        db.append_message("gone", role="user", content="hi")
        db.update_token_counts("gone", input_tokens=1000, output_tokens=0)
        db.record_auxiliary_usage(
            "gone", "compression", input_tokens=500, output_tokens=0)
        db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET archived = 1 WHERE id = 'gone'"))

        totals = db.usage_totals()

        assert totals["tokens"] == 100
