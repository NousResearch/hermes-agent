"""RED test for #126979: sidebar usage total must include subagent
sessions and auxiliary (task != '') model usage."""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


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
