"""Tests for ArtifactLedger service."""

import pytest
from pathlib import Path


@pytest.fixture()
def ledger(tmp_path):
    from hermes_cli.code.artifact_ledger import ArtifactLedger
    return ArtifactLedger(db_path=tmp_path / "state.db")


class TestArtifactLedger:
    def test_create_task_intake_artifact(self, ledger):
        art = ledger.create_artifact(
            category="task_intake",
            content="# Task\n\nImplement login flow.",
            title="Login Task",
        )
        assert art["id"]
        assert art["category"] == "task_intake"
        assert art["title"] == "Login Task"
        assert "login" in art["content"]

    def test_create_adr_artifact(self, ledger):
        art = ledger.create_artifact(
            category="adr",
            content="# ADR-001: Use PostgreSQL",
        )
        assert art["category"] == "adr"

    def test_invalid_category_raises(self, ledger):
        with pytest.raises(ValueError, match="Unknown artifact category"):
            ledger.create_artifact(category="not_a_real_category", content="x")

    def test_list_by_code_session(self, ledger):
        session_id = "sess-abc"
        ledger.create_artifact(category="task_intake", content="a", code_session_id=session_id)
        ledger.create_artifact(category="prd_lite", content="b", code_session_id=session_id)
        ledger.create_artifact(category="adr", content="c", code_session_id="other-session")

        arts = ledger.list_artifacts(code_session_id=session_id)
        assert len(arts) == 2
        categories = {a["category"] for a in arts}
        assert "task_intake" in categories
        assert "prd_lite" in categories
        assert "adr" not in categories

    def test_list_by_category(self, ledger):
        ledger.create_artifact(category="task_intake", content="a")
        ledger.create_artifact(category="task_intake", content="b")
        ledger.create_artifact(category="adr", content="c")

        arts = ledger.list_artifacts(category="task_intake")
        assert all(a["category"] == "task_intake" for a in arts)
        assert len(arts) == 2

    def test_list_by_orchestrated_run(self, ledger):
        run_id = "run-xyz"
        ledger.create_artifact(category="implementation_plan", content="plan", orchestrated_run_id=run_id)
        ledger.create_artifact(category="test_report", content="results", orchestrated_run_id=run_id)
        ledger.create_artifact(category="adr", content="other")

        arts = ledger.list_artifacts(orchestrated_run_id=run_id)
        assert len(arts) == 2

    def test_get_artifact_by_id(self, ledger):
        art = ledger.create_artifact(category="diff_summary", content="added 5 lines")
        fetched = ledger.get_artifact(art["id"])
        assert fetched is not None
        assert fetched["id"] == art["id"]

    def test_get_missing_artifact_returns_none(self, ledger):
        assert ledger.get_artifact("nonexistent-id") is None

    def test_update_artifact_content(self, ledger):
        art = ledger.create_artifact(category="review_report", content="initial")
        updated = ledger.update_artifact(art["id"], content="revised review")
        assert updated is not None
        assert updated["content"] == "revised review"

    def test_categories_list(self, ledger):
        cats = ledger.categories()
        assert "task_intake" in cats
        assert "adr" in cats
        assert "deploy_plan" in cats
        assert "memory_update" in cats

    def test_metadata_roundtrip(self, ledger):
        meta = {"source": "test", "confidence": 0.9}
        art = ledger.create_artifact(category="architecture_note", content="arch", metadata=meta)
        fetched = ledger.get_artifact(art["id"])
        assert fetched["metadata"]["source"] == "test"
        assert fetched["metadata"]["confidence"] == pytest.approx(0.9)
