"""Tests for ArtifactLedger."""

import pytest

from hermes_cli.code.artifact_ledger import ArtifactLedger


@pytest.fixture()
def ledger(tmp_path):
    return ArtifactLedger(db_path=tmp_path / "state.db")


def test_create_and_list_artifact(ledger):
    artifact = ledger.create_artifact(
        "task_intake",
        "Implement parser",
        title="Parser Task",
        workspace_id="ws-1",
        code_session_id="cs-1",
        orchestrated_run_id="run-1",
    )
    assert artifact["id"]
    assert artifact["artifact_type"] == "task_intake"

    rows = ledger.list_artifacts(code_session_id="cs-1")
    assert any(row["id"] == artifact["id"] for row in rows)


def test_invalid_artifact_type_raises(ledger):
    with pytest.raises(ValueError):
        ledger.create_artifact("unknown_type", "x")


def test_artifact_filters(ledger):
    ledger.create_artifact("task_intake", "a", code_session_id="cs-a")
    ledger.create_artifact("review_report", "b", code_session_id="cs-b")
    results = ledger.list_artifacts(code_session_id="cs-a")
    assert len(results) == 1
    assert results[0]["code_session_id"] == "cs-a"
