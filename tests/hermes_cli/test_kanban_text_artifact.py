"""Completion cannot release dependants on an attachment-only adviser handoff."""
import copy
import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


def contract(**changes):
    value = {"metadata_key": "advice_artifact", "assignee": "architecture",
        "identity": {"schema_version": 1, "repository": "owner/repo", "item_number": 70,
            "role": "architecture", "source_revision_key": "a" * 64, "snapshot_revision": "b" * 40},
        "max_characters": 4000, "evidence_source": "hermes_completion_metadata"}
    value.update(changes)
    return "text-artifact-v1:" + json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_tool_handoff_rejects_incomplete_or_mismatched_artifacts_then_releases_exact_text(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_PROFILE", "architecture")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from tools import kanban_tools  # register the actual tool handlers
    from tools.registry import registry

    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="Advice", assignee="architecture", completion_contract=contract())
        child = kb.create_task(conn, title="Planner", assignee="planner", parents=[tid])
        assert kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        artifact = {**json.loads(contract().split(":", 1)[1])["identity"], "task_id": tid,
            "complete": True, "truncated": False, "text": "é" * 3999 + "終"}
        attachment = tmp_path / "advice_final.md"
        attachment.write_text(artifact["text"])
        invalid = [None, {}, {"advice_artifact": {**artifact, "task_id": "t_00000000"}},
            {"advice_artifact": {**artifact, "source_revision_key": "c" * 64}},
            {"advice_artifact": {**artifact, "schema_version": True}},
            {"advice_artifact": {**artifact, "text": "x" * 4001}},
            {"advice_artifact": {**artifact, "text": "Short summary"}},
            {"advice_artifact": {**artifact, "text": " "}},
            {"advice_artifact": {**artifact, "complete": False}},
            {"advice_artifact": {**artifact, "truncated": True}},
            {"advice_artifact": {**artifact, "evidence_source": "invented"}}]
        for metadata in invalid:
            result = json.loads(registry.dispatch("kanban_complete", {
                "summary": "Short summary", "metadata": copy.deepcopy(metadata), "artifacts": [str(attachment)]}))
            assert "Completion requires metadata.advice_artifact" in result["error"]
            task = kb.get_task(conn, tid)
            assert task.status == "running" and task.current_run_id == run_id
            assert kb.latest_run(conn, tid).ended_at is None
            assert kb.get_task(conn, child).status == "todo"
            assert attachment.read_text() == artifact["text"]
        result = json.loads(registry.dispatch("kanban_complete", {
            "summary": "Short summary", "metadata": {"advice_artifact": artifact}}))
        assert result["ok"]
        assert kb.get_task(conn, tid).status == "done"
        assert kb.get_task(conn, child).status == "ready"
        run = kb.latest_run(conn, tid)
        assert run.id == run_id and run.metadata["advice_artifact"] == artifact


def test_contract_cannot_be_substituted_or_bypassed_by_direct_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="Advice", assignee="architecture",
            idempotency_key="stable", completion_contract=contract())
        assert kb.create_task(conn, title="Advice", assignee="architecture",
            idempotency_key="stable", completion_contract=contract()) == tid
        for changed in [None, "local-only", contract(max_characters=5000)]:
            with pytest.raises(ValueError, match="different text artifact"):
                kb.create_task(conn, title="Advice", idempotency_key="stable", completion_contract=changed)
        for bad in [contract(max_characters=True), contract(identity={"task_id": "other"}),
                    "text-artifact-v1:{", contract(metadata_key=""), contract(identity={})]:
            with pytest.raises(ValueError):
                kb.create_task(conn, title="Malformed", completion_contract=bad)
        with pytest.raises(ValueError, match="active run is required"):
            kb.complete_task(conn, tid, summary="Implicit final", force=True)
        artifact = {**json.loads(contract().split(":", 1)[1])["identity"], "task_id": tid,
            "complete": True, "truncated": False, "text": "Full advice including a finding."}
        with pytest.raises(ValueError, match="active run is required"):
            kb.complete_task(conn, tid, summary="Summary", metadata={"advice_artifact": artifact}, force=True)
        assert kb.assign_task(conn, tid, "another-profile")
        with pytest.raises(ValueError, match="assignee differs"):
            kb.complete_task(conn, tid, summary="Summary", metadata={"advice_artifact": artifact})
        assert kb.get_task(conn, tid).status != "done"
        assert not kb.list_runs(conn, tid)
        legacy = kb.create_task(conn, title="Generic task", assignee="architecture")
        assert kb.complete_task(conn, legacy, summary="Legacy completion remains supported")
