from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


pytestmark = pytest.mark.real_completion_evidence_gate


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _evidence_file(tmp_path: Path, name: str = "evidence.md") -> str:
    path = tmp_path / name
    path.write_text("verified evidence\n", encoding="utf-8")
    return str(path)


def test_complete_task_unknown_id_returns_false_before_evidence_gate(kanban_home):
    conn = kb.connect()
    try:
        assert kb.complete_task(conn, "t_deadbeef", summary="done") is False
    finally:
        conn.close()


@pytest.mark.parametrize("state", ["ready", "running", "blocked"])
def test_complete_task_rejects_done_without_evidence_path(kanban_home, state):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title=f"{state} task", assignee="worker")
        if state == "running":
            assert kb.claim_task(conn, tid)
        elif state == "blocked":
            assert kb.block_task(conn, tid, reason="waiting")
            task = kb.get_task(conn, tid)
            assert task is not None
            assert task.status == "blocked"
        else:
            task = kb.get_task(conn, tid)
            assert task is not None
            assert task.status == "ready"

        with pytest.raises(kb.CompletionEvidenceError) as exc:
            kb.complete_task(conn, tid, summary="done but no proof")

        assert "evidence path" in str(exc.value)
        assert "existing, absolute" in str(exc.value)
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == state
        events = kb.list_events(conn, tid)
        assert any(e.kind == "completion_blocked_missing_evidence" for e in events)
    finally:
        conn.close()


@pytest.mark.parametrize(
    "completion_kwargs",
    [
        pytest.param({}, id="missing-fields"),
        pytest.param({"evidence_paths": None}, id="top-level-paths-null"),
        pytest.param({"evidence_paths": ""}, id="top-level-path-empty-string"),
        pytest.param({"evidence_paths": "   "}, id="top-level-path-whitespace"),
        pytest.param({"evidence_paths": []}, id="top-level-paths-empty-list"),
        pytest.param(
            {"evidence_paths": [None, "", "   "]},
            id="top-level-paths-null-and-empty",
        ),
        pytest.param({"evidence_refs": None}, id="top-level-refs-null"),
        pytest.param({"evidence_refs": ""}, id="top-level-ref-empty-string"),
        pytest.param(
            {"metadata": {"evidence_refs": None}},
            id="metadata-refs-null",
        ),
        pytest.param(
            {"metadata": {"evidence_refs": ""}},
            id="metadata-ref-empty-string",
        ),
        pytest.param(
            {"metadata": {"evidence_refs": [None, "", "   "]}},
            id="metadata-refs-null-and-empty",
        ),
        pytest.param(
            {"metadata": {"evidence_paths": None}},
            id="metadata-paths-null",
        ),
        pytest.param(
            {"metadata": {"artifacts": []}},
            id="metadata-artifacts-empty-list",
        ),
    ],
)
def test_complete_task_rejects_missing_empty_and_null_evidence_fields(
    kanban_home, completion_kwargs
):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="no meaningful evidence", assignee="worker")

        with pytest.raises(kb.CompletionEvidenceError) as exc:
            kb.complete_task(
                conn,
                tid,
                summary="done without meaningful proof",
                **completion_kwargs,
            )

        assert "evidence path" in str(exc.value)
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        events = kb.list_events(conn, tid)
        blocked = [e for e in events if e.kind == "completion_blocked_missing_evidence"]
        assert len(blocked) == 1
    finally:
        conn.close()


def test_complete_task_rejects_relative_and_missing_evidence_paths(kanban_home, tmp_path):
    conn = kb.connect()
    try:
        rel_tid = kb.create_task(conn, title="relative", assignee="worker")
        with pytest.raises(kb.CompletionEvidenceError) as rel_exc:
            kb.complete_task(
                conn,
                rel_tid,
                summary="done with relative proof",
                evidence_paths=["relative-proof.md"],
            )
        assert "relative path" in str(rel_exc.value)
        rel_task = kb.get_task(conn, rel_tid)
        assert rel_task is not None
        assert rel_task.status == "ready"

        missing_tid = kb.create_task(conn, title="missing", assignee="worker")
        with pytest.raises(kb.CompletionEvidenceError) as missing_exc:
            kb.complete_task(
                conn,
                missing_tid,
                summary="done with missing proof",
                metadata={"evidence_refs": [str(tmp_path / "missing.md")]},
            )
        assert "missing path" in str(missing_exc.value)
        missing_task = kb.get_task(conn, missing_tid)
        assert missing_task is not None
        assert missing_task.status == "ready"
    finally:
        conn.close()


def test_complete_task_rejects_sensitive_looking_evidence_path(kanban_home, tmp_path):
    secretish_dir = tmp_path / "token"
    secretish_dir.mkdir()
    secretish = secretish_dir / "proof.md"
    secretish.write_text("do not read me\n", encoding="utf-8")

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="sensitive", assignee="worker")
        with pytest.raises(kb.CompletionEvidenceError) as exc:
            kb.complete_task(
                conn,
                tid,
                summary="done with unsafe proof ref",
                evidence_paths=[str(secretish)],
            )
        msg = str(exc.value)
        assert "sensitive-looking path" in msg
        assert str(secretish) not in msg
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
    finally:
        conn.close()


def test_complete_task_accepts_existing_evidence_and_persists_refs(
    kanban_home, tmp_path
):
    evidence = _evidence_file(tmp_path)
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="valid", assignee="worker")
        assert kb.claim_task(conn, tid)

        ok = kb.complete_task(
            conn,
            tid,
            summary="done with proof",
            metadata={"changed_files": ["hermes_cli/kanban_db.py"]},
            evidence_paths=[evidence],
        )

        assert ok is True
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "done"
        run = kb.latest_run(conn, tid)
        assert run is not None
        assert run.metadata["changed_files"] == ["hermes_cli/kanban_db.py"]
        assert run.metadata["evidence_refs"] == [evidence]
        completed = [e for e in kb.list_events(conn, tid) if e.kind == "completed"]
        assert len(completed) == 1
        assert completed[0].payload is not None
        assert completed[0].payload["evidence_refs"] == [evidence]
    finally:
        conn.close()


def test_complete_task_allows_existing_artifact_as_evidence(kanban_home, tmp_path):
    artifact = _evidence_file(tmp_path, "artifact.txt")
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="artifact", assignee="worker")
        ok = kb.complete_task(
            conn,
            tid,
            summary="done with artifact proof",
            metadata={"artifacts": [artifact]},
        )

        assert ok is True
        run = kb.latest_run(conn, tid)
        assert run is not None
        assert run.metadata["artifacts"] == [artifact]
        assert run.metadata["evidence_refs"] == [artifact]
        completed = [e for e in kb.list_events(conn, tid) if e.kind == "completed"]
        assert completed[0].payload is not None
        assert completed[0].payload["artifacts"] == [artifact]
        assert completed[0].payload["evidence_refs"] == [artifact]
    finally:
        conn.close()
