"""Tests for the first-class task_artifacts registry.

Artifacts a task produced (file paths, commit SHAs, URLs, deploy links) are
promoted from the per-run ``metadata["artifacts"]`` string-list into a typed,
queryable ``task_artifacts`` table at completion. All rows land ``unchecked``
(verification is a separate out-of-band pass). The legacy bare-string
``metadata["artifacts"]`` path (which the gateway notifier consumes) is
preserved unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _artifacts(conn, tid):
    return conn.execute(
        "SELECT kind, ref, label, verified FROM task_artifacts "
        "WHERE task_id=? ORDER BY id",
        (tid,),
    ).fetchall()


def test_artifacts_captured_typed_and_inferred(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship", created_by="hive")
        kb.complete_task(
            conn, tid, result="done",
            metadata={"artifacts": ["src/foo.py"]},
            artifacts=[
                "a1b2c3d4e5f6",                       # → commit (inferred)
                "https://edgelesslab.com/blog/x",     # → url (inferred)
                {"kind": "deploy", "ref": "https://app.edgeless.dev", "label": "prod"},
            ],
        )
        rows = _artifacts(conn, tid)
        kinds = {r["kind"]: r["ref"] for r in rows}
        assert kinds["file"] == "src/foo.py"
        assert kinds["commit"] == "a1b2c3d4e5f6"
        assert kinds["url"] == "https://edgelesslab.com/blog/x"
        assert kinds["deploy"] == "https://app.edgeless.dev"
        assert all(r["verified"] == "unchecked" for r in rows)
        # typed label preserved
        assert next(r for r in rows if r["kind"] == "deploy")["label"] == "prod"


def test_artifact_dedup(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", created_by="hive")
        kb.complete_task(
            conn, tid, result="done",
            metadata={"artifacts": ["src/foo.py"]},
            artifacts=["src/foo.py"],  # same ref via both paths
        )
        rows = _artifacts(conn, tid)
        assert len([r for r in rows if r["ref"] == "src/foo.py"]) == 1


def test_artifact_validation_rejects_oversized(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", created_by="hive")
        kb.complete_task(conn, tid, result="done",
                         artifacts=["x" * 5000, "ok/path.txt"])
        rows = _artifacts(conn, tid)
        assert len(rows) == 1 and rows[0]["ref"] == "ok/path.txt"


def test_artifacts_per_run_capped(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", created_by="hive")
        kb.complete_task(conn, tid, result="done",
                         artifacts=[f"f{i}/x.txt" for i in range(200)])
        assert len(_artifacts(conn, tid)) <= kb._MAX_ARTIFACTS_PER_RUN


def test_notifier_backcompat_metadata_stays_bare_strings(kanban_home):
    """The gateway notifier only delivers str items off the completed event —
    typed artifacts must NOT poison that path."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", created_by="hive")
        kb.complete_task(conn, tid, result="done",
                         metadata={"artifacts": ["src/foo.py"]})
        ev = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='completed'",
            (tid,)).fetchone()
        payload = json.loads(ev["payload"])
        assert payload.get("artifacts") == ["src/foo.py"]
        assert all(isinstance(a, str) for a in payload["artifacts"])


def test_no_artifact_completion_unchanged(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", created_by="hive")
        assert kb.complete_task(conn, tid, result="done") is True
        assert kb.get_task(conn, tid).status == "done"
        assert _artifacts(conn, tid) == []


def test_kind_inference():
    assert kb._infer_artifact_kind("https://x.com/y") == "url"
    assert kb._infer_artifact_kind("src/main.py") == "file"
    assert kb._infer_artifact_kind("README.md") == "file"
    assert kb._infer_artifact_kind("a1b2c3d4") == "commit"
    assert kb._infer_artifact_kind("deadbeefcafe1234") == "commit"
    assert kb._infer_artifact_kind("hello world") == "message"
