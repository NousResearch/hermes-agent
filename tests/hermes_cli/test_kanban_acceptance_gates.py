import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _graph(conn, *, required="SHIPPED", evidence_type="implementation", declared=True):
    body = ""
    if declared:
        body = f"Required classification: {required}\nRequired evidence type: {evidence_type}\n"
    parent = kb.create_task(conn, title="parent", body=body, assignee="implementer")
    child = kb.create_task(conn, title="child", parents=[parent])
    return parent, child


def _comment(conn, task_id):
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO task_comments(task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
        (task_id, "reviewer", "verified evidence", now),
    )
    conn.commit()
    return int(cur.lastrowid)


def _verifier_run(conn, task_id, *, profile="reviewer", metadata=None):
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO task_runs(task_id, profile, status, started_at, ended_at, outcome, metadata) "
        "VALUES (?, ?, 'done', ?, ?, 'completed', ?)",
        (task_id, profile, now, now, json.dumps(metadata) if metadata is not None else None),
    )
    conn.commit()
    return int(cur.lastrowid)


def _accepted_verifier(conn, task_id, *, classification="SHIPPED", evidence_type="implementation", profile="reviewer"):
    comment_id = _comment(conn, task_id)
    metadata = {
        "verification": {
            "target_task_id": task_id,
            "classification": classification,
            "verdict": "PASS",
            "evidence_type": evidence_type,
            "evidence_refs": [{"kind": "comment", "id": comment_id}],
        }
    }
    return _verifier_run(conn, task_id, profile=profile, metadata=metadata)


def _complete(conn, task_id, metadata=None):
    return kb.complete_task(conn, task_id, summary="execution finished", metadata=metadata)


def test_completed_undeclared_parent_does_not_release_dependency(kanban_home):
    with kb.connect() as conn:
        parent, child = _graph(conn, declared=False)
        assert _complete(conn, parent)
        assert kb.get_task(conn, child).status == "todo"


def test_standalone_legacy_task_can_complete(kanban_home):
    with kb.connect() as conn:
        task = kb.create_task(conn, title="standalone")
        assert _complete(conn, task)
        assert kb.get_task(conn, task).status == "done"


def test_archived_unaccepted_parent_does_not_release_dependency(kanban_home):
    with kb.connect() as conn:
        parent, child = _graph(conn)
        assert kb.archive_task(conn, parent)
        assert kb.get_task(conn, child).status == "todo"


def test_missing_verifier_run_id_fails_closed(kanban_home):
    with kb.connect() as conn:
        parent, child = _graph(conn)
        assert _complete(conn, parent, {})
        assert kb.get_task(conn, child).status == "todo"


def test_implementation_run_cannot_self_verify(kanban_home):
    with kb.connect() as conn:
        parent, child = _graph(conn)
        claim = kb.claim_task(conn, parent, claimer="implementer")
        assert claim is not None
        run_id = kb.get_task(conn, parent).current_run_id
        assert _complete(conn, parent, {"verifier_run_id": run_id})
        assert kb.get_task(conn, child).status == "todo"


def test_same_worker_identity_cannot_verify(kanban_home):
    with kb.connect() as conn:
        parent, child = _graph(conn)
        verifier = _accepted_verifier(conn, parent, profile="implementer")
        assert _complete(conn, parent, {"verifier_run_id": verifier})
        assert kb.get_task(conn, child).status == "todo"


@pytest.mark.parametrize("verifier_metadata", [None, [], {"verification": []}])
def test_malformed_verifier_metadata_fails_closed(kanban_home, verifier_metadata):
    with kb.connect() as conn:
        parent, child = _graph(conn)
        verifier = _verifier_run(conn, parent, metadata=verifier_metadata)
        assert _complete(conn, parent, {"verifier_run_id": verifier})
        assert kb.get_task(conn, child).status == "todo"


@pytest.mark.parametrize("refs", [[], ["invented"], [{"kind": "comment"}], [{"kind": "comment", "id": 99999}]])
def test_malformed_or_unknown_evidence_fails_closed(kanban_home, refs):
    with kb.connect() as conn:
        parent, child = _graph(conn)
        verifier = _verifier_run(conn, parent, metadata={"verification": {
            "target_task_id": parent, "classification": "SHIPPED", "verdict": "PASS",
            "evidence_type": "implementation", "evidence_refs": refs,
        }})
        assert _complete(conn, parent, {"verifier_run_id": verifier})
        assert kb.get_task(conn, child).status == "todo"


def test_independent_verifier_with_durable_evidence_releases_dependency(kanban_home):
    with kb.connect() as conn:
        parent, child = _graph(conn)
        verifier = _accepted_verifier(conn, parent)
        assert _complete(conn, parent, {"verifier_run_id": verifier})
        assert kb.get_task(conn, child).status == "ready"


def test_claim_time_recheck_blocks_when_accepted_parent_is_archived(kanban_home):
    with kb.connect() as conn:
        parent, child = _graph(conn)
        verifier = _accepted_verifier(conn, parent)
        assert _complete(conn, parent, {"verifier_run_id": verifier})
        assert kb.get_task(conn, child).status == "ready"
        assert kb.archive_task(conn, parent)
        assert kb.claim_task(conn, child, claimer="worker") is None
        assert kb.get_task(conn, child).status == "todo"


def test_malformed_closing_metadata_has_no_partial_transition(kanban_home):
    with kb.connect() as conn:
        parent, child = _graph(conn)
        assert _complete(conn, parent, []) is False
        assert kb.get_task(conn, parent).status == "ready"
        assert kb.get_task(conn, child).status == "todo"


def test_research_classification_accepts_research_evidence(kanban_home):
    with kb.connect() as conn:
        parent, child = _graph(conn, required="RESEARCH_READY", evidence_type="research")
        verifier = _accepted_verifier(conn, parent, classification="RESEARCH_READY", evidence_type="research")
        assert _complete(conn, parent, {"verifier_run_id": verifier})
        assert kb.get_task(conn, child).status == "ready"