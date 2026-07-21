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


def _graph(conn, *, required="SHIPPED", evidence_type="implementation"):
    parent = kb.create_task(
        conn,
        title="parent",
        body=(
            f"Required classification: {required}\n"
            f"Required evidence type: {evidence_type}\n"
        ),
    )
    child = kb.create_task(conn, title="child", parents=[parent])
    return parent, child


def _complete(conn, task_id, **metadata):
    assert kb.complete_task(conn, task_id, summary="execution finished", metadata=metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"outcome_classification": "SHIPPED", "evidence_refs": ["pr:1"]},
        {
            "outcome_classification": "SHIPPED",
            "verifier_verdict": "FAIL",
            "evidence_refs": ["pr:1"],
            "evidence_type": "implementation",
        },
    ],
    ids=["no-classification", "no-verifier-pass", "verifier-fail"],
)
def test_unaccepted_done_parent_does_not_release_dependency(kanban_home, metadata):
    with kb.connect() as conn:
        parent, child = _graph(conn)
        _complete(conn, parent, **metadata)
        assert kb.get_task(conn, parent).status == "done"
        assert kb.get_task(conn, child).status == "todo"


def test_matching_classification_evidence_and_pass_releases_dependency(kanban_home):
    with kb.connect() as conn:
        parent, child = _graph(conn)
        _complete(conn, parent, outcome_classification="SHIPPED", verifier_verdict="PASS", evidence_refs=["pr:1", "tests:focused"], evidence_type="implementation")
        assert kb.get_task(conn, child).status == "ready"


def test_reconnaissance_cannot_satisfy_implementation_gate(kanban_home):
    with kb.connect() as conn:
        parent, child = _graph(conn)
        _complete(conn, parent, outcome_classification="SHIPPED", verifier_verdict="PASS", evidence_refs=["comment:research"], evidence_type="research")
        assert kb.get_task(conn, child).status == "todo"


def test_parent_programme_cannot_complete_with_unaccepted_child(kanban_home):
    with kb.connect() as conn:
        child = kb.create_task(
            conn,
            title="required child",
            body="Required classification: CHILD_ACCEPTED\n",
        )
        programme = kb.create_task(conn, title="programme", parents=[child])
        _complete(conn, child)
        assert kb.get_task(conn, programme).status == "todo"
        assert kb.complete_task(conn, programme, summary="premature") is False


def test_research_classification_accepts_research_evidence(kanban_home):
    with kb.connect() as conn:
        parent, child = _graph(conn, required="RESEARCH_DECISION_READY", evidence_type="research")
        _complete(conn, parent, outcome_classification="RESEARCH_DECISION_READY", verifier_verdict="PASS", evidence_refs=["decision:compose", "sources:official-docs"], evidence_type="research")
        assert kb.get_task(conn, child).status == "ready"
