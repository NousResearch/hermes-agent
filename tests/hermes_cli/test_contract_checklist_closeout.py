from __future__ import annotations

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_complex_requirements_with_done_checklist_are_complete_candidate():
    from hermes_cli.closeout_state import classify_closeout_response

    verdict = classify_closeout_response(
        """
        Final result: all requested work is complete.

        Requirement checklist:
        - requirement: inspect README.md; status: done; evidence: README.md reviewed; residual risk: none
        - requirement: run focused tests; status: done; evidence: pytest closeout tests passed; residual risk: none
        """,
        task_contract="""
        Requirements:
        1. Inspect README.md.
        2. Run focused tests.
        """,
    )

    assert verdict["status"] == "complete_candidate"
    assert verdict["reasons"] == []
    assert [item["status"] for item in verdict["contract_checklist"]] == ["done", "done"]


def test_partial_requirement_keeps_closeout_recoverable_incomplete():
    from hermes_cli.closeout_state import classify_closeout_response

    verdict = classify_closeout_response(
        """
        Final result: implementation is mostly done.

        Requirement checklist:
        - requirement: update closeout classifier; status: done; evidence: hermes_cli/closeout_state.py; residual risk: none
        - requirement: run full baseline suite; status: partial; evidence: focused tests only; residual risk: baseline suite not rerun; next action: rerun baseline suite
        """,
        task_contract="""
        Requirements:
        1. Update closeout classifier.
        2. Run full baseline suite.
        """,
    )

    assert verdict["status"] == "recoverable_incomplete"
    assert "contract_checklist_incomplete" in verdict["reasons"]
    assert verdict["contract_checklist"][1]["status"] == "partial"
    assert verdict["contract_checklist"][1]["next_action"] == "rerun baseline suite"


def test_prose_only_final_for_explicit_requirements_is_not_silently_complete():
    from hermes_cli.closeout_state import classify_closeout_response

    verdict = classify_closeout_response(
        "I inspected the files and ran the tests. Everything is done.",
        task_contract="""
        Requirements:
        1. Inspect closeout_state.py.
        2. Add focused tests.
        3. Run focused tests.
        """,
    )

    assert verdict["status"] == "recoverable_incomplete"
    assert "contract_checklist_missing" in verdict["reasons"]
    assert verdict["contract_requirements"] == [
        "Inspect closeout_state.py.",
        "Add focused tests.",
        "Run focused tests.",
    ]


def test_simple_closeout_without_explicit_requirements_does_not_need_checklist():
    from hermes_cli.closeout_state import classify_closeout_response

    verdict = classify_closeout_response("Done: answered the version question.")

    assert verdict["status"] == "complete_candidate"
    assert "contract_checklist_missing" not in verdict["reasons"]
    assert verdict["contract_checklist"] == []


def test_write_closeout_state_persists_contract_checklist(hermes_home):
    from hermes_cli.closeout_state import write_closeout_state
    from hermes_cli.closure_artifacts import read_closure_artifact

    path = write_closeout_state(
        session_id="root-session",
        task_id="task-contract",
        task_contract="""
        Requirements:
        1. Update closeout classifier.
        2. Run focused tests.
        """,
        final_response="""
        Requirement checklist:
        - requirement: update closeout classifier; status: done; evidence: hermes_cli/closeout_state.py; residual risk: none
        - requirement: run focused tests; status: done; evidence: pytest contract closeout passed; residual risk: none
        """,
    )

    data = read_closure_artifact(path)

    assert data["status"] == "complete_candidate"
    assert data["contract_checklist"][0]["requirement"] == "update closeout classifier"
    assert data["contract_checklist"][1]["evidence"] == "pytest contract closeout passed"
