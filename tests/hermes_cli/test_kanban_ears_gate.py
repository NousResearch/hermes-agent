"""INVARIANT tests for Rule 6 — the EARS-ification gate in the
decomposer's aux-LLM call path (``hermes_cli/kanban_decompose.py``).

Owner choice under test (documented in kanban_decompose.py and the Rule 6
commit message): when the aux LLM cannot EARS-ify a child task without
inventing missing details, the decomposer does NOT block task creation — it
files a decision-hud problem report via
``plugin_bridges.decision_hud.push_problem_report`` and proceeds with
``ears_sentence=None`` on that child, never a guessed sentence.

Two behavior contracts:
  1. success path — a real, template-conformant EARS sentence from the LLM
     is stored verbatim, and no problem report is filed.
  2. failure path — a refusal (or an LLM-claimed sentence that fails the
     mechanical EARS-shape check) results in ears_sentence staying None,
     and exactly one problem report being filed; the code never invents a
     plausible-looking sentence to paper over the gap.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _patch_aux_client(content: str):
    return patch("agent.auxiliary_client.call_llm", return_value=_fake_aux_response(content))


def _patch_extra_body():
    return patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={})


def _patch_list_profiles(names: list[str]):
    from types import SimpleNamespace
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
    ]


# --- Pure mechanical EARS-shape validator ------------------------------------

@pytest.mark.parametrize("sentence", [
    "The dispatcher shall retry a failed task at most twice.",
    "When a task is claimed, the dispatcher shall record a claim_lock.",
    "While the board is paused, the dispatcher shall not dispatch new tasks.",
    "If a task fails three times, then the dispatcher shall block it.",
    "Where notifications are enabled, the dispatcher shall wake the origin session.",
])
def test_validate_ears_sentence_accepts_all_five_templates(sentence):
    assert decomp.validate_ears_sentence(sentence) is True


@pytest.mark.parametrize("sentence", [
    "Fix the bug in the dispatcher.",
    "This task is about making things better somehow.",
    "",
    "   ",
    "Improve performance.",
])
def test_validate_ears_sentence_rejects_non_ears_text(sentence):
    assert decomp.validate_ears_sentence(sentence) is False


# --- Integration: success path ------------------------------------------------

def test_ears_success_stores_real_sentence_and_files_no_report(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {
                "title": "research options",
                "body": "look it up",
                "assignee": "researcher",
                "parents": [],
                "ears_sentence": "When a new dependency is proposed, the researcher shall document its license.",
            },
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._dh_bridge.push_problem_report",
        ) as mock_report:
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.child_ids and len(outcome.child_ids) == 1
    mock_report.assert_not_called()

    with kbc.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    assert child is not None
    assert child.ears_sentence == (
        "When a new dependency is proposed, the researcher shall document its license."
    )


# --- Integration: failure path ------------------------------------------------

def test_ears_explicit_refusal_files_report_and_does_not_invent_sentence(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="vague idea", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {
                "title": "do the vague thing",
                "body": "not enough detail to specify behavior",
                "assignee": "researcher",
                "parents": [],
                "ears_refusal": "original task gives no concrete trigger or system response to restate",
            },
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._dh_bridge.push_problem_report",
            return_value=(True, "report-123"),
        ) as mock_report:
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.child_ids and len(outcome.child_ids) == 1
    mock_report.assert_called_once()

    with kbc.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    assert child is not None
    assert child.ears_sentence is None


def test_ears_malformed_sentence_is_treated_as_failure_not_invented(kanban_home):
    """The LLM claims success but the sentence doesn't match any EARS
    template — the mechanical validator catches this so the decomposer
    never trusts the LLM's own claim of success (the plan's flagged
    'classifier grading its own homework' risk)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="vague idea 2", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {
                "title": "do the vague thing",
                "body": "not enough detail to specify behavior",
                "assignee": "researcher",
                "parents": [],
                "ears_sentence": "Make things better somehow.",
            },
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._dh_bridge.push_problem_report",
            return_value=(True, "report-456"),
        ) as mock_report:
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    mock_report.assert_called_once()
    with kbc.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    assert child.ears_sentence is None


def test_ears_refusal_report_not_filed_when_decompose_write_fails(kanban_home):
    """Adversarial-review regression: if decompose_triage_task's DB write
    fails/no-ops (e.g. the task already got decomposed by a concurrent
    sweep), no EARS problem report should have been filed for a child that
    was never actually created — the report must be deferred until AFTER
    the write is confirmed."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="vague idea 3", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {
                "title": "do the vague thing",
                "body": "not enough detail to specify behavior",
                "assignee": "researcher",
                "parents": [],
                "ears_refusal": "no concrete trigger to restate",
            },
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose.decompose_triage_task",
            return_value=None,  # simulates "already decomposed / moved out of triage"
        ), patch(
            "hermes_cli.kanban_decompose._dh_bridge.push_problem_report",
            return_value=(True, "report-789"),
        ) as mock_report:
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert not outcome.ok
    assert "already decomposed" in outcome.reason
    # The write never succeeded -> no report should have been filed for a
    # child that doesn't exist.
    mock_report.assert_not_called()
