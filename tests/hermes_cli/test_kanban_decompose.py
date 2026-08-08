"""Tests for the decomposer module + `hermes kanban decompose` CLI surface.

The auxiliary LLM client is mocked — no network calls. Tests exercise the
prompt plumbing, response parsing, DB writes (via the real DB helper),
and the assignee-fallback logic.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
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


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    # decompose_task now routes through call_llm (see #35566) — mock it at
    # the source module so task config, extra_body, and retries stay out of
    # unit-test scope.
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_extra_body():
    # No-op shim retained for call-site compatibility: extra_body plumbing
    # now lives inside call_llm, which _patch_aux_client already mocks.
    return patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={})


def _patch_list_profiles(names: list[str]):
    """Pretend the named profiles exist. The decomposer uses
    profiles_mod.list_profiles() to build the roster + valid-set, and
    profiles_mod.profile_exists() to resolve orchestrator/default."""
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


def test_decompose_with_fanout_creates_children(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
            {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 2

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert root.status == "todo"
    assert c0.status == "ready"
    assert c1.status == "todo"
    assert c0.assignee == "researcher"
    assert c1.assignee == "engineer"


def test_decompose_fanout_false_invalid_llm_assignee_uses_default(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="route me safely", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Route to fallback.",
        "assignee": "made_up",
    })

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.assignee == "fallback"


def _escalate_via_block_loop(conn, task_id, kind=None):
    """Replay the #79738 sequence: block -> unblock -> re-block same cause.

    The unblock-loop breaker routes the second same-cause block to ``triage``
    (``block_loop_detected`` event) once ``block_recurrences`` reaches
    ``BLOCK_RECURRENCE_LIMIT``. Returns the task row after escalation.
    """
    assert kb.block_task(
        conn, task_id, reason="review-required: please review", kind=kind,
    )
    assert kb.unblock_task(conn, task_id)
    assert kb.block_task(
        conn, task_id, reason="review-required: please review", kind=kind,
    )
    return kb.get_task(conn, task_id)


def test_block_loop_triage_without_kind_is_not_auto_decomposed(kanban_home):
    """#79738: review-blocked task (kind omitted -> block_kind stays NULL)
    must not be fed to the auto-decomposer after the block-loop breaker
    routes it to triage."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="ship the widget", body="impl + PR")
        task = _escalate_via_block_loop(conn, tid, kind=None)
        assert task.status == "triage"
        assert task.block_kind is None  # reporter's case: kind was omitted
        # The auto-decompose feed must exclude block-loop escalations.
        assert tid not in decomp.list_triage_ids()
        # decompose_task must refuse before touching the LLM or the spec.
        with patch("agent.auxiliary_client.call_llm") as call_llm:
            outcome = decomp.decompose_task(tid, author="auto-decomposer")
            assert outcome.ok is False
            call_llm.assert_not_called()
        task = kb.get_task(conn, tid)
        assert task.status == "triage"
        assert task.title == "ship the widget"  # spec must not be rewritten


def test_typed_block_loop_triage_is_not_auto_decomposed(kanban_home):
    """Typed-kind escalation (block_kind preserved) is excluded too."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="needs capability")
        task = _escalate_via_block_loop(conn, tid, kind="capability")
        assert task.status == "triage"
        assert task.block_kind == "capability"
        assert tid not in decomp.list_triage_ids()


def test_recover_escalated_triage_restores_decomposability(kanban_home):
    """The audited operator recovery action makes an escalated triage card
    decomposable again (and gives it a fresh loop budget)."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="needs capability")
        _escalate_via_block_loop(conn, tid, kind="capability")
        assert tid not in decomp.list_triage_ids()
        assert kb.recover_escalated_triage_task(conn, tid) is True
        task = kb.get_task(conn, tid)
        assert task.block_kind is None
        assert task.block_recurrences == 0
        assert tid in decomp.list_triage_ids()


def test_fresh_triage_remains_auto_decomposable(kanban_home):
    """Freshly created triage cards are untouched by the escalation guard."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)
        assert tid in decomp.list_triage_ids()


def test_manual_decompose_of_escalated_triage_recovers_and_proceeds(kanban_home):
    """Explicit `hermes kanban decompose <id>` on an escalated card IS the
    human-in-the-loop decision: acknowledge (audited) and decompose (#79728)."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="needs capability")
        _escalate_via_block_loop(conn, tid, kind="capability")
        assert kb.is_block_loop_escalated(conn, tid) is True

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "operator wants a single unit",
        "title": "Tightened title",
        "body": "After human review.",
    })
    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "ready"  # specify + recompute_ready (no parents)
        assert task.title == "Tightened title"
        assert task.block_kind is None
        assert kb.is_block_loop_escalated(conn, tid) is False
        events = kb.list_events(conn, tid)
        assert any(e.kind == "triage_escalation_recovered" for e in events)


def test_manual_decompose_failure_keeps_escalation(kanban_home):
    """A FAILED manual decompose must not clear the escalation: the card
    stays escalated and out of the auto-decompose feed, and the recovery
    event is only written when an attempt actually succeeds (#81353)."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="needs capability")
        _escalate_via_block_loop(conn, tid, kind="capability")
        assert kb.is_block_loop_escalated(conn, tid) is True

    # Manual decompose fails at the LLM call (API error).
    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ), patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("api down"),
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "triage"
        assert task.title == "needs capability"  # spec must not be rewritten
        # Escalation survives the failed attempt...
        assert kb.is_block_loop_escalated(conn, tid) is True
        assert tid not in decomp.list_triage_ids()
        events = kb.list_events(conn, tid)
        assert not any(e.kind == "triage_escalation_recovered" for e in events)
    # ...and the auto-decomposer still refuses the card.
    with patch("agent.auxiliary_client.call_llm") as call_llm:
        outcome = decomp.decompose_task(tid, author=decomp.AUTO_DECOMPOSER_AUTHOR)
        assert outcome.ok is False
        call_llm.assert_not_called()


def test_manual_decompose_fanout_recovers_on_success(kanban_home):
    """The fanout=true success path also acknowledges the escalation
    (audited) after the children are created."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="needs capability")
        _escalate_via_block_loop(conn, tid, kind="capability")
        assert kb.is_block_loop_escalated(conn, tid) is True

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "operator split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
            {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
        ],
    })
    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 2
    with kb.connect_closing() as conn:
        root = kb.get_task(conn, tid)
        assert root.status == "todo"
        assert root.block_kind is None
        assert root.block_recurrences == 0
        assert kb.is_block_loop_escalated(conn, tid) is False
        events = kb.list_events(conn, tid)
        assert any(e.kind == "triage_escalation_recovered" for e in events)


def test_decompose_returns_false_when_task_not_triage(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x")  # ready, not triage

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False
    assert "not in triage" in outcome.reason


