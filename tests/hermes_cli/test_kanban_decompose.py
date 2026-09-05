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
    with kbc.connect() as conn:
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

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert root.status == "todo"
    assert c0.status == "ready"
    assert c1.status == "todo"
    assert c0.assignee == "researcher"
    assert c1.assignee == "engineer"
    assert f"root task `{tid}`" in (c0.body or "")
    assert "This is a leaf work item" in (c0.body or "")
    assert "Call kanban_show" in (c0.body or "")


def test_decompose_makes_leaf_handoff_self_contained(kanban_home):
    """A fresh child must retain the root brief and board-navigation seam.

    Previously the LLM-produced child body was stored verbatim. That left a
    leaf with phrases such as "use the hypothesis from another card" but no
    reliable way to discover that card, so workers converted ordinary missing
    context into sticky ``needs_input`` blocks.
    """
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="investigate stream ordering",
            body="Inspect websocket_stream.py and report the concrete ordering risk.",
            triage=True,
        )

    payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "one bounded investigation",
        "tasks": [{
            "title": "Inspect stream ordering",
            "body": "Read the named source and report evidence.",
            "assignee": "researcher",
            "parents": [],
        }],
    })
    patches = _patch_list_profiles(["orchestrator", "researcher"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    assert child is not None
    assert "root task `" + tid + "`" in (child.body or "")
    assert "investigate stream ordering" in (child.body or "")
    assert "Inspect websocket_stream.py" in (child.body or "")
    assert "do not decompose this task" in (child.body or "").lower()


def test_decompose_rejects_placeholder_child_scope_before_graph_write(kanban_home):
    """A generic monolith placeholder must not become a runnable leaf card."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="continue monolith burndown",
            body="Split only concrete source targets.",
            triage=True,
        )

    payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "bad placeholder",
        "tasks": [{
            "title": "Analyze extraction",
            "body": "Examine the target monolith component and define seams.",
            "assignee": "researcher",
            "parents": [],
        }],
    })
    patches = _patch_list_profiles(["orchestrator", "researcher"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "placeholder target" in outcome.reason
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"
        assert kb.child_ids(conn, tid) == []


def test_decompose_inherits_recent_root_handoffs(kanban_home):
    """Comments added after creation must reach fresh phase workers."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="continue monolith burndown",
            triage=True,
        )
        kb.add_comment(
            conn,
            tid,
            "operator",
            "Phase 1 target is the live_runner.py Stage-1 adapter seam. "
            "Produce an exact-head PR before starting the next phase.",
        )

    payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "bounded phase",
        "tasks": [{
            "title": "Define phase one seam",
            "body": "Inspect the named target and document entry and exit points.",
            "assignee": "researcher",
            "parents": [],
        }],
    })
    patches = _patch_list_profiles(["orchestrator", "researcher"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    assert child is not None
    assert "Recent root handoffs/comments" in (child.body or "")
    assert "live_runner.py Stage-1 adapter seam" in (child.body or "")
    assert "exact-head PR" in (child.body or "")
    assert "untrusted" in (child.body or "")


def test_decompose_fanout_false_invalid_llm_assignee_uses_default(kanban_home):
    with kbc.connect() as conn:
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
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.assignee == "fallback"


def test_decompose_returns_false_when_task_not_triage(kanban_home):
    with kbc.connect() as conn:
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


@pytest.mark.parametrize(
    ("idempotency_key", "body"),
    [
        (
            "github-pr-feedback:repair:mrkillbob/luna-bot:132:abc123",
            "Repair the pull request and push the exact-head fix.",
        ),
        (
            None,
            jsonlib.dumps(
                {
                    "repository": "mrkillbob/luna-bot",
                    "pr_number": 132,
                    "expected_head_sha": "a" * 40,
                    "action": "repair_and_push",
                }
            ),
        ),
    ],
)
def test_decompose_refuses_atomic_pr_automation_before_llm(
    kanban_home, idempotency_key, body
):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Repair ExampleApp PR #132",
            body=body,
            triage=True,
            idempotency_key=idempotency_key,
        )

    with patch(
        "agent.auxiliary_client.call_llm",
        side_effect=AssertionError("atomic PR work must never reach the decomposer LLM"),
    ):
        outcome = decomp.decompose_task(tid, author="auto-decomposer")

    assert outcome.ok is False
    assert "atomic PR automation" in outcome.reason
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"


def test_decompose_refuses_governed_research_intake_before_llm(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="[Lab] Research intake",
            body=(
                "Choose only source-backed hypotheses and complete as "
                "RESEARCH_LAB_IDLE when none qualify."
            ),
            assignee="exampleapp-research-lab-director",
            triage=True,
            idempotency_key="research-lab-intake-20260826-12",
        )

    with patch(
        "agent.auxiliary_client.call_llm",
        side_effect=AssertionError("governed research intake must retain its typed owner"),
    ):
        outcome = decomp.decompose_task(tid, author="auto-decomposer")

    assert outcome.ok is False
    assert "governed research intake" in outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "triage"
    assert task.assignee == "exampleapp-research-lab-director"
