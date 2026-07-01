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


def _patch_call_llm(content: str):
    mocked = MagicMock(return_value=_fake_aux_response(content))
    return patch(
        "agent.auxiliary_client.call_llm",
        side_effect=mocked,
    ), mocked


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
        patcher, call_llm_mock = _patch_call_llm(llm_payload)
        with patcher:
            outcome = decomp.decompose_task(tid, author="me")
        call_llm_mock.assert_called_once()
        assert call_llm_mock.call_args.kwargs["task"] == "kanban_decomposer"
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


def test_decompose_fanout_children_inherit_goal_mode(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="ship a goal-mode feature",
            triage=True,
            goal_mode=True,
            goal_max_turns=3,
        )

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
        patcher, call_llm_mock = _patch_call_llm(llm_payload)
        with patcher:
            outcome = decomp.decompose_task(tid, author="me")
        call_llm_mock.assert_called_once()
        assert call_llm_mock.call_args.kwargs["task"] == "kanban_decomposer"
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.child_ids and len(outcome.child_ids) == 2

    with kb.connect() as conn:
        children = [kb.get_task(conn, child_id) for child_id in outcome.child_ids]
        rows = [
            conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'created'",
                (child_id,),
            ).fetchone()
            for child_id in outcome.child_ids
        ]

    assert all(child is not None for child in children)
    assert all(child.goal_mode is True for child in children)
    assert all(child.goal_max_turns == 3 for child in children)
    assert all(jsonlib.loads(row["payload"])["goal_mode"] is True for row in rows)


def test_decompose_fanout_children_inherit_original_hard_constraints(kanban_home):
    body = (
        "Repo: /Users/yuxiansheng/dev/devrun-smoke. "
        "This is read-only; do not modify files; final verdict only."
    )
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="verify isolated board routing",
            body=body,
            triage=True,
        )

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "inspect route", "body": "Check the board.", "assignee": "researcher", "parents": []},
            {"title": "write verdict", "body": "Summarize outcome.", "assignee": "engineer", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_call_llm(llm_payload)[0]:
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.child_ids and len(outcome.child_ids) == 2
    with kb.connect() as conn:
        children = [kb.get_task(conn, child_id) for child_id in outcome.child_ids]

    for child in children:
        assert child is not None
        assert "Inherited hard constraints" in child.body
        assert "read-only" in child.body
        assert "do not modify files" in child.body
        assert "final verdict only" in child.body
        assert "/Users/yuxiansheng/dev/devrun-smoke" in child.body


def test_decompose_stabilizes_closure_graph_dependencies(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Post-patch final closure",
            body=(
                "Verify Budget Gate Evidence, isolated-board routing, and "
                "worker terminal states; read-only; do not modify files; "
                "final verdict only."
            ),
            triage=True,
        )

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {
                "title": "Verify Budget Gate Evidence",
                "body": "Check the current Kanban card body.",
                "assignee": "reviewer",
                "parents": [],
            },
            {
                "title": "Verify isolated-board routing",
                "body": "Check the target board.",
                "assignee": "ops",
                "parents": [],
            },
            {
                "title": "Verify Kanban workers reach terminal states",
                "body": "Inspect done/blocked states.",
                "assignee": "reviewer",
                "parents": [],
            },
            {
                "title": "Cross-check closure evidence files",
                "body": "Compare closure_verdict.md against the current run.",
                "assignee": "researcher",
                "parents": [],
            },
            {
                "title": "Synthesize final post-patch closure verdict",
                "body": "Produce PASS, NEEDS_CHANGES, or BLOCKED.",
                "assignee": "reviewer",
                "parents": [0],
            },
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "reviewer", "ops", "researcher"])
    for p in patches:
        p.start()
    try:
        with _patch_call_llm(llm_payload)[0]:
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.child_ids and len(outcome.child_ids) == 5

    with kb.connect() as conn:
        children = [kb.get_task(conn, child_id) for child_id in outcome.child_ids]

    terminal = children[2]
    closure_files = children[3]
    synthesis = children[4]

    assert terminal.status == "todo"
    assert "Terminal-state timing guard" in terminal.body
    assert "do not return FAIL" in terminal.body
    assert closure_files.status == "ready"
    assert "Current-run evidence guard" in closure_files.body
    assert synthesis.status == "todo"
    assert "Final synthesis evidence guard" in synthesis.body

    with kb.connect() as conn:
        terminal_parent_ids = {
            row["parent_id"]
            for row in conn.execute(
                "SELECT parent_id FROM task_links WHERE child_id = ?",
                (terminal.id,),
            ).fetchall()
        }
        synthesis_parent_ids = {
            row["parent_id"]
            for row in conn.execute(
                "SELECT parent_id FROM task_links WHERE child_id = ?",
                (synthesis.id,),
            ).fetchall()
        }

    assert terminal_parent_ids == {children[0].id, children[1].id, children[3].id}
    assert synthesis_parent_ids == {child.id for child in children[:4]}


def test_decompose_does_not_gate_synthesis_on_root_terminal_child(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Final closure without root self-deadlock",
            body=(
                "Verify all workers/root reach done or blocked; read-only; "
                "do not modify files; final verdict only."
            ),
            triage=True,
        )

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {
                "title": "Verify evidence worker completed",
                "body": "Check the evidence worker handoff.",
                "assignee": "reviewer",
                "parents": [],
            },
            {
                "title": "Verify all workers and root reach done or blocked",
                "body": "Confirm root reaches done or blocked.",
                "assignee": "ops",
                "parents": [],
            },
            {
                "title": "Verify terminal-state verifier waits for siblings",
                "body": "Inspect terminal-state verifier wiring.",
                "assignee": "ops",
                "parents": [],
            },
            {
                "title": "Synthesize final closure verdict from live Kanban evidence",
                "body": "Produce the final verdict.",
                "assignee": "reviewer",
                "parents": [0, 1, 2],
            },
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "reviewer", "ops"])
    for p in patches:
        p.start()
    try:
        with _patch_call_llm(llm_payload)[0]:
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.child_ids and len(outcome.child_ids) == 4

    with kb.connect() as conn:
        children = [kb.get_task(conn, child_id) for child_id in outcome.child_ids]
        terminal_parent_ids = {
            row["parent_id"]
            for row in conn.execute(
                "SELECT parent_id FROM task_links WHERE child_id = ?",
                (children[2].id,),
            ).fetchall()
        }
        synthesis_parent_ids = {
            row["parent_id"]
            for row in conn.execute(
                "SELECT parent_id FROM task_links WHERE child_id = ?",
                (children[3].id,),
            ).fetchall()
        }

    assert "Root-status self-reference guard" in children[1].body
    assert "root cannot close until its children close" in children[1].body
    assert terminal_parent_ids == {children[0].id}
    assert synthesis_parent_ids == {children[0].id, children[2].id}
    assert children[1].id not in synthesis_parent_ids


def test_decompose_live_closure_uses_deterministic_fallback_on_empty_response(kanban_home):
    body = (
        "Final no-regret closure test after root self-deadlock fix: verify "
        "Budget Gate Evidence appears in the Kanban card body, Current live "
        "verification scope appears in the card body, isolated-board routing "
        "works, terminal-state verifier waits for sibling workers instead of "
        "premature FAIL, final synthesis uses current live Kanban evidence over "
        "stale closure files, and the Kanban graph can close without any child "
        "requiring the root to already be done before children finish; read-only; "
        "do not modify files; final verdict only"
    )
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Final no-regret closure",
            body=body,
            triage=True,
            goal_mode=True,
            goal_max_turns=3,
        )

    patches = _patch_list_profiles(["orchestrator", "reviewer", "ops"])
    for p in patches:
        p.start()
    try:
        with _patch_call_llm("")[0]:
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 5
    assert "deterministic closure fallback" in outcome.reason

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        children = [kb.get_task(conn, child_id) for child_id in outcome.child_ids]
        synthesis = children[-1]
        synthesis_parent_ids = {
            row["parent_id"]
            for row in conn.execute(
                "SELECT parent_id FROM task_links WHERE child_id = ?",
                (synthesis.id,),
            ).fetchall()
        }

    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    assert all(child.goal_mode is True for child in children)
    assert all(child.goal_max_turns == 3 for child in children)
    assert "Verify Budget Gate Evidence" in children[0].title
    assert "Verify Current live verification scope" in children[1].title
    assert "Verify isolated-board routing" in children[2].title
    assert "Verify terminal-state verifier waits" in children[3].title
    assert "Synthesize final closure verdict" in children[4].title
    assert synthesis_parent_ids == {child.id for child in children[:4]}
    assert all("root reach done" not in child.title.lower() for child in children)
    assert all("root to already be done" not in child.title.lower() for child in children)


def test_decompose_fanout_false_assigns_default_when_unassigned(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="just one thing", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "**Goal**\nDo the thing.",
    })

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_call_llm(llm_payload)[0], patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is False
    assert outcome.new_title == "Tightened title"
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    # specify path with no parents -> recompute_ready flips to 'ready'
    assert task.status == "ready"
    assert task.title == "Tightened title"
    assert task.assignee == "fallback"


def test_decompose_fanout_false_preserves_existing_assignee(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="already routed",
            assignee="engineer",
            triage=True,
        )

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Keep existing lane.",
        "assignee": "fallback",
    })

    patches = _patch_list_profiles(["orchestrator", "engineer", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_call_llm(llm_payload)[0], patch(
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
    assert task.assignee == "engineer"
    assert task.title == "Tightened title"


def test_decompose_fanout_false_uses_valid_llm_assignee(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="route me", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Route to specialist.",
        "assignee": "engineer",
    })

    patches = _patch_list_profiles(["orchestrator", "engineer", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_call_llm(llm_payload)[0], patch(
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
    assert task.assignee == "engineer"


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
        with _patch_call_llm(llm_payload)[0], patch(
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


def test_decompose_unknown_assignee_falls_back_to_default(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", triage=True)

    # Roster only has 'orchestrator' and 'fallback'; LLM picks 'made_up'.
    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test",
        "tasks": [
            {"title": "do X", "body": "", "assignee": "made_up", "parents": []},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with patch.dict(
            "os.environ", {}, clear=False,
        ), _patch_call_llm(llm_payload)[0], \
            patch(
                "hermes_cli.kanban_decompose._load_config",
                return_value={
                    "kanban": {
                        "orchestrator_profile": "orchestrator",
                        "default_assignee": "fallback",
                    }
                },
            ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.child_ids and len(outcome.child_ids) == 1
    with kb.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    # 'made_up' wasn't in roster, so assignee rewritten to 'fallback'
    assert child.assignee == "fallback"


def test_decompose_malformed_llm_json_promotes_guarded_single_task(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="verify safely",
            body="read-only; do not modify files; final verdict only",
            triage=True,
        )

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_call_llm("not json at all, sorry")[0], patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is True
    assert outcome.fanout is False
    assert "malformed JSON fallback" in outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert task.assignee == "fallback"
    assert "decomposer returned malformed JSON" in task.body
    assert "read-only" in task.body
    assert "do not modify files" in task.body
    assert "final verdict only" in task.body


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


def test_decompose_no_aux_client_configured(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", triage=True)

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("No LLM provider configured"),
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "no auxiliary client" in outcome.reason
