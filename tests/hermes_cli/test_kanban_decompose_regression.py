"""Regression tests for the 09-13 auto-decomposer strandings.

Failure shape t_72cf3f84: a "promote #260" triage root was decomposed into
children that restated the root's own scope AND gated the root on them
(root wakes only when all children complete — but the root's job WAS the
promotion). Root claim then rejects with ``parents_not_done`` forever.
Failure shape #2: the same decompose routed a merge+deploy root to a
zero-write (research/read-only) profile.

These tests pin the three invariants that must hold for every decompose:
  1. a child that duplicates the parent's scope is REJECTED (decompose
     returns ok=False, nothing is written);
  2. when every leaf child would be the root's own job anyway, the root is
     never gated on its own duplicates (it is left executable);
  3. the roster prompt annotates read-only profiles so the LLM cannot
     mistake a zero-write lane for a write-capable one.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp
from hermes_cli.kanban_db_graph import decompose_triage_task, _leaves_assigned_to_root


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
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


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


def _create_triage(conn, title, body=None):
    return kb.create_task(conn, title=title, body=body, triage=True)


# --- t_72cf3f84 shape: child duplicates the parent scope -------------------

def test_decompose_rejects_child_that_duplicates_root_scope(kanban_home):
    """A "promote #260" root must not decompose into another "promote #260"
    child. The decomposer rejects the graph and writes NOTHING."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, "promote PR #260 to production")

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "split",
        "tasks": [
            {"title": "promote PR #260 to production", "body": "do the promote", "assignee": "work", "parents": []},
            {"title": "verify deploy", "body": "check containers", "assignee": "test", "parents": []},
        ],
    })
    patches = _patch_list_profiles(["orch", "work", "test"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "duplicates the root" in outcome.reason
    # Nothing written: root untouched, no children created.
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        assert root.status == "triage"
        assert kb.list_events(conn, tid) == [] or not any(
            ev.kind == "decomposed" for ev in kb.list_events(conn, tid)
        )


def test_decompose_allows_children_that_partition_root_scope(kanban_home):
    """Legitimate sub-steps sharing one root keyword must still decompose —
    only a child that covers most of the root's title is a duplicate."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, "promote PR #260 to production")

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "real partition",
        "tasks": [
            {"title": "merge dev branch", "body": "merge code", "assignee": "work", "parents": []},
            {"title": "restart containers", "body": "bounce services", "assignee": "test", "parents": []},
        ],
    })
    patches = _patch_list_profiles(["orch", "work", "test"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True


def test_leaves_assigned_to_root_means_self_gated_graph():
    """The pure DB-layer predicate: when every leaf child's assignee is the
    root's own assignee, the graph is the root's own job restated."""
    children = [
        {"title": "a", "assignee": "orch", "parents": []},
        {"title": "b", "assignee": "orch", "parents": [0]},
    ]
    assert _leaves_assigned_to_root(children, "orch") is True
    # A genuinely different leaf assignee breaks the pattern.
    children[1]["assignee"] = "work"
    assert _leaves_assigned_to_root(children, "orch") is False
    # Unset child assignees inherit the root's, so they count as "same".
    assert _leaves_assigned_to_root(
        [{"title": "a", "parents": []}], "orch"
    ) is True


def test_db_decompose_does_not_gate_root_on_duplicate_leaves(kanban_home):
    """DB-layer guard: when the graph's leaves all belong to the root's own
    assignee, the root stays executable (ready) instead of waiting on its
    own duplicates — the stranded-wiring deadlock cannot recur."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, "promote PR #260 to production")

    children = [
        {"title": "promote step A", "assignee": "orch", "parents": []},
        {"title": "promote step B", "assignee": "orch", "parents": [0]},
    ]
    with kbc.connect() as conn:
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch", children=children, author="decomposer",
        )
    assert child_ids is not None

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
    # Root NOT gated on the duplicate leaves: it is claimable.
    assert root.status == "ready"
    # The degraded wiring is audited on the card.
    comments = kb.list_comments(conn, tid)
    assert any("DEGRADED DECOMPOSITION" in (c.body or "") for c in comments)
    # And no root-gating links exist for the duplicate-leaf case.
    with kbc.connect() as conn:
        gated = conn.execute(
            "SELECT COUNT(*) AS n FROM task_links WHERE child_id = ? AND parent_id IN "
            "(" + ",".join("?" * len(child_ids)) + ")",
            tuple([tid, *child_ids]),
        ).fetchone()["n"]
    assert gated == 0


def test_db_decompose_normal_path_still_gates_root(kanban_home):
    """The fix must not over-fire: a healthy graph with distinct leaf
    assignees keeps the root waiting (todo) for its children."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, "ship the feature")

    children = [
        {"title": "research", "assignee": "researcher", "parents": []},
        {"title": "build", "assignee": "engineer", "parents": [0]},
    ]
    with kbc.connect() as conn:
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch", children=children, author="decomposer",
        )
    assert child_ids is not None
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])
    assert root.status == "todo"
    assert c0.status == "ready"
    assert c1.status == "todo"
    comments = kb.list_comments(conn, tid)
    assert not any("DEGRADED DECOMPOSITION" in (c.body or "") for c in comments)


def test_decompose_rejects_action_sequence_chain_of_root_pipeline(kanban_home):
    """A "merge + deploy" root whose children are just its own pipeline split
    one step per child (every child's action drawn from the root title) is
    the parent restated — rejected, nothing written."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, "Promote PR #260 to production (merge dev->main + deploy)")

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "pipeline split",
        "tasks": [
            {"title": "Merge dev into main for PR #260", "body": "merge", "assignee": "work", "parents": []},
            {"title": "Deploy main to production", "body": "deploy", "assignee": "work", "parents": [0]},
        ],
    })
    patches = _patch_list_profiles(["orch", "work", "test"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "action sequence" in outcome.reason
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"


def test_decompose_allows_children_with_new_deliverables(kanban_home):
    """Children that introduce genuinely different work (not merely the
    root's verbs split up) still decompose fine."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, "Promote PR #260 to production (merge dev->main + deploy)")

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "real partition",
        "tasks": [
            {"title": "Audit dashboard bundle for stale cache headers", "body": "audit work", "assignee": "work", "parents": []},
            {"title": "Write rollback playbook for deploy night", "body": "playbook", "assignee": "test", "parents": []},
        ],
    })
    patches = _patch_list_profiles(["orch", "work", "test"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason


# --- zero-write mis-assignment shape ----------------------------------------

def test_roster_annotates_readonly_profiles():
    """A zero-write/research profile description gets the no-write marker so
    the LLM prompt (and the human reading the prompt) can't mistake it for a
    write-capable lane."""
    roster = [
        {"name": "weatherman-data", "description": "SCALING ORACLE research lane: read/sims only", "has_description": True},
        {"name": "weatherman-work", "description": "OpenCode DRIVER: implements cards, merges to dev", "has_description": True},
        {"name": "bare", "description": "no markers here", "has_description": True},
    ]
    formatted = decomp._format_roster(roster)
    assert "weatherman-data ⚠ no-write" in formatted
    assert "weatherman-work ⚠ no-write" not in formatted
    assert "bare ⚠ no-write" not in formatted


def test_readonly_markers_cover_vp_documented_case():
    """The documented stranded assignment: weatherman-data (read/sim only)
    receiving a MERGE+DEPLOY card. Every marker phrase used across the VP
    governance docs must trigger the annotation."""
    for desc in (
        "WeatherMan data scientist — SCALING ORACLE (research lane)",
        "read-only profile: reads and sims, never writes",
        "zero-write gateway",
        "never merges own PRs, never promotes/deploys",
    ):
        assert decomp._roster_capability_note(desc) == " ⚠ no-write", desc
    assert decomp._roster_capability_note("implements code, opens PRs") == ""
