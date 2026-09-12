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


def test_decompose_false_when_task_not_triage_or_blocked(kanban_home):
    """A card in a non-fanoutable status (here: done) is rejected at entry."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x")
        with conn:
            conn.execute(
                "UPDATE tasks SET status = 'done' WHERE id = ?", (tid,)
            )
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


def test_decompose_blocked_resume_fanout_via_entry_path(kanban_home):
    """The 2026-09-01 resume fan-out must work through the REAL entry point.

    A ``blocked`` card split into 3 independent children yields at least one
    immediately-claimable child (ready) and nothing waits on the root. This is
    the executable expectation the original card t_b5958dab demanded — now
    exercised end-to-end through ``decompose_task`` (which previously rejected
    any non-triage task and made the blocked branch unreachable dead code).
    """
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="stuck card", assignee="orchestrator",
            initial_status="blocked",
        )

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "resume split",
        "tasks": [
            {"title": "research", "assignee": "researcher", "parents": []},
            {"title": "build", "assignee": "engineer", "parents": []},
            {"title": "verify", "assignee": "default", "parents": []},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer", "default"])
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
    assert outcome.child_ids and len(outcome.child_ids) == 3

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        kids = [kb.get_task(conn, cid) for cid in outcome.child_ids]
        # At least one child is immediately claimable.
        assert any(k.status == "ready" for k in kids), "no child immediately claimable"
        # No child is gated under the root (the 2026-09-01 deadlock bug).
        gated_under_root = [
            cid for cid in outcome.child_ids
            if root.id in {p for p in kb.parent_ids(conn, cid)}
        ]
        assert gated_under_root == []
        # Root waits on the whole graph: it is a child of every child.
        for cid in outcome.child_ids:
            assert root.id in set(kb.child_ids(conn, cid))
    # Root flipped to todo (gated on children completion).
    assert root.status == "todo"


# --- Guard 1 (t_c52b9bc3): NO SPLIT MID-REVIEW ---

def _insert_run(conn, tid, outcome, *, status="done"):
    """Insert a terminal run with the given outcome for a task."""
    import time as _time
    now = _time.time()
    cur = conn.execute(
        "INSERT INTO task_runs "
        "(task_id, profile, status, outcome, started_at, ended_at) "
        "VALUES (?, 'test', ?, ?, ?, ?)",
        (tid, status, outcome, int(now), int(now)),
    )
    conn.commit()
    return cur.lastrowid


def test_decompose_refuses_when_review_cycle_change_requested(kanban_home):
    """A card with a changes_requested round that ends blocked is NOT split.

    Guard 1: a blocked card whose newest decisive run outcome is
    ``changes_requested`` must be RESUMED (same card+worktree), never fanned
    out. decompose_task must refuse the split BEFORE calling the aux LLM (no
    fan-out choreography at all).
    """
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="mid-review card", assignee="orchestrator",
            initial_status="blocked",
        )
        # Simulate the review loop: round-1 changes_requested is the newest
        # decisive outcome; the fix run then ended BLOCKED (a trailing
        # non-decisive outcome that must not clear it).
        _insert_run(conn, tid, "changes_requested")
        _insert_run(conn, tid, "blocked")

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        # Guard fires before any LLM call — the aux client mock is intentionally
        # NOT installed, so reaching the LLM would raise and fail the test.
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "review cycle" in outcome.reason or "resuming" in outcome.reason
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        assert root.status == "blocked"  # card untouched — not fanned out
        # No children were created.
        child_ids = kb.child_ids(conn, tid)
    assert child_ids == []


# --- AC1/AC2: auto-decomposer decision-shaped children land in triage ---

def _auto_decompose(llm_payload, *, tid, profiles):
    """Run decompose_task with author='auto-decomposer' against a mocked aux LLM."""
    patches = _patch_list_profiles(profiles)
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            return decomp.decompose_task(tid, author=decomp.AUTO_DECOMPOSER_AUTHOR)
    finally:
        for p in patches:
            p.stop()


def test_auto_decompose_decision_child_lands_in_triage(kanban_home):
    """AC1: a decision-shaped auto-decomposer child lands in triage, not ready."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="analyze the People flow", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "split",
        "tasks": [
            {"title": "Decide the mention-derived People source design",
             "body": "pick source", "assignee": "jobsy", "parents": []},
            {"title": "Implement the chosen integration",
             "body": "build it", "assignee": "engineer", "parents": [0]},
        ],
    })

    profiles = ["orchestrator", "jobsy", "engineer"]
    outcome = _auto_decompose(llm_payload, tid=tid, profiles=profiles)
    assert outcome.ok, outcome.reason
    assert outcome.fanout is True and len(outcome.child_ids) == 2

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])  # decision card
        c1 = kb.get_task(conn, outcome.child_ids[1])   # downstream impl

    # Root flips to 'todo' as usual (gated by the parked decision).
    assert root.status == "todo"
    # AC1: decision-shaped card -> triage, NOT ready. AC2: it must not auto-promote.
    assert c0.status == "triage"
    assert c0.assignee == "jobsy"
    # Downstream depends on the decision (triage parent) -> stays 'todo' (no promote).
    assert c1.status == "todo"
    # created_by stamped so the re-entry guard (list_triage_ids) can exclude it.
    assert c0.created_by == decomp.AUTO_DECOMPOSER_AUTHOR


def test_auto_decompose_decision_child_does_not_auto_promote_on_recompute(kanban_home):
    """AC2: recompute_ready never promotes a triage card to ready."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="parallel decision fan", triage=True)
        child_ids = kb.decompose_triage_task(
            conn, tid,
            root_assignee="orch",
            children=[
                {"title": "Decide the data model orientation",
                 "body": "a or b", "assignee": "jobsy", "triage": True},
                {"title": "Then implement it",
                 "body": "build", "assignee": "engineer", "parents": [0]},
            ],
            author=decomp.AUTO_DECOMPOSER_AUTHOR,
        )
    assert child_ids is not None and len(child_ids) == 2

    with kb.connect() as conn:
        decision = kb.get_task(conn, child_ids[0])
        downstream = kb.get_task(conn, child_ids[1])
    # Decision parked; never upgraded by recompute_ready (only 'todo'/'blocked' do).
    assert decision.status == "triage"
    # Downstream is gated on an un-promoted decision -> still todo.
    assert downstream.status == "todo"


def test_auto_decompose_non_decision_child_still_promotes(kanban_home):
    """AC5: a non-decision auto-decomposer child keeps current behavior (ready)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="feature build", triage=True)
        child_ids = kb.decompose_triage_task(
            conn, tid,
            root_assignee="orch",
            children=[
                {"title": "Implement feature Y",
                 "body": "build", "assignee": "engineer"},
            ],
            author=decomp.AUTO_DECOMPOSER_AUTHOR,
        )
    assert child_ids is not None
    with kb.connect() as conn:
        c0 = kb.get_task(conn, child_ids[0])
    assert c0.status == "ready"
    assert c0.created_by == decomp.AUTO_DECOMPOSER_AUTHOR


def test_list_triage_ids_excludes_auto_decomposer_created(kanban_home):
    """Re-entry guard: auto-decomposer-created triage is not re-decomposed."""
    with kb.connect() as conn:
        user_triage = kb.create_task(conn, title="user dropped", triage=True, assignee="someone")
        park_root = kb.create_task(conn, title="park me", triage=True)
        decision = kb.decompose_triage_task(
            conn, park_root,
            root_assignee="orch",
            children=[{"title": "Approve the API", "assignee": "jobsy", "triage": True}],
            author=decomp.AUTO_DECOMPOSER_AUTHOR,
        )[0]
    ids = decomp.list_triage_ids()
    assert user_triage in ids          # real user triage still decomposes
    assert decision not in ids         # auto-decomposer-parked decision excluded


def test_list_triage_ids_excludes_loop_breaker_triage(kanban_home):
    """Charter §6 (2026-09-03): a card that block_task routed to triage because
    it re-blocked for the same kind (the loop breaker) is parked for a HUMAN.
    The auto-decomposer must not pick it up and re-run it without Richie.
    """
    with kb.connect() as conn:
        looped = kb.create_task(conn, title="keeps failing", assignee="bob")
        conn.execute(
            "UPDATE tasks SET status='triage', block_kind='transient', block_recurrences=? WHERE id=?",
            (kb.BLOCK_RECURRENCE_LIMIT, looped),
        )
        conn.commit()
        fresh = kb.create_task(conn, title="user dropped", triage=True, assignee="someone")
    ids = decomp.list_triage_ids()
    assert fresh in ids
    assert looped not in ids


# --- AC1 (t_405f7f1f): decision-verb regex tightened => no over-match ---

def test_decision_regex_matches_only_decision_shaped_titles():
    """The routing regex must match titles that demand a PM decision and NOT
    implementation titles that merely contain an ambiguous decision verb.
    Asserts behaviour of the regex via _is_decision_shaped, not snapshots.
    See kanban-decompose t_405f7f1f AC1 for the source matrix.
    """
    cases = {
        # Genuine decision shapes -> match (keep).
        "Decide the X approach": True,
        "Approve the X design": True,
        "Spec the X interface": True,
        "Ratify the PRD amendment": True,
        "Amend the design doc": True,
        # Ambiguous verbs in implementation titles -> do NOT match (the fix).
        "Lock the report row rendering": False,
        "Sign off on the lint config": False,
        "Approve the merge button": False,
        # Unrelated implementation titles -> keep no-match.
        "Implement useCardReorder hook": False,
        "Add photo affordance": False,
        "Extract useCardReorder hook (GlobalAside + ActionList dedup)": False,
    }
    for title, expected in cases.items():
        assert decomp._is_decision_shaped(title) is expected, title


# --- dry-run: compute the graph but write nothing ---

def test_dry_run_fanout_writes_nothing(kanban_home):
    """--dry-run computes children + routing decisions but performs no DB write."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "Decide the X approach", "body": "a or b",
             "assignee": "jobsy", "parents": []},
            {"title": "Implement feature Y", "body": "build it",
             "assignee": "engineer", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "jobsy", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(
                tid, author=decomp.AUTO_DECOMPOSER_AUTHOR, dry_run=True,
            )
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    # No child ids assigned — nothing was written.
    assert outcome.child_ids is None
    # The plan carries the routing decisions a manual E2E needs to inspect.
    assert outcome.dry_run_plan is not None
    assert len(outcome.dry_run_plan) == 2
    assert outcome.dry_run_plan[0]["title"] == "Decide the X approach"
    assert outcome.dry_run_plan[0]["assignee"] == "jobsy"
    assert outcome.dry_run_plan[0]["triage"] is True   # decision-shaped, parked
    assert outcome.dry_run_plan[1]["title"] == "Implement feature Y"
    assert outcome.dry_run_plan[1]["triage"] is False

    # The board is untouched: the root is still triage, and no children exist.
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        remaining = kb.list_tasks(conn, status="triage")
    assert root.status == "triage"
    assert all(r.id != tid for r in remaining) is False  # root still in triage set
    # Only the root task was ever created.
    with kb.connect() as conn:
        all_tasks = kb.list_tasks(conn, limit=1000)
    assert len(all_tasks) == 1


def test_decompose_role_map_routes_lane_children_deterministically(kanban_home):
    """AC2 (t_bc08efc3): a lane-shaped child title is routed by the role-map,
    NOT the LLM's free-text roster string — Implement->bob, Review->rodge,
    Verify->steve-o — regardless of what the LLM emitted."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    # The LLM mis-emits every assignee to a DIFFERENT lane; the role-map must win.
    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "route by lane",
        "tasks": [
            {"title": "Implement the widget", "body": "build it",
             "assignee": "rodge", "parents": []},
            {"title": "Review the widget diff", "body": "check it",
             "assignee": "steve-o", "parents": [0]},
            {"title": "Verify the widget renders", "body": "qa it",
             "assignee": "bob", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "bob", "rodge", "steve-o"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me", dry_run=True)
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    plan = outcome.dry_run_plan
    assert plan is not None and len(plan) == 3
    assert plan[0]["title"] == "Implement the widget"
    assert plan[0]["assignee"] == "bob"      # LLM said rodge; role-map wins
    assert plan[1]["title"] == "Review the widget diff"
    assert plan[1]["assignee"] == "rodge"    # LLM said steve-o; role-map wins
    assert plan[2]["title"] == "Verify the widget renders"
    assert plan[2]["assignee"] == "steve-o"  # LLM said bob; role-map wins


def test_dry_run_single_task_writes_nothing(kanban_home):
    """--dry-run on a fanout=false response returns the spec but writes nothing."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="single thing", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "one unit",
        "title": "Tightened title",
        "body": "Do the one thing.",
        "assignee": "jobsy",
    })

    patches = _patch_list_profiles(["orchestrator", "jobsy"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me", dry_run=True)
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is False
    assert outcome.dry_run_plan is not None
    assert outcome.dry_run_plan[0]["title"] == "Tightened title"

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        all_tasks = kb.list_tasks(conn, limit=1000)
    assert root.status == "triage"
    assert len(all_tasks) == 1




# --- Change A (2026-09-04 GlobalAside): deploy children are operator-held ---


def test_deploy_regex_matches_production_shipping_titles():
    """``_is_deploy_shaped`` matches titles that would ship code / release to
    a human, and does NOT over-match incidental uses. This is the trigger that
    forces a real operator_hold so a deployment can never auto-promote past an
    owner approval gate."""
    cases = {
        "Deploy GlobalAside restore to production": True,
        "Deploy the backend fix": True,
        "Release v1.2 to prod": True,
        "Rollout the new build": True,
        "Go live with the sidebar": True,
        "Ship-to-prod the dashboard": True,
        "Implement feature Y": False,
        "Verify aside renders": False,
        "Design the card layout": False,
        "Run the regression suite": False,
        "Review the GlobalAside diff": False,
        "Investigate the release notes wording": True,   # 'release' noun — still deploy-ish
    }
    for title, expected in cases.items():
        assert decomp._is_deploy_shaped(title) is expected, title


def test_auto_decompose_deploy_child_forced_hold(kanban_home):
    """Change A backstop: an AUTO-decomposed child whose title is deploy-shaped
    gets ``hold=True`` even when the LLM omits the flag — so it is created as a
    real operator_hold downstream and can never auto-promote past approval.
    A manually-fan-out (non-auto) deploy child keeps ``hold=False`` unless the
    caller explicitly flags it."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test",
        "tasks": [
            {"title": "Implement GlobalAside", "body": "build",
             "assignee": "engineer", "parents": []},                       # idx 0
            {"title": "Deploy to production", "body": "ship",
             "assignee": "engineer", "parents": [0]},                      # idx 1, deploy-shaped
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(
                tid, author=decomp.AUTO_DECOMPOSER_AUTHOR, dry_run=True,
            )
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout
    plan = outcome.dry_run_plan
    assert plan is not None and len(plan) == 2
    # Impl child: not held.
    assert plan[0]["hold"] is False
    # Deploy-shaped child: forced hold TRUE (LLM didn't set it).
    assert plan[1]["hold"] is True, (
        "auto-decomposed deploy child not forced to operator_hold — Change A "
        "backstop regression"
    )


def test_manual_decompose_deploy_child_not_forced(kanban_home):
    """A MANUAL (non-auto-decomposer) fan-out of a deploy-shaped child must NOT
    be force-held — the caller (owner/PM) is already committed, mirroring the
    AC1 decision-shaped asymmetry. Only auto-decomposer children get the
    backstop hold."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="release feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test",
        "tasks": [
            {"title": "Deploy to production", "body": "ship",
             "assignee": "engineer", "parents": []},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me", dry_run=True)
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    plan = outcome.dry_run_plan
    assert plan is not None and len(plan) == 1
    assert plan[0]["hold"] is False  # manual deploy NOT auto-held


def test_auto_decompose_explicit_hold_respected(kanban_home):
    """When the LLM explicitly sets hold:true on a non-deploy-shaped child, it
    is honoured (terminal children like 'release to human' that don't match the
    deploy regex are still held by explicit intent)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="publish", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test",
        "tasks": [
            {"title": "Send approval to owner", "body": "ask",
             "assignee": "jobsy", "parents": [], "hold": True},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "jobsy"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(
                tid, author=decomp.AUTO_DECOMPOSER_AUTHOR, dry_run=True,
            )
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.dry_run_plan is not None
    assert outcome.dry_run_plan[0]["hold"] is True
