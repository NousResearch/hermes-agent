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
        with _patch_aux_client(llm_payload):
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


def test_decompose_fanout_children_inherit_root_assignee_when_unrouted(kanban_home):
    """Unrouted children fall back to the ROOT task's assignee, not
    the decomposer's active profile (#114294). The active profile here is ``private``
    (an incognito profile with no credentials), so the old fallback spawned
    workers that deadlocked on capability blockers."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="zdr", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "made_up", "parents": []},
            {"title": "build", "body": "code it", "assignee": None, "parents": [0]},
        ],
    })

    # get_active_profile_name() is mocked to names[0] = "private" — the
    # global default chain would resolve there without kanban.default_assignee.
    patches = _patch_list_profiles(["private", "zdr"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), patch(
            "hermes_cli.config.load_config_readonly",
            return_value={},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert c0.assignee == "zdr"
    assert c1.assignee == "zdr"
    # Same class for the root: no ``orchestrator_profile`` must not hand the
    # orchestration card to the dispatcher's own (here: incognito) profile.
    assert root.assignee == "zdr"


def test_decompose_explicit_default_assignee_wins_over_root_assignee(kanban_home):
    """An explicitly configured ``kanban.default_assignee`` stays
    authoritative for unroutable children; the root task's assignee only
    fills in when no explicit default is set (explicit config → card
    assignee → active profile)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="engineer", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "made_up", "parents": []},
            {"title": "build", "body": "code it", "assignee": None, "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["engineer", "docs", "private"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"kanban": {"default_assignee": "docs"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert c0.assignee == "docs"
    assert c1.assignee == "docs"


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
        with _patch_aux_client(llm_payload), patch(
            "hermes_cli.config.load_config_readonly",
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


def test_load_routing_falls_back_to_defaults_when_config_unreadable(kanban_home, monkeypatch):
    """decompose_task promises ok=False on expected failures; a config read that raises (missing
    profile home, HomeInitializationError) must not escape _load_routing as an exception."""
    from hermes_cli import config as config_mod

    def _boom():
        raise FileNotFoundError("profile home is gone")

    monkeypatch.setattr(config_mod, "load_config_readonly", _boom)
    routing = decomp._load_routing()
    assert routing.default_assignee == "default" and routing.auto_promote is True


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


# --- duplicate suppression (t_2e0cd7e4) -------------------------------------
# A parked defect card decomposed into children that re-did the parent's own
# work and into two siblings editing one file, so one bug became three
# concurrent edits. The decomposer must refuse to spawn work it already owns.

_PARENT_TITLE = "Fleet routing refuses execution-class cards on review lanes"
_PARENT_BODY = "The refusal comes from hermes_cli/kanban_fleet_routing.py."


def _decompose_with(kanban_home, tid, tasks, profiles=("orchestrator", "engineer")):
    payload = jsonlib.dumps({"fanout": True, "rationale": "split", "tasks": tasks})
    patches = _patch_list_profiles(list(profiles))
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(payload):
            return decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()


def test_decompose_drops_a_child_that_duplicates_the_parents_scope(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=_PARENT_TITLE, body=_PARENT_BODY, triage=True)

    outcome = _decompose_with(kanban_home, tid, [
        {"title": _PARENT_TITLE, "body": "", "assignee": "engineer", "parents": []},
        {"title": "Add regression test for reviewer-raised defect cards",
         "body": "", "assignee": "engineer", "parents": []},
    ])

    assert outcome.ok, outcome.reason
    assert outcome.child_ids is not None and len(outcome.child_ids) == 1
    with kbc.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    assert child is not None
    assert child.title == "Add regression test for reviewer-raised defect cards"


def test_decompose_drops_siblings_that_edit_the_same_file(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Ship the routing governance release",
                             body="Coordinate the work in one pass.", triage=True)

    outcome = _decompose_with(kanban_home, tid, [
        {"title": "Patch kanban_fleet_routing.py to accept execution cards on review lane",
         "body": "", "assignee": "engineer", "parents": []},
        {"title": "Update kanban_fleet_routing.py so execution cards on review lanes are accepted",
         "body": "", "assignee": "engineer", "parents": []},
        {"title": "Write the changelog note", "body": "", "assignee": "engineer", "parents": []},
    ])

    assert outcome.ok, outcome.reason
    assert outcome.child_ids is not None and len(outcome.child_ids) == 2
    with kbc.connect() as conn:
        titles = set()
        for cid in outcome.child_ids:
            child = kb.get_task(conn, cid)
            assert child is not None
            titles.add(child.title)
    assert titles == {
        "Patch kanban_fleet_routing.py to accept execution cards on review lane",
        "Write the changelog note",
    }, "the second same-file child must not be spawned"


def test_decompose_with_only_duplicates_leaves_the_card_in_triage(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=_PARENT_TITLE, body=_PARENT_BODY, triage=True)

    outcome = _decompose_with(kanban_home, tid, [
        {"title": _PARENT_TITLE, "body": "", "assignee": "engineer", "parents": []},
        {"title": _PARENT_TITLE, "body": "again", "assignee": "engineer", "parents": []},
    ])

    assert outcome.ok is False
    assert "every proposed child duplicates" in outcome.reason
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
    assert root is not None and root.status == "triage", "a human must still pick it up"


def test_a_dropped_sibling_leaves_no_child_waiting_on_it(kanban_home):
    """Round-1 review: dropping a child silently deleted the edge of any kept child naming it.

    The kept child then promoted with its prerequisite owned by nobody. A sibling absorbed into an
    accepted child must repoint the edge at the survivor.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Ship the routing governance release",
                             body="Coordinate the work in one pass.", triage=True)

    outcome = _decompose_with(kanban_home, tid, [
        {"title": "Add regression test for reviewer-raised defect cards",
         "body": "", "assignee": "engineer", "parents": []},
        # exact duplicate of the previous child -> dropped, absorbed into it
        {"title": "Add regression test for reviewer-raised defect cards",
         "body": "", "assignee": "engineer", "parents": []},
        {"title": "Write the changelog note", "body": "", "assignee": "engineer", "parents": [1]},
    ])

    assert outcome.ok, outcome.reason
    assert outcome.child_ids is not None and len(outcome.child_ids) == 2
    survivor, changelog = outcome.child_ids
    with kbc.connect() as conn:
        assert kb.parent_ids(conn, changelog) == [survivor], (
            "the changelog's prerequisite was dropped, so the edge must point at the survivor")


def test_a_child_waiting_on_parent_scope_duplicate_is_cascaded_away(kanban_home):
    """Nothing else owns the parent's own scope, so waiting on it can only deadlock.

    The root waits on every child, so the edge cannot be satisfied by the parent either; the
    dependent child is dropped with its prerequisite and the card stays in triage.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=_PARENT_TITLE, body=_PARENT_BODY, triage=True)

    outcome = _decompose_with(kanban_home, tid, [
        {"title": _PARENT_TITLE, "body": "", "assignee": "engineer", "parents": []},
        {"title": "Write the changelog note", "body": "", "assignee": "engineer", "parents": [0]},
    ])

    assert outcome.ok is False
    assert "every proposed child duplicates" in outcome.reason
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"
        assert kb.list_events(conn, tid), "the drop must be recorded, not silent"


