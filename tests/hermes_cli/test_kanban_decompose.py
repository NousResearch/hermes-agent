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


# ---------------------------------------------------------------------------
# auto_decompose_excluded_profiles (#83046)
# ---------------------------------------------------------------------------

def _capturing_aux(content: str):
    """Patch call_llm with a side_effect that records the prompt messages.

    Returns ``(patcher, captured)`` — ``captured["messages"]`` holds the
    [system, user] message list the decomposer actually sent.
    """
    captured: dict = {}

    def _fake_call_llm(*, task, messages, **kwargs):
        captured["messages"] = messages
        return _fake_aux_response(content)

    return (
        patch("agent.auxiliary_client.call_llm", side_effect=_fake_call_llm),
        captured,
    )


def test_excluded_profile_absent_from_roster_prompt_and_valid_set(kanban_home):
    """The reporter's core invariant: an excluded profile appears in NEITHER
    the roster the LLM is prompted with NOR the post-hoc valid-assignee set
    — so a response that names it anyway cannot create a child for it."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "worker", "parents": []},
            # The LLM disobeys the (now shorter) roster and names the
            # excluded profile directly.
            {"title": "review it", "body": "check the work", "assignee": "reviewer", "parents": []},
        ],
    })

    aux_patch, captured = _capturing_aux(llm_payload)
    # Mixed-case + padding in config must match the on-disk id.
    cfg_patch = patch(
        "hermes_cli.kanban_decompose._load_config",
        return_value={
            "kanban": {
                "default_assignee": "worker",
                "auto_decompose_excluded_profiles": [" Reviewer "],
            },
        },
    )

    patches = _patch_list_profiles(["orchestrator", "reviewer", "worker"]) + [
        aux_patch,
        cfg_patch,
    ]
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="me")

        # Unit-level: the valid-assignee surface excludes the profile too.
        excluded = decomp._resolve_excluded_profiles(
            {"auto_decompose_excluded_profiles": [" Reviewer "]}
        )
        assert excluded == frozenset({"reviewer"})
        roster, valid = decomp._build_roster(excluded=excluded)
        assert {e["name"] for e in roster} == {"orchestrator", "worker"}
        assert "reviewer" not in valid
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    user_msg = next(
        m["content"] for m in captured["messages"] if m["role"] == "user"
    )
    assert "reviewer" not in user_msg  # absent from the prompted roster
    assert "orchestrator" in user_msg and "worker" in user_msg

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        children = [kb.get_task(conn, cid) for cid in outcome.child_ids]
    assert root.status == "todo"
    by_title = {c.title: c for c in children}
    assert by_title["review it"].assignee == "worker"


def test_fanout_false_naming_excluded_profile_falls_back_to_default(kanban_home):
    """Single-task promotion shares the routing guarantee: an LLM assignee
    naming an excluded profile falls back to default_assignee, never to
    the excluded card."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="route me safely", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Route to fallback.",
        "assignee": "REVIEWER",  # excluded (case-insensitively)
    })

    patches = _patch_list_profiles(["orchestrator", "reviewer", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={
                "kanban": {
                    "default_assignee": "fallback",
                    "auto_decompose_excluded_profiles": ["reviewer"],
                },
            },
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.assignee == "fallback"


def test_directly_assigned_card_ignores_exclusions(kanban_home):
    """Scope guarantee: explicit assignment is never rewritten by the
    exclusion list. A triage card already assigned to the excluded profile
    promotes with its assignee intact; a non-triage card assigned to it is
    simply none of the decomposer's business."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="pre-assigned", assignee="reviewer", triage=True)
        other_tid = kb.create_task(conn, title="explicit ready card", assignee="reviewer")

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Keep the existing assignee.",
        "assignee": None,
    })

    patches = _patch_list_profiles(["orchestrator", "reviewer", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={
                "kanban": {
                    "default_assignee": "fallback",
                    "auto_decompose_excluded_profiles": ["reviewer"],
                },
            },
        ):
            outcome = decomp.decompose_task(tid, author="me")
            skipped = decomp.decompose_task(other_tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        promoted = kb.get_task(conn, tid)
        untouched = kb.get_task(conn, other_tid)
    # Parentless specified tasks flip straight to ready (recompute_ready
    # runs inline after specify) — the point here is the assignee.
    assert promoted.status == "ready"
    assert promoted.assignee == "reviewer"
    assert skipped.ok is False
    assert "not in triage" in skipped.reason
    assert untouched.status == "ready"
    assert untouched.assignee == "reviewer"


def test_excluded_default_assignee_fails_closed(kanban_home):
    """Excluding the resolved default_assignee would silently route every
    unmatched child at the forbidden profile — fail closed instead: no
    LLM call, no children, root stays in Triage, one board notice."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="dangerous config", triage=True)

    aux_mock = MagicMock(return_value=_fake_aux_response("{}"))
    aux_patch = patch("agent.auxiliary_client.call_llm", aux_mock)
    cfg_patch = patch(
        "hermes_cli.kanban_decompose._load_config",
        return_value={
            "kanban": {
                "default_assignee": "fallback",
                "auto_decompose_excluded_profiles": ["fallback"],
            },
        },
    )
    patches = _patch_list_profiles(["orchestrator", "fallback"]) + [
        aux_patch,
        cfg_patch,
    ]
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="auto-decomposer")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "auto_decompose_excluded_profiles" in outcome.reason

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)
    assert root.status == "triage"  # fail-closed: root untouched
    assert len(comments) == 1
    assert comments[0].body.startswith("DECOMPOSE SKIPPED:")
    assert comments[0].author == "auto-decomposer"
    assert sum(1 for e in events if e.kind == "commented") == 1
    aux_mock.assert_not_called()


def test_exclusion_emptying_roster_fails_closed(kanban_home):
    """Excluding every installed profile leaves nothing routable — skip
    loudly rather than prompting the LLM with an empty roster."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="no one left", triage=True)

    aux_mock = MagicMock(
        return_value=_fake_aux_response(jsonlib.dumps({"fanout": True, "tasks": []}))
    )
    aux_patch = patch("agent.auxiliary_client.call_llm", aux_mock)
    cfg_patch = patch(
        "hermes_cli.kanban_decompose._load_config",
        return_value={"kanban": {"auto_decompose_excluded_profiles": ["solo"]}},
    )
    # For the roster-empty guard (not the default-assignee guard) to fire,
    # the resolved default assignee must sit outside BOTH the exclusion set
    # and the roster. Simulate the real shape where the active profile
    # resolves to a name absent from list_profiles() (list_profiles omits
    # the built-in default when the default home dir doesn't exist).
    # Started after _patch_list_profiles' version so this one wins.
    active_patch = patch(
        "hermes_cli.profiles.get_active_profile_name",
        return_value="default",
    )
    patches = (
        _patch_list_profiles(["solo"]) + [aux_patch, cfg_patch, active_patch]
    )
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="auto-decomposer")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "no routable profiles remain" in outcome.reason

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        comments = kb.list_comments(conn, tid)
    assert root.status == "triage"
    assert len(comments) == 1
    assert comments[0].body.startswith("DECOMPOSE SKIPPED:")
    aux_mock.assert_not_called()


def test_empty_exclusion_list_preserves_existing_behavior(kanban_home):
    """The default (empty) list is a no-op: every profile stays routable
    and fan-out proceeds exactly as before #83046."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "review it", "body": "check the work", "assignee": "reviewer", "parents": []},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "reviewer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"auto_decompose_excluded_profiles": []}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    assert child.assignee == "reviewer"


