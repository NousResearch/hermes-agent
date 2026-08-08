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
            name=n,
            is_default=(i == 0),
            description=f"desc for {n}",
            description_auto=False,
            model="m",
            provider="p",
            skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch(
            "hermes_cli.profiles.get_active_profile_name",
            return_value=names[0] if names else "default",
        ),
    ]


def test_decompose_with_fanout_creates_children(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {
                "title": "research",
                "body": "look it up",
                "assignee": "researcher",
                "parents": [],
            },
            {
                "title": "build",
                "body": "code it",
                "assignee": "engineer",
                "parents": [0],
            },
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


def test_decompose_fails_closed_on_malformed_parent_indices(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship safely", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "bad dependency shape",
        "tasks": [
            {
                "title": "research",
                "body": "look it up",
                "assignee": "researcher",
                "parents": [],
            },
            {
                "title": "build",
                "body": "code it",
                "assignee": "engineer",
                "parents": ["0"],
            },
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

    assert not outcome.ok
    assert "parents[0] is not a valid task index" in outcome.reason
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        children = kb.child_ids(conn, tid)
    assert root is not None
    assert root.status == "triage"
    assert children == []


@pytest.mark.parametrize("malformed", [None, 0, False, "", {}])
def test_decompose_rejects_falsy_non_list_parents(kanban_home, malformed):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship safely", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "bad dependency container",
        "tasks": [
            {
                "title": "research",
                "body": "look it up",
                "assignee": "researcher",
                "parents": malformed,
            }
        ],
    })
    patches = _patch_list_profiles(["orchestrator", "researcher"])
    for item in patches:
        item.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for item in patches:
            item.stop()

    assert not outcome.ok
    assert "parents must be a list" in outcome.reason


def test_decompose_rejects_boolean_and_duplicate_parent_indices(kanban_home):
    with kb.connect() as conn:
        bool_tid = kb.create_task(conn, title="bool parent", triage=True)
        duplicate_tid = kb.create_task(conn, title="duplicate parent", triage=True)

    def run(task_id, parents):
        payload = jsonlib.dumps({
            "fanout": True,
            "rationale": "bad dependency entry",
            "tasks": [
                {"title": "first", "assignee": "researcher", "parents": []},
                {"title": "second", "assignee": "researcher", "parents": parents},
            ],
        })
        patches = _patch_list_profiles(["orchestrator", "researcher"])
        for item in patches:
            item.start()
        try:
            with _patch_aux_client(payload), _patch_extra_body():
                return decomp.decompose_task(task_id, author="me")
        finally:
            for item in patches:
                item.stop()

    bool_outcome = run(bool_tid, [False])
    duplicate_outcome = run(duplicate_tid, [0, 0])

    assert not bool_outcome.ok
    assert "not a valid task index" in bool_outcome.reason
    assert not duplicate_outcome.ok
    assert "duplicate task index" in duplicate_outcome.reason


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
        with (
            _patch_aux_client(llm_payload),
            _patch_extra_body(),
            patch(
                "hermes_cli.kanban_decompose._load_config",
                return_value={"kanban": {"default_assignee": "fallback"}},
            ),
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
