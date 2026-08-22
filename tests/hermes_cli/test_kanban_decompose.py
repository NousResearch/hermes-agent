"""Tests for the decomposer module + `hermes kanban decompose` CLI surface.

The auxiliary LLM client is mocked — no network calls. Tests exercise the
prompt plumbing, response parsing, DB writes (via the real DB helper),
and the assignee-fallback logic.
"""

from __future__ import annotations

import json as jsonlib
import sqlite3
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
        patch.object(decomp.profiles_mod, "list_profiles", return_value=fake_profiles),
        patch.object(
            decomp.profiles_mod,
            "profile_exists",
            side_effect=lambda x: x in names,
        ),
        patch.object(
            decomp.profiles_mod,
            "get_active_profile_name",
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
            {"title": "research", "body": "look it up", "assignee": "researcher"},
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


@pytest.mark.parametrize(
    "parent_fields",
    [
        [{"parents": "not-a-list"}],
        [{"parents": None}],
        [{"parents": ["0"]}],
        [{"parents": [1]}],
        [{"parents": [0]}],
        [{"parents": [1]}, {"parents": [0]}],
        [{}, {"parents": [False]}],
        [{}, {"parents": [0, 0]}],
    ],
    ids=[
        "non-list",
        "null",
        "string-index",
        "out-of-range",
        "self-reference",
        "cycle",
        "boolean-index-non-self",
        "duplicate-parent",
    ],
)
def test_decompose_rejects_invalid_dependency_graph_without_rewriting(
    kanban_home, parent_fields
):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="preserve this root", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "invalid graph",
        "tasks": [
            {
                "title": f"child {idx}",
                "body": "work",
                "assignee": "worker",
                **fields,
            }
            for idx, fields in enumerate(parent_fields)
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "worker"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert outcome.reason.startswith("DB rejected graph:")
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        task_ids = [row["id"] for row in conn.execute("SELECT id FROM tasks")]
        rejection_events = [
            event for event in kb.list_events(conn, tid)
            if event.kind == "decompose_rejected"
        ]
    assert root is not None
    assert root.status == "triage"
    assert root.title == "preserve this root"
    assert task_ids == [tid]
    assert len(rejection_events) == 1
    payload = rejection_events[0].payload
    assert payload is not None
    assert payload["class"] == "invalid_dependency_graph"
    assert payload["reason"] == outcome.reason.removeprefix(
        "DB rejected graph: "
    )
    assert payload["author"] == "me"


@pytest.mark.parametrize("mutation", ["transition", "delete"])
def test_decompose_invalid_graph_does_not_reject_a_stale_root(
    kanban_home, mutation
):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="preserve this root", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "invalid graph after stale read",
        "tasks": [
            {"title": "first", "body": "work", "assignee": "worker"},
            {
                "title": "second",
                "body": "work",
                "assignee": "worker",
                "parents": ["0"],
            },
        ],
    })

    def mutate_root_during_llm(*args, **kwargs):
        with kb.connect() as conn:
            if mutation == "transition":
                with kb.write_txn(conn):
                    conn.execute(
                        "UPDATE tasks SET status = 'todo' WHERE id = ?",
                        (tid,),
                    )
            else:
                assert kb.delete_task(conn, tid)
        return _fake_aux_response(llm_payload)

    patches = _patch_list_profiles(["orchestrator", "worker"])
    for p in patches:
        p.start()
    try:
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=mutate_root_during_llm,
        ), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert outcome.reason == "task moved out of triage before decomposition"
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        task_ids = [row["id"] for row in conn.execute("SELECT id FROM tasks")]
        rejection_count = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind = 'decompose_rejected'"
        ).fetchone()[0]
    assert rejection_count == 0
    if mutation == "transition":
        assert root is not None
        assert root.status == "todo"
        assert root.title == "preserve this root"
        assert task_ids == [tid]
    else:
        assert root is None
        assert task_ids == []


def test_decompose_event_write_failure_is_not_reported_as_durable_rejection(
    kanban_home,
):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="preserve this root", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "invalid graph",
        "tasks": [
            {
                "title": "child",
                "body": "work",
                "assignee": "worker",
                "parents": ["0"],
            },
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "worker"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch.object(
            decomp.kb,
            "_append_event",
            side_effect=sqlite3.OperationalError("event write failed"),
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert outcome.reason == "DB error: OperationalError"
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        task_ids = [row["id"] for row in conn.execute("SELECT id FROM tasks")]
        rejection_count = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind = 'decompose_rejected'"
        ).fetchone()[0]
    assert root is not None
    assert root.status == "triage"
    assert root.title == "preserve this root"
    assert task_ids == [tid]
    assert rejection_count == 0


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
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch.object(
            decomp,
            "_load_config",
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


