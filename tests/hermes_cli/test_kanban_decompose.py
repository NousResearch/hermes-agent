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


def test_decompose_auxiliary_error_is_reported(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", triage=True)

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("no auxiliary provider configured"),
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert outcome.reason == "LLM error: RuntimeError"


def test_linear_bridge_origin_task_is_not_decomposed(kanban_home):
    """Already-specced Linear bridge cards must remain a single work item."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="already specced from Linear",
            assignee="engineer",
            triage=True,
            created_by="linear_bridge",
            idempotency_key="linear:00000000-0000-0000-0000-000000000001",
        )

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "would split without the provenance gate",
        "tasks": [
            {"title": "unexpected child", "body": "must not exist", "assignee": "engineer", "parents": []},
        ],
    })
    client = _mock_client_returning(llm_payload)
    patches = _patch_list_profiles(["engineer"])
    for p in patches:
        p.start()
    try:
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(client, "test-model"),
        ), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="auto-decomposer")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "linear bridge" in outcome.reason.lower()
    assert client.chat.completions.create.call_count == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE created_by = 'auto-decomposer'"
        ).fetchone()[0] == 0


def test_block_recurrence_triage_is_not_decomposed(kanban_home):
    """Loop-breaker triage is a human escalation, never decomposition intent."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="needs repeated human review", assignee="worker")
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        assert kb.block_task(conn, tid, reason="first review hold", kind="needs_input")
        assert kb.unblock_task(conn, tid)
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        assert kb.block_task(conn, tid, reason="second distinct review hold", kind="needs_input")
        escalated = kb.get_task(conn, tid)
        assert escalated.status == "triage"
        assert escalated.triage_origin == "block_recurrence"

    client = _mock_client_returning(jsonlib.dumps({
        "fanout": True,
        "rationale": "must not run",
        "tasks": [
            {"title": "unexpected child", "body": "must not exist", "assignee": "worker", "parents": []},
        ],
    }))
    patches = _patch_list_profiles(["worker"])
    for p in patches:
        p.start()
    try:
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(client, "test-model"),
        ), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="auto-decomposer")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "human escalation" in outcome.reason.lower()
    assert client.chat.completions.create.call_count == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE created_by = 'auto-decomposer'"
        ).fetchone()[0] == 0


def test_decomposition_rejects_children_over_configured_hard_cap(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="oversized rough idea", triage=True)

    tasks = [
        {
            "title": f"child {idx}",
            "body": "bounded child",
            "assignee": "engineer",
            "parents": [],
        }
        for idx in range(7)
    ]
    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "intentionally over the hard cap",
        "tasks": tasks,
    })
    patches = _patch_list_profiles(["engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), patch.object(
            decomp,
            "_load_config",
            return_value={"kanban": {"decomposition_max_children": 6}},
        ), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="auto-decomposer")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "7" in outcome.reason and "maximum 6" in outcome.reason
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE created_by = 'auto-decomposer'"
        ).fetchone()[0] == 0


def test_decomposition_child_cannot_recursively_decompose_at_max_depth(kanban_home):
    with kb.connect() as conn:
        root_id = kb.create_task(conn, title="root rough idea", triage=True)
        child_ids = kb.decompose_triage_task(
            conn,
            root_id,
            root_assignee="orchestrator",
            children=[
                {
                    "title": "first-generation child",
                    "body": "must stay bounded",
                    "assignee": "engineer",
                    "parents": [],
                }
            ],
            author="auto-decomposer",
        )
        assert child_ids and len(child_ids) == 1
        child_id = child_ids[0]
        child = kb.get_task(conn, child_id)
        assert child.decomposition_depth == 1
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'triage', triage_origin = ? WHERE id = ?",
                (kb.TRIAGE_ORIGIN_DECOMPOSE, child_id),
            )

    client = _mock_client_returning(jsonlib.dumps({
        "fanout": True,
        "rationale": "must not run past configured depth",
        "tasks": [
            {"title": "grandchild", "body": "must not exist", "assignee": "engineer", "parents": []},
        ],
    }))
    patches = _patch_list_profiles(["engineer", "orchestrator"])
    for p in patches:
        p.start()
    try:
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(client, "test-model"),
        ), patch.object(
            decomp,
            "_load_config",
            return_value={"kanban": {"decomposition_max_depth": 1}},
        ), _patch_extra_body():
            outcome = decomp.decompose_task(child_id, author="auto-decomposer")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "maximum decomposition depth" in outcome.reason.lower()
    assert client.chat.completions.create.call_count == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, child_id).status == "triage"
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE title = 'grandchild'"
        ).fetchone()[0] == 0


def test_auto_decompose_sweep_only_lists_intended_triage(kanban_home):
    with kb.connect() as conn:
        intended_id = kb.create_task(conn, title="rough idea", triage=True)
        bridge_id = kb.create_task(
            conn,
            title="linear card",
            triage=True,
            created_by="linear_bridge",
            idempotency_key="linear:00000000-0000-0000-0000-000000000002",
        )
        escalated_id = kb.create_task(
            conn,
            title="human escalation",
            triage=True,
            triage_origin=kb.TRIAGE_ORIGIN_BLOCK_RECURRENCE,
        )
        root_id = kb.create_task(conn, title="parent rough idea", triage=True)
        child_ids = kb.decompose_triage_task(
            conn,
            root_id,
            root_assignee="orchestrator",
            children=[{"title": "depth-one child", "assignee": "worker", "parents": []}],
            author="auto-decomposer",
        )
        assert child_ids
        depth_child_id = child_ids[0]
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'triage', triage_origin = ? WHERE id = ?",
                (kb.TRIAGE_ORIGIN_DECOMPOSE, depth_child_id),
            )

    with patch.object(
        decomp,
        "_load_config",
        return_value={"kanban": {"decomposition_max_depth": 1}},
    ):
        eligible_ids = decomp.list_auto_decompose_ids()

    assert eligible_ids == [intended_id]
    assert bridge_id not in eligible_ids
    assert escalated_id not in eligible_ids
    assert depth_child_id not in eligible_ids


def test_auto_decompose_sweep_does_not_starve_eligible_rows_after_page(kanban_home):
    with kb.connect() as conn:
        for idx in range(1000):
            kb.create_task(
                conn,
                title=f"bridge card {idx}",
                triage=True,
                created_by="linear_bridge",
                idempotency_key=f"linear:starvation-{idx}",
                priority=1,
            )
        intended_id = kb.create_task(
            conn,
            title="eligible rough idea behind first page",
            triage=True,
            priority=0,
        )

    assert decomp.list_auto_decompose_ids(limit=1) == [intended_id]


def test_decomposition_safety_limits_have_safe_config_defaults():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["kanban"]["decomposition_max_children"] == 6
    assert DEFAULT_CONFIG["kanban"]["decomposition_max_depth"] == 1


def test_decomposition_child_block_loop_cannot_recursively_decompose(kanban_home):
    with kb.connect() as conn:
        root_id = kb.create_task(conn, title="root", triage=True)
        child_ids = kb.decompose_triage_task(
            conn,
            root_id,
            root_assignee="orchestrator",
            children=[{"title": "generated child", "assignee": "worker", "parents": []}],
            author="auto-decomposer",
        )
        assert child_ids
        child_id = child_ids[0]
        assert kb.claim_task(conn, child_id, claimer="worker") is not None
        assert kb.block_task(conn, child_id, reason="review hold one", kind="needs_input")
        assert kb.unblock_task(conn, child_id)
        assert kb.claim_task(conn, child_id, claimer="worker") is not None
        assert kb.block_task(conn, child_id, reason="review hold two", kind="needs_input")
        child = kb.get_task(conn, child_id)
        assert child.status == "triage"
        assert child.triage_origin == kb.TRIAGE_ORIGIN_BLOCK_RECURRENCE
        assert child.decomposition_depth == 1

    client = _mock_client_returning(jsonlib.dumps({
        "fanout": True,
        "tasks": [{"title": "forbidden grandchild", "assignee": "worker", "parents": []}],
    }))
    with patch(
        "agent.auxiliary_client.get_text_auxiliary_client",
        return_value=(client, "test-model"),
    ):
        outcome = decomp.decompose_task(child_id, author="auto-decomposer")

    assert outcome.ok is False
    assert "human escalation" in outcome.reason.lower()
    assert client.chat.completions.create.call_count == 0
