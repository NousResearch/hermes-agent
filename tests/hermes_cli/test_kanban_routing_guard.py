"""Behavioral tests for the Sprint 3 Kanban routing guard.

These tests exercise the real create / assign / decompose / dispatch
paths. They do not touch a production Kanban DB or spawn live workers.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp

try:
    from hermes_cli import kanban_routing as routing
except ImportError:  # parent runtime 2019da689 has no controller helper
    routing = None  # type: ignore[assignment]


def _guard_error():
    if routing is None:
        return AssertionError
    return routing.RoutingGuardError


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    authorized = {
        "engineer-grok", "architect-sol", "reviewer", "reviewer-grok",
        "architect-grok", "engineer", "engineer38",
    }
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists",
        lambda name: str(name).strip() in authorized,
    )
    try:
        import hermes_cli.kanban_routing as _routing_mod
    except ImportError:
        _routing_mod = None
    if _routing_mod is not None and hasattr(_routing_mod, "profile_is_available"):
        monkeypatch.setattr(
            "hermes_cli.kanban_routing.profile_is_available",
            lambda name: str(name or "").strip() in authorized,
        )
    kb._INITIALIZED_PATHS.clear()
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


def _routing_cfg(**overrides):
    cfg = {"kanban": routing.default_routing_config()}
    cfg["kanban"].update(overrides.pop("kanban_extra", {}))
    if overrides:
        cfg["kanban"]["routing"].update(overrides)
    return cfg


def _decompose_payload(tasks):
    return jsonlib.dumps(
        {
            "fanout": True,
            "rationale": "test split",
            "tasks": tasks,
        }
    )


def test_critical_implementation_routes_to_engineer_grok(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="fix auth",
            assignee="engineer-grok",
            routing_role="implementation",
        )
        task = kb.get_task(conn, tid)
    assert task.assignee == "engineer-grok"
    assert task.routing_criticality == "critical"
    assert task.routing_role == "implementation"


def test_critical_architecture_routes_to_architect_sol(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="design store",
            assignee="architect-sol",
            routing_role="architecture",
        )
        task = kb.get_task(conn, tid)
    assert task.assignee == "architect-sol"
    assert task.routing_role == "architecture"


def test_critical_review_routes_to_reviewer_or_architect_sol(kanban_home):
    with kb.connect() as conn:
        reviewer = kb.create_task(
            conn, title="review a", assignee="reviewer", routing_role="review",
        )
        architect = kb.create_task(
            conn, title="review b", assignee="architect-sol", routing_role="review",
        )
        assert kb.get_task(conn, reviewer).assignee == "reviewer"
        assert kb.get_task(conn, architect).assignee == "architect-sol"
        with pytest.raises(_guard_error()) as exc:
            kb.create_task(
                conn,
                title="review c",
                assignee="reviewer-grok",
                routing_role="review",
            )
        assert exc.value.code == routing.REASON_DENIED


def test_missing_criticality_is_treated_critical(kanban_home):
    crit, role, _ = routing.resolve_routing_fields(None, None)
    assert crit == "critical"
    assert role == "implementation"
    with kb.connect() as conn:
        with pytest.raises(_guard_error()) as exc:
            kb.create_task(conn, title="mystery", assignee="engineer")
        assert exc.value.code == routing.REASON_DENIED
        with pytest.raises(_guard_error()):
            kb.create_task(conn, title="mystery-38", assignee="engineer38")


def test_engineer_grok_unavailable_blocks_without_downgrade(kanban_home):
    llm_payload = _decompose_payload(
        [
            {
                "title": "implement fix",
                "body": "do it",
                "assignee": "engineer-grok",
                "role": "implementation",
                "parents": [],
            }
        ]
    )
    patches = _patch_list_profiles(["engineer", "engineer38", "fallback"])
    for p in patches:
        p.start()
    try:
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="root", triage=True)
        with _patch_aux_client(llm_payload), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={
                "kanban": {
                    "default_assignee": "fallback",
                    "auto_promote_children": False,
                    "routing": routing.default_routing_config(),
                }
            },
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False
    assert "engineer-grok" in outcome.reason or "ROUTING" in outcome.reason
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        children = kb.list_tasks(conn)
    assert root.status == "triage"
    assert [c.id for c in children if c.id != tid] == []
    assert all(c.assignee != "engineer" for c in children)
    assert all(c.assignee != "engineer38" for c in children)
    assert all(c.assignee != "fallback" for c in children)


def test_engineer38_cannot_receive_critical_implementation(kanban_home):
    with kb.connect() as conn:
        with pytest.raises(_guard_error()) as exc:
            kb.create_task(
                conn,
                title="critical impl",
                assignee="engineer38",
                routing_role="implementation",
            )
        assert exc.value.code == routing.REASON_DENIED
        tid = kb.create_task(
            conn, title="parked", assignee="engineer-grok", routing_role="implementation",
        )
        with pytest.raises(_guard_error()):
            kb.assign_task(conn, tid, "engineer38")
        assert kb.get_task(conn, tid).assignee == "engineer-grok"


def test_engineer_cannot_receive_critical_implementation(kanban_home):
    with kb.connect() as conn:
        with pytest.raises(_guard_error()) as exc:
            kb.create_task(conn, title="critical impl", assignee="engineer")
        assert exc.value.code == routing.REASON_DENIED


def test_explicit_noncritical_may_use_engineer(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="bounded bench",
            assignee="engineer",
            routing_criticality="noncritical",
            routing_role="noncritical",
        )
        task = kb.get_task(conn, tid)
    assert task.assignee == "engineer"
    assert task.routing_criticality == "noncritical"


def test_explicit_noncritical_may_use_engineer38(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="bounded bench 38",
            assignee="engineer38",
            routing_criticality="noncritical",
        )
        task = kb.get_task(conn, tid)
    assert task.assignee == "engineer38"
    assert task.routing_criticality == "noncritical"


def test_decompose_without_auto_promote_stays_todo(kanban_home):
    llm_payload = _decompose_payload(
        [
            {
                "title": "implement",
                "body": "code it",
                "assignee": "engineer-grok",
                "role": "implementation",
                "criticality": "critical",
                "parents": [],
            },
            {
                "title": "review",
                "body": "review it",
                "assignee": "reviewer",
                "role": "review",
                "parents": [0],
            },
        ]
    )
    names = ["engineer-grok", "reviewer", "architect-sol"]
    patches = _patch_list_profiles(names)
    for p in patches:
        p.start()
    try:
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="sprint graph", triage=True)
        with _patch_aux_client(llm_payload), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={
                "kanban": {
                    "auto_promote_children": False,
                    "orchestrator_profile": "architect-sol",
                    "routing": routing.default_routing_config(),
                }
            },
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason
    assert outcome.child_ids and len(outcome.child_ids) == 2
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        children = [kb.get_task(conn, cid) for cid in outcome.child_ids]
    assert root.status == "todo"
    assert all(child.status == "todo" for child in children)
    assert children[0].assignee == "engineer-grok"
    assert children[1].assignee == "reviewer"
    assert children[0].routing_criticality == "critical"
    assert children[0].routing_role == "implementation"
    assert children[1].routing_role == "review"
    assert not any(child.status == "ready" for child in children)


def test_dispatch_before_preflight_does_not_claim_or_spawn(kanban_home):
    llm_payload = _decompose_payload(
        [
            {
                "title": "implement",
                "body": "code it",
                "assignee": "engineer-grok",
                "role": "implementation",
                "parents": [],
            }
        ]
    )
    patches = _patch_list_profiles(["engineer-grok", "architect-sol"])
    for p in patches:
        p.start()
    try:
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="preflight", triage=True)
        with _patch_aux_client(llm_payload), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={
                "kanban": {
                    "auto_promote_children": False,
                    "routing": routing.default_routing_config(),
                }
            },
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason
    spawned = []

    def _spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    with kb.connect() as conn:
        result = kb.dispatch_once(conn, spawn_fn=_spawn)
        child = kb.get_task(conn, outcome.child_ids[0])
    assert result.spawned == []
    assert spawned == []
    assert child.status == "todo"
    assert child.claim_lock is None
    assert child.worker_pid is None


def test_allowed_assignment_can_become_claimable(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="approved impl",
            assignee="engineer-grok",
            routing_role="implementation",
            triage=True,
        )
        assert kb.get_task(conn, tid).status == "triage"
        ok = kb.specify_triage_task(
            conn, tid, assignee="engineer-grok", author="human",
        )
        assert ok is True
        kb.recompute_ready(conn)
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert claimed.assignee == "engineer-grok"
        assert claimed.status == "running"


def test_critical_unknown_assignee_does_not_use_default_assignee(kanban_home):
    llm_payload = _decompose_payload(
        [
            {
                "title": "implement",
                "body": "code it",
                "assignee": "made_up",
                "role": "implementation",
                "parents": [],
            }
        ]
    )
    patches = _patch_list_profiles(["engineer-grok", "fallback", "engineer"])
    for p in patches:
        p.start()
    try:
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="unknown child", triage=True)
        with _patch_aux_client(llm_payload), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={
                "kanban": {
                    "default_assignee": "fallback",
                    "auto_promote_children": False,
                    "routing": routing.default_routing_config(),
                }
            },
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False
    with kb.connect() as conn:
        children = [t for t in kb.list_tasks(conn) if t.id != tid]
    assert children == []


def test_dispatcher_cannot_stamp_denied_profiles_on_critical_ready(kanban_home):
    spawned = []

    def _spawn(task, workspace, board=None):
        spawned.append((task.id, task.assignee))
        return 99

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="unassigned critical",
            assignee=None,
            routing_role="implementation",
        )
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.get_task(conn, tid).assignee is None
        result = kb.dispatch_once(
            conn,
            spawn_fn=_spawn,
            default_assignee="engineer",
        )
        task = kb.get_task(conn, tid)
    assert tid in result.skipped_unassigned or getattr(
        result, "skipped_routing_denied", [],
    )
    assert task.assignee is None
    assert task.claim_lock is None
    assert task.worker_pid is None
    assert spawned == []

    with kb.connect() as conn:
        tid38 = kb.create_task(
            conn,
            title="unassigned critical 38",
            assignee=None,
        )
        result38 = kb.dispatch_once(
            conn,
            spawn_fn=_spawn,
            default_assignee="engineer38",
        )
        task38 = kb.get_task(conn, tid38)
    assert task38.assignee is None
    assert result38.spawned == []


def test_parent_ordering_and_completion_integrity_unchanged(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    import subprocess

    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "T"], check=True)
    (repo / "README.md").write_text("init\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    from hermes_cli import kanban_completion_integrity as kci

    contract = {
        "schema_version": 1,
        "type": "git_revision",
        "repository": str(repo),
        "worktree": str(repo),
        "git_common_dir": kci.resolve_git_common_dir(str(repo)),
    }
    with kb.connect() as conn:
        parent = kb.create_task(
            conn,
            title="parent",
            assignee="implementer",
            completion_contract=contract,
            routing_criticality="noncritical",
            routing_role="noncritical",
        )
        child = kb.create_task(
            conn,
            title="child",
            assignee="implementer",
            parents=[parent],
            routing_criticality="noncritical",
            routing_role="noncritical",
        )
        assert kb.get_task(conn, child).status == "todo"
        claimed = kb.claim_task(conn, parent)
        assert claimed is not None
        assert kb.complete_task(
            conn,
            parent,
            summary="done",
            metadata={
                "terminal_result": {
                    "attempt_id": claimed.attempt_id,
                    "repository": str(repo),
                    "worktree": str(repo),
                    "commit_sha": sha,
                }
            },
        )
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_second_opinion_reviewer_grok_requires_explicit_flag(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="second look",
            assignee="reviewer-grok",
            routing_role="review",
            routing_second_opinion=True,
        )
        assert kb.get_task(conn, tid).assignee == "reviewer-grok"
        assert kb.get_task(conn, tid).routing_second_opinion is True


def _plant_review_task(
    conn,
    *,
    assignee: str | None,
    routing_role: str = "review",
    routing_criticality: str | None = None,
    routing_second_opinion: bool = False,
    routing_preflight: int = 0,
    title: str = "critical review",
) -> str:
    tid = kb.create_task(
        conn,
        title=title,
        assignee="reviewer",
        routing_role=routing_role,
        routing_criticality=routing_criticality,
        routing_second_opinion=routing_second_opinion,
    )
    conn.execute(
        "UPDATE tasks SET status = 'review', assignee = ?, "
        "routing_preflight = ? WHERE id = ?",
        (assignee, routing_preflight, tid),
    )
    return tid


def _plant_ready_task(
    conn,
    *,
    assignee: str | None,
    routing_role: str = "implementation",
    routing_criticality: str | None = None,
    routing_preflight: int = 0,
    title: str = "critical impl",
) -> str:
    tid = kb.create_task(
        conn,
        title=title,
        assignee=assignee or "engineer-grok",
        routing_role=routing_role,
        routing_criticality=routing_criticality,
    )
    conn.execute(
        "UPDATE tasks SET status = 'ready', assignee = ?, "
        "routing_preflight = ? WHERE id = ?",
        (assignee, routing_preflight, tid),
    )
    return tid


def test_create_critical_implementation_rejects_made_up_profile(kanban_home):
    with kb.connect() as conn:
        with pytest.raises(_guard_error()) as exc:
            kb.create_task(
                conn,
                title="critical impl",
                assignee="made-up-impl",
                routing_role="implementation",
            )
        assert exc.value.code in {
            routing.REASON_UNKNOWN,
            routing.REASON_UNAVAILABLE,
            routing.REASON_DENIED,
        }


def test_assign_critical_implementation_rejects_made_up_profile(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="critical impl",
            assignee="engineer-grok",
            routing_role="implementation",
        )
        with pytest.raises(_guard_error()) as exc:
            kb.assign_task(conn, tid, "made-up-impl")
        assert kb.get_task(conn, tid).assignee == "engineer-grok"
        assert exc.value.code in {
            routing.REASON_UNKNOWN,
            routing.REASON_UNAVAILABLE,
            routing.REASON_DENIED,
        }


def test_create_critical_implementation_rejects_unavailable_engineer_grok(kanban_home, monkeypatch):
    if routing is not None and hasattr(routing, "profile_is_available"):
        monkeypatch.setattr(
            "hermes_cli.kanban_routing.profile_is_available",
            lambda name: str(name or "").strip() != "engineer-grok",
        )
    else:
        monkeypatch.setattr(
            "hermes_cli.profiles.profile_exists",
            lambda name: str(name or "").strip() != "engineer-grok",
        )
    with kb.connect() as conn:
        with pytest.raises(_guard_error()) as exc:
            kb.create_task(
                conn,
                title="critical impl",
                assignee="engineer-grok",
                routing_role="implementation",
            )
        assert exc.value.code == routing.REASON_UNAVAILABLE


def test_unassigned_critical_ready_claim_rejected(kanban_home):
    with kb.connect() as conn:
        tid = _plant_ready_task(conn, assignee=None)
        claimed = kb.claim_task(conn, tid)
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
    assert claimed is None
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.worker_pid is None
    assert any(
        ev.kind == "claim_rejected"
        and (ev.payload or {}).get("reason") in {
            routing.REASON_EMPTY,
            routing.REASON_DENIED,
            routing.REASON_UNAVAILABLE,
        }
        for ev in events
    )


def test_unavailable_critical_assignee_claim_rejected(kanban_home, monkeypatch):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="critical impl",
            assignee="engineer-grok",
            routing_role="implementation",
        )
    if routing is not None and hasattr(routing, "profile_is_available"):
        monkeypatch.setattr(
            "hermes_cli.kanban_routing.profile_is_available",
            lambda name: str(name or "").strip() != "engineer-grok",
        )
    else:
        monkeypatch.setattr(
            "hermes_cli.profiles.profile_exists",
            lambda name: str(name or "").strip() != "engineer-grok",
        )
    with kb.connect() as conn:
        claimed = kb.claim_task(conn, tid)
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
    assert claimed is None
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.worker_pid is None
    assert any(
        ev.kind == "claim_rejected"
        and (ev.payload or {}).get("reason") == routing.REASON_UNAVAILABLE
        for ev in events
    )


def test_critical_review_made_up_claim_review_rejected(kanban_home):
    with kb.connect() as conn:
        tid = _plant_review_task(conn, assignee="made-up-reviewer")
        claimed = kb.claim_review_task(conn, tid)
        task = kb.get_task(conn, tid)
    assert claimed is None
    assert task.status == "review"
    assert task.claim_lock is None
    assert task.worker_pid is None


def test_critical_review_engineer_claim_review_rejected(kanban_home):
    with kb.connect() as conn:
        tid = _plant_review_task(conn, assignee="engineer")
        claimed = kb.claim_review_task(conn, tid)
        task = kb.get_task(conn, tid)
    assert claimed is None
    assert task.status == "review"
    assert task.claim_lock is None


def test_critical_review_reviewer_claim_allowed(kanban_home):
    with kb.connect() as conn:
        tid = _plant_review_task(conn, assignee="reviewer")
        claimed = kb.claim_review_task(conn, tid)
        task = kb.get_task(conn, tid)
    assert claimed is not None
    assert task.status == "running"
    assert task.assignee == "reviewer"


def test_critical_review_architect_sol_claim_allowed(kanban_home):
    with kb.connect() as conn:
        tid = _plant_review_task(conn, assignee="architect-sol")
        claimed = kb.claim_review_task(conn, tid)
    assert claimed is not None
    assert claimed.assignee == "architect-sol"
    assert claimed.status == "running"


def test_critical_review_unauthorized_installed_profile_review_dispatch_blocked(kanban_home):
    spawned: list[str] = []

    def _spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    with kb.connect() as conn:
        tid = _plant_review_task(conn, assignee="engineer")
        result = kb.dispatch_once(conn, spawn_fn=_spawn)
        task = kb.get_task(conn, tid)
    assert result.spawned == []
    assert spawned == []
    assert task.status == "review"
    assert task.claim_lock is None
    assert task.worker_pid is None
    assert tid in getattr(result, "skipped_routing_denied", [])


def test_preflight_task_promote_rejected(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="parked impl",
            assignee="engineer-grok",
            routing_role="implementation",
            triage=True,
        )
        conn.execute("UPDATE tasks SET status = 'todo', routing_preflight = 1 WHERE id = ?", (tid,))
        ok, reason = kb.promote_task(conn, tid, actor="human")
        task = kb.get_task(conn, tid)
    assert ok is False
    assert reason is not None
    assert "preflight" in reason.lower() or "ROUTING" in reason
    assert task.status == "todo"
    assert task.routing_preflight is True


def test_preflight_task_ordinary_claim_rejected(kanban_home):
    with kb.connect() as conn:
        tid = _plant_ready_task(
            conn,
            assignee="engineer-grok",
            routing_preflight=1,
        )
        claimed = kb.claim_task(conn, tid)
        task = kb.get_task(conn, tid)
    assert claimed is None
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.worker_pid is None


def test_preflight_review_claim_rejected(kanban_home):
    with kb.connect() as conn:
        tid = _plant_review_task(
            conn,
            assignee="reviewer",
            routing_preflight=1,
        )
        claimed = kb.claim_review_task(conn, tid)
        task = kb.get_task(conn, tid)
    assert claimed is None
    assert task.status == "review"
    assert task.claim_lock is None
    assert task.worker_pid is None


def test_explicit_human_routing_approval_clears_preflight_and_allows_execution(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="parked impl",
            assignee="engineer-grok",
            routing_role="implementation",
            triage=True,
        )
        conn.execute("UPDATE tasks SET status = 'todo', routing_preflight = 1 WHERE id = ?", (tid,))
        assert kb.get_task(conn, tid).routing_preflight is True
        ok = kb.assign_task(conn, tid, "engineer-grok")
        assert ok is True
        task = kb.get_task(conn, tid)
        assert task.routing_preflight is False
        promoted, reason = kb.promote_task(conn, tid, actor="human")
        assert promoted is True, reason
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert claimed.status == "running"
        assert claimed.assignee == "engineer-grok"
