"""Focused contracts for the Architecture-First Kanban gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _architect_context() -> "kb.MutationContext":
    return kb.MutationContext(
        board_key="default",
        principal="orchestrator-session",
        actor_type="orchestrator_agent",
        session_id="session-1",
        request_scope_id="turn-1",
        mode="enforce",
        phase="architecture",
    )


def _implementation_context() -> "kb.MutationContext":
    return kb.MutationContext(
        board_key="default",
        principal="orchestrator-session",
        actor_type="orchestrator_agent",
        session_id="session-1",
        request_scope_id="turn-1",
        mode="enforce",
        phase="implementation",
    )


def _formal_handoff() -> dict:
    return {
        "role": "architect",
        "design_depth": "formal",
        "chosen_approach": "Use one transactional gate projection.",
        "alternatives_rejected": ["prompt-only guard"],
        "slices": [{"name": "core", "verification": ["focused test"]}],
        "acceptance_criteria": ["protected work is denied before approval"],
        "verification_plan": ["run focused tests"],
        "human_approval_required": False,
        "rollout": {"mode": "shadow"},
        "rollback": {"mode": "off"},
    }


def test_schema_migrates_architecture_gates_projection(kanban_home):
    with kb.connect() as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(architecture_gates)")
        }

    assert "architecture_gates" in tables
    assert {
        "gate_id",
        "board_key",
        "architect_task_id",
        "state",
        "design_digest",
        "accepted_snapshot",
        "row_version",
    } <= columns


def test_handoff_digest_is_stable_and_domain_separated():
    handoff = _formal_handoff()
    canonical = kb.canonicalize_architecture_handoff(handoff)
    first = kb.architecture_handoff_digest(
        policy_version="v1",
        canonicalization_version="v1",
        trusted_scope={"board_key": "default", "request_scope_id": "turn-1"},
        architect_task_id="t_architect",
        accepted_run_id=7,
        canonical_handoff_json=canonical,
    )
    second = kb.architecture_handoff_digest(
        policy_version="v1",
        canonicalization_version="v1",
        trusted_scope={"request_scope_id": "turn-1", "board_key": "default"},
        architect_task_id="t_architect",
        accepted_run_id=7,
        canonical_handoff_json=canonical,
    )
    different_scope = kb.architecture_handoff_digest(
        policy_version="v1",
        canonicalization_version="v1",
        trusted_scope={"board_key": "default", "request_scope_id": "turn-2"},
        architect_task_id="t_architect",
        accepted_run_id=7,
        canonical_handoff_json=canonical,
    )

    assert first == second
    assert first != different_scope
    with pytest.raises(ValueError, match="unknown top-level"):
        kb.canonicalize_architecture_handoff({**handoff, "forged": True})


def test_enforce_gate_blocks_protected_create_link_and_direct_claim(kanban_home):
    with kb.connect() as conn:
        architect = kb.create_task(
            conn,
            title="Design workflow",
            assignee="architect",
            mutation_context=_architect_context(),
        )
        gate = kb.get_architecture_gate_for_task(conn, architect)

        assert gate is not None
        assert gate.state == "open"
        assert gate.architect_task_id == architect

        with pytest.raises(kb.ArchitectureGateError, match="architecture_gate_open"):
            kb.create_task(
                conn,
                title="Implement workflow",
                assignee="coder",
                parents=[architect],
                mutation_context=_implementation_context(),
            )

        bypass = kb.create_task(conn, title="unsupported direct mutation", assignee="coder")
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (architect, bypass),
        )
        with pytest.raises(kb.ArchitectureGateError, match="architecture_gate_open"):
            kb.link_tasks(
                conn,
                architect,
                bypass,
                mutation_context=_implementation_context(),
            )
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (bypass,))

        assert kb.claim_task(conn, bypass) is None
        assert kb.get_task(conn, bypass).status == "todo"
        assert any(
            event.kind == "claim_blocked"
            and event.payload == {"reason": "architecture_gate_open", "gate_id": gate.gate_id}
            for event in kb.list_events(conn, bypass)
        )


def test_valid_completed_architect_handoff_accepts_exact_snapshot_and_allows_create(kanban_home):
    with kb.connect() as conn:
        architect = kb.create_task(
            conn,
            title="Design workflow",
            assignee="architect",
            mutation_context=_architect_context(),
        )
        claimed = kb.claim_task(conn, architect)
        assert claimed is not None and claimed.current_run_id is not None
        assert kb.complete_task(
            conn,
            architect,
            metadata=_formal_handoff(),
            expected_run_id=claimed.current_run_id,
        )

        gate = kb.get_architecture_gate_for_task(conn, architect)
        accepted = kb.accept_architecture_handoff(conn, gate.gate_id)
        assert accepted.state == "policy_accepted"
        assert accepted.design_digest
        assert accepted.accepted_snapshot == kb.canonicalize_architecture_handoff(_formal_handoff())

        implementation = kb.create_task(
            conn,
            title="Implement workflow",
            assignee="coder",
            parents=[architect],
            mutation_context=_implementation_context(),
        )
        claimed_implementation = kb.claim_task(conn, implementation)
        assert claimed_implementation is not None and claimed_implementation.current_run_id is not None
        assert not kb.complete_task(conn, implementation)
        assert kb.complete_task(
            conn,
            implementation,
            expected_run_id=claimed_implementation.current_run_id,
        )
        completed = kb.get_task(conn, implementation)
        assert completed is not None
        assert completed.status == "done"


def test_architect_invalidation_requires_fresh_acceptance_before_claim(kanban_home):
    with kb.connect() as conn:
        architect = kb.create_task(
            conn,
            title="Design workflow",
            assignee="architect",
            mutation_context=_architect_context(),
        )
        claimed = kb.claim_task(conn, architect)
        assert claimed is not None
        assert kb.complete_task(
            conn,
            architect,
            metadata=_formal_handoff(),
            expected_run_id=claimed.current_run_id,
        )
        gate = kb.get_architecture_gate_for_task(conn, architect)
        kb.accept_architecture_handoff(conn, gate.gate_id)
        kb.invalidate_architecture_gate(conn, gate.gate_id, reason="architect_retry")

        with pytest.raises(kb.ArchitectureGateError, match="architecture_gate_open"):
            kb.create_task(
                conn,
                title="attempt after invalidation",
                assignee="coder",
                mutation_context=_implementation_context(),
            )

        direct = kb.create_task(conn, title="direct child", assignee="coder")
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (architect, direct),
        )
        assert kb.claim_task(conn, direct) is None
        assert kb.get_architecture_gate(conn, gate.gate_id).state == "invalidated"
