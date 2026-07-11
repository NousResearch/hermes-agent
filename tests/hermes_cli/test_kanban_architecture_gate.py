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


def _human_context(*, surface: str = "cli") -> "kb.MutationContext":
    return kb.MutationContext(
        board_key="default",
        principal="human-1",
        actor_type="human",
        session_id="session-1",
        request_scope_id="turn-1",
        surface=surface,
        mode="enforce",
        phase="approval",
    )


def _awaiting_human_approval(conn) -> "kb.ArchitectureGate":
    architect = kb.create_task(
        conn,
        title="Design workflow",
        assignee="architect",
        mutation_context=_architect_context(),
    )
    claimed = kb.claim_task(conn, architect)
    assert claimed is not None and claimed.current_run_id is not None
    handoff = {**_formal_handoff(), "human_approval_required": True}
    assert kb.complete_task(conn, architect, metadata=handoff, expected_run_id=claimed.current_run_id)
    gate = kb.get_architecture_gate_for_task(conn, architect)
    assert gate is not None
    accepted = kb.accept_architecture_handoff(conn, gate.gate_id)
    assert accepted.state == "validated_awaiting_approval"
    return accepted


def test_human_approval_requires_authenticated_exact_digest_and_is_idempotent(kanban_home):
    with kb.connect() as conn:
        gate = _awaiting_human_approval(conn)

        with pytest.raises(kb.ArchitectureGateError, match="approval_requires_human"):
            kb.approve_architecture_gate(conn, gate.gate_id, _implementation_context(), gate.design_digest)
        with pytest.raises(kb.ArchitectureGateError, match="approval_surface_not_authenticated"):
            kb.approve_architecture_gate(
                conn, gate.gate_id, _human_context(surface="model"), gate.design_digest,
            )
        with pytest.raises(kb.ArchitectureGateError, match="approval_digest_mismatch"):
            kb.approve_architecture_gate(conn, gate.gate_id, _human_context(), "forged")

        approved = kb.approve_architecture_gate(conn, gate.gate_id, _human_context(), gate.design_digest)
        replay = kb.approve_architecture_gate(conn, gate.gate_id, _human_context(), gate.design_digest)

        assert approved.state == replay.state == "human_approved"
        assert approved.approved_digest == gate.design_digest
        assert replay.row_version == approved.row_version
        assert approved.approval_actor_id == "human-1"


def test_accepted_edit_invalidation_and_wrong_state_deny_human_approval(kanban_home):
    with kb.connect() as conn:
        gate = _awaiting_human_approval(conn)
        kb.invalidate_architecture_gate(conn, gate.gate_id, reason="accepted_edit")

        with pytest.raises(kb.ArchitectureGateError, match="approval_invalidated"):
            kb.approve_architecture_gate(conn, gate.gate_id, _human_context(), gate.design_digest)


def test_discovery_capability_is_bound_single_use_and_never_allows_protected_work(kanban_home):
    with kb.connect() as conn:
        gate = _awaiting_human_approval(conn)
        capability = kb.issue_discovery_capability(
            conn,
            gate.gate_id,
            _human_context(),
            principal="orchestrator-session",
            session_id="session-1",
            request_scope_id="turn-1",
            profile="scout",
        )
        discovery_context = kb.MutationContext(
            board_key="default",
            principal="orchestrator-session",
            actor_type="orchestrator_agent",
            session_id="session-1",
            request_scope_id="turn-1",
            gate_id=gate.gate_id,
            profile="scout",
            discovery_capability=capability.token,
            mode="enforce",
            phase="discovery",
        )
        discovery = kb.create_task(
            conn, title="Read-only research", assignee="scout", mutation_context=discovery_context,
        )
        assert kb.get_task(conn, discovery).status == "ready"

        with pytest.raises(kb.ArchitectureGateError, match="discovery_capability_used"):
            kb.create_task(
                conn, title="Replay discovery", assignee="scout", mutation_context=discovery_context,
            )

        expired = kb.issue_discovery_capability(
            conn, gate.gate_id, _human_context(), principal="orchestrator-session",
            session_id="session-1", request_scope_id="turn-1", profile="scout",
        )
        conn.execute("UPDATE discovery_capabilities SET expires_at = 0 WHERE token = ?", (expired.token,))
        expired_context = kb.MutationContext(
            **{**discovery_context.__dict__, "discovery_capability": expired.token}
        )
        with pytest.raises(kb.ArchitectureGateError, match="discovery_capability_expired"):
            kb.create_task(conn, title="Expired discovery", assignee="scout", mutation_context=expired_context)

        fresh = kb.issue_discovery_capability(
            conn, gate.gate_id, _human_context(), principal="orchestrator-session",
            session_id="session-1", request_scope_id="turn-1", profile="scout",
        )
        protected = kb.MutationContext(
            **{
                **discovery_context.__dict__,
                "discovery_capability": fresh.token,
                "phase": "implementation",
            }
        )
        with pytest.raises(kb.ArchitectureGateError, match="architecture_gate_open"):
            kb.create_task(conn, title="Forbidden implementation", assignee="coder", mutation_context=protected)


def test_five_card_strava_incident_is_classified_without_any_external_action(kanban_home):
    with kb.connect() as conn:
        cards = [
            kb.create_task(conn, title=title, assignee="coder")
            for title in (
                "Strava ingestion", "Google Drive export", "delivery worker",
                "retry worker", "finalizer",
            )
        ]
        for parent, child in zip(cards, cards[1:]):
            kb.link_tasks(conn, parent, child)
        architect = kb.create_task(
            conn, title="Architect remediation", assignee="architect", mutation_context=_architect_context(),
        )
        kb.link_tasks(conn, architect, cards[0])
        gate = kb.get_architecture_gate_for_task(conn, architect)
        assert gate is not None

        events_before = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        assert {item.task_id for item in kb.classify_policy_quarantine(conn, gate.gate_id)} == set(cards)
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == events_before
        assert kb.apply_policy_quarantine(conn, gate.gate_id) == set(cards)
        for card in cards:
            task = kb.get_task(conn, card)
            assert task is not None and task.policy_quarantined


def test_policy_quarantine_dominates_readiness_claim_dependencies_and_stale_completion(kanban_home):
    with kb.connect() as conn:
        premature = kb.create_task(conn, title="Premature implementation", assignee="coder")
        claimed = kb.claim_task(conn, premature)
        assert claimed is not None and claimed.current_run_id is not None
        descendant = kb.create_task(conn, title="Premature finalizer", assignee="reviewer", parents=[premature])
        architect = kb.create_task(
            conn, title="Design workflow", assignee="architect", mutation_context=_architect_context(),
        )
        kb.link_tasks(conn, architect, premature)

        gate = kb.get_architecture_gate_for_task(conn, architect)
        assert gate is not None
        classified = kb.classify_policy_quarantine(conn, gate.gate_id)
        assert {item.task_id for item in classified} == {premature, descendant}
        assert kb.apply_policy_quarantine(conn, gate.gate_id) == {premature, descendant}

        assert kb.claim_task(conn, descendant) is None
        assert not kb.complete_task(conn, premature, expected_run_id=claimed.current_run_id)
        assert kb.get_task(conn, premature).policy_quarantined
        assert kb.get_task(conn, descendant).policy_quarantined

        dependent = kb.create_task(conn, title="Depends on invalid work", assignee="coder", parents=[premature])
        assert kb.get_task(conn, dependent).status == "todo"
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, dependent).status == "todo"
