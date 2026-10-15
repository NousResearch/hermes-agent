"""Level 5B-P0 Governance Regression Test Suite (TDD RED Baseline).

Enforces 6 non-negotiable invariants (R1 to R6):
- R1_policy_missing_fail_closed: Governed project with missing policy -> DENIED.
- R2_policy_malformed_fail_closed: Malformed YAML policy -> DENIED.
- R3_concurrent_budget_atomicity: Limit 1 + 2 concurrent requests -> exactly 1 success, 1 denial, final count 1.
- R4_decomposition_cannot_bypass_budget: Decomposition cannot bypass budget -> DENIED, 0 child tasks created.
- R5_agent_cannot_self_authorize: Self-asserted actor_class=HUMAN -> DENIED.
- R6_missing_required_business_behavior_blocks_acceptance: Missing signature verification -> NOT_COMPLETE (exit code != 0).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import pytest

REPO = Path(__file__).resolve().parents[2]

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_governance as gov
from hermes_cli import kanban_acceptance as ac
from hermes_cli.kanban_db_graph import decompose_triage_task


def make_test_human_context(
    principal_id: str = "human_admin",
    action: str = "*",
    target: str = "*",
) -> gov.TrustedPrincipalContext:
    token = gov.issue_human_proof_token(principal_id=principal_id, action=action, target=target)
    return gov.TrustedPrincipalContext(
        principal_id=principal_id,
        principal_type="human",
        auth_method="TRUSTED_CONTROL_SURFACE",
        proof_token=token,
    )



def test_r1_policy_missing_fail_closed(tmp_path, monkeypatch):
    """R1: When governance policy is missing on a governed project, graph mutation must be DENIED."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-r1"
    kb.create_board(board_slug, name="Governed Board R1")
    b_dir = kb.board_dir(board_slug)

    missing_policy = b_dir / "governance_policy.yaml"
    monkeypatch.setenv("HERMES_KANBAN_GOVERNANCE_POLICY", str(missing_policy))
    monkeypatch.setenv("HERMES_KANBAN_GOVERNANCE_REQUIRED", "1")

    with kbc.connect_closing(board=board_slug) as conn:
        with pytest.raises((gov.GovernanceBudgetExceeded, RuntimeError, FileNotFoundError)):
            kb.create_task(conn, title="R1 Task", assignee="builder", board=board_slug)
        count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count == 0, f"Graph mutation allowed despite missing policy! Tasks count: {count}"


def test_r2_policy_malformed_fail_closed(tmp_path, monkeypatch):
    """R2: When governance YAML is malformed, graph mutation must be DENIED (fail-closed)."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-r2"
    kb.create_board(board_slug, name="Governed Board R2")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text("budget: [\n", encoding="utf-8")

    with kbc.connect_closing(board=board_slug) as conn:
        with pytest.raises((gov.GovernanceBudgetExceeded, RuntimeError, ValueError)):
            kb.create_task(conn, title="R2 Task", assignee="builder", board=board_slug)
        count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count == 0, f"Graph mutation allowed despite malformed policy! Tasks count: {count}"


def test_r3_concurrent_budget_atomicity(tmp_path, monkeypatch):
    """R3: Given max_tasks: 1, 2 concurrent create requests must result in exactly 1 success, 1 denial, final_task_count: 1."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-r3"
    kb.create_board(board_slug, name="Governed Board R3")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text(
        "policy_id: r3-test\nimmutable_for_run: true\nbudget:\n  max_total_tasks: 1\n",
        encoding="utf-8",
    )
    db_path = b_dir / "kanban.db"

    original_check = gov.check_governance_mutation
    barrier = threading.Barrier(2)

    def scheduled_check(*args, **kwargs):
        original_check(*args, **kwargs)
        try:
            barrier.wait(timeout=5)
        except Exception:
            pass

    monkeypatch.setattr(gov, "check_governance_mutation", scheduled_check)

    def do_create(idx):
        local_conn = kbc.connect(db_path)
        try:
            kb.create_task(local_conn, title=f"Writer-{idx}", assignee="builder", board=board_slug)
            return True
        except Exception:
            return False
        finally:
            local_conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(do_create, [0, 1]))

    with kbc.connect_closing(board=board_slug) as conn:
        task_count = conn.execute("SELECT COUNT(*) FROM tasks WHERE status != 'archived'").fetchone()[0]

    assert task_count == 1, f"Budget overrun! Expected 1 task, found {task_count}"
    assert results.count(True) == 1, f"Expected exactly 1 successful create, got {results.count(True)}"
    assert results.count(False) == 1, f"Expected exactly 1 denial, got {results.count(False)}"


def test_r4_decomposition_cannot_bypass_budget(tmp_path, monkeypatch):
    """R4: Given remaining_capacity: 1, decomposition requiring 3 tasks must be DENIED (0 created, 0 linked)."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-r4"
    kb.create_board(board_slug, name="Governed Board R4")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text(
        "policy_id: r4-test\nimmutable_for_run: true\nbudget:\n  max_total_tasks: 1\n",
        encoding="utf-8",
    )

    with kbc.connect_closing(board=board_slug) as conn:
        root_id = kb.create_task(conn, title="Root Triage", assignee="p0-orchestrator", triage=True, board=board_slug)
        assert root_id is not None

        with pytest.raises(gov.GovernanceBudgetExceeded):
            children = decompose_triage_task(
                conn,
                root_id,
                root_assignee="p0-orchestrator",
                children=[
                    {"title": "Child-1", "assignee": "builder"},
                    {"title": "Child-2", "assignee": "builder"},
                ],
                author="test",
                auto_promote=False,
            )
            assert children is None, "Decomposition should not return children when over budget"

        total_tasks = conn.execute("SELECT COUNT(*) FROM tasks WHERE status != 'archived'").fetchone()[0]
        assert total_tasks == 1, f"Decomposition bypassed budget! Expected 1 task, found {total_tasks}"


def test_r5_agent_cannot_self_authorize(tmp_path, monkeypatch):
    """R5: Caller self-asserting actor_class=HUMAN without cryptographic/authenticated human proof must be DENIED."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-r5"
    kb.create_board(board_slug, name="Governed Board R5")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text(
        "policy_id: r5-test\nimmutable_for_run: true\nbudget:\n  max_total_tasks: 1\n",
        encoding="utf-8",
    )

    with kbc.connect_closing(board=board_slug) as conn:
        with pytest.raises((PermissionError, RuntimeError)):
            eid = gov.record_reauthorization_event(
                conn,
                previous_policy={"budget": {"max_total_tasks": 1}},
                new_policy={"budget": {"max_total_tasks": 2}},
                actor="agent-spoofing-human",
                actor_class="HUMAN",
                reason="agent self-granted quota",
            )
            row = conn.execute("SELECT decision FROM governance_events WHERE event_id = ?", (eid,)).fetchone()
            assert row is None or row[0] == "DENY", f"Self-authorization succeeded with decision: {row[0] if row else None}"


def test_r6_missing_required_business_behavior_blocks_acceptance(tmp_path, monkeypatch):
    """R6: Even when all tasks are done and agents self-declare PASS, missing required behavior blocks project acceptance."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    # 1. Kanban board: all tasks marked DONE with self-declared pass summaries
    board_slug = "r6-board"
    kb.create_board(board_slug, name="R6 Board")
    with kbc.connect_closing(board=board_slug) as conn:
        t_p0 = kb.create_task(conn, title="P0 Planning", assignee="p0-orchestrator", board=board_slug)
        t_b = kb.create_task(conn, title="Builder Task", assignee="builder", board=board_slug)
        t_v = kb.create_task(conn, title="Verifier Task", assignee="verifier", board=board_slug)
        # Mark all tasks done with self-declared success
        for tid, summary in [(t_p0, "P0: Goal achieved"), (t_b, "Builder: 32 tests passing"), (t_v, "Verifier: Independent review passed")]:
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (tid,))
            kb.add_comment(conn, tid, author="audit", body=summary)
        done_count = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'done'").fetchone()[0]
        assert done_count == 3

    # 2. Workspace setup with repository-local intentionally permissive validator fixture
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    release_script = Path(__file__).parent / "fixtures" / "release_decision_no_signature_check.py"
    assert release_script.is_file(), f"Missing repository fixture: {release_script}"

    # Copy the repository fixture into the isolated project workspace.
    (workspace_dir / "release_decision.py").write_bytes(release_script.read_bytes())

    manifest_data = {
        "release_id": "r6-audit",
        "cases": [
            {
                "case_id": "c1",
                "historical_verdict": "PASS",
                "assertions": ["PASS"],
            }
        ],
    }
    manifest_file = workspace_dir / "unsigned_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    # 3. Freeze P0_SPEC requiring signature validation fail closed (exit code 1)
    spec_dict = {
        "p0_spec_version": 1,
        "project": {"project_id": "r6-project", "spec_id": "r6-spec"},
        "scope": {"source_manifest": ["release_decision.py"]},
        "acceptance_criteria": [
            {
                "id": "AC-SIGNATURE-VALIDATION",
                "statement": "Missing signature must fail closed with exit code 1",
                "required": True,
                "verification": {
                    "type": "command",
                    "argv": [sys.executable, "-B", "release_decision.py", "unsigned_manifest.json"],
                    "expected": {
                        "exit_code": 1,
                        "stdout_contains": ["MISSING_SIGNATURE"],
                    },
                },
            }
        ],
        "completion_policy": {"required_criteria": "ALL_PASS"},
    }
    frozen_spec = ac.freeze_acceptance_spec(workspace_dir, spec_dict)

    # 4. Acceptance Runner executes the frozen verification command
    receipts = ac.execute_acceptance_run(workspace_dir, frozen_spec)
    assert len(receipts) == 1
    # Receipt captures reality: release_decision.py returned exit code 0 instead of 1
    assert receipts[0].predicate_satisfied is False
    assert receipts[0].exit_code == 0

    # 5. Acceptance Authority computes derived status
    acceptance = ac.derive_project_acceptance(workspace_dir, board_slug=board_slug)

    # All tasks are DONE, but project acceptance is strictly NOT_COMPLETE / FAILED_ACCEPTANCE
    assert acceptance.status != "PASS", "Self-declared task completion falsely produced project PASS!"
    assert acceptance.status in {"FAILED_ACCEPTANCE", "NOT_COMPLETE"}
    assert acceptance.criteria_results["AC-SIGNATURE-VALIDATION"]["status"] == "FAILED"
    assert acceptance.required_passed == 0
    assert acceptance.required_total == 1


def test_e1_non_governed_project(tmp_path, monkeypatch):
    """E1: Non-governed project with missing policy should ALLOW normal task creation."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()
    for var in ("HERMES_KANBAN_GOVERNANCE_REQUIRED", "HERMES_KANBAN_GOVERNANCE_POLICY", "HERMES_KANBAN_GOVERNANCE_POLICY_HASH"):
        monkeypatch.delenv(var, raising=False)

    board_slug = "plain-board-e1"
    kb.create_board(board_slug, name="Plain Board E1")

    with kbc.connect_closing(board=board_slug) as conn:
        tid = kb.create_task(conn, title="E1 Normal Task", assignee="builder", board=board_slug)
        assert tid is not None
        count = conn.execute("SELECT COUNT(*) FROM tasks WHERE status != 'archived'").fetchone()[0]
        assert count == 1


def test_e2_governed_policy_hash_mismatch(tmp_path, monkeypatch):
    """E2: Governed project with policy hash mismatch must DENY graph mutation."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-e2"
    kb.create_board(board_slug, name="Governed Board E2")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text("policy_id: e2\nbudget:\n  max_total_tasks: 2\n", encoding="utf-8")

    # Set expected hash to an incorrect value
    monkeypatch.setenv("HERMES_KANBAN_GOVERNANCE_POLICY_HASH", "0000000000000000000000000000000000000000000000000000000000000000")

    with kbc.connect_closing(board=board_slug) as conn:
        with pytest.raises(gov.GovernancePolicyHashMismatch):
            kb.create_task(conn, title="E2 Task", assignee="builder", board=board_slug)
        count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count == 0, f"Graph mutation allowed despite hash mismatch! Tasks count: {count}"


def test_e3_governed_unsupported_policy_version(tmp_path, monkeypatch):
    """E3: Governed project with unsupported policy version must DENY graph mutation."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()
    monkeypatch.delenv("HERMES_KANBAN_GOVERNANCE_POLICY_HASH", raising=False)

    board_slug = "governed-e3"
    kb.create_board(board_slug, name="Governed Board E3")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text("policy_id: e3\nversion: 999\nbudget:\n  max_total_tasks: 2\n", encoding="utf-8")

    with kbc.connect_closing(board=board_slug) as conn:
        with pytest.raises(gov.GovernancePolicyUnsupportedVersion):
            kb.create_task(conn, title="E3 Task", assignee="builder", board=board_slug)
        count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count == 0, f"Graph mutation allowed despite unsupported version! Tasks count: {count}"


def test_e4_concurrent_budget_atomicity_8_callers(tmp_path, monkeypatch):
    """E4: 8 concurrent callers competing for max_total_tasks: 1 -> exactly 1 succeeds, 7 denied, 0 overrun."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-e4"
    kb.create_board(board_slug, name="Governed Board E4")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text(
        "policy_id: e4-test\nimmutable_for_run: true\nbudget:\n  max_total_tasks: 1\n",
        encoding="utf-8",
    )
    db_path = b_dir / "kanban.db"

    def do_create(idx):
        local_conn = kbc.connect(db_path)
        try:
            kb.create_task(local_conn, title=f"Caller-{idx}", assignee="builder", board=board_slug)
            return True
        except Exception:
            return False
        finally:
            local_conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(do_create, range(8)))

    with kbc.connect_closing(board=board_slug) as conn:
        task_count = conn.execute("SELECT COUNT(*) FROM tasks WHERE status != 'archived'").fetchone()[0]

    assert task_count == 1, f"Budget overrun! Expected 1 task, found {task_count}"
    assert results.count(True) == 1, f"Expected exactly 1 success, got {results.count(True)}"
    assert results.count(False) == 7, f"Expected exactly 7 denials, got {results.count(False)}"


def test_e5_batch_wave_contention_all_or_nothing(tmp_path, monkeypatch):
    """E5: Remaining quota 3. Wave A requests 3 tasks; Wave B requests 2 tasks.
    Concurrently submitted. Exactly one whole wave wins (either 3 or 2 tasks in DB),
    the other is completely denied. Never 5, never 4, never a partial wave.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-e5"
    kb.create_board(board_slug, name="Governed Board E5")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text(
        "policy_id: e5-test\nimmutable_for_run: true\nbudget:\n  max_total_tasks: 3\n",
        encoding="utf-8",
    )
    db_path = b_dir / "kanban.db"

    wave_a = {
        "reason": "wave_a",
        "create_tasks": [
            {"temp_id": "a1", "title": "Wave A Builder 1", "assignee": "builder"},
            {"temp_id": "a2", "title": "Wave A Verifier 1", "assignee": "verifier"},
            {"temp_id": "a3", "title": "Wave A Checkpoint", "assignee": "p0-orchestrator"},
        ],
        "create_edges": [
            {"from": "a1", "to": "a2"},
            {"from": "a2", "to": "a3"},
        ],
    }

    wave_b = {
        "reason": "wave_b",
        "create_tasks": [
            {"temp_id": "b1", "title": "Wave B Builder 1", "assignee": "builder"},
            {"temp_id": "b2", "title": "Wave B Verifier 1", "assignee": "verifier"},
        ],
        "create_edges": [
            {"from": "b1", "to": "b2"},
        ],
    }

    def submit_wave(plan):
        local_conn = kbc.connect(db_path)
        try:
            res = kb.commit_graph_mutation(local_conn, board_id=board_slug, mutation_plan=plan)
            return True, len(res.created_task_ids)
        except Exception:
            return False, 0
        finally:
            local_conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(submit_wave, wave_a)
        fut_b = pool.submit(submit_wave, wave_b)
        res_a = fut_a.result()
        res_b = fut_b.result()

    with kbc.connect_closing(board=board_slug) as conn:
        total_tasks = conn.execute("SELECT COUNT(*) FROM tasks WHERE status != 'archived'").fetchone()[0]

    # Exactly one succeeded, one failed
    assert (res_a[0] and not res_b[0]) or (res_b[0] and not res_a[0]), f"Unexpected wave outcomes: A={res_a}, B={res_b}"
    if res_a[0]:
        assert total_tasks == 3, f"Expected whole Wave A (3 tasks), found {total_tasks}"
    else:
        assert total_tasks == 2, f"Expected whole Wave B (2 tasks), found {total_tasks}"


def test_e6_budget_archiving_does_not_restore_capacity(tmp_path, monkeypatch):
    """E6: Capacity consumed once remains consumed. Archiving a task does not restore quota."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-e6"
    kb.create_board(board_slug, name="Governed Board E6")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text(
        "policy_id: e6-test\nimmutable_for_run: true\nbudget:\n  max_total_tasks: 1\n",
        encoding="utf-8",
    )

    with kbc.connect_closing(board=board_slug) as conn:
        t1 = kb.create_task(conn, title="Initial Task", assignee="builder", board=board_slug)
        assert t1 is not None

        # Archive task 1
        conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (t1,))
        archived_count = conn.execute("SELECT COUNT(*) FROM tasks WHERE status != 'archived'").fetchone()[0]
        assert archived_count == 0

        # Attempting to create another task must be DENIED because cumulative budget is exhausted
        with pytest.raises(gov.GovernanceBudgetExceeded):
            kb.create_task(conn, title="Budget Laundering Attempt", assignee="builder", board=board_slug)


def test_e7_agent_cannot_invoke_grant_surface(tmp_path, monkeypatch):
    """E7: Agent principal attempting to invoke grant surface must be DENIED (PermissionError)."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    agent_ctx = gov.TrustedPrincipalContext(
        principal_id="p0-orchestrator",
        principal_type="agent",
        auth_method="agent_session",
    )
    with pytest.raises(PermissionError) as exc_info:
        gov.create_reauthorization_grant(
            board_slug="governed-e7",
            previous_policy_hash="dummy_hash",
            authorized_delta=3,
            principal_context=agent_ctx,
        )
    assert "unauthorized" in str(exc_info.value).lower()


def test_e8_forged_authorization_proof(tmp_path, monkeypatch):
    """E8: Forged authorization proof signature must be DENIED."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-e8"
    kb.create_board(board_slug, name="Governed Board E8")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text("policy_id: e8\nbudget:\n  max_total_tasks: 1\n", encoding="utf-8")
    expected_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()

    human_ctx = make_test_human_context(
        principal_id="human_admin",
        action="*",
        target="*",
    )
    grant = gov.create_reauthorization_grant(
        board_slug=board_slug,
        previous_policy_hash=expected_hash,
        authorized_delta=2,
        principal_context=human_ctx,
    )
    tampered_grant = dataclasses.replace(grant, signature_proof="forged_signature_hex_1234567890")

    with kbc.connect_closing(board=board_slug) as conn:
        with pytest.raises(PermissionError) as exc_info:
            gov.apply_reauthorization_grant(
                conn,
                board_slug=board_slug,
                grant=tampered_grant,
                principal_context=human_ctx,
            )
        assert "proof" in str(exc_info.value).lower() or "signature" in str(exc_info.value).lower()


def test_e9_reauthorization_replay(tmp_path, monkeypatch):
    """E9: Reauthorization grant replay protection: same grant used twice must be DENIED on 2nd attempt; budget incremented once."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-e9"
    kb.create_board(board_slug, name="Governed Board E9")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text("policy_id: e9\nbudget:\n  max_total_tasks: 1\n", encoding="utf-8")
    expected_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()

    human_ctx = make_test_human_context(
        principal_id="human_admin",
        action="*",
        target="*",
    )
    grant = gov.create_reauthorization_grant(
        board_slug=board_slug,
        previous_policy_hash=expected_hash,
        authorized_delta=2,
        principal_context=human_ctx,
    )

    with kbc.connect_closing(board=board_slug) as conn:
        new_pol, _ = gov.apply_reauthorization_grant(conn, board_slug=board_slug, grant=grant, principal_context=human_ctx)
        assert new_pol["budget"]["max_total_tasks"] == 3

        with pytest.raises(PermissionError) as exc_info:
            gov.apply_reauthorization_grant(conn, board_slug=board_slug, grant=grant, principal_context=human_ctx)
        assert "replay" in str(exc_info.value).lower()

        loaded_pol = gov.load_governance_policy(conn, board_slug=board_slug)
        assert loaded_pol is not None
        pol, _, _ = loaded_pol
        assert pol["budget"]["max_total_tasks"] == 3


def test_e10_scope_escalation(tmp_path, monkeypatch):
    """E10: Scope escalation: grant authorizes +3 tasks, caller attempts +4 tasks -> DENIED."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-e10"
    kb.create_board(board_slug, name="Governed Board E10")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text("policy_id: e10\nbudget:\n  max_total_tasks: 1\n", encoding="utf-8")
    expected_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()

    human_ctx = make_test_human_context(
        principal_id="human_admin",
        action="*",
        target="*",
    )
    grant = gov.create_reauthorization_grant(
        board_slug=board_slug,
        previous_policy_hash=expected_hash,
        authorized_delta=3,
        principal_context=human_ctx,
    )

    with kbc.connect_closing(board=board_slug) as conn:
        with pytest.raises(PermissionError) as exc_info:
            gov.apply_reauthorization_grant(
                conn,
                board_slug=board_slug,
                grant=grant,
                principal_context=human_ctx,
                attempted_delta=4,
            )
        assert "scope" in str(exc_info.value).lower() or "exceeds" in str(exc_info.value).lower()


def test_e11_cross_project_reuse(tmp_path, monkeypatch):
    """E11: Cross project reuse: grant issued for Project A attempted on Project B -> DENIED."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_a = "governed-project-a"
    board_b = "governed-project-b"
    kb.create_board(board_a, name="Governed Project A")
    kb.create_board(board_b, name="Governed Project B")

    b_a_dir = kb.board_dir(board_a)
    b_b_dir = kb.board_dir(board_b)
    policy_a = b_a_dir / "governance_policy.yaml"
    policy_b = b_b_dir / "governance_policy.yaml"
    policy_a.write_text("policy_id: a\nbudget:\n  max_total_tasks: 1\n", encoding="utf-8")
    policy_b.write_text("policy_id: b\nbudget:\n  max_total_tasks: 1\n", encoding="utf-8")
    hash_a = hashlib.sha256(policy_a.read_bytes()).hexdigest()

    human_ctx = make_test_human_context(
        principal_id="human_admin",
        action="*",
        target="*",
    )
    grant_for_a = gov.create_reauthorization_grant(
        board_slug=board_a,
        previous_policy_hash=hash_a,
        authorized_delta=2,
        principal_context=human_ctx,
    )

    with kbc.connect_closing(board=board_b) as conn:
        with pytest.raises(PermissionError) as exc_info:
            gov.apply_reauthorization_grant(
                conn,
                board_slug=board_b,
                grant=grant_for_a,
                principal_context=human_ctx,
            )
        assert "scope" in str(exc_info.value).lower() or "board" in str(exc_info.value).lower()


def test_e12_stale_policy_binding(tmp_path, monkeypatch):
    """E12: Stale policy binding: grant previous_policy_hash != current active policy hash (CAS check) -> DENIED."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-e12"
    kb.create_board(board_slug, name="Governed Board E12")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text("policy_id: e12\nbudget:\n  max_total_tasks: 1\n", encoding="utf-8")

    human_ctx = make_test_human_context(
        principal_id="human_admin",
        action="*",
        target="*",
    )
    grant = gov.create_reauthorization_grant(
        board_slug=board_slug,
        previous_policy_hash="outdated_stale_hash_1111",
        authorized_delta=2,
        principal_context=human_ctx,
    )

    with kbc.connect_closing(board=board_slug) as conn:
        with pytest.raises(PermissionError) as exc_info:
            gov.apply_reauthorization_grant(
                conn,
                board_slug=board_slug,
                grant=grant,
                principal_context=human_ctx,
            )
        assert "stale" in str(exc_info.value).lower() or "match" in str(exc_info.value).lower()


def test_e13_valid_human_grant(tmp_path, monkeypatch):
    """E13: Valid human grant: trusted human principal, exact scope match -> policy v2 created, durable event recorded."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-e13"
    kb.create_board(board_slug, name="Governed Board E13")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text("policy_id: e13\nbudget:\n  max_total_tasks: 1\n", encoding="utf-8")
    expected_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()

    human_ctx = make_test_human_context(
        principal_id="human_admin",
        action="*",
        target="*",
    )
    grant = gov.create_reauthorization_grant(
        board_slug=board_slug,
        previous_policy_hash=expected_hash,
        authorized_delta=3,
        principal_context=human_ctx,
    )

    with kbc.connect_closing(board=board_slug) as conn:
        new_policy, new_hash = gov.apply_reauthorization_grant(
            conn,
            board_slug=board_slug,
            grant=grant,
            principal_context=human_ctx,
        )
        assert new_policy["budget"]["max_total_tasks"] == 4
        assert new_hash != expected_hash

        event = conn.execute(
            "SELECT decision, reason_code, details_json FROM governance_events WHERE requested_operation = 'policy_reauthorization' ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        assert event is not None
        assert event["decision"] == "ALLOW"
        assert event["reason_code"] == "REAUTHORIZED"
        details = json.loads(event["details_json"])
        assert details["principal"]["type"] == "HUMAN"
        assert details["principal"]["derived_from_trusted_context"] is True


def test_e14_request_does_not_equal_authorization(tmp_path, monkeypatch):
    """E14: Creating a ReauthorizationRequest does NOT alter budget or grant capacity."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb._INITIALIZED_PATHS.clear()

    board_slug = "governed-e14"
    kb.create_board(board_slug, name="Governed Board E14")
    b_dir = kb.board_dir(board_slug)
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text("policy_id: e14\nbudget:\n  max_total_tasks: 1\n", encoding="utf-8")
    expected_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()

    with kbc.connect_closing(board=board_slug) as conn:
        t1 = kb.create_task(conn, title="Initial Task", assignee="builder", board=board_slug)
        assert t1 is not None

        req = gov.request_reauthorization(
            conn,
            board_slug=board_slug,
            current_policy_hash=expected_hash,
            requested_delta=5,
            reason="P0 requesting remediation capacity",
            requester="p0-orchestrator",
        )
        assert req.status == "PENDING"

        policy, _, _ = gov.load_governance_policy(conn, board_slug=board_slug)
        assert policy["budget"]["max_total_tasks"] == 1

        with pytest.raises(gov.GovernanceBudgetExceeded):
            kb.create_task(conn, title="Over-budget Task", assignee="builder", board=board_slug)


def test_e15_all_tasks_done_but_failed_criterion(tmp_path):
    """E15: All board tasks done, but 1 required AC failed -> FAILED_ACCEPTANCE (NOT PASS)."""
    ws = tmp_path / "ws_e15"
    ws.mkdir()
    (ws / "service.py").write_text("import sys\nsys.exit(2)\n", encoding="utf-8")

    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e15"},
        "scope": {"source_manifest": ["service.py"]},
        "acceptance_criteria": [
            {
                "id": "AC-E15",
                "statement": "Service must exit 0",
                "required": True,
                "verification": {"type": "command", "argv": [sys.executable, "service.py"], "expected": {"exit_code": 0}},
            }
        ],
    }
    frozen = ac.freeze_acceptance_spec(ws, spec)
    ac.execute_acceptance_run(ws, frozen)
    res = ac.derive_project_acceptance(ws)
    assert res.status == "FAILED_ACCEPTANCE"


def test_e16_agent_writes_pass_marker(tmp_path):
    """E16: Agent attempts to write a PASS marker in project status or board metadata -> derived status ignored."""
    ws = tmp_path / "ws_e16"
    ws.mkdir()
    (ws / "service.py").write_text("import sys\nsys.exit(1)\n", encoding="utf-8")

    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e16"},
        "scope": {"source_manifest": ["service.py"]},
        "acceptance_criteria": [
            {
                "id": "AC-E16",
                "required": True,
                "verification": {"type": "command", "argv": [sys.executable, "service.py"], "expected": {"exit_code": 0}},
            }
        ],
    }
    frozen = ac.freeze_acceptance_spec(ws, spec)
    ac.execute_acceptance_run(ws, frozen)

    # Agent writes fake PASS marker
    (ws / "PROJECT_STATUS.md").write_text("# Project Status: PASS\nAll gates passed!\n")
    (ws / "board.json").write_text(json.dumps({"status": "PASS", "project_acceptance": "PASS"}))

    res = ac.derive_project_acceptance(ws)
    assert res.status != "PASS"
    assert res.status == "FAILED_ACCEPTANCE"


def test_e17_old_receipt_after_source_change(tmp_path):
    """E17: Source file modified after verification receipt generated -> receipt becomes STALE_EVIDENCE."""
    ws = tmp_path / "ws_e17"
    ws.mkdir()
    src = ws / "logic.py"
    src.write_text("def run(): return 0\nif __name__ == '__main__': exit(run())\n", encoding="utf-8")

    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e17"},
        "scope": {"source_manifest": ["logic.py"]},
        "acceptance_criteria": [
            {
                "id": "AC-E17",
                "required": True,
                "verification": {"type": "command", "argv": [sys.executable, "logic.py"], "expected": {"exit_code": 0}},
            }
        ],
    }
    frozen = ac.freeze_acceptance_spec(ws, spec)
    ac.execute_acceptance_run(ws, frozen)

    # Fresh state derives PASS
    res1 = ac.derive_project_acceptance(ws)
    assert res1.status == "PASS"

    # Now modify source file
    src.write_text("def run(): return 0\n# modified!\nif __name__ == '__main__': exit(run())\n", encoding="utf-8")

    # Now state must immediately be STALE_EVIDENCE
    res2 = ac.derive_project_acceptance(ws)
    assert res2.status == "STALE_EVIDENCE"
    assert res2.criteria_results["AC-E17"]["status"] == "STALE_EVIDENCE"


def test_e18_old_receipt_after_spec_change(tmp_path):
    """E18: Spec hash changed after receipt -> STALE_EVIDENCE."""
    ws = tmp_path / "ws_e18"
    ws.mkdir()
    src = ws / "logic.py"
    src.write_text("exit(0)\n", encoding="utf-8")

    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e18"},
        "scope": {"source_manifest": ["logic.py"]},
        "acceptance_criteria": [
            {
                "id": "AC-E18",
                "required": True,
                "verification": {"type": "command", "argv": [sys.executable, "logic.py"], "expected": {"exit_code": 0}},
            }
        ],
    }
    frozen = ac.freeze_acceptance_spec(ws, spec)
    ac.execute_acceptance_run(ws, frozen)
    assert ac.derive_project_acceptance(ws).status == "PASS"

    # Human amends spec to add another requirement
    human_ctx = make_test_human_context(principal_id="human_admin", action="amend_spec", target=str(ws))
    spec_v2 = dict(spec)
    spec_v2["version"] = 2
    spec_v2["acceptance_criteria"] = list(spec["acceptance_criteria"]) + [
        {
            "id": "AC-E18-B",
            "required": True,
            "verification": {"type": "command", "argv": [sys.executable, "logic.py"], "expected": {"exit_code": 0}},
        }
    ]
    ac.amend_acceptance_spec(ws, spec_v2, human_ctx)

    # Old receipts now point to old spec_hash -> STALE_EVIDENCE / NOT_COMPLETE
    res = ac.derive_project_acceptance(ws)
    assert res.status in {"STALE_EVIDENCE", "NOT_COMPLETE"}


def test_e19_missing_required_receipt(tmp_path):
    """E19: Required criterion in spec has no receipt generated -> NOT_COMPLETE."""
    ws = tmp_path / "ws_e19"
    ws.mkdir()
    src = ws / "logic.py"
    src.write_text("exit(0)\n", encoding="utf-8")

    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e19"},
        "scope": {"source_manifest": ["logic.py"]},
        "acceptance_criteria": [
            {
                "id": "AC-E19-REQUIRED",
                "required": True,
                "verification": {"type": "command", "argv": [sys.executable, "logic.py"], "expected": {"exit_code": 0}},
            }
        ],
    }
    ac.freeze_acceptance_spec(ws, spec)
    # Do NOT run execute_acceptance_run
    res = ac.derive_project_acceptance(ws)
    assert res.status == "NOT_COMPLETE"
    assert res.criteria_results["AC-E19-REQUIRED"]["status"] == "NO_EVIDENCE"


def test_e20_forged_exit_code_or_stdout(tmp_path):
    """E20: Tampered receipt file on disk -> CORRUPT_EVIDENCE / FAILED_ACCEPTANCE."""
    ws = tmp_path / "ws_e20"
    ws.mkdir()
    src = ws / "logic.py"
    src.write_text("exit(0)\n", encoding="utf-8")

    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e20"},
        "scope": {"source_manifest": ["logic.py"]},
        "acceptance_criteria": [
            {
                "id": "AC-E20",
                "required": True,
                "verification": {"type": "command", "argv": [sys.executable, "logic.py"], "expected": {"exit_code": 0}},
            }
        ],
    }
    frozen = ac.freeze_acceptance_spec(ws, spec)
    ac.execute_acceptance_run(ws, frozen)
    assert ac.derive_project_acceptance(ws).status == "PASS"

    # Corrupt receipt file
    receipt_file = ws / ".governance" / "receipts" / "AC-E20.json"
    receipt_file.write_text("{corrupt: json [", encoding="utf-8")

    res = ac.derive_project_acceptance(ws)
    assert res.status == "FAILED_ACCEPTANCE"
    assert res.criteria_results["AC-E20"]["status"] == "CORRUPT_EVIDENCE"


def test_e21_current_source_all_criteria_pass(tmp_path):
    """E21: Compliant source code + fresh receipts for all criteria -> PASS."""
    ws = tmp_path / "ws_e21"
    ws.mkdir()
    src = ws / "app.py"
    src.write_text("print('ALL_SYSTEMS_GO')\nexit(0)\n", encoding="utf-8")

    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e21"},
        "scope": {"source_manifest": ["app.py"]},
        "acceptance_criteria": [
            {
                "id": "AC-E21-A",
                "statement": "Exit code 0",
                "required": True,
                "verification": {"type": "command", "argv": [sys.executable, "app.py"], "expected": {"exit_code": 0}},
            },
            {
                "id": "AC-E21-B",
                "statement": "Stdout contains ALL_SYSTEMS_GO",
                "required": True,
                "verification": {"type": "command", "argv": [sys.executable, "app.py"], "expected": {"exit_code": 0, "stdout_contains": ["ALL_SYSTEMS_GO"]}},
            },
        ],
    }
    frozen = ac.freeze_acceptance_spec(ws, spec)
    ac.execute_acceptance_run(ws, frozen)

    res = ac.derive_project_acceptance(ws)
    assert res.status == "PASS"
    assert res.required_passed == 2
    assert res.required_total == 2


def test_e22_one_optional_criterion_missing(tmp_path):
    """E22: Required criteria pass, 1 optional criterion missing receipt -> PASS (required gate met)."""
    ws = tmp_path / "ws_e22"
    ws.mkdir()
    src = ws / "app.py"
    src.write_text("exit(0)\n", encoding="utf-8")

    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e22"},
        "scope": {"source_manifest": ["app.py"]},
        "acceptance_criteria": [
            {
                "id": "AC-E22-REQ",
                "required": True,
                "verification": {"type": "command", "argv": [sys.executable, "app.py"], "expected": {"exit_code": 0}},
            },
            {
                "id": "AC-E22-OPT",
                "statement": "Optional performance benchmark",
                "required": False,
                "verification": {"type": "command", "argv": [sys.executable, "app.py"], "expected": {"exit_code": 0}},
            },
        ],
    }
    frozen = ac.freeze_acceptance_spec(ws, spec)
    # Only run the required criterion
    ac.execute_acceptance_run(ws, frozen, criterion_id="AC-E22-REQ")

    res = ac.derive_project_acceptance(ws)
    assert res.status == "PASS"
    assert res.criteria_results["AC-E22-REQ"]["status"] == "PASS"
    assert res.criteria_results["AC-E22-OPT"]["status"] == "NO_EVIDENCE"


def test_e23_builder_modifies_acceptance_spec(tmp_path):
    """E23: Builder / Agent attempts to modify frozen acceptance spec -> PermissionError: DENIED."""
    ws = tmp_path / "ws_e23"
    ws.mkdir()
    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e23"},
        "scope": {"source_manifest": []},
        "acceptance_criteria": [
            {"id": "AC-E23", "required": True, "verification": {"type": "command", "argv": ["true"], "expected": {"exit_code": 0}}}
        ],
    }
    ac.freeze_acceptance_spec(ws, spec)

    agent_ctx = gov.TrustedPrincipalContext(
        principal_id="builder-agent",
        principal_type="agent",
        auth_method="agent_session",
    )
    tampered_spec = dict(spec)
    tampered_spec["acceptance_criteria"][0]["verification"]["expected"]["exit_code"] = 99

    with pytest.raises(PermissionError) as exc_info:
        ac.amend_acceptance_spec(ws, tampered_spec, agent_ctx)
    assert "cannot amend" in str(exc_info.value).lower() or "unauthorized" in str(exc_info.value).lower()


def test_e24_receipt_tampered_exit_code_and_predicate(tmp_path):
    """E24: Tampered receipt (exit code 17 with flipped predicate_satisfied=True) must be caught as TAMPERED_EVIDENCE."""
    ws = tmp_path / "ws_e24"
    ws.mkdir()
    (ws / "app.py").write_text("import sys\nsys.exit(17)\n", encoding="utf-8")

    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e24"},
        "scope": {"source_manifest": ["app.py"]},
        "acceptance_criteria": [
            {
                "id": "AC-E24",
                "required": True,
                "verification": {"type": "command", "argv": [sys.executable, "app.py"], "expected": {"exit_code": 0}},
            }
        ],
    }
    frozen = ac.freeze_acceptance_spec(ws, spec)
    ac.execute_acceptance_run(ws, frozen)

    receipt_file = ws / ".governance" / "receipts" / "AC-E24.json"
    receipt_data = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert receipt_data["exit_code"] == 17
    assert receipt_data["predicate_satisfied"] is False

    # Attacker / Agent tampers with receipt: flips predicate_satisfied to True
    receipt_data["predicate_satisfied"] = True
    receipt_file.write_text(json.dumps(receipt_data), encoding="utf-8")

    # derive_project_acceptance MUST catch tampering via signature mismatch and exit code mismatch
    res = ac.derive_project_acceptance(ws)
    assert res.status != "PASS"
    assert res.status == "FAILED_ACCEPTANCE"
    assert res.criteria_results["AC-E24"]["status"] == "TAMPERED_EVIDENCE"


def test_e25_freeze_cannot_overwrite_existing_spec_without_human(tmp_path):
    """E25: Calling freeze_acceptance_spec on an already frozen spec without human auth must raise PermissionError."""
    ws = tmp_path / "ws_e25"
    ws.mkdir()
    spec1 = {
        "p0_spec_version": 1,
        "project": {"project_id": "e25"},
        "scope": {"source_manifest": []},
        "acceptance_criteria": [
            {"id": "AC-E25-HARD", "required": True, "verification": {"type": "command", "argv": ["false"], "expected": {"exit_code": 0}}}
        ],
    }
    ac.freeze_acceptance_spec(ws, spec1)

    # Calling freeze again to overwrite with an easy criterion without human authority
    spec2 = {
        "p0_spec_version": 1,
        "project": {"project_id": "e25"},
        "scope": {"source_manifest": []},
        "acceptance_criteria": [
            {"id": "AC-E25-EASY", "required": True, "verification": {"type": "command", "argv": ["true"], "expected": {"exit_code": 0}}}
        ],
    }
    with pytest.raises(PermissionError) as exc_info:
        ac.freeze_acceptance_spec(ws, spec2)
    assert "already frozen" in str(exc_info.value).lower()


def test_e26_cross_process_valid_receipt_verification(tmp_path):
    """E26: Unmodified valid evidence must remain valid across separate Python processes (deterministic key persistence)."""
    ws = tmp_path / "ws_e26"
    ws.mkdir()
    (ws / "service.py").write_text("print('ALL_GOOD')\nexit(0)\n", encoding="utf-8")

    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e26"},
        "scope": {"source_manifest": ["service.py"]},
        "acceptance_criteria": [
            {
                "id": "AC-E26",
                "required": True,
                "verification": {
                    "type": "command",
                    "argv": [sys.executable, "service.py"],
                    "expected": {"exit_code": 0, "stdout_contains": ["ALL_GOOD"]},
                },
            }
        ],
    }
    frozen = ac.freeze_acceptance_spec(ws, spec)
    ac.execute_acceptance_run(ws, frozen)

    # 1. In-process check
    in_proc_res = ac.derive_project_acceptance(ws)
    assert in_proc_res.status == "PASS"

    # 2. Cross-process check in a fresh, separate Python process
    code = (
        "import sys, json; from pathlib import Path; "
        "sys.path.insert(0, sys.argv[1]); "
        "from hermes_cli import kanban_acceptance as ac; "
        "r = ac.derive_project_acceptance(Path(sys.argv[2])); "
        "print(json.dumps({'status': r.status, 'criteria': r.criteria_results}))"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code, str(REPO), str(ws)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["status"] == "PASS", f"Cross-process verification failed: {out}"
    assert out["criteria"]["AC-E26"]["status"] == "PASS"


def test_e27_mutated_spec_object_cannot_bypass_frozen_command_or_predicates(tmp_path):
    """E27: Mutating caller's in-memory P0Spec object must NOT allow running different commands or dropping predicates."""
    # Part 1: Mutating argv
    ws1 = tmp_path / "ws_e27_cmd"
    ws1.mkdir()
    spec1 = {
        "p0_spec_version": 1,
        "project": {"project_id": "e27_cmd"},
        "scope": {"source_manifest": []},
        "acceptance_criteria": [
            {
                "id": "AC-E27-CMD",
                "required": True,
                "verification": {
                    "type": "command",
                    "argv": [sys.executable, "-c", "import sys; sys.exit(17)"],
                    "expected": {"exit_code": 0},
                },
            }
        ],
    }
    frozen1 = ac.freeze_acceptance_spec(ws1, spec1)
    ac.execute_acceptance_run(ws1, frozen1)
    assert ac.derive_project_acceptance(ws1).status == "FAILED_ACCEPTANCE"

    # Mutate in-memory argv to exit 0
    frozen1.acceptance_criteria[0].verification["argv"] = [sys.executable, "-c", "import sys; sys.exit(0)"]
    ac.execute_acceptance_run(ws1, frozen1)

    res1 = ac.derive_project_acceptance(ws1)
    assert res1.status != "PASS", "Mutated argv in memory falsely produced PASS!"
    assert res1.status in {"FAILED_ACCEPTANCE", "TAMPERED_EVIDENCE"}

    # Part 2: Mutating predicates (dropping required stdout_contains)
    ws2 = tmp_path / "ws_e27_pred"
    ws2.mkdir()
    spec2 = {
        "p0_spec_version": 1,
        "project": {"project_id": "e27_pred"},
        "scope": {"source_manifest": []},
        "acceptance_criteria": [
            {
                "id": "AC-E27-PRED",
                "required": True,
                "verification": {
                    "type": "command",
                    "argv": [sys.executable, "-c", "print('WRONG_OUTPUT')"],
                    "expected": {"exit_code": 0, "stdout_contains": ["MANDATORY_CORRECT_OUTPUT"]},
                },
            }
        ],
    }
    frozen2 = ac.freeze_acceptance_spec(ws2, spec2)
    ac.execute_acceptance_run(ws2, frozen2)
    assert ac.derive_project_acceptance(ws2).status == "FAILED_ACCEPTANCE"

    # Mutate in-memory predicate
    frozen2.acceptance_criteria[0].verification["expected"]["stdout_contains"] = []
    ac.execute_acceptance_run(ws2, frozen2)

    res2 = ac.derive_project_acceptance(ws2)
    assert res2.status != "PASS", "Mutated predicate in memory falsely produced PASS!"
    assert res2.status in {"FAILED_ACCEPTANCE", "TAMPERED_EVIDENCE"}


def test_e28_self_asserted_human_cannot_overwrite_or_amend_spec_without_valid_proof(tmp_path):
    """E28: Self-asserted human context with arbitrary label and missing/invalid proof token must be DENIED."""
    ws = tmp_path / "ws_e28"
    ws.mkdir()
    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e28"},
        "scope": {"source_manifest": []},
        "acceptance_criteria": [
            {
                "id": "AC-E28-STRICT",
                "required": True,
                "verification": {
                    "type": "command",
                    "argv": [sys.executable, "-c", "import sys; sys.exit(17)"],
                    "expected": {"exit_code": 0},
                },
            }
        ],
    }
    ac.freeze_acceptance_spec(ws, spec)

    # 1. Self-asserted human with arbitrary auth_method and NO proof_token
    spoofed_ctx_no_token = gov.TrustedPrincipalContext(
        principal_id="audit-agent",
        principal_type="human",
        auth_method="arbitrary-label",
    )
    easy_spec = dict(spec)
    easy_spec["acceptance_criteria"][0]["verification"]["argv"] = [sys.executable, "-c", "exit(0)"]

    with pytest.raises(PermissionError) as exc_info1:
        ac.freeze_acceptance_spec(ws, easy_spec, principal_context=spoofed_ctx_no_token)
    assert "proof" in str(exc_info1.value).lower() or "unauthorized" in str(exc_info1.value).lower() or "token" in str(exc_info1.value).lower()

    with pytest.raises(PermissionError) as exc_info2:
        ac.amend_acceptance_spec(ws, easy_spec, principal_context=spoofed_ctx_no_token)
    assert "proof" in str(exc_info2.value).lower() or "unauthorized" in str(exc_info2.value).lower() or "token" in str(exc_info2.value).lower()

    # 2. Self-asserted human with INVALID/FORGED proof_token
    spoofed_ctx_bad_token = gov.TrustedPrincipalContext(
        principal_id="audit-agent",
        principal_type="human",
        auth_method="TRUSTED_CONTROL_SURFACE",
        proof_token="forged-token-abc-123",
    )
    with pytest.raises(PermissionError) as exc_info3:
        ac.amend_acceptance_spec(ws, easy_spec, principal_context=spoofed_ctx_bad_token)
    assert "proof" in str(exc_info3.value).lower() or "token" in str(exc_info3.value).lower() or "invalid" in str(exc_info3.value).lower()


def test_e29_direct_disk_yaml_tampering_denied(tmp_path):
    """E29: Direct tampering of P0_SPEC.yaml on disk without authorized amend must be caught and DENIED."""
    ws = tmp_path / "ws_e29"
    ws.mkdir()
    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e29"},
        "scope": {"source_manifest": []},
        "acceptance_criteria": [
            {
                "id": "AC-E29",
                "required": True,
                "verification": {"type": "command", "argv": [sys.executable, "-c", "import sys; sys.exit(17)"], "expected": {"exit_code": 0}},
            }
        ],
    }
    ac.freeze_acceptance_spec(ws, spec)
    ac.execute_acceptance_run(ws)
    assert ac.derive_project_acceptance(ws).status == "FAILED_ACCEPTANCE"

    # Attacker directly modifies P0_SPEC.yaml on disk to exit(0)
    spec_tampered = dict(spec)
    spec_tampered["acceptance_criteria"][0]["verification"]["argv"] = [sys.executable, "-c", "exit(0)"]
    (ws / ".governance" / "P0_SPEC.yaml").write_text(gov.dump_governance_yaml(spec_tampered), encoding="utf-8")

    # Runner & Deriver must reject tampered spec
    with pytest.raises(PermissionError) as exc_info:
        ac.execute_acceptance_run(ws)
    assert "tampered" in str(exc_info.value).lower()

    res = ac.derive_project_acceptance(ws)
    assert res.status == "FAILED_ACCEPTANCE"
    assert "SPEC_TAMPERED" in res.criteria_results


def test_e30_dotted_target_token_verification(tmp_path):
    """E30: Target paths containing dots (e.g. .hermes or project.v1) must encode and verify cleanly."""
    ws = tmp_path / "ws_e30"
    ws.mkdir()
    target = str(ws / ".hermes" / "project.v1")
    token = gov.issue_human_proof_token("admin_user", action="amend_spec", target=target)
    valid, reason = gov.verify_human_proof_token(token, "admin_user", action="amend_spec", target=target)
    assert valid is True, f"Dotted target token verification failed: {reason}"


def test_e31_verification_subprocess_environment_scrubbed(tmp_path, monkeypatch):
    """E31: Verification commands executed by acceptance runner must NEVER inherit governance secrets."""
    ws = tmp_path / "ws_e31"
    ws.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_GOVERNANCE_SECRET", "LEAK_PROBE_SECRET_12345")
    monkeypatch.setenv("HERMES_GOVERNANCE_SECRET", "LEAK_PROBE_SECRET_67890")

    probe_code = (
        "import os; "
        "has_sec1 = 'HERMES_KANBAN_GOVERNANCE_SECRET' in os.environ; "
        "has_sec2 = 'HERMES_GOVERNANCE_SECRET' in os.environ; "
        "print('LEAKED' if (has_sec1 or has_sec2) else 'CLEAN')"
    )
    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e31"},
        "scope": {"source_manifest": []},
        "acceptance_criteria": [
            {
                "id": "AC-E31",
                "required": True,
                "verification": {
                    "type": "command",
                    "argv": [sys.executable, "-c", probe_code],
                    "expected": {"exit_code": 0, "stdout_contains": ["CLEAN"]},
                },
            }
        ],
    }
    ac.freeze_acceptance_spec(ws, spec)
    receipts = ac.execute_acceptance_run(ws)
    assert len(receipts) == 1
    assert "LEAKED" not in receipts[0].stdout_snippet
    assert "CLEAN" in receipts[0].stdout_snippet
    assert receipts[0].predicate_satisfied is True


def test_e32_production_module_lacks_test_human_helper():
    """E32: Production governance module must NOT expose self-issuing human context helper."""
    assert not hasattr(gov, "create_test_human_context"), (
        "CRITICAL SECURITY DEFECT: Production module hermes_cli.kanban_governance "
        "must not expose create_test_human_context helper."
    )


def test_e33_registry_missing_fail_closed(tmp_path, monkeypatch):
    """E33: When control plane approved spec registry is missing, loading & derived acceptance must FAIL CLOSED."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    ws = tmp_path / "ws_e33"
    ws.mkdir()

    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e33"},
        "scope": {"source_manifest": []},
        "acceptance_criteria": [
            {"id": "AC1", "verification": {"type": "command", "argv": ["true"], "expected": {"exit_code": 0}}}
        ],
    }
    ac.freeze_acceptance_spec(ws, spec)

    # Delete registry file
    registry = home / ".governance_authority" / "approved_specs.json"
    assert registry.is_file()
    registry.unlink()

    with pytest.raises(PermissionError) as exc_info:
        ac.load_frozen_spec(ws)
    assert "missing" in str(exc_info.value).lower()

    res = ac.derive_project_acceptance(ws)
    assert res.status == "FAILED_ACCEPTANCE"
    assert "SPEC_TAMPERED" in res.criteria_results


def test_e34_registry_corrupt_fail_closed(tmp_path, monkeypatch):
    """E34: When control plane approved spec registry is corrupt, loading & derived acceptance must FAIL CLOSED."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    ws = tmp_path / "ws_e34"
    ws.mkdir()

    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e34"},
        "scope": {"source_manifest": []},
        "acceptance_criteria": [
            {"id": "AC1", "verification": {"type": "command", "argv": ["true"], "expected": {"exit_code": 0}}}
        ],
    }
    ac.freeze_acceptance_spec(ws, spec)

    registry = home / ".governance_authority" / "approved_specs.json"
    registry.write_text("{corrupt json [", encoding="utf-8")

    with pytest.raises(PermissionError) as exc_info:
        ac.load_frozen_spec(ws)
    assert "corrupt" in str(exc_info.value).lower()

    res = ac.derive_project_acceptance(ws)
    assert res.status == "FAILED_ACCEPTANCE"
    assert "SPEC_TAMPERED" in res.criteria_results


def test_e35_unknown_project_identity_fail_closed(tmp_path, monkeypatch):
    """E35: Spec YAML modified to an unknown/unregistered project_id must FAIL CLOSED."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    ws = tmp_path / "ws_e35"
    ws.mkdir()

    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e35_registered"},
        "scope": {"source_manifest": []},
        "acceptance_criteria": [
            {"id": "AC1", "verification": {"type": "command", "argv": ["true"], "expected": {"exit_code": 0}}}
        ],
    }
    ac.freeze_acceptance_spec(ws, spec)

    # Tamper YAML and sidecar to unregistered project_id
    import copy, hashlib
    tampered = copy.deepcopy(spec)
    tampered["project"]["project_id"] = "e35_UNREGISTERED"
    raw = gov.dump_governance_yaml(tampered).encode("utf-8")
    (ws / ".governance" / "P0_SPEC.yaml").write_bytes(raw)
    (ws / ".governance" / "P0_SPEC.sha256").write_text(hashlib.sha256(raw).hexdigest(), encoding="utf-8")

    with pytest.raises(PermissionError) as exc_info:
        ac.load_frozen_spec(ws)
    assert "unregistered" in str(exc_info.value).lower() or "no approved" in str(exc_info.value).lower()

    res = ac.derive_project_acceptance(ws)
    assert res.status == "FAILED_ACCEPTANCE"
    assert "SPEC_TAMPERED" in res.criteria_results


def test_e36_runner_env_cannot_reintroduce_governance_secrets(tmp_path):
    """E36: runner_env parameter cannot re-introduce governance secrets into the subprocess environment."""
    ws = tmp_path / "ws_e36"
    ws.mkdir()

    probe_code = (
        "import os; "
        "has_sec = 'HERMES_GOVERNANCE_SECRET' in os.environ or 'HERMES_KANBAN_GOVERNANCE_SECRET' in os.environ; "
        "print('LEAKED' if has_sec else 'CLEAN')"
    )
    spec = {
        "p0_spec_version": 1,
        "project": {"project_id": "e36"},
        "scope": {"source_manifest": []},
        "acceptance_criteria": [
            {
                "id": "AC-E36",
                "required": True,
                "verification": {
                    "type": "command",
                    "argv": [sys.executable, "-c", probe_code],
                    "expected": {"exit_code": 0, "stdout_contains": ["CLEAN"]},
                },
            }
        ],
    }
    ac.freeze_acceptance_spec(ws, spec)

    # Attempt to reintroduce via runner_env
    receipts = ac.execute_acceptance_run(
        ws,
        runner_env={"HERMES_GOVERNANCE_SECRET": "OVERRIDE_SECRET_TEST", "HERMES_KANBAN_GOVERNANCE_SECRET": "OVERRIDE_SECRET_TEST"},
    )
    assert len(receipts) == 1
    assert "LEAKED" not in receipts[0].stdout_snippet
    assert "CLEAN" in receipts[0].stdout_snippet
    assert receipts[0].predicate_satisfied is True
