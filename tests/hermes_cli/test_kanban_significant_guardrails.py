"""Hard guardrails for significant Kanban work.

These tests pin the non-advisory C/D/E mixed-agent gate:
significant work cannot dispatch downstream work directly unless it has an
explicit single-controller waiver, and review lanes/reconcilers must provide
structured custody metadata before the graph can advance.
"""

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


def _significant(group="gate-1"):
    return {"significant_work": True, "guardrail_group": group}


def _lane(role: str, group="gate-1"):
    return {"guardrail_group": group, "guardrail_role": role}


def _handoff(model: str, verdict="APPROVE", artifact="abc123"):
    return {
        "artifact_commit": artifact,
        "findings_artifact": f"reviews/{model}.md",
        "model": model,
        "verdict": verdict,
    }


def _reconciler_handoff(verdict="APPROVE", artifact="abc123"):
    return {
        "artifact_commit": artifact,
        "findings_artifact": "reviews/reconciler.md",
        "verdict": verdict,
    }


def _create_gate(conn, parent: str, group="gate-1"):
    codex = kb.create_task(
        conn,
        title="Codex review lane",
        assignee="codex-reviewer",
        parents=[parent],
        metadata=_lane("codex_lane", group),
    )
    claude = kb.create_task(
        conn,
        title="Claude review lane",
        assignee="claude-reviewer",
        parents=[parent],
        metadata=_lane("claude_lane", group),
    )
    recon = kb.create_task(
        conn,
        title="Reconcile reviews",
        assignee="controller",
        parents=[codex, claude],
        metadata=_lane("reconciler", group),
    )
    return codex, claude, recon


def test_significant_task_rejects_direct_downstream_without_waiver(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial control-plane work", metadata=_significant())
        with pytest.raises(kb.SignificantWorkGuardrailError, match="direct downstream"):
            kb.create_task(conn, title="implement directly", parents=[parent])


def test_single_controller_waiver_allows_direct_downstream(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(
            conn,
            title="mechanical typo",
            metadata={
                "significant_work": True,
                "single_controller_acceptable": True,
                "guardrail_group": "gate-1",
            },
        )
        child = kb.create_task(conn, title="mechanical follow-up", parents=[parent])
        assert kb.get_task(conn, child).status == "todo"


def test_significant_task_allows_codex_claude_reconciler_gate_shape(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, recon = _create_gate(conn, parent)
        assert kb.parent_ids(conn, recon) == sorted([claude, codex])
        assert kb.get_task(conn, codex).status == "todo"
        assert kb.get_task(conn, claude).status == "todo"
        assert kb.get_task(conn, recon).status == "todo"


def test_downstream_cannot_parent_directly_on_review_lane(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, _claude, _recon = _create_gate(conn, parent)
        with pytest.raises(kb.SignificantWorkGuardrailError, match="review lane"):
            kb.create_task(conn, title="bad downstream", parents=[codex])


def test_review_lane_completion_requires_structured_metadata(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, _claude, _recon = _create_gate(conn, parent)
        kb.complete_task(conn, parent, result="ready for review")
        with pytest.raises(kb.SignificantWorkGuardrailError, match="artifact_commit"):
            kb.complete_task(conn, codex, result="approved", metadata={"model": "gpt-5.5"})


def test_standalone_guardrail_role_cannot_be_created_without_group(kanban_home):
    with kb.connect() as conn:
        with pytest.raises(kb.SignificantWorkGuardrailError, match="guardrail_group"):
            kb.create_task(
                conn,
                title="ungrouped codex lane",
                assignee="codex-reviewer",
                metadata={"guardrail_role": "codex_lane"},
            )


def test_reconciler_rejects_swapped_lane_model_families(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, recon = _create_gate(conn, parent)
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("claude-opus-4-7"))
        kb.complete_task(conn, claude, metadata=_handoff("gpt-5.5"))
        with pytest.raises(kb.SignificantWorkGuardrailError, match="codex_lane.*OpenAI"):
            kb.complete_task(conn, recon, metadata=_reconciler_handoff())


def test_reconciler_rejects_same_model_family(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, recon = _create_gate(conn, parent)
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("gpt-5.5"))
        kb.complete_task(conn, claude, metadata=_handoff("openai-codex/gpt-5.5"))
        with pytest.raises(kb.SignificantWorkGuardrailError, match="different model families"):
            kb.complete_task(conn, recon, metadata=_reconciler_handoff())


def test_reconciler_rejects_artifact_commit_mismatch(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, recon = _create_gate(conn, parent)
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("gpt-5.5", artifact="abc123"))
        kb.complete_task(conn, claude, metadata=_handoff("claude-opus-4-7", artifact="def456"))
        with pytest.raises(kb.SignificantWorkGuardrailError, match="artifact_commit mismatch"):
            kb.complete_task(conn, recon, metadata=_reconciler_handoff())


def test_reconciler_rejects_verdict_disagreement(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, recon = _create_gate(conn, parent)
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("gpt-5.5", verdict="APPROVE"))
        kb.complete_task(conn, claude, metadata=_handoff("claude-opus-4-7", verdict="BLOCK"))
        with pytest.raises(kb.SignificantWorkGuardrailError, match="verdict disagreement"):
            kb.complete_task(conn, recon, metadata=_reconciler_handoff())


def test_reconciler_rejects_non_approving_consensus(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, recon = _create_gate(conn, parent)
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("gpt-5.5", verdict="BLOCK"))
        kb.complete_task(conn, claude, metadata=_handoff("claude-opus-4-7", verdict="BLOCK"))
        with pytest.raises(kb.SignificantWorkGuardrailError, match="non-approving"):
            kb.complete_task(conn, recon, metadata=_reconciler_handoff(verdict="BLOCK"))


def test_string_false_is_not_a_single_controller_waiver(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(
            conn,
            title="nontrivial work",
            metadata={
                "significant_work": True,
                "single_controller_acceptable": "false",
                "guardrail_group": "gate-1",
            },
        )
        with pytest.raises(kb.SignificantWorkGuardrailError, match="direct downstream"):
            kb.create_task(conn, title="bad downstream", parents=[parent])


def test_significant_and_gate_roles_require_guardrail_group(kanban_home):
    with kb.connect() as conn:
        with pytest.raises(kb.SignificantWorkGuardrailError, match="guardrail_group"):
            kb.create_task(conn, title="nontrivial work", metadata={"significant_work": True})
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        with pytest.raises(kb.SignificantWorkGuardrailError, match="guardrail_group"):
            kb.create_task(
                conn,
                title="ungrouped lane",
                parents=[parent],
                assignee="codex-reviewer",
                metadata={"guardrail_role": "codex_lane"},
            )


def test_openai_o_series_counts_as_openai_family(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, recon = _create_gate(conn, parent)
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("o3"))
        kb.complete_task(conn, claude, metadata=_handoff("openai-codex/gpt-5.5"))
        with pytest.raises(kb.SignificantWorkGuardrailError, match="different model families"):
            kb.complete_task(conn, recon, metadata=_reconciler_handoff())


def test_reconciler_requires_own_custody_metadata(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, recon = _create_gate(conn, parent)
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("gpt-5.5"))
        kb.complete_task(conn, claude, metadata=_handoff("claude-opus-4-7"))
        with pytest.raises(kb.SignificantWorkGuardrailError, match="reconciler completion missing"):
            kb.complete_task(conn, recon, metadata=None)


def test_same_profile_cannot_complete_both_review_lanes(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex = kb.create_task(
            conn,
            title="Codex lane",
            assignee="codex-claude-controller",
            parents=[parent],
            metadata=_lane("codex_lane"),
        )
        claude = kb.create_task(
            conn,
            title="Claude lane",
            assignee="codex-claude-controller",
            parents=[parent],
            metadata=_lane("claude_lane"),
        )
        recon = kb.create_task(
            conn,
            title="Reconcile",
            assignee="controller",
            parents=[codex, claude],
            metadata=_lane("reconciler"),
        )
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("gpt-5.5"))
        kb.complete_task(conn, claude, metadata=_handoff("claude-opus-4-7"))
        with pytest.raises(kb.SignificantWorkGuardrailError, match="different worker profiles"):
            kb.complete_task(conn, recon, metadata=_reconciler_handoff())


def test_unlink_cannot_remove_guardrail_dependency_to_promote_downstream(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        _codex, _claude, recon = _create_gate(conn, parent)
        downstream = kb.create_task(conn, title="safe downstream", parents=[recon])
        assert kb.get_task(conn, downstream).status == "todo"
        with pytest.raises(kb.SignificantWorkGuardrailError, match="cannot unlink guardrail"):
            kb.unlink_tasks(conn, recon, downstream)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, downstream).status == "todo"


def test_reconciler_lanes_must_share_same_significant_parent(kanban_home):
    with kb.connect() as conn:
        parent_a = kb.create_task(conn, title="nontrivial work A", metadata=_significant("gate-1"))
        parent_b = kb.create_task(conn, title="nontrivial work B", metadata=_significant("gate-1"))
        codex = kb.create_task(
            conn,
            title="Codex lane A",
            assignee="codex-reviewer",
            parents=[parent_a],
            metadata=_lane("codex_lane"),
        )
        claude = kb.create_task(
            conn,
            title="Claude lane B",
            assignee="claude-reviewer",
            parents=[parent_b],
            metadata=_lane("claude_lane"),
        )
        with pytest.raises(kb.SignificantWorkGuardrailError, match="same significant parent"):
            kb.create_task(
                conn,
                title="Bad reconcile",
                assignee="controller",
                parents=[codex, claude],
                metadata=_lane("reconciler"),
            )


def test_reconciler_own_verdict_must_approve(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, recon = _create_gate(conn, parent)
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("gpt-5.5", verdict="APPROVE"))
        kb.complete_task(conn, claude, metadata=_handoff("claude-opus-4-7", verdict="APPROVE"))
        with pytest.raises(kb.SignificantWorkGuardrailError, match="reconciler verdict"):
            kb.complete_task(conn, recon, metadata=_reconciler_handoff(verdict="BLOCK"))


def test_completed_fake_lane_cannot_be_linked_under_significant_parent(kanban_home):
    with kb.connect() as conn:
        significant = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        unrelated_parent = kb.create_task(conn, title="unrelated source")
        codex = kb.create_task(
            conn,
            title="precompleted fake codex lane",
            assignee="codex-reviewer",
            parents=[unrelated_parent],
            metadata=_lane("codex_lane"),
        )
        kb.complete_task(conn, unrelated_parent, result="not the reviewed work")
        kb.complete_task(conn, codex, metadata=_handoff("gpt-5.5"))
        with pytest.raises(kb.SignificantWorkGuardrailError, match="completed guardrail lane"):
            kb.link_tasks(conn, significant, codex)


def test_significant_waiver_still_requires_guardrail_group(kanban_home):
    with kb.connect() as conn:
        with pytest.raises(kb.SignificantWorkGuardrailError, match="non-empty guardrail_group"):
            kb.create_task(
                conn,
                title="mechanical but significant",
                metadata={"significant_work": True, "single_controller_acceptable": True},
            )


def test_lane_reassignment_preserves_profile_family(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, _recon = _create_gate(conn, parent)
        with pytest.raises(kb.SignificantWorkGuardrailError, match="Codex/OpenAI"):
            kb.assign_task(conn, codex, "claude-reviewer")
        with pytest.raises(kb.SignificantWorkGuardrailError, match="Claude/Anthropic"):
            kb.assign_task(conn, claude, "codex-reviewer")


def test_edit_completed_review_lane_cannot_fabricate_approval_metadata(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, _claude, _recon = _create_gate(conn, parent)
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("gpt-5.5", verdict="BLOCK"))
        with pytest.raises(kb.SignificantWorkGuardrailError, match="completed guardrail review metadata"):
            kb.edit_completed_task_result(
                conn,
                codex,
                result="retroactive approval",
                metadata=_handoff("gpt-5.5", verdict="APPROVE"),
            )


def test_running_downstream_cannot_be_linked_under_unfinished_reconciler(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        _codex, _claude, recon = _create_gate(conn, parent)
        downstream = kb.create_task(conn, title="already running downstream", assignee="worker")
        run = kb.claim_task(conn, downstream, claimer="worker")
        assert run is not None
        assert kb.get_task(conn, downstream).status == "running"
        with pytest.raises(kb.SignificantWorkGuardrailError, match="running downstream"):
            kb.link_tasks(conn, recon, downstream)
        assert kb.complete_task(conn, downstream, result="finished unrelated work") is True


def test_reconciler_requires_current_lanes_still_done(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, recon = _create_gate(conn, parent)
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("gpt-5.5"))
        kb.complete_task(conn, claude, metadata=_handoff("claude-opus-4-7"))
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (codex,))
        with pytest.raises(kb.SignificantWorkGuardrailError, match="current status"):
            kb.complete_task(conn, recon, metadata=_reconciler_handoff())


def test_blocked_downstream_linked_under_reconciler_is_not_unblocked_ready(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, recon = _create_gate(conn, parent)
        downstream = kb.create_task(conn, title="ship")
        kb.block_task(conn, downstream, reason="manual hold")
        kb.link_tasks(conn, recon, downstream)
        assert kb.get_task(conn, downstream).status == "blocked"
        assert kb.unblock_task(conn, downstream) is False
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("gpt-5.5"))
        kb.complete_task(conn, claude, metadata=_handoff("claude-opus-4-7"))
        kb.complete_task(conn, recon, metadata=_reconciler_handoff())
        assert kb.get_task(conn, downstream).status == "blocked"


def test_done_downstream_cannot_be_retroactively_linked_under_reconciler(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        _codex, _claude, recon = _create_gate(conn, parent)
        downstream = kb.create_task(conn, title="already completed work", assignee="worker")
        kb.complete_task(conn, downstream, result="ran before the gate")
        with pytest.raises(kb.SignificantWorkGuardrailError, match="completed downstream"):
            kb.link_tasks(conn, recon, downstream)


def test_reconciler_rejects_duplicate_review_lane_parents(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex_block, claude, recon = _create_gate(conn, parent)
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex_block, metadata=_handoff("gpt-5.5", verdict="BLOCK"))
        kb.complete_task(conn, claude, metadata=_handoff("claude-opus-4-7", verdict="APPROVE"))
        codex_approve = kb.create_task(
            conn,
            title="duplicate approving codex lane",
            assignee="codex-reviewer-2",
            parents=[parent],
            metadata=_lane("codex_lane"),
        )
        kb.complete_task(conn, codex_approve, metadata=_handoff("gpt-5.5", verdict="APPROVE"))
        with pytest.raises(kb.SignificantWorkGuardrailError, match="exactly one codex_lane"):
            kb.link_tasks(conn, codex_approve, recon)
        with pytest.raises(kb.SignificantWorkGuardrailError, match="verdict disagreement"):
            kb.complete_task(conn, recon, metadata=_reconciler_handoff())


def test_guardrail_gate_cards_cannot_be_archived_after_completion(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, recon = _create_gate(conn, parent)
        downstream = kb.create_task(conn, title="safe downstream", assignee="worker", parents=[recon])
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("gpt-5.5"))
        kb.complete_task(conn, claude, metadata=_handoff("claude-opus-4-7"))
        kb.complete_task(conn, recon, metadata=_reconciler_handoff())
        assert kb.get_task(conn, downstream).status == "ready"
        with pytest.raises(kb.SignificantWorkGuardrailError, match="cannot archive guardrail"):
            kb.archive_task(conn, recon)
        assert kb.get_task(conn, recon).status == "done"
        assert kb.claim_task(conn, downstream, claimer="worker") is not None


def test_downstream_becomes_ready_only_after_valid_reconciler(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="nontrivial work", metadata=_significant())
        codex, claude, recon = _create_gate(conn, parent)
        downstream = kb.create_task(conn, title="safe downstream", parents=[recon])
        kb.complete_task(conn, parent, result="ready for review")
        kb.complete_task(conn, codex, metadata=_handoff("gpt-5.5"))
        assert kb.get_task(conn, downstream).status == "todo"
        kb.complete_task(conn, claude, metadata=_handoff("claude-opus-4-7"))
        assert kb.get_task(conn, downstream).status == "todo"
        kb.complete_task(conn, recon, metadata=_reconciler_handoff())
        assert kb.get_task(conn, downstream).status == "ready"
