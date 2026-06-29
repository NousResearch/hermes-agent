from __future__ import annotations

import json

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture(autouse=True)
def clear_approval_records_between_tests():
    from tools import approval

    approval.clear_all_approval_records_for_tests()
    yield
    approval.clear_all_approval_records_for_tests()


def test_safe_engineering_actions_do_not_request_micro_approval(monkeypatch):
    from tools import approval

    monkeypatch.setattr(approval, "_clio_mvp_safe_actions_enabled", lambda: True)
    safe_commands = [
        "pytest tests/test_clio_mvp_execution_profile.py -q",
        "pytest tests -q",
        "npm run typecheck",
        "npm run lint",
        "npm run build",
        "git diff --check",
        "git add hermes_cli/goal_os.py tests/test_clio_mvp_execution_v2.py && git commit -m \"fix: enforce Clio MVP execution evidence and approval gates\"",
        "git push niko-fork ai/clio-mvp-execution-v2-evidence-gates",
        "gh pr create --draft --title \"fix: enforce Clio MVP execution evidence and approval gates\"",
        "gh pr checks 123 --watch",
        "ssh build-staging-clio 'sudo -n /usr/local/sbin/buidl-staging-deploy-no-secret --dry-run 615bbefe3c36b9d23d6b1bdf655bfa0b57fc8a83'",
        "ssh build-staging-clio 'sudo -n /usr/local/sbin/buidl-staging-controlled-provider-setup --self-check'",
    ]

    for command in safe_commands:
        assert approval._clio_mvp_auto_approve("terminal", command, "safe delegated engineering work") is True, command


def test_safe_engineering_auto_approval_still_blocks_hard_gates(monkeypatch):
    from tools import approval

    monkeypatch.setattr(approval, "_clio_mvp_safe_actions_enabled", lambda: True)
    hard_gate_commands = [
        "deploy production",
        "git push origin main",
        "run provider prompt against Anthropic",
        "enable image generation",
        "run database migration",
        "enable worker",
    ]

    for command in hard_gate_commands:
        assert approval._clio_mvp_auto_approve("terminal", command, "hard gate") is False, command


def test_approval_status_reports_last_request_category(monkeypatch):
    from tools import approval

    monkeypatch.setattr(approval, "_clio_mvp_safe_actions_enabled", lambda: True)
    approval._clio_mvp_auto_approve("terminal", "pytest tests -q", "focused tests")

    status = approval.approval_status_snapshot()
    assert status["mvp_execution_profile_active"] in {True, False}
    assert status["safe_action_auto_approval_active"] is True
    assert status["hard_gates_active"] is True
    assert status["last_approval_request_category"] == "SAFE_AUTONOMOUS"
    assert "pytest tests -q" in status["last_approval_request_summary"]
    assert "automatically approved" in status["why_it_asked_niko"].lower() or "auto-approved" in status["why_it_asked_niko"].lower()


def test_safe_approval_request_resolves_without_niko(monkeypatch):
    from tools import approval

    approval.clear_all_approval_records_for_tests()
    monkeypatch.setattr(approval, "_clio_mvp_safe_actions_enabled", lambda: True)

    decision = approval.resolve_approval_request(
        action="terminal",
        payload="pytest tests/test_clio_mvp_execution_v2.py -q",
        requested_by="Builder Agent",
        reason="focused test command",
    )

    assert decision["approved"] is True
    assert decision["category"] == "SAFE_AUTONOMOUS"
    assert decision["resolved_by"] == "policy"
    assert decision["niko_required"] is False
    assert approval.approval_status_snapshot()["pending_approvals"] == []


def test_safe_feature_branch_git_and_pr_actions_auto_approve(monkeypatch):
    from tools import approval

    approval.clear_all_approval_records_for_tests()
    monkeypatch.setattr(approval, "_clio_mvp_safe_actions_enabled", lambda: True)
    commands = [
        "git commit -m 'fix: resolve safe approvals without Niko'",
        "git push niko-fork ai/fix-approval-resolution-engine",
        "gh pr create --draft --title 'fix: resolve safe approvals without Niko'",
        "gh pr checks 148 --watch",
    ]

    for command in commands:
        decision = approval.resolve_approval_request(
            action="terminal",
            payload=command,
            requested_by="Builder Agent",
            reason="safe feature branch work",
        )
        assert decision["approved"] is True, command
        assert decision["category"] == "SAFE_AUTONOMOUS", command


def test_goal_envelope_wrapper_commands_auto_approve(monkeypatch):
    from tools import approval

    approval.clear_all_approval_records_for_tests()
    monkeypatch.setattr(approval, "_clio_mvp_safe_actions_enabled", lambda: True)
    commands = [
        "ssh build-staging-clio 'sudo -n /usr/local/sbin/buidl-staging-deploy-no-secret --apply 3395c11d3bfd5625ee5bce48f3bdb285c1c86095'",
        "ssh build-staging-clio 'sudo -n /usr/local/sbin/buidl-staging-controlled-provider-setup --self-check'",
    ]

    for command in commands:
        decision = approval.resolve_approval_request(
            action="terminal",
            payload=command,
            requested_by="Ops Agent",
            reason="approved goal envelope provider-safe setup-only wrapper",
            goal_envelope_approved=True,
        )
        assert decision["approved"] is True, command
        assert decision["category"] == "GOAL_ENVELOPE_APPROVED", command


def test_hard_gates_remain_pending_red_and_visible(monkeypatch):
    from tools import approval

    approval.clear_all_approval_records_for_tests()
    monkeypatch.setattr(approval, "_clio_mvp_safe_actions_enabled", lambda: True)

    for payload in [
        "provider",
        "providers",
        "run real provider call",
        "prompt",
        "prompts",
        "execute live blind prompt",
        "enable image generation",
        "production",
        "deploy production",
        "change DNS records",
        "DB",
        "run DB migration",
        "charge billing credits payments",
    ]:
        decision = approval.resolve_approval_request(
            action="terminal",
            payload=payload,
            requested_by="Builder Agent",
            reason="hard gate probe",
        )
        assert decision["approved"] is False, payload
        assert decision["category"] == "NIKO_HARD_GATE", payload
        assert decision["niko_required"] is True, payload

    status = approval.approval_status_snapshot()
    assert status["current_blocker_status"] == "RED"
    assert len(status["pending_approvals"]) == 13
    assert all(item["category"] == "NIKO_HARD_GATE" for item in status["pending_approvals"])


def test_goal_envelope_never_overrides_hard_gates(monkeypatch):
    from tools import approval

    approval.clear_all_approval_records_for_tests()
    monkeypatch.setattr(approval, "_clio_mvp_safe_actions_enabled", lambda: True)

    for payload in ["configure providers", "edit prompts", "touch production config", "run DB task"]:
        decision = approval.resolve_approval_request(
            action="terminal",
            payload=payload,
            requested_by="Builder Agent",
            reason="approved goal envelope",
            goal_envelope_approved=True,
        )
        assert decision["approved"] is False, payload
        assert decision["category"] == "NIKO_HARD_GATE", payload
        assert decision["niko_required"] is True, payload


def test_unknown_action_is_red_not_auto_approved(monkeypatch):
    from tools import approval

    approval.clear_all_approval_records_for_tests()
    monkeypatch.setattr(approval, "_clio_mvp_safe_actions_enabled", lambda: True)

    decision = approval.resolve_approval_request(
        action="terminal",
        payload="custom risky operator action",
        requested_by="Builder Agent",
        reason="unknown action",
    )

    assert decision["approved"] is False
    assert decision["category"] == "UNKNOWN_RED"
    assert decision["current_blocker_status"] == "RED"


def test_timeout_routes_by_category(monkeypatch):
    from tools import approval

    approval.clear_all_approval_records_for_tests()
    monkeypatch.setattr(approval, "_clio_mvp_safe_actions_enabled", lambda: True)

    safe = approval.resolve_approval_timeout(
        action="terminal",
        payload="pytest tests/test_clio_mvp_execution_v2.py -q",
        requested_by="Builder Agent",
        reason="safe timed out",
    )
    delegated = approval.resolve_approval_timeout(
        action="decision",
        payload="rerun failed safe test",
        requested_by="Builder Agent",
        reason="safe ambiguous test rerun",
    )
    hard = approval.resolve_approval_timeout(
        action="terminal",
        payload="run real provider call",
        requested_by="Builder Agent",
        reason="hard timed out",
    )

    assert safe["approved"] is True
    assert safe["timeout_status"] == "auto_resolved_after_timeout"
    assert delegated["approved"] is True
    assert delegated["category"] == "DELEGATED_AGENT_APPROVAL"
    assert delegated["resolved_by"] in {"Verifier Agent", "Reviewer Agent", "Ops Agent"}
    assert hard["approved"] is False
    assert hard["timeout_status"] == "blocked_after_timeout"


def test_delegated_approval_records_role_evidence_and_reason(monkeypatch):
    from tools import approval

    approval.clear_all_approval_records_for_tests()
    monkeypatch.setattr(approval, "_clio_mvp_safe_actions_enabled", lambda: True)

    decision = approval.resolve_approval_request(
        action="decision",
        payload="decide whether non-blocking CI warning is NOISE",
        requested_by="Builder Agent",
        reason="safe ambiguous CI classification",
        delegated_role="Verifier Agent",
        evidence="CI passed, warning does not affect acceptance criteria",
    )

    assert decision["approved"] is True
    assert decision["category"] == "DELEGATED_AGENT_APPROVAL"
    assert decision["resolved_by"] == "Verifier Agent"
    assert decision["evidence"]
    assert "not a hard gate" in decision["resolution_reason"].lower()


def test_approval_status_contains_pending_and_resolved_details(monkeypatch):
    from tools import approval

    approval.clear_all_approval_records_for_tests()
    monkeypatch.setattr(approval, "_clio_mvp_safe_actions_enabled", lambda: True)

    approval.resolve_approval_request(action="terminal", payload="pytest tests -q", requested_by="Builder Agent", reason="safe tests")
    approval.resolve_approval_request(action="terminal", payload="deploy production", requested_by="Builder Agent", reason="hard gate")

    rendered = approval.format_approval_status()
    assert "Pending approvals" in rendered
    assert "Resolved approvals" in rendered
    assert "SAFE_AUTONOMOUS" in rendered
    assert "NIKO_HARD_GATE" in rendered
    assert "Niko required: yes" in rendered


def test_green_report_guard_blocks_pending_approval(hermes_home):
    from hermes_cli.goal_os import GoalOSManager, evaluate_goal_report_readiness
    from tools import approval

    approval.clear_all_approval_records_for_tests()
    manager = GoalOSManager()
    goal = manager.create_goal("Approval pending checkpoint")
    goal.evidence_log.append({"role": "Verifier Agent", "summary": "tests passed", "acceptance_checked": True, "commands": ["pytest"]})
    approval.resolve_approval_request(action="terminal", payload="deploy production", requested_by="Builder Agent", reason="hard gate")

    verdict = evaluate_goal_report_readiness(goal, requested_label="GREEN")
    assert verdict.classification == "RED"
    assert "approval" in verdict.reason.lower()


def test_builder_self_report_cannot_close_card_while_approval_pending(hermes_home):
    from hermes_cli.goal_os import GoalOSManager
    from tools import approval

    approval.clear_all_approval_records_for_tests()
    manager = GoalOSManager()
    goal = manager.create_goal("Pending approval blocks card close")
    builder = next(card for card in goal.cards if card.owner_role == "Builder Agent")
    approval.resolve_approval_request(action="terminal", payload="deploy production", requested_by="Builder Agent", reason="hard gate")

    report = manager.close_card(builder.card_id, actor_role="Builder Agent", evidence={"summary": "I built it"})
    assert report.classification == "RED"
    assert "approval" in report.message.lower()


def test_blockers_report_shows_pending_hard_gate(hermes_home):
    from hermes_cli.goal_os import GoalOSManager
    from tools import approval

    approval.clear_all_approval_records_for_tests()
    manager = GoalOSManager()
    manager.create_goal("Pending approval visible in blockers")
    approval.resolve_approval_request(action="terminal", payload="deploy production", requested_by="Builder Agent", reason="hard gate")

    report = manager.blockers_report()
    assert report.classification == "RED"
    assert "NIKO_HARD_GATE" in report.message
    assert "approval_" in report.message


def test_green_report_guard_requires_verifier_evidence(hermes_home):
    from hermes_cli.goal_os import GoalOSManager, evaluate_goal_report_readiness

    manager = GoalOSManager()
    goal = manager.create_goal("Implement profile v2")

    verdict = evaluate_goal_report_readiness(goal, requested_label="GREEN")
    assert verdict.classification == "RED"
    assert "verifier evidence" in verdict.reason.lower()


def test_builder_self_report_does_not_close_card_without_verifier(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    manager = GoalOSManager()
    goal = manager.create_goal("Builder self-report should not close")
    builder = next(card for card in goal.cards if card.owner_role == "Builder Agent")
    report = manager.close_card(builder.card_id, actor_role="Builder Agent", evidence={"summary": "I built it"})

    assert report.classification == "RED"
    stored = manager.get_goal(goal.goal_id)
    assert stored is not None
    stored_builder = next(card for card in stored.cards if card.card_id == builder.card_id)
    assert stored_builder.status != "done"


def test_verifier_evidence_can_close_non_ui_card(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    manager = GoalOSManager()
    goal = manager.create_goal("Verifier closes backend card")
    builder = next(card for card in goal.cards if card.owner_role == "Builder Agent")
    builder.status = "verification"
    manager.save_goal(goal)

    report = manager.close_card(
        builder.card_id,
        actor_role="Verifier Agent",
        evidence={"summary": "pytest passed", "commands": ["pytest tests/test_clio_mvp_execution_v2.py -q"], "acceptance_checked": True},
    )

    assert report.classification == "GREEN"
    stored = manager.get_goal(goal.goal_id)
    assert stored is not None
    assert next(card for card in stored.cards if card.card_id == builder.card_id).status == "done"


def test_ui_card_requires_product_qa_and_design_qa_before_done(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    manager = GoalOSManager()
    goal = manager.create_goal("Browser UI checkpoint")
    card = next(card for card in goal.cards if card.owner_role == "Product QA Agent")
    card.status = "verification"
    card.acceptance_criteria.append("UI/browser-facing work requires product QA and design QA evidence.")
    manager.save_goal(goal)

    missing = manager.close_card(card.card_id, actor_role="Verifier Agent", evidence={"summary": "tests passed", "commands": ["pytest"], "acceptance_checked": True})
    assert missing.classification == "RED"
    assert "product qa" in missing.message.lower()
    assert "design qa" in missing.message.lower()

    with_qa = manager.close_card(
        card.card_id,
        actor_role="Verifier Agent",
        evidence={
            "summary": "browser evidence reviewed",
            "commands": ["pytest"],
            "acceptance_checked": True,
            "product_qa": "prompt-domain fit and no fake preview checked",
            "design_qa": "layout quality checked",
            "browser_evidence": "human browser report accepted",
        },
    )
    assert with_qa.classification == "GREEN"


def test_browser_checkpoint_setup_ready_is_not_product_green(hermes_home):
    from hermes_cli.goal_os import GoalOSManager, controlled_provider_status_report

    manager = GoalOSManager()
    goal = manager.create_goal("Buidl browser checkpoint")
    report = controlled_provider_status_report(goal, ["CONTROLLED_SETUP_READY"])

    assert report.classification == "NOISE"
    assert "READY_FOR_BROWSER_TESTING=yes" in report.message
    assert "CHECKPOINT_ACCEPTED" not in report.message


def test_controlled_provider_workflow_cannot_skip_to_checkpoint_accepted(hermes_home):
    from hermes_cli.goal_os import GoalOSManager, controlled_provider_status_report

    goal = GoalOSManager().create_goal("Controlled provider status")
    report = controlled_provider_status_report(goal, ["CONTROLLED_SETUP_READY", "CHECKPOINT_ACCEPTED"])

    assert report.classification == "RED"
    assert "may not skip" in report.message.lower()


def test_green_blocked_by_browser_setup_contradiction(hermes_home):
    from hermes_cli.goal_os import GoalOSManager, evaluate_goal_report_readiness

    manager = GoalOSManager()
    goal = manager.create_goal("Contradicted browser checkpoint")
    goal.evidence_log.append({"role": "Verifier Agent", "summary": "tests passed", "acceptance_checked": True, "commands": ["pytest"]})
    goal.evidence_log.append({"role": "Product QA Agent", "summary": "browser still shows provider-safe mode", "contradiction": True})
    manager.save_goal(goal)

    verdict = evaluate_goal_report_readiness(goal, requested_label="GREEN")
    assert verdict.classification == "RED"
    assert "contradiction" in verdict.reason.lower()


def test_noise_for_harmless_wording_difference():
    from hermes_cli.goal_os import classify_report

    assert classify_report("Harmless wording difference in wrapper output") == "NOISE"


def test_approval_status_command_is_registered():
    from hermes_cli.commands import COMMAND_REGISTRY

    names = {cmd.name for cmd in COMMAND_REGISTRY}
    assert "approval-status" in names


def test_green_for_buidl_goal_requires_product_qa_and_memory_agent(hermes_home):
    from hermes_cli.goal_os import GoalOSManager, evaluate_goal_report_readiness

    manager = GoalOSManager()
    goal = manager.create_goal(
        "Buidl backend checkpoint",
        business_outcome="Advance Buidl safely",
    )
    goal.evidence_log.append({"role": "Verifier Agent", "acceptance_checked": True, "summary": "tests/build passed"})
    verdict = evaluate_goal_report_readiness(goal, requested_label="GREEN")
    assert verdict.classification == "RED"
    assert "Product QA evidence" in verdict.reason
    assert "Memory Agent evidence" in verdict.reason

    goal.evidence_log.append({"role": "Product QA Agent", "summary": "business goal checked"})
    goal.evidence_log.append({"role": "Memory Agent", "summary": "durable result recorded"})
    goal.evidence_log.extend([
        {
            "role": "Claude Code Builder Agent",
            "executor": "claude-code",
            "claude_code_builder_pass": True,
            "changed_files": ["hermes_cli/goal_os.py"],
            "commands": ["python -m pytest tests/test_buidl_goal_os.py -q"],
        },
        {
            "role": "Codex Reviewer Agent",
            "executor": "openai-codex",
            "codex_reviewer_pass": True,
            "files_inspected": ["hermes_cli/goal_os.py"],
            "independent_findings": ["lane ownership checked"],
            "safety_review": "No secrets printed.",
        },
        {
            "role": "Codex Verifier Agent",
            "executor": "openai-codex",
            "codex_verifier_pass": True,
            "files_inspected": ["hermes_cli/goal_os.py"],
            "commands": ["python -m pytest tests/test_buidl_goal_os.py -q", "git diff --check"],
            "acceptance_checked": True,
        },
    ])
    assert evaluate_goal_report_readiness(goal, requested_label="GREEN").classification == "GREEN"
