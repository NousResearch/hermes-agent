from __future__ import annotations

import json

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


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
    assert status["last_approval_request_category"] == "safe"
    assert "pytest tests -q" in status["last_approval_request_summary"]
    assert "safe delegated" in status["why_it_asked_niko"].lower() or "auto-approved" in status["why_it_asked_niko"].lower()


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
    assert evaluate_goal_report_readiness(goal, requested_label="GREEN").classification == "GREEN"
