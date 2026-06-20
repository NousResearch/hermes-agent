from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


def test_goal_creates_durable_goal_contract(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    manager = GoalOSManager()
    report = manager.handle_command("goal", "Ship Buidl MVP", target_repo="/tmp/buidl", target_branch="ai/mvp")

    assert report.classification == "GREEN"
    assert report.goal is not None
    assert report.goal.title == "Ship Buidl MVP"
    assert report.goal.status == "pursuing"
    assert report.goal.target_repo == "/tmp/buidl"
    assert report.goal.target_branch == "ai/mvp"
    assert report.goal.cards
    assert {card.owner_role for card in report.goal.cards} >= {"Builder Agent", "Reviewer Agent", "Verifier Agent"}

    stored = json.loads((hermes_home / "goal-os" / "goals.json").read_text())
    assert report.goal.goal_id in stored["goals"]
    assert stored["goals"][report.goal.goal_id]["title"] == "Ship Buidl MVP"


def test_goal_contract_schema_includes_required_fields(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    goal = GoalOSManager().create_goal("Implement safe agent server change")
    data = goal.to_dict()
    required = {
        "goal_id", "title", "business_outcome", "target_repo", "target_branch",
        "target_environment", "allowed_actions", "forbidden_actions", "approval_gates",
        "acceptance_criteria", "verification_commands", "stop_conditions", "rollback_plan",
        "cards", "blockers", "status", "next_action", "evidence_log",
    }
    assert required <= set(data)

    card_data = data["cards"][0]
    card_required = {
        "card_id", "goal_id", "title", "owner_role", "status", "branch",
        "files_expected", "acceptance_criteria", "verification_commands", "blocker_reason",
        "evidence", "next_action",
    }
    assert card_required <= set(card_data)


def test_command_registry_exposes_goal_os_commands():
    from hermes_cli.commands import COMMAND_REGISTRY

    names = {cmd.name for cmd in COMMAND_REGISTRY}
    assert {"goal", "status", "blockers", "approve", "stop", "resume", "review", "ship"} <= names


def test_status_reports_active_goals_and_next_actions(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    manager = GoalOSManager()
    manager.create_goal("Add durable orchestration")
    report = manager.handle_command("status", "")

    assert report.classification == "GREEN"
    assert "Add durable orchestration" in report.message
    assert "Next action" in report.message


def test_blockers_lists_true_blockers_only(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    manager = GoalOSManager()
    goal = manager.create_goal("Fix product QA")
    manager.add_blocker(goal.goal_id, "DB migration approval needed", true_blocker=True)
    manager.add_blocker(goal.goal_id, "Harmless wording difference", true_blocker=False)

    report = manager.handle_command("blockers", "")
    assert report.classification == "RED"
    assert "DB migration approval needed" in report.message
    assert "Harmless wording difference" not in report.message


def test_approve_records_approval_gate(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    manager = GoalOSManager()
    goal = manager.create_goal("Prepare release")
    report = manager.handle_command("approve", f"{goal.goal_id} production deploy")

    assert report.classification == "GREEN"
    reloaded = GoalOSManager().get_goal(goal.goal_id)
    assert reloaded is not None
    assert any(item["gate"] == "production deploy" for item in reloaded.evidence_log)


def test_stop_pauses_and_resume_resumes_goal(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    manager = GoalOSManager()
    goal = manager.create_goal("Run safe loop")

    stopped = manager.handle_command("stop", goal.goal_id)
    assert stopped.classification == "GREEN"
    paused_goal = GoalOSManager().get_goal(goal.goal_id)
    assert paused_goal is not None
    assert paused_goal.status == "paused"

    resumed = manager.handle_command("resume", goal.goal_id)
    assert resumed.classification == "GREEN"
    resumed_goal = GoalOSManager().get_goal(goal.goal_id)
    assert resumed_goal is not None
    assert resumed_goal.status == "pursuing"


def test_review_shows_cards_waiting_for_review(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    manager = GoalOSManager()
    goal = manager.create_goal("Review changes")
    card = goal.cards[0]
    card.status = "review"
    manager.save_goal(goal)

    report = manager.handle_command("review", "")
    assert report.classification == "GREEN"
    assert card.title in report.message


def test_ship_refuses_without_verifier_evidence(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    manager = GoalOSManager()
    goal = manager.create_goal("Ship without evidence")
    report = manager.handle_command("ship", goal.goal_id)

    assert report.classification == "RED"
    assert "verifier evidence" in report.message.lower()
    blocked_goal = GoalOSManager().get_goal(goal.goal_id)
    assert blocked_goal is not None
    assert blocked_goal.status != "achieved"


def test_ship_requires_verifier_evidence_not_builder_self_report(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    manager = GoalOSManager()
    goal = manager.create_goal("Builder claims done")
    builder_card = goal.cards[0]
    builder_card.owner_role = "Builder Agent"
    builder_card.status = "done"
    builder_card.evidence.append({"role": "Builder Agent", "summary": "done"})
    manager.save_goal(goal)

    report = manager.handle_command("ship", goal.goal_id)
    assert report.classification == "RED"
    assert "Verifier Agent" in report.message


def test_ship_marks_achieved_with_verifier_evidence(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    manager = GoalOSManager()
    goal = manager.create_goal("Verified ship")
    for card in goal.cards:
        card.status = "done"
    goal.evidence_log.append({"role": "Verifier Agent", "summary": "tests passed", "commands": ["pytest"]})
    manager.save_goal(goal)

    report = manager.handle_command("ship", goal.goal_id)
    assert report.classification == "GREEN"
    shipped_goal = GoalOSManager().get_goal(goal.goal_id)
    assert shipped_goal is not None
    assert shipped_goal.status == "achieved"


def test_hard_gates_cannot_be_bypassed(hermes_home):
    from hermes_cli.goal_os import GoalOSManager, is_hard_gate

    manager = GoalOSManager()
    report = manager.handle_command("goal", "Deploy production and run provider prompt")
    assert report.goal is not None
    assert report.goal.status == "blocked"
    assert report.classification == "RED"
    assert is_hard_gate("production deploy")
    assert is_hard_gate("real provider calls")
    assert is_hard_gate("real prompt execution")


def test_blind_prompt_cannot_be_stored_or_printed(hermes_home):
    from hermes_cli.goal_os import BLIND_PROMPT_PLACEHOLDER, GoalOSManager

    live_prompt = "known live prompt: make a hotel landing page with secret phrase"
    manager = GoalOSManager()
    report = manager.handle_command("goal", f"Provider test using {live_prompt}")

    raw = (hermes_home / "goal-os" / "goals.json").read_text()
    assert live_prompt not in raw
    assert live_prompt not in report.message
    assert BLIND_PROMPT_PLACEHOLDER in raw
    assert BLIND_PROMPT_PLACEHOLDER in report.message


def test_niko_is_not_asked_for_terminal_or_credentials_in_safe_tasks(hermes_home):
    from hermes_cli.goal_os import GoalOSManager

    report = GoalOSManager().handle_command("goal", "Implement repo-safe tests and commit branch")
    assert report.goal is not None
    text = json.dumps(report.goal.to_dict()) + report.message
    forbidden = ["ask Niko for terminal", "ask Niko for credentials", "ask Niko for sudo", "provider keys"]
    assert not any(item.lower() in text.lower() for item in forbidden)


def test_report_classifier_green_red_noise():
    from hermes_cli.goal_os import classify_report, hard_gates_in_text

    assert classify_report("proceed with evidence") == "GREEN"
    assert classify_report("blocked by production deploy approval") == "RED"
    assert classify_report("harmless wording difference in label") == "NOISE"
    assert hard_gates_in_text("deploy production") == ["production deploy"]
    assert hard_gates_in_text(
        "Verify safe mode without running providers, prompts, production, DNS, DB migrations, billing, credits, payments, images or worker actions."
    ) == []


def test_no_known_live_prompt_or_provider_keys_embedded(hermes_home):
    root = Path(__file__).resolve().parents[1]
    combined = (root / "hermes_cli" / "goal_os.py").read_text(errors="ignore")
    credential_prefix = "sk" + "-ant" + "-"
    assert credential_prefix not in combined
    assert "ANTHROPIC" + "_API_KEY=" not in combined
    assert "CLAUDE" + "-FABLE-5.md" not in combined


def test_model_config_remains_env_driven():
    from hermes_cli.clio_profile import resolve_clio_anthropic_model

    assert resolve_clio_anthropic_model({}, env={"CLIO_ANTHROPIC_MODEL": "claude-fable-5"}) == "claude-fable-5"
    assert resolve_clio_anthropic_model({"clio": {"anthropic_model": "claude-sonnet-4"}}, env={}) == "claude-sonnet-4"
