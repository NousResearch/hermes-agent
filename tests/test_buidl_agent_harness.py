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


def test_capability_registry_loads_with_safe_defaults():
    from hermes_cli.capabilities import load_registry

    registry = load_registry()
    assert registry.ecc_commit == "34faa39bd3cd496a0aece0245f2b7e38b7923abc"
    assert registry.capabilities
    assert all(cap.status in {"disabled", "review", "approved", "active"} for cap in registry.capabilities)
    assert not any(cap.status == "active" and cap.risk in {"medium", "high"} for cap in registry.capabilities)
    assert all(cap.source in {"buidl-native", "ecc-inspired", "custom"} for cap in registry.capabilities)


def test_curated_buidl_agent_team_exists_and_is_config_only():
    from hermes_cli.capabilities import load_registry

    agents = {cap.name: cap for cap in load_registry().by_type("agent")}
    expected = {
        "Clio Orchestrator",
        "Planner Agent",
        "Architect Agent",
        "Builder Agent",
        "Code Reviewer Agent",
        "Security Reviewer Agent",
        "Test Runner Agent",
        "Build Error Resolver",
        "Ops/Staging Agent",
        "Product QA Agent",
        "Memory Curator",
        "Verifier Agent",
    }
    assert expected <= set(agents)
    assert all(agents[name].status in {"approved", "review"} for name in expected)
    assert all("invoke separate llm" not in agents[name].description.lower() for name in expected)


def test_dangerous_hooks_default_to_review_or_disabled():
    from hermes_cli.capabilities import load_registry

    hooks = {cap.name: cap for cap in load_registry().by_type("hook")}
    expected = {
        "no-niko-terminal-work",
        "no-niko-credentials",
        "no-live-blind-prompt-storage",
        "provider-call-approval-gate",
        "image-generation-approval-gate",
        "production-domain-db-money-gate",
        "no-caddy-hash-printing",
        "no-env-value-printing",
        "obsidian-partial-read-write-guard",
        "no-local-gym-regression",
        "no-fake-generated-preview",
        "staging-apply-marker-verification",
    }
    assert expected <= set(hooks)
    assert all(hooks[name].status in {"disabled", "review"} for name in expected)


def test_buidl_commands_exist_and_connect_to_goal_os():
    from hermes_cli.capabilities import load_registry
    from hermes_cli.commands import COMMAND_REGISTRY
    from hermes_cli.goal_os import GoalOSManager

    expected = {"goal", "status", "blockers", "plan", "execute", "review", "verify", "fix-ci", "ship", "learn", "checkpoint"}
    registry_names = {cap.name for cap in load_registry().by_type("command")}
    command_names = {cmd.name for cmd in COMMAND_REGISTRY}
    assert expected <= registry_names
    assert expected <= command_names

    manager = GoalOSManager()
    manager.handle_command("goal", "Build harness")
    for command in expected:
        report = manager.handle_command(command, "safe session summary" if command == "learn" else "")
        assert report.classification in {"GREEN", "RED", "NOISE"}


def test_learn_creates_sanitized_lesson_candidate_without_secrets(hermes_home):
    from hermes_cli.capabilities.learning import LearningManager
    from hermes_cli.goal_os import BLIND_PROMPT_PLACEHOLDER

    manager = LearningManager()
    candidate = manager.create_candidate(
        "Session found live blind prompt and token=abc123secret456 while fixing provider gates."
    )
    assert candidate.status == "review"
    assert BLIND_PROMPT_PLACEHOLDER in candidate.summary
    assert "abc123secret456" not in candidate.summary
    stored = json.loads((hermes_home / "capabilities" / "memory" / "lesson-candidates.json").read_text())
    assert candidate.candidate_id in stored["candidates"]


def test_guard_hooks_block_repeated_failure_patterns():
    from hermes_cli.capabilities.guards import evaluate_buidl_guardrails

    blocked_cases = [
        "Niko, please run this terminal command for me",
        "Send me the provider credentials",
        "Store this live blind prompt in a fixture",
        "Call the provider without approval",
        "Generate an image now",
        "Deploy production and change DNS",
        "Run DB migrations and update billing credits payments",
        "Print the Caddy hash and env value",
    ]
    for case in blocked_cases:
        result = evaluate_buidl_guardrails(case)
        assert not result.allowed, case
        assert result.classification == "RED"

    assert evaluate_buidl_guardrails("Harmless wording difference in status copy").classification == "NOISE"


def test_blind_prompt_cannot_be_stored_in_goal_or_learning(hermes_home):
    from hermes_cli.capabilities.learning import LearningManager
    from hermes_cli.goal_os import BLIND_PROMPT_PLACEHOLDER, GoalOSManager

    goal_report = GoalOSManager().handle_command("goal", "known live prompt: do the secret hidden test")
    lesson = LearningManager().create_candidate("known live prompt: do the secret hidden test")
    assert goal_report.goal is not None
    assert goal_report.goal.title == BLIND_PROMPT_PLACEHOLDER
    assert lesson.summary == BLIND_PROMPT_PLACEHOLDER


def test_all_imported_capabilities_are_allowlisted_and_not_wholesale_ecc():
    from hermes_cli.capabilities import load_registry

    registry = load_registry()
    assert registry.context_bloat_limit == 64
    assert len(registry.capabilities) <= registry.context_bloat_limit
    for cap in registry.capabilities:
        assert cap.source in {"buidl-native", "ecc-inspired", "custom"}
        assert cap.ecc_reference is None or cap.ecc_reference.startswith(("agents/", "skills/", "commands/", "hooks/", "rules/"))
        assert not cap.wholesale_copied


def test_context_bloat_guard_rejects_loading_everything():
    from hermes_cli.capabilities import load_registry

    registry = load_registry()
    with pytest.raises(ValueError, match="context bloat"):
        registry.select_context_bundle(limit=999, include_all=True)


def test_ecc_audit_inventory_is_pinned_and_curated():
    from hermes_cli.capabilities import load_ecc_audit_inventory

    inventory = load_ecc_audit_inventory()
    assert inventory["commit"] == "34faa39bd3cd496a0aece0245f2b7e38b7923abc"
    assert inventory["context_bloat_risks"]["skills"] >= 200
    assert "wholesale install" in inventory["rejected_for_buidl"]
    assert "hook review-mode pattern" in inventory["recommended_for_buidl"]
