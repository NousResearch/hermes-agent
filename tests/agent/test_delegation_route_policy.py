"""Contract tests for the short-denylist delegation route policy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agent.delegation_route_policy import RouteDecision, decide_delegation_route


ENABLED_CONFIG = {
    "enabled": True,
    "profiles": ["default"],
    "default_route": "gemini",
    "default_data_classification": "standard",
}


@pytest.mark.parametrize(
    "task",
    [
        {"goal": "Condense these meeting notes."},
        {"goal": "Draft a Python function that parses this record."},
        {"goal": "Draft the deployment command for parent review."},
        {"goal": "Deploy this service now."},
        {"goal": 'Preserve this quoted source unchanged: "deploy on Friday".'},
        {"goal": "Extract the headings.", "route": "auto"},
        {"goal": "Return a JSON summary.", "route": "gemini"},
    ],
    ids=[
        "ordinary-text",
        "code-draft",
        "deployment-command-draft",
        "live-action-prose-is-output-only",
        "quoted-deploy-word",
        "explicit-auto",
        "explicit-gemini",
    ],
)
def test_eligible_leaf_work_routes_to_gemini_without_prose_classification(task):
    decision = decide_delegation_route(
        task=task,
        role="worker",
        profile="default",
        config=ENABLED_CONFIG,
    )

    assert decision.route == "gemini"
    assert decision.eligible_for_daily_review is True


@pytest.mark.parametrize(
    ("task", "role", "profile", "config", "reason_fragment"),
    [
        ({"goal": "Draft a summary."}, "orchestrator", "default", ENABLED_CONFIG, "orchestrator"),
        ({"goal": "Draft a summary.", "route": "sol"}, "worker", "default", ENABLED_CONFIG, "Sol"),
        (
            {"goal": "Draft a summary.", "data_classification": "restricted"},
            "worker",
            "default",
            ENABLED_CONFIG,
            "restricted",
        ),
        ({"goal": "Draft a summary."}, "worker", "client", ENABLED_CONFIG, "profile"),
        (
            {"goal": "Draft a summary."},
            "worker",
            "default",
            {**ENABLED_CONFIG, "enabled": False},
            "disabled",
        ),
    ],
    ids=["orchestrator", "explicit-sol", "restricted-data", "profile-excluded", "disabled"],
)
def test_hard_exclusions_route_to_sol(task, role, profile, config, reason_fragment):
    decision = decide_delegation_route(
        task=task,
        role=role,
        profile=profile,
        config=config,
    )

    assert decision.route == "sol"
    assert decision.eligible_for_daily_review is False
    assert reason_fragment.lower() in decision.reason.lower()


@pytest.mark.parametrize(
    ("task", "role", "profile", "config", "reason_fragment"),
    [
        ({"goal": "Draft a summary.", "route": "gemini"}, "orchestrator", "default", ENABLED_CONFIG, "orchestrator"),
        (
            {
                "goal": "Draft a summary.",
                "route": "gemini",
                "data_classification": "restricted",
            },
            "worker",
            "default",
            ENABLED_CONFIG,
            "restricted",
        ),
        ({"goal": "Draft a summary.", "route": "gemini"}, "worker", "client", ENABLED_CONFIG, "profile"),
    ],
    ids=["orchestrator", "restricted-data", "profile-excluded"],
)
def test_explicit_gemini_request_reports_denial_when_a_hard_exclusion_applies(
    task, role, profile, config, reason_fragment
):
    decision = decide_delegation_route(
        task=task,
        role=role,
        profile=profile,
        config=config,
    )

    assert decision.route == "sol"
    assert decision.eligible_for_daily_review is False
    assert "gemini" in decision.reason.lower()
    assert "denied" in decision.reason.lower()
    assert reason_fragment.lower() in decision.reason.lower()


def test_task_defaults_come_from_policy_config():
    decision = decide_delegation_route(
        task={"goal": "Draft a summary."},
        role="worker",
        profile="default",
        config={
            **ENABLED_CONFIG,
            "default_route": "sol",
            "default_data_classification": "standard",
        },
    )

    assert decision.route == "sol"
    assert "Sol" in decision.reason


def test_route_decision_is_frozen():
    decision = RouteDecision(
        route="gemini",
        reason="eligible leaf delegation",
        eligible_for_daily_review=True,
    )

    with pytest.raises(FrozenInstanceError):
        decision.route = "sol"  # type: ignore[misc]
