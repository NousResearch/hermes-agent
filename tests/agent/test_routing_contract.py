"""Regression tests for visible Routing Decision contract enforcement."""

from __future__ import annotations

import json

import pytest

from agent.routing_contract import check_route_contract


@pytest.fixture(autouse=True)
def model_routing_table(monkeypatch):
    from tools import kanban_tools

    table = {
        "task_lanes": {
            "architecture_design": {
                "provider": "openai-codex",
                "model": "gpt-5.5",
                "reasoning_effort": "xhigh",
                "route_key": "openai-codex/gpt-5.5",
            },
            "verification_leaf": {
                "provider": "nous",
                "model": "deepseek/deepseek-v4-flash:free",
                "reasoning_effort": "low",
                "route_key": "nous/deepseek/deepseek-v4-flash:free",
            },
        }
    }
    monkeypatch.setattr(
        kanban_tools,
        "_load_model_routing_table",
        lambda: (table, "test-routing-table"),
    )


def _check(text: str, tool_calls=()):
    return check_route_contract(
        text=text,
        tool_calls=tool_calls,
        active_provider="openai-codex",
        active_model="gpt-5.5",
        active_effort="high",
    )


def _tool_call(name: str, args: dict) -> dict:
    return {"function": {"name": name, "arguments": json.dumps(args)}}


def test_architecture_label_cannot_downgrade_to_high():
    check = _check(
        "Routing Decision: config check -> hermes-agent -> inline(read-only) "
        "-> architecture_design/gpt-5.5-high"
    )

    assert check.violation is not None
    assert check.violation.code == "lane_label_effort_mismatch"


def test_architecture_xhigh_cannot_be_inline_from_high_session():
    check = _check(
        "Routing Decision: config check -> hermes-agent -> inline(read-only) "
        "-> architecture_design/gpt-5.5-xhigh"
    )

    assert check.violation is not None
    assert check.violation.code == "xhigh_lane_not_executed"


def test_architecture_xhigh_is_satisfied_by_matching_delegate_task():
    check = _check(
        "Routing Decision: config check -> hermes-agent -> delegate_task "
        "-> architecture_design/gpt-5.5-xhigh",
        [
            _tool_call(
                "delegate_task",
                {
                    "provider": "openai-codex",
                    "model": "gpt-5.5",
                    "reasoning_effort": "xhigh",
                    "task": "inspect the architecture",
                },
            )
        ],
    )

    assert check.violation is None
    assert check.execution_surface == "delegate_task"


def test_front_door_high_label_is_allowed_for_inline_current_session():
    check = _check(
        "Routing Decision: quick check -> current profile -> inline(read-only) "
        "-> front_door/gpt-5.5-high"
    )

    assert check.violation is None
    assert check.execution_surface == "active_session"


def test_front_door_label_ignores_sentence_punctuation():
    check = _check(
        "Routing Decision: quick check -> current profile -> inline(read-only) "
        "-> front_door/gpt-5.5-high."
    )

    assert check.violation is None
    assert check.execution_surface == "active_session"


def test_front_door_label_ignores_parenthetical_note():
    check = _check(
        "Routing Decision: quick check -> current profile -> inline(read-only) "
        "-> front_door/openai-codex/gpt-5.5-high (quota guard inactive)."
    )

    assert check.violation is None
    assert check.execution_surface == "active_session"


def test_kanban_model_routing_satisfies_architecture_xhigh_lane():
    check = _check(
        "Routing Decision: architecture work -> kanban-orchestrator -> kanban_create "
        "-> architecture_design/gpt-5.5-xhigh",
        [_tool_call("kanban_create", {"model_routing": "architecture_design"})],
    )

    assert check.violation is None
    assert check.execution_surface == "kanban_create"
