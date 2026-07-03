"""Deterministic eval buckets for Hermes tool-restraint behavior.

The scorer splits route-packet review into small dimensions so a regression can
say whether the model selected a wrong tool, ignored loaded context, failed to
recover from an error, or simply missed the task.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DIMENSIONS = (
    "tool_selection",
    "argument_extraction",
    "result_utilization",
    "error_recovery",
    "plan_coherence",
    "task_completion",
)
VALID_DIMENSION_LABELS = {"pass", "fail", "not_applicable"}
NO_TOOL_ACTIONS = {
    "no_reply",
    "ask_clarifying_question",
    "refuse",
    "answer_from_loaded_context",
}


@dataclass(frozen=True)
class Fixture:
    id: str
    category: str
    prompt: str
    expected_action: str
    must_not_call_tools: bool = False
    must_call_tool: bool = False
    allowed_tools: tuple[str, ...] = ()
    expected_answer_contains: tuple[str, ...] = ()
    expected_dimensions: Mapping[str, str] | None = None


@dataclass(frozen=True)
class CandidateTrace:
    final_action: str
    tools_called: tuple[str, ...] = ()
    final_text: str = ""
    dimension_labels: Mapping[str, str] | None = None


def load_fixtures(path: Path) -> list[Fixture]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    fixtures: list[Fixture] = []
    for item in raw["fixtures"]:
        fixtures.append(
            Fixture(
                id=item["id"],
                category=item["category"],
                prompt=item["prompt"],
                expected_action=item["expected_action"],
                must_not_call_tools=bool(item.get("must_not_call_tools", False)),
                must_call_tool=bool(item.get("must_call_tool", False)),
                allowed_tools=tuple(item.get("allowed_tools", ())),
                expected_answer_contains=tuple(item.get("expected_answer_contains", ())),
                expected_dimensions=item.get("expected_dimensions"),
            )
        )
    return fixtures


def validate_fixture(fixture: Fixture) -> list[str]:
    errors: list[str] = []
    if not fixture.id:
        errors.append("id is required")
    if fixture.must_not_call_tools and fixture.must_call_tool:
        errors.append("fixture cannot require both must_not_call_tools and must_call_tool")
    if fixture.must_not_call_tools and fixture.expected_action not in NO_TOOL_ACTIONS:
        errors.append("must_not_call_tools fixture must use a no-tool expected_action")
    if fixture.must_call_tool and not fixture.allowed_tools:
        errors.append("must_call_tool fixture must list allowed_tools")
    labels = dict(fixture.expected_dimensions or {})
    missing = [name for name in DIMENSIONS if name not in labels]
    if missing:
        errors.append("missing dimension labels: " + ", ".join(missing))
    invalid = {name: value for name, value in labels.items() if value not in VALID_DIMENSION_LABELS}
    if invalid:
        errors.append("invalid dimension labels: " + repr(invalid))
    return errors


def score_trace(fixture: Fixture, trace: CandidateTrace) -> dict[str, Any]:
    failures: list[str] = []
    labels = {name: "pass" for name in DIMENSIONS}

    if trace.final_action != fixture.expected_action:
        failures.append(f"action expected {fixture.expected_action!r}, got {trace.final_action!r}")
        labels["task_completion"] = "fail"
        labels["plan_coherence"] = "fail"

    if fixture.must_not_call_tools and trace.tools_called:
        failures.append("called tools in a no-tool fixture: " + ", ".join(trace.tools_called))
        labels["tool_selection"] = "fail"
        labels["task_completion"] = "fail"

    if fixture.must_call_tool:
        if not trace.tools_called:
            failures.append("required a tool call but no tool was called")
            labels["tool_selection"] = "fail"
            labels["task_completion"] = "fail"
        elif fixture.allowed_tools and not any(tool in fixture.allowed_tools for tool in trace.tools_called):
            failures.append("called disallowed tool(s): " + ", ".join(trace.tools_called))
            labels["tool_selection"] = "fail"
            labels["task_completion"] = "fail"

    lower_text = trace.final_text.lower()
    for required in fixture.expected_answer_contains:
        if required.lower() not in lower_text:
            failures.append(f"final text missing required phrase: {required!r}")
            labels["result_utilization"] = "fail"
            labels["task_completion"] = "fail"

    if trace.dimension_labels:
        for name, value in trace.dimension_labels.items():
            if name in labels and value in VALID_DIMENSION_LABELS:
                labels[name] = value

    fixture_errors = validate_fixture(fixture)
    return {
        "fixture_id": fixture.id,
        "ok": not failures and not fixture_errors,
        "failures": failures + fixture_errors,
        "dimensions": labels,
    }


def summarize_fixture_set(fixtures: list[Fixture]) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    invalid: dict[str, list[str]] = {}
    for fixture in fixtures:
        by_category[fixture.category] = by_category.get(fixture.category, 0) + 1
        errors = validate_fixture(fixture)
        if errors:
            invalid[fixture.id] = errors
    return {
        "fixture_count": len(fixtures),
        "categories": by_category,
        "invalid": invalid,
        "dimension_count": len(DIMENSIONS),
        "dimensions": list(DIMENSIONS),
    }
