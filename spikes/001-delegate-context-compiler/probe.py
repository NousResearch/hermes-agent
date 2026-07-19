#!/usr/bin/env python3
"""Throwaway deterministic probe for delegation context projection.

This file deliberately does not import Hermes production code or call an LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FIXTURES_PATH = ROOT / "fixtures.json"
TASK_HEADER = "=== TASK ===\n"
TRUNCATION_MARKER = "\n...[deterministic head/tail truncation]...\n"
OOB_OPEN = (
    "[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered "
    "mid-turn; not tool output]"
)
OOB_CLOSE = "[/OUT-OF-BAND USER MESSAGE]"


def expand_fixture(value: Any) -> Any:
    if isinstance(value, list):
        return [expand_fixture(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"text", "count"}:
            return str(value["text"]) * int(value["count"])
        expanded = {key: expand_fixture(item) for key, item in value.items()}
        if "content_repeat" in expanded:
            expanded["content"] = expanded.pop("content_repeat")
        return expanded
    return value


def task_block(goal: str) -> str:
    return f"{TASK_HEADER}{goal}"


def extract_oob(content: str) -> str:
    opening = f"{OOB_OPEN}\n"
    closing = f"\n{OOB_CLOSE}"
    if not content.startswith(opening) or not content.endswith(closing):
        return content
    body = content[len(opening) : -len(closing)]
    return f"[OOB user]\n{body}"


def textual_messages(
    messages: list[dict[str, Any]],
    *,
    allowed_roles: set[str] | None,
    process_oob: bool = True,
) -> list[str]:
    rendered: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        if allowed_roles is not None and role not in allowed_roles:
            continue
        if role == "user" and process_oob:
            content = extract_oob(content)
        rendered.append(f"[{role}]\n{content}")
    return rendered


def fit_context(context: str, available: int) -> str:
    if available <= 0:
        return ""
    if len(context) <= available:
        return context
    if available <= len(TRUNCATION_MARKER):
        return context[:available]
    payload = available - len(TRUNCATION_MARKER)
    head = (payload + 1) // 2
    tail = payload - head
    return context[:head] + TRUNCATION_MARKER + (context[-tail:] if tail else "")


def compile_output(context_parts: list[str], goal: str, max_chars: int) -> str:
    task = task_block(goal)
    if len(task) > max_chars:
        raise ValueError("task block alone exceeds max_chars")
    context = "\n\n".join(part for part in context_parts if part)
    if not context:
        return task
    separator = "\n\n"
    available = max_chars - len(task) - len(separator)
    fitted = fit_context(context, available)
    return f"{fitted}{separator}{task}" if fitted else task


def compile_prioritized_output(
    explicit: str | None,
    projected_parts: list[str],
    goal: str,
    max_chars: int,
) -> str:
    """Reserve context budget for explicit text before optional projection."""
    task = task_block(goal)
    if len(task) > max_chars:
        raise ValueError("task block alone exceeds max_chars")
    outer_separator = "\n\n"
    available = max_chars - len(task) - len(outer_separator)
    if available <= 0:
        return task

    explicit_text = explicit if isinstance(explicit, str) and explicit else ""
    if len(explicit_text) >= available:
        context = fit_context(explicit_text, available)
        return f"{context}{outer_separator}{task}" if context else task

    context = explicit_text
    projection = "\n\n".join(part for part in projected_parts if part)
    if projection:
        inner_separator = "\n\n" if context else ""
        projection_budget = available - len(context) - len(inner_separator)
        fitted_projection = fit_context(projection, projection_budget)
        if fitted_projection:
            context = f"{context}{inner_separator}{fitted_projection}"
    return f"{context}{outer_separator}{task}" if context else task


def explicit_output(fixture: dict[str, Any]) -> str:
    explicit = fixture.get("explicit_context")
    parts = [explicit] if isinstance(explicit, str) and explicit else []
    return compile_output(parts, fixture["goal"], int(fixture["max_chars"]))


def projection_output(fixture: dict[str, Any]) -> str:
    explicit = fixture.get("explicit_context")
    projected_parts: list[str] = []
    if fixture.get("projection_enabled"):
        eligible = textual_messages(fixture["parent_messages"], allowed_roles={"user", "assistant"})
        recent = eligible[-int(fixture["recent_turns"]):]
        projected_parts.extend(recent)
    return compile_prioritized_output(
        explicit if isinstance(explicit, str) else None,
        projected_parts,
        fixture["goal"],
        int(fixture["max_chars"]),
    )


def naive_output(fixture: dict[str, Any]) -> str:
    parts: list[str] = []
    explicit = fixture.get("explicit_context")
    if isinstance(explicit, str) and explicit:
        parts.append(explicit)
    parts.extend(
        textual_messages(
            fixture["parent_messages"], allowed_roles=None, process_oob=False
        )
    )
    parts.append(task_block(fixture["goal"]))
    return "\n\n".join(parts)


def marker_recall(output: str, markers: list[str]) -> tuple[int, float]:
    found = sum(marker in output for marker in markers)
    return found, (found / len(markers) if markers else 1.0)


def evaluate(fixture: dict[str, Any]) -> dict[str, Any]:
    naive = naive_output(fixture)
    explicit = explicit_output(fixture)
    projection = projection_output(fixture)
    projection_again = projection_output(fixture)
    required = list(fixture["required_markers"])
    forbidden = list(fixture["forbidden_markers"])
    explicit_found, explicit_rate = marker_recall(explicit, required)
    projection_found, projection_rate = marker_recall(projection, required)
    leaks = [marker for marker in forbidden if marker in projection]
    max_chars = int(fixture["max_chars"])
    expected_task = task_block(fixture["goal"])
    result = {
        "name": fixture["name"],
        "projection_enabled": bool(fixture["projection_enabled"]),
        "naive_chars": len(naive),
        "explicit_chars": len(explicit),
        "projection_chars": len(projection),
        "reduction_vs_naive": round(1 - (len(projection) / len(naive)), 4),
        "explicit_required_found": explicit_found,
        "projection_required_found": projection_found,
        "required_total": len(required),
        "explicit_recall_rate": round(explicit_rate, 4),
        "projection_recall_rate": round(projection_rate, 4),
        "recall_improved": projection_found > explicit_found,
        "forbidden_leaks": leaks,
        "within_budget": len(projection) <= max_chars,
        "task_final": projection.endswith(expected_task),
        "deterministic": projection == projection_again,
    }
    assert result["within_budget"], fixture["name"]
    assert result["task_final"], fixture["name"]
    assert result["deterministic"], fixture["name"]
    assert not leaks, (fixture["name"], leaks)
    if not fixture["projection_enabled"]:
        assert projection == explicit, fixture["name"]
    return result


def verdict(results: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    projected = [result for result in results if result["projection_enabled"]]
    all_reduce = all(result["projection_chars"] < result["naive_chars"] for result in projected)
    zero_leaks = all(not result["forbidden_leaks"] for result in results)
    all_invariants = all(
        result["within_budget"] and result["task_final"] and result["deterministic"]
        for result in results
    )
    useful_improvements = sum(result["recall_improved"] for result in projected)
    harmful_losses = sum(
        result["projection_required_found"] < result["explicit_required_found"]
        for result in projected
    )
    criteria = {
        "all_projection_fixtures_reduce_vs_naive": all_reduce,
        "zero_forbidden_leaks": zero_leaks,
        "all_budget_task_final_deterministic": all_invariants,
        "useful_recall_improvements": useful_improvements,
        "harmful_losses_vs_explicit": harmful_losses,
    }
    if all_reduce and zero_leaks and all_invariants and useful_improvements >= 2 and harmful_losses == 0:
        return "VALIDATED", criteria
    if zero_leaks and all_invariants:
        return "PARTIAL", criteria
    return "INVALIDATED", criteria


def main() -> int:
    fixtures = expand_fixture(json.loads(FIXTURES_PATH.read_text(encoding="utf-8")))
    results = [evaluate(fixture) for fixture in fixtures]
    final_verdict, criteria = verdict(results)
    payload = {"fixtures": results, "criteria": criteria, "verdict": final_verdict}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
