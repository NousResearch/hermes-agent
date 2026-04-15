from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

_ERROR_RE = re.compile(r"\b(error|failed|failure|exception|traceback|timeout|timed out|not found|invalid)\b", re.I)
_CORRECTION_RE = re.compile(
    r"\b(don't|do not|instead|use .* instead|should have|should've|actually|remember|stop)\b",
    re.I,
)
_BLOCKAGE_RE = re.compile(r"\b(sorry|i can't|i cannot|unable to|not available|not supported|failed to)\b", re.I)


def load_trajectory_entries(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL trajectory entries, skipping blank and malformed rows."""
    entries: list[dict[str, Any]] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            entries.append(row)
    return entries


def analyze_sessions(
    sessions: Iterable[dict[str, Any]],
    trajectory_entries: Iterable[dict[str, Any]] | None = None,
    min_count: int = 2,
) -> dict[str, Any]:
    counters: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)

    session_list = list(sessions)
    for session in session_list:
        for message in session.get("messages", []):
            _record_message_findings(message, counters, examples)

    trajectory_list = list(trajectory_entries or [])
    for entry in trajectory_list:
        _record_message_findings(entry, counters, examples)

    findings = []
    for (category, detail), count in sorted(counters.items(), key=lambda item: (-item[1], item[0])):
        if count < min_count:
            continue
        findings.append(
            {
                "key": f"{category}:{detail}",
                "category": category,
                "label": _build_label(category, detail),
                "count": count,
                "examples": examples[(category, detail)][:3],
            }
        )

    recommendations = _build_recommendations(findings)
    prompt_deltas = _build_prompt_deltas(findings)

    return {
        "summary": {
            "sessions_analyzed": len(session_list),
            "trajectory_files": 1 if trajectory_list else 0,
            "findings": len(findings),
        },
        "findings": findings,
        "recommendations": recommendations,
        "prompt_deltas": prompt_deltas,
    }



def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Hermes Self-Evolution Report",
        "",
        f"- Sessions analyzed: {summary.get('sessions_analyzed', 0)}",
        f"- Trajectory files analyzed: {summary.get('trajectory_files', 0)}",
        f"- Repeated findings: {summary.get('findings', 0)}",
        "",
        "## Findings",
    ]

    findings = report.get("findings", [])
    if not findings:
        lines.extend(["", "No repeated patterns met the reporting threshold."])
    else:
        for finding in findings:
            lines.append("")
            lines.append(f"### {finding['label']} ({finding['count']}x)")
            for example in finding.get("examples", []):
                lines.append(f"- {example}")

    lines.extend(["", "## Recommendations"])
    recommendations = report.get("recommendations", [])
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- No recommendations yet.")

    lines.extend(["", "## Candidate Prompt Deltas"])
    prompt_deltas = report.get("prompt_deltas", [])
    if prompt_deltas:
        lines.extend(f"- {item}" for item in prompt_deltas)
    else:
        lines.append("- No prompt delta candidates yet.")

    return "\n".join(lines).rstrip() + "\n"



def _record_message_findings(
    message: dict[str, Any],
    counters: Counter[tuple[str, str]],
    examples: dict[tuple[str, str], list[str]],
) -> None:
    role = (message.get("role") or "").lower()
    content = (message.get("content") or "").strip()
    if role == "tool":
        tool_name = (message.get("tool_name") or "").strip()
        tool_failure = _extract_tool_failure(content, fallback_tool_name=tool_name)
        if tool_failure is not None:
            resolved_tool_name, example_content = tool_failure
            _bump(("tool_failure", resolved_tool_name), example_content, counters, examples)
    elif role == "user" and _is_user_correction(content):
        _bump(("user_correction", "tool_choice"), content, counters, examples)
    elif role == "assistant" and _BLOCKAGE_RE.search(content):
        _bump(("assistant_blockage", "fallback_needed"), content, counters, examples)



def _extract_tool_failure(content: str, fallback_tool_name: str = "") -> tuple[str, str] | None:
    if not content:
        return None

    parsed = _maybe_parse_json_object(content)
    if isinstance(parsed, dict):
        exit_code = parsed.get("exit_code")
        error_value = parsed.get("error")
        success_value = parsed.get("success")
        exit_code_meaning = str(parsed.get("exit_code_meaning") or "")
        if success_value is True and not error_value and exit_code in (0, None):
            return None
        if "not an error" in exit_code_meaning.lower():
            return None
        if success_value is False or error_value or exit_code not in (0, None):
            tool_name = _infer_tool_name(parsed, fallback_tool_name)
            return tool_name, content
        return None

    if not _ERROR_RE.search(content):
        return None

    tool_name = fallback_tool_name or "unknown_tool"
    return tool_name, content



def _infer_tool_name(parsed: dict[str, Any], fallback_tool_name: str = "") -> str:
    explicit_name = str(parsed.get("tool_name") or parsed.get("name") or fallback_tool_name or "").strip()
    if explicit_name:
        return explicit_name
    if "tool_calls_made" in parsed and "duration_seconds" in parsed:
        return "execute_code"
    if "exit_code" in parsed and "output" in parsed:
        return "terminal"
    return "unknown_tool"



def _is_user_correction(content: str) -> bool:
    if not content:
        return False
    if content.lstrip().startswith("[SYSTEM:"):
        return False
    return bool(_CORRECTION_RE.search(content))



def _maybe_parse_json_object(content: str) -> dict[str, Any] | None:
    if not content.startswith("{"):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None



def _bump(
    key: tuple[str, str],
    content: str,
    counters: Counter[tuple[str, str]],
    examples: dict[tuple[str, str], list[str]],
) -> None:
    counters[key] += 1
    if content and len(examples[key]) < 3 and content not in examples[key]:
        examples[key].append(content)



def _build_label(category: str, detail: str) -> str:
    if category == "tool_failure":
        return f"Repeated failures for tool `{detail}`"
    if category == "user_correction":
        return "Repeated user corrections about tool choice"
    if category == "assistant_blockage":
        return "Repeated assistant blockage/apology responses"
    return f"Repeated pattern: {category}:{detail}"



def _build_recommendations(findings: list[dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []
    for finding in findings:
        if finding["category"] == "tool_failure":
            tool_name = finding["key"].split(":", 1)[1]
            recommendations.append(
                f"Investigate `{tool_name}` reliability or strengthen fallback guidance."
            )
        elif finding["category"] == "user_correction":
            recommendations.append(
                "Promote recurring user corrections into durable memory or a skill if the pattern is stable."
            )
        elif finding["category"] == "assistant_blockage":
            recommendations.append(
                "Tighten action-oriented fallback guidance so the assistant switches tools before apologizing."
            )
    return _dedupe(recommendations)



def _build_prompt_deltas(findings: list[dict[str, Any]]) -> list[str]:
    deltas: list[str] = []
    for finding in findings:
        if finding["category"] == "tool_failure":
            tool_name = finding["key"].split(":", 1)[1]
            deltas.append(
                f"When `{tool_name}` fails with URL/schema/runtime errors, immediately fall back to terminal or browser retrieval instead of retrying the same broken path."
            )
        elif finding["category"] == "user_correction":
            deltas.append(
                "When the user corrects tool choice or workflow, prefer saving the stable correction to memory or a skill after finishing the task."
            )
        elif finding["category"] == "assistant_blockage":
            deltas.append(
                "Before saying you cannot do something, try another available tool or explain the concrete blocked capability and the next best fallback."
            )
    return _dedupe(deltas)



def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
