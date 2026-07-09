"""Compact finalization prompt helpers for recoverable closeout states."""

from __future__ import annotations

from typing import Any, Mapping


FINALIZATION_RECOMMENDED_ACTION = "compact_finalization_prompt"


def _clean_text(value: Any, *, max_chars: int = 2_000) -> str:
    from hermes_cli.closure_artifacts import _clean_text as _closure_clean_text

    return _closure_clean_text(value, max_chars=max_chars)


def _clean_list(values: Any, *, max_items: int = 12, max_chars: int = 300) -> list[str]:
    from hermes_cli.closure_artifacts import _clean_list as _closure_clean_list

    return _closure_clean_list(values, max_items=max_items, max_chars=max_chars)


def _first_text(values: list[str], fallback: str) -> str:
    return values[0] if values else fallback


def _section(title: str, values: list[str], fallback: str) -> list[str]:
    lines = [f"{title}:"]
    items = values or [fallback]
    lines.extend(f"- {item}" for item in items)
    return lines


def _clean_contract_checklist(values: Any) -> list[dict[str, str]]:
    if not isinstance(values, list):
        return []
    cleaned: list[dict[str, str]] = []
    for item in values[:12]:
        if not isinstance(item, Mapping):
            continue
        row = {
            "requirement": _clean_text(item.get("requirement") or "", max_chars=220),
            "status": _clean_text(item.get("status") or "", max_chars=80),
            "evidence": _clean_text(item.get("evidence") or "", max_chars=220),
            "residual_risk": _clean_text(item.get("residual_risk") or "", max_chars=220),
            "next_action": _clean_text(item.get("next_action") or "", max_chars=220),
        }
        if any(row.values()):
            cleaned.append(row)
    return cleaned


def _contract_checklist_section(values: list[dict[str, str]]) -> list[str]:
    lines = ["Requirement checklist:"]
    if not values:
        lines.append(
            "- No stored checklist yet; create one only when the task contract has multiple explicit requirements."
        )
        return lines
    for item in values:
        line = (
            f"- requirement: {item.get('requirement') or '(missing)'}; "
            f"status: {item.get('status') or '(missing)'}; "
            f"evidence: {item.get('evidence') or '(missing)'}; "
            f"residual risk: {item.get('residual_risk') or '(unspecified)'}"
        )
        if item.get("next_action"):
            line += f"; next action: {item['next_action']}"
        lines.append(line)
    return lines


def build_compact_finalization_prompt(
    packet: Mapping[str, Any],
    *,
    required_model: str | None = None,
) -> str:
    """Build a bounded prompt for final/blocked answers from closeout state.

    The prompt is intentionally allowlist-based: only compact packet fields used
    for finalization are read, so raw logs, queued prompts, and transcript
    excerpts cannot leak merely because they exist in the packet.
    """

    session_id = _clean_text(packet.get("session_id") or "unknown-session", max_chars=160)
    latest_session_id = _clean_text(
        packet.get("latest_session_id") or session_id,
        max_chars=160,
    )
    task_contract = _clean_text(
        packet.get("task_contract") or packet.get("task_id") or "unknown task contract",
        max_chars=900,
    )
    verified_artifacts = _clean_list(
        packet.get("verified_artifacts") or packet.get("changed_files"),
        max_items=12,
        max_chars=260,
    )
    tests_run = _clean_list(packet.get("tests_run"), max_items=12, max_chars=260)
    blockers = _clean_list(
        packet.get("blockers")
        or packet.get("failing_tests")
        or packet.get("remaining_closeout_tasks")
        or packet.get("remaining_checklist"),
        max_items=12,
        max_chars=260,
    )
    contract_checklist = _clean_contract_checklist(packet.get("contract_checklist"))
    next_safe_action = _clean_text(
        packet.get("next_safe_action")
        or _first_text(
            _clean_list(
                packet.get("remaining_closeout_tasks") or packet.get("remaining_checklist"),
                max_items=1,
                max_chars=260,
            ),
            "inspect current state and return a compact final or blocked answer",
        ),
        max_chars=500,
    )
    model = _clean_text(required_model or packet.get("required_model") or "gpt-5.5", max_chars=80)

    lines: list[str] = [
        "COMPACT FINALIZATION PROMPT",
        f"Session: {session_id}",
        f"Latest child/session to resume: {latest_session_id}",
        f"Required model: {model}",
        "",
        "Task contract:",
        f"- {task_contract}",
        "",
        *_contract_checklist_section(contract_checklist),
        "",
        *_section("Verified artifacts", verified_artifacts, "verify current repo/test state first"),
        "",
        *_section("Tests run", tests_run, "run the focused verification required by the task"),
        "",
        *_section("Known blockers or incompletes", blockers, "no known blocker recorded"),
        "",
        "Next safe action:",
        f"- {next_safe_action}",
        "",
        "Operating rules:",
        "- Use this compact packet plus current repo, DB, and test state only.",
        "- Do not replay broad history or load stale assistant/tool transcripts.",
        "- Do not include raw logs, secrets, queued prompt text, private URLs, provider payloads, or giant transcript excerpts.",
        f"- Keep recovery and finalization on {model}; do not suggest a lower-model fallback.",
        "- When the task contract has multiple explicit requirements, include a compact Requirement checklist with requirement, status, evidence, residual risk, and next action when not done/not_applicable.",
        "- Treat partial, blocked, or not_started requirements as incomplete unless the operator explicitly accepted that outcome.",
        "- Simple tasks without explicit multi-item requirements should stay concise; do not invent bureaucratic closeout.",
        "- Use chat --query-file for long Windows prompts.",
        "- If completion is still unverified, give a blocked answer with the exact missing verification.",
    ]
    return "\n".join(lines).strip() + "\n"
