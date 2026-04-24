from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


_COMPLETE_HINTS = (
    "complete",
    "completed",
    "done",
    "finished",
    "implemented",
    "created",
    "added",
    "wrote",
    "written",
    "verified",
    "passed",
)
_BLOCKED_HINTS = (
    "blocked",
    "blocker",
    "cannot",
    "can't",
    "unable",
    "waiting",
    "missing dependency",
    "not found",
    "failed because",
)
_READ_ONLY_CALLER_ROLES = {"", "reviewer", "verifier"}


@dataclass(frozen=True)
class GapCheckResult:
    status: str
    missing_items: list[str]
    blocked_items: list[str]
    remediation_tasks: list[str]
    next_prompt: str | None
    read_only_safe: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_gap(
    plan: Any,
    result: Any,
    evidence: Any,
    *,
    caller_role: str | None = None,
) -> GapCheckResult:
    normalized_role = str(caller_role or "").strip().lower()
    if normalized_role not in _READ_ONLY_CALLER_ROLES:
        normalized_role = ""

    plan_items = _extract_plan_items(plan)
    haystack = _flatten_to_text([result, evidence])
    missing_items: list[str] = []
    blocked_items: list[str] = []

    for item in plan_items:
        if _item_is_blocked(item, result, evidence, haystack):
            blocked_items.append(item)
        elif not _item_is_satisfied(item, result, evidence, haystack):
            missing_items.append(item)

    if blocked_items:
        remediation_tasks = [f"Resolve blocker for plan item: {item}" for item in blocked_items]
        next_prompt = _build_next_prompt(
            prefix="Resolve each blocked item before continuing:",
            items=blocked_items,
        )
        return GapCheckResult(
            status="blocked",
            missing_items=[],
            blocked_items=blocked_items,
            remediation_tasks=remediation_tasks,
            next_prompt=next_prompt,
        )

    if missing_items:
        remediation_tasks = [f"Complete missing plan item: {item}" for item in missing_items]
        next_prompt = _build_next_prompt(
            prefix="Complete the remaining missing items before ending the loop:",
            items=missing_items,
        )
        return GapCheckResult(
            status="missing",
            missing_items=missing_items,
            blocked_items=[],
            remediation_tasks=remediation_tasks,
            next_prompt=next_prompt,
        )

    return GapCheckResult(
        status="complete",
        missing_items=[],
        blocked_items=[],
        remediation_tasks=[],
        next_prompt=None,
    )


def should_skip_next_iteration(gap_result: GapCheckResult | dict[str, Any]) -> bool:
    if isinstance(gap_result, GapCheckResult):
        return gap_result.status == "complete"
    return str((gap_result or {}).get("status") or "").strip().lower() == "complete"


def _extract_plan_items(plan: Any) -> list[str]:
    extracted = _extract_string_items(plan)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in extracted:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _extract_string_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(_extract_string_items(item))
        return items
    if isinstance(value, tuple):
        items: list[str] = []
        for item in value:
            items.extend(_extract_string_items(item))
        return items
    if isinstance(value, dict):
        preferred_keys = (
            "steps",
            "plan",
            "items",
            "tasks",
            "acceptance",
            "checklist",
            "completed",
            "missing",
            "blocked",
            "summary",
            "artifacts",
            "evidence",
            "result",
        )
        items: list[str] = []
        seen_any_preferred = False
        for key in preferred_keys:
            if key in value:
                seen_any_preferred = True
                items.extend(_extract_string_items(value.get(key)))
        if seen_any_preferred:
            return items
        for nested_value in value.values():
            items.extend(_extract_string_items(nested_value))
        return items
    return []


def _flatten_to_text(value: Any) -> str:
    return " ".join(_extract_string_items(value)).casefold()


def _item_is_blocked(item: str, result: Any, evidence: Any, haystack: str) -> bool:
    direct_blocked = _extract_normalized_lookup(_extract_from_mapping_key(result, "blocked"))
    direct_blocked.update(_extract_normalized_lookup(_extract_from_mapping_key(evidence, "blocked")))
    if item.casefold() in direct_blocked:
        return True

    item_text = item.casefold()
    if item_text in haystack and any(hint in haystack for hint in _BLOCKED_HINTS):
        return True
    return False


def _item_is_satisfied(item: str, result: Any, evidence: Any, haystack: str) -> bool:
    direct_completed = _extract_normalized_lookup(_extract_from_mapping_key(result, "completed"))
    direct_completed.update(_extract_normalized_lookup(_extract_from_mapping_key(evidence, "completed")))
    if item.casefold() in direct_completed:
        return True

    item_text = item.casefold()
    if item_text in haystack:
        return True

    tokens = [token for token in item_text.replace("-", " ").split() if len(token) > 2]
    if tokens and all(token in haystack for token in tokens):
        return True

    return any(f"{hint} {item_text}" in haystack or f"{item_text} {hint}" in haystack for hint in _COMPLETE_HINTS)


def _extract_from_mapping_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return None


def _extract_normalized_lookup(items: Any) -> set[str]:
    return {item.casefold() for item in _extract_string_items(items)}


def _build_next_prompt(*, prefix: str, items: list[str]) -> str:
    bullet_list = "\n".join(f"- {item}" for item in items)
    label = "blocked items" if prefix.lower().startswith("resolve") else "missing items"
    return f"{prefix}\n{bullet_list}\nFocus on these {label} and report concrete evidence."
