"""Lightweight task feature extraction for smart model routing.

This module is intentionally heuristic-only.  It gives the router a cheap,
explainable first pass before any auxiliary LLM judge is considered.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


_CODE_FILE_RE = re.compile(
    r"(?i)\b[\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|swift|c|cc|cpp|h|hpp|"
    r"cs|rb|php|md|yaml|yml|json|toml|sql|sh|zsh|fish|css|scss|html)\b"
)
_PATH_RE = re.compile(r"(?:^|\s)(?:/[\w./-]+|\.{1,2}/[\w./-]+)")


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


@dataclass(frozen=True)
class TaskFeatures:
    """Router-facing summary of the current user turn."""

    requires_code_edit: bool = False
    likely_needs_file_access: bool = False
    likely_needs_tests: bool = False
    likely_needs_web: bool = False
    architecture_or_analysis: bool = False
    review_or_debug: bool = False
    high_risk_domain: bool = False
    destructive_intent: bool = False
    has_explicit_file_reference: bool = False
    has_multimodal_hint: bool = False
    asks_for_latest: bool = False
    asks_for_plan: bool = False
    complexity_score: int = 0
    signals: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_task_features(user_message: Any, *, message_count: int = 0) -> TaskFeatures:
    """Extract cheap, deterministic routing features from a user request."""

    text = _stringify_user_message(user_message)
    lowered = text.lower()
    signals: list[str] = []

    requires_code_edit = _contains_any(
        lowered,
        (
            "fix",
            "implement",
            "add",
            "modify",
            "patch",
            "refactor",
            "delete",
            "remove",
            "update the code",
            "\u4fee\u590d",
            "\u5b9e\u73b0",
            "\u4fee\u6539",
            "\u91cd\u6784",
        ),
    )
    if requires_code_edit:
        signals.append("code_edit_terms")

    has_explicit_file_reference = bool(_CODE_FILE_RE.search(text) or _PATH_RE.search(text))
    if has_explicit_file_reference:
        signals.append("file_or_path_reference")

    likely_needs_file_access = has_explicit_file_reference or _contains_any(
        lowered,
        (
            "repo",
            "codebase",
            "project",
            "branch",
            "module",
            "\u9879\u76ee",
            "\u4ee3\u7801\u5e93",
            "\u6587\u4ef6",
            "\u6a21\u5757",
        ),
    )
    if likely_needs_file_access:
        signals.append("file_access_likely")

    likely_needs_tests = _contains_any(
        lowered,
        ("test", "pytest", "vitest", "unit test", "e2e", "\u6d4b\u8bd5"),
    )
    if likely_needs_tests:
        signals.append("tests_likely")

    asks_for_latest = _contains_any(
        lowered,
        (
            "latest",
            "today",
            "current",
            "recent",
            "newest",
            "\u6700\u65b0",
            "\u4eca\u5929",
            "\u5f53\u524d",
            "\u8fd1\u671f",
        ),
    )
    likely_needs_web = asks_for_latest or _contains_any(
        lowered,
        ("search", "browse", "look up", "verify online", "\u641c", "\u67e5\u4e00\u4e0b"),
    )
    if likely_needs_web:
        signals.append("web_or_freshness_likely")

    architecture_or_analysis = _contains_any(
        lowered,
        (
            "architecture",
            "design",
            "analyze",
            "analysis",
            "tradeoff",
            "proposal",
            "\u67b6\u6784",
            "\u8bbe\u8ba1",
            "\u5206\u6790",
            "\u65b9\u6848",
        ),
    )
    if architecture_or_analysis:
        signals.append("architecture_or_analysis")

    review_or_debug = _contains_any(
        lowered,
        ("review", "bug", "debug", "regression", "stack trace", "\u4ee3\u7801\u5ba1\u67e5", "\u62a5\u9519"),
    )
    if review_or_debug:
        signals.append("review_or_debug")

    high_risk_domain = _contains_any(
        lowered,
        (
            "credential",
            "secret",
            "token",
            "security",
            "auth",
            "payment",
            "legal",
            "finance",
            "\u51ed\u8bc1",
            "\u5bc6\u94a5",
            "\u5b89\u5168",
            "\u652f\u4ed8",
            "\u6cd5\u5f8b",
            "\u91d1\u878d",
        ),
    )
    if high_risk_domain:
        signals.append("high_risk_domain")

    destructive_intent = _contains_any(
        lowered,
        ("delete", "remove", "drop table", "reset", "wipe", "\u5220\u9664", "\u6e05\u7a7a", "\u91cd\u7f6e"),
    )
    if destructive_intent:
        signals.append("destructive_terms")

    has_multimodal_hint = _contains_any(
        lowered,
        ("image", "screenshot", "photo", "pdf", "\u56fe\u7247", "\u622a\u56fe", "\u7167\u7247"),
    )
    if has_multimodal_hint:
        signals.append("multimodal_hint")

    asks_for_plan = _contains_any(
        lowered,
        ("plan", "todo", "roadmap", "steps", "\u8ba1\u5212", "\u6b65\u9aa4", "\u8def\u7ebf\u56fe"),
    )
    if asks_for_plan:
        signals.append("planning_request")

    score = 0
    score += 2 if requires_code_edit else 0
    score += 2 if likely_needs_file_access else 0
    score += 1 if likely_needs_tests else 0
    score += 1 if likely_needs_web else 0
    score += 2 if architecture_or_analysis else 0
    score += 2 if review_or_debug else 0
    score += 3 if high_risk_domain else 0
    score += 2 if destructive_intent else 0
    score += 1 if has_multimodal_hint else 0
    score += 1 if asks_for_plan else 0
    score += 1 if message_count > 20 else 0
    score = min(score, 10)

    return TaskFeatures(
        requires_code_edit=requires_code_edit,
        likely_needs_file_access=likely_needs_file_access,
        likely_needs_tests=likely_needs_tests,
        likely_needs_web=likely_needs_web,
        architecture_or_analysis=architecture_or_analysis,
        review_or_debug=review_or_debug,
        high_risk_domain=high_risk_domain,
        destructive_intent=destructive_intent,
        has_explicit_file_reference=has_explicit_file_reference,
        has_multimodal_hint=has_multimodal_hint,
        asks_for_latest=asks_for_latest,
        asks_for_plan=asks_for_plan,
        complexity_score=score,
        signals=signals,
    )


def _stringify_user_message(user_message: Any) -> str:
    if isinstance(user_message, str):
        return user_message
    if isinstance(user_message, list):
        parts: list[str] = []
        for item in user_message:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(user_message or "")
