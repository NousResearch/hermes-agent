"""Deterministic Slack intent routing helpers.

This module keeps Slack async-routing policy and classification in one place so
the gateway runner does not grow scattered phrase checks.  The first
implementation is intentionally local and deterministic: it handles obvious
force prefixes, control/clarify replies, project/code work, and long ad-hoc
work without making an LLM call before dispatch.
"""

from __future__ import annotations

import hashlib
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional


class SlackIntentKind(str, Enum):
    QUICK = "quick"
    LONG_AD_HOC = "long_ad_hoc"
    PROJECT_CODE = "project_code"
    CONTROL = "control"
    CLARIFICATION = "clarification"


@dataclass(frozen=True)
class SlackIntentDecision:
    kind: SlackIntentKind
    reason: str
    text: str
    forced: bool = False

    @property
    def should_foreground(self) -> bool:
        return self.kind in {
            SlackIntentKind.QUICK,
            SlackIntentKind.CONTROL,
            SlackIntentKind.CLARIFICATION,
        }


@dataclass(frozen=True)
class SlackAsyncPolicy:
    enabled: bool = True
    foreground_max_iterations: int = 6
    async_on_budget_exceeded: bool = True
    long_word_threshold: int = 55
    project_routing: str = "auto"  # auto | kanban | background | foreground
    project_assignee: str = ""
    project_workspace: str = "scratch"
    project_board: str = ""
    project_goal: bool = True
    project_goal_max_turns: int = 12
    project_max_runtime: str = "2h"
    project_skills: tuple[str, ...] = ()
    ad_hoc_ack: str = "I'll take this into the background and report back here."
    budget_ack: str = "This is running past the Slack foreground budget, so I'll continue in the background and report back here."
    project_ack_prefix: str = "Queued for project work"
    foreground_prefixes: tuple[str, ...] = ("fg:", "quick:", "foreground:")
    background_prefixes: tuple[str, ...] = ("bg:", "background:", "async:")
    project_prefixes: tuple[str, ...] = ("project:", "kanban:", "code:")
    long_markers: tuple[str, ...] = field(default_factory=tuple)
    project_markers: tuple[str, ...] = field(default_factory=tuple)
    quick_markers: tuple[str, ...] = field(default_factory=tuple)
    control_markers: tuple[str, ...] = field(default_factory=tuple)
    clarification_markers: tuple[str, ...] = field(default_factory=tuple)


DEFAULT_LONG_MARKERS = (
    "take your time",
    "when you have time",
    "in the background",
    "run in background",
    "background task",
    "deep dive",
    "thorough",
    "comprehensive",
    "write a report",
    "full report",
    "find every",
    "look through all",
    "analyze all",
    "summarize the whole",
)

DEFAULT_PROJECT_MARKERS = (
    "implement",
    "fix the bug",
    "fix this bug",
    "debug this",
    "refactor",
    "add tests",
    "write tests",
    "run tests",
    "open a pr",
    "pull request",
    "make a pr",
    "create a pr",
    "commit",
    "push branch",
    "codebase",
    "repo",
    "repository",
    "failing test",
    "ci failure",
    "regression",
    "deploy",
    "migration",
)

DEFAULT_QUICK_MARKERS = (
    "quick",
    "brief",
    "tl;dr",
    "tldr",
    "what is",
    "who is",
    "when is",
    "where is",
    "why is",
    "why did",
    "what does",
    "can you explain",
    "how do i",
    "explain",
)

DEFAULT_CONTROL_MARKERS = (
    "status",
    "stop",
    "cancel",
    "nevermind",
    "never mind",
    "done",
    "pause",
    "resume",
)

DEFAULT_CLARIFICATION_MARKERS = (
    "yes",
    "no",
    "ok",
    "okay",
    "that one",
    "the first",
    "the second",
    "option 1",
    "option 2",
    "option 3",
)


def _boolish(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default
    return bool(value)


def _intish(value: Any, default: int, *, minimum: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _string_tuple(value: Any, default: Iterable[str] = ()) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        parts = [str(p).strip() for p in value]
    else:
        parts = [str(value).strip()]
    return tuple(p for p in parts if p)


def _policy_block(config: dict[str, Any]) -> dict[str, Any]:
    candidates: list[Any] = []
    slack = config.get("slack") if isinstance(config.get("slack"), dict) else {}
    candidates.append(slack.get("async_routing"))
    gateway = config.get("gateway") if isinstance(config.get("gateway"), dict) else {}
    gateway_platforms = gateway.get("platforms") if isinstance(gateway.get("platforms"), dict) else {}
    gateway_slack = gateway_platforms.get("slack") if isinstance(gateway_platforms.get("slack"), dict) else {}
    candidates.append(gateway_slack.get("async_routing"))
    platforms = config.get("platforms") if isinstance(config.get("platforms"), dict) else {}
    platforms_slack = platforms.get("slack") if isinstance(platforms.get("slack"), dict) else {}
    candidates.append(platforms_slack.get("async_routing"))
    extra = platforms_slack.get("extra") if isinstance(platforms_slack.get("extra"), dict) else {}
    candidates.append(extra.get("async_routing"))

    merged: dict[str, Any] = {}
    for candidate in candidates:
        if isinstance(candidate, dict):
            merged.update(candidate)
    return merged


def policy_from_config(config: Optional[dict[str, Any]]) -> SlackAsyncPolicy:
    config = config if isinstance(config, dict) else {}
    block = _policy_block(config)
    kanban_cfg = config.get("kanban") if isinstance(config.get("kanban"), dict) else {}

    project_assignee = str(
        block.get("project_assignee")
        or block.get("kanban_assignee")
        or kanban_cfg.get("default_assignee")
        or ""
    ).strip()

    def markers(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        if name in block:
            base = _string_tuple(block.get(name), default)
        else:
            base = default
        extra = _string_tuple(block.get(f"additional_{name}"))
        return tuple(dict.fromkeys([*base, *extra]))

    return SlackAsyncPolicy(
        enabled=_boolish(block.get("enabled"), True),
        foreground_max_iterations=_intish(
            block.get("foreground_max_iterations"),
            6,
            minimum=1,
        ),
        async_on_budget_exceeded=_boolish(
            block.get("async_on_budget_exceeded"),
            True,
        ),
        long_word_threshold=_intish(block.get("long_word_threshold"), 55, minimum=1),
        project_routing=str(block.get("project_routing") or "auto").strip().lower(),
        project_assignee=project_assignee,
        project_workspace=str(block.get("project_workspace") or "scratch").strip() or "scratch",
        project_board=str(block.get("project_board") or block.get("kanban_board") or "").strip(),
        project_goal=_boolish(block.get("project_goal"), True),
        project_goal_max_turns=_intish(block.get("project_goal_max_turns"), 12, minimum=1),
        project_max_runtime=str(block.get("project_max_runtime") or "2h").strip() or "2h",
        project_skills=_string_tuple(block.get("project_skills")),
        ad_hoc_ack=str(
            block.get("ad_hoc_ack")
            or "I'll take this into the background and report back here."
        ).strip(),
        budget_ack=str(
            block.get("budget_ack")
            or "This is running past the Slack foreground budget, so I'll continue in the background and report back here."
        ).strip(),
        project_ack_prefix=str(block.get("project_ack_prefix") or "Queued for project work").strip(),
        foreground_prefixes=_string_tuple(
            block.get("foreground_prefixes"),
            ("fg:", "quick:", "foreground:"),
        ),
        background_prefixes=_string_tuple(
            block.get("background_prefixes"),
            ("bg:", "background:", "async:"),
        ),
        project_prefixes=_string_tuple(
            block.get("project_prefixes"),
            ("project:", "kanban:", "code:"),
        ),
        long_markers=markers("long_markers", DEFAULT_LONG_MARKERS),
        project_markers=markers("project_markers", DEFAULT_PROJECT_MARKERS),
        quick_markers=markers("quick_markers", DEFAULT_QUICK_MARKERS),
        control_markers=markers("control_markers", DEFAULT_CONTROL_MARKERS),
        clarification_markers=markers(
            "clarification_markers",
            DEFAULT_CLARIFICATION_MARKERS,
        ),
    )


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _strip_force_prefix(text: str, prefixes: Iterable[str]) -> tuple[bool, str, str]:
    raw = str(text or "").lstrip()
    lowered = raw.lower()
    for prefix in prefixes:
        p = str(prefix or "").strip().lower()
        if p and lowered.startswith(p):
            return True, raw[len(prefix):].lstrip(), p
    return False, str(text or ""), ""


def _contains_any(normalized: str, markers: Iterable[str]) -> Optional[str]:
    for marker in markers:
        needle = _normalize_text(marker)
        if needle and needle in normalized:
            return marker
    return None


def classify_slack_intent(
    text: str,
    policy: Optional[SlackAsyncPolicy] = None,
) -> SlackIntentDecision:
    policy = policy or SlackAsyncPolicy()
    raw = str(text or "").strip()

    forced, cleaned, prefix = _strip_force_prefix(raw, policy.foreground_prefixes)
    if forced:
        return SlackIntentDecision(
            SlackIntentKind.QUICK,
            f"forced foreground prefix {prefix}",
            cleaned,
            forced=True,
        )
    forced, cleaned, prefix = _strip_force_prefix(raw, policy.background_prefixes)
    if forced:
        return SlackIntentDecision(
            SlackIntentKind.LONG_AD_HOC,
            f"forced background prefix {prefix}",
            cleaned,
            forced=True,
        )
    forced, cleaned, prefix = _strip_force_prefix(raw, policy.project_prefixes)
    if forced:
        return SlackIntentDecision(
            SlackIntentKind.PROJECT_CODE,
            f"forced project prefix {prefix}",
            cleaned,
            forced=True,
        )

    normalized = _normalize_text(raw)
    if not normalized:
        return SlackIntentDecision(SlackIntentKind.QUICK, "empty text", raw)

    if normalized in {_normalize_text(m) for m in (policy.control_markers or DEFAULT_CONTROL_MARKERS)}:
        return SlackIntentDecision(SlackIntentKind.CONTROL, "control word", raw)
    if normalized in {_normalize_text(m) for m in (policy.clarification_markers or DEFAULT_CLARIFICATION_MARKERS)}:
        return SlackIntentDecision(
            SlackIntentKind.CLARIFICATION,
            "clarification reply",
            raw,
        )

    quick_marker = _contains_any(normalized, policy.quick_markers or DEFAULT_QUICK_MARKERS)
    project_marker = _contains_any(normalized, policy.project_markers or DEFAULT_PROJECT_MARKERS)
    quick_project_question = bool(
        quick_marker
        and normalized.startswith(
            (
                "how do i ",
                "how would i ",
                "how can i ",
                "what is ",
                "what does ",
                "why is ",
                "why did ",
                "where is ",
                "can you explain ",
                "explain ",
            )
        )
    )
    if project_marker and not quick_project_question:
        return SlackIntentDecision(
            SlackIntentKind.PROJECT_CODE,
            f"project marker {project_marker!r}",
            raw,
        )

    long_marker = _contains_any(normalized, policy.long_markers or DEFAULT_LONG_MARKERS)
    if long_marker:
        return SlackIntentDecision(
            SlackIntentKind.LONG_AD_HOC,
            f"long marker {long_marker!r}",
            raw,
        )

    words = normalized.split()
    work_verbs = (
        "investigate",
        "research",
        "audit",
        "analyze",
        "compare",
        "summarize",
        "review",
        "trace",
    )
    if len(words) >= policy.long_word_threshold and any(v in normalized for v in work_verbs):
        return SlackIntentDecision(
            SlackIntentKind.LONG_AD_HOC,
            f"word threshold {len(words)}",
            raw,
        )

    return SlackIntentDecision(SlackIntentKind.QUICK, "default quick", raw)


def stable_slack_handoff_id(*parts: Any, prefix: str = "slack_bg") -> str:
    raw = ":".join(str(p or "") for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def slack_idempotency_key(*parts: Any) -> str:
    raw = ":".join(str(p or "") for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()[:24]
    return f"slack:{digest}"


def build_kanban_create_text(
    *,
    decision: SlackIntentDecision,
    policy: SlackAsyncPolicy,
    chat_id: str,
    thread_id: str,
    message_id: str,
    user_id: str,
) -> str:
    """Build a public `/kanban create ...` command for Slack project routing."""
    body = decision.text.strip()
    title = body.splitlines()[0].strip() if body else "Slack project task"
    if len(title) > 140:
        title = title[:137].rstrip() + "..."
    idem = slack_idempotency_key(chat_id, thread_id, message_id, body)

    parts: list[str] = ["/kanban"]
    if policy.project_board:
        parts.extend(["--board", policy.project_board])
    parts.extend(["create", title])
    if body:
        parts.extend(["--body", body])
    if policy.project_assignee:
        parts.extend(["--assignee", policy.project_assignee])
    if policy.project_workspace:
        parts.extend(["--workspace", policy.project_workspace])
    if policy.project_max_runtime:
        parts.extend(["--max-runtime", policy.project_max_runtime])
    if policy.project_goal:
        parts.append("--goal")
        parts.extend(["--goal-max-turns", str(policy.project_goal_max_turns)])
    for skill in policy.project_skills:
        parts.extend(["--skill", skill])
    if user_id:
        parts.extend(["--created-by", f"slack:{user_id}"])
    parts.extend(["--idempotency-key", idem])

    return " ".join(shlex.quote(part) for part in parts)
