"""Mechanical completion-report gate for consecutive tool batches.

A reportable tool batch arms the gate. If the model attempts another tool batch
without visible status text, the conversation loop emits a deterministic
user-facing progress message before dispatching the new tools. The gate never
mutates model context or injects synthetic user messages, preserving provider
role alternation and prompt-cache stability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from agent.agent_runtime_helpers import message_content_to_text


# Internal bookkeeping/context tools do not represent a user-visible work
# phase. Everything else defaults to reportable so newly added operational
# tools cannot silently bypass the gate.
_NON_REPORTABLE_TOOLS = frozenset(
    {
        "clarify",
        "memory",
        "session_search",
        "skill_view",
        "skills_list",
        "todo",
    }
)

_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think>\s*", re.DOTALL | re.IGNORECASE
)


@dataclass(frozen=True)
class GateDecision:
    """Decision returned before a consecutive tool batch is dispatched."""

    action: str  # "allow" or "report"
    message: str = ""


class CompletionReportGate:
    """Require user-visible status between reportable tool batches."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.pending_tools: tuple[str, ...] = ()

    @property
    def pending(self) -> bool:
        return bool(self.pending_tools)

    def arm(self, tool_names: Iterable[str]) -> bool:
        """Require a report before a subsequent tool batch.

        A batch containing only internal bookkeeping tools is ignored. Tool
        names are deduplicated in call order so deterministic reports remain
        compact and cannot contain tool outputs or credentials.
        """

        if not self.enabled:
            return False

        reportable: list[str] = []
        seen: set[str] = set()
        for raw_name in tool_names:
            name = str(raw_name or "").strip()
            if not name or name in _NON_REPORTABLE_TOOLS or name in seen:
                continue
            seen.add(name)
            reportable.append(name)

        if not reportable:
            return False

        self.pending_tools = tuple(reportable)
        return True

    def before_tool_batch(self, visible_content: Any) -> GateDecision:
        """Allow the batch or require a deterministic interim report first."""

        if not self.enabled or not self.pending:
            return GateDecision("allow")

        if _has_visible_report(visible_content):
            self.pending_tools = ()
            return GateDecision("allow")

        names = ", ".join(name.replace("_", " ") for name in self.pending_tools[:6])
        if len(self.pending_tools) > 6:
            names += f", +{len(self.pending_tools) - 6} more"
        self.pending_tools = ()
        return GateDecision(
            "report",
            f"Previous tool phase ended ({names}). Continuing with the next step.",
        )


def _has_visible_report(content: Any) -> bool:
    """Return whether assistant content contains user-visible non-thinking text."""

    visible = message_content_to_text(content)
    visible = _THINK_BLOCK_RE.sub("", visible).strip()
    return bool(visible and visible != "(empty)")


def completion_report_gate_enabled(config: dict[str, Any] | None = None) -> bool:
    """Resolve the persisted ``agent.completion_report_gate`` setting."""

    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}

    agent_cfg = (config or {}).get("agent") if isinstance(config, dict) else None
    value = (
        agent_cfg.get("completion_report_gate")
        if isinstance(agent_cfg, dict)
        else False
    )
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


__all__ = [
    "CompletionReportGate",
    "GateDecision",
    "completion_report_gate_enabled",
]
