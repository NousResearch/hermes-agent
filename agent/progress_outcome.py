"""Turn-level progress outcome canary.

Tool-specific guardrails catch known loops such as repeated empty searches or
terminal usage errors. This module watches the whole turn for a broader smell:
multiple tool rounds have happened, but none of them produced an observable
work outcome.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from agent.tool_result_classification import (
    FILE_MUTATING_TOOL_NAMES,
    file_mutation_result_landed,
)
from utils import safe_json_loads


@dataclass(frozen=True)
class ProgressOutcomeConfig:
    """Thresholds for the turn-level progress canary."""

    enabled: bool = True
    warn_after_tool_rounds: int = 3
    max_warnings: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ProgressOutcomeConfig":
        if not isinstance(data, Mapping):
            return cls()
        defaults = cls()
        return cls(
            enabled=_as_bool(data.get("enabled"), defaults.enabled),
            warn_after_tool_rounds=_positive_int(
                data.get("warn_after_tool_rounds", data.get("warn_after")),
                defaults.warn_after_tool_rounds,
            ),
            max_warnings=_non_negative_int(data.get("max_warnings"), defaults.max_warnings),
        )


@dataclass(frozen=True)
class ProgressOutcomeDecision:
    """Decision emitted when a turn needs a progress nudge."""

    action: str = "allow"  # allow | warn
    code: str = "allow"
    message: str = ""
    tool_rounds: int = 0
    tool_calls: int = 0
    outcomes: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "code": self.code,
            "message": self.message,
            "tool_rounds": self.tool_rounds,
            "tool_calls": self.tool_calls,
            "outcomes": list(self.outcomes),
        }


@dataclass
class _ToolCall:
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)


class ProgressOutcomeTracker:
    """Per-turn tracker for observable progress outcomes."""

    def __init__(self, config: ProgressOutcomeConfig | None = None):
        self.config = config or ProgressOutcomeConfig()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        self.tool_rounds = 0
        self.tool_calls = 0
        self.outcomes: list[str] = []
        self.warning_count = 0

    def after_tool_round(
        self,
        tool_calls: Sequence[Any],
        tool_results: Sequence[Mapping[str, Any]],
    ) -> ProgressOutcomeDecision:
        calls = [_coerce_tool_call(call) for call in tool_calls]
        self.tool_rounds += 1
        self.tool_calls += len(calls)

        for outcome in _classify_progress_outcomes(calls, tool_results):
            if outcome not in self.outcomes:
                self.outcomes.append(outcome)

        if (
            not self.config.enabled
            or self.outcomes
            or self.warning_count >= self.config.max_warnings
            or self.tool_rounds < self.config.warn_after_tool_rounds
        ):
            return ProgressOutcomeDecision(
                tool_rounds=self.tool_rounds,
                tool_calls=self.tool_calls,
                outcomes=tuple(self.outcomes),
            )

        self.warning_count += 1
        return ProgressOutcomeDecision(
            action="warn",
            code="progress_outcome_warning",
            message=_progress_outcome_message(self.tool_rounds, self.tool_calls),
            tool_rounds=self.tool_rounds,
            tool_calls=self.tool_calls,
            outcomes=tuple(self.outcomes),
        )


def _classify_progress_outcomes(
    tool_calls: Sequence[_ToolCall],
    tool_results: Sequence[Mapping[str, Any]],
) -> list[str]:
    outcomes: list[str] = []
    for idx, call in enumerate(tool_calls):
        result_msg = tool_results[idx] if idx < len(tool_results) else {}
        content = result_msg.get("content", "") if isinstance(result_msg, Mapping) else ""
        result_tool_name = ""
        if isinstance(result_msg, Mapping):
            result_tool_name = str(result_msg.get("tool_name") or result_msg.get("name") or "")
        tool_name = result_tool_name or call.name
        if tool_name in FILE_MUTATING_TOOL_NAMES and file_mutation_result_landed(tool_name, content):
            outcomes.append(f"{tool_name}_landed")
            continue
        if tool_name == "terminal" and _terminal_command_landed(call.args, content):
            outcomes.append("terminal_state_change")
            continue
        if tool_name == "delegate_task" and not _tool_result_failed(content):
            outcomes.append("delegate_task_started")
            continue
        if tool_name == "clarify":
            outcomes.append("clarification_requested")
            continue
    return outcomes


def _terminal_command_landed(args: Mapping[str, Any], result: Any) -> bool:
    command = args.get("command", args.get("cmd", ""))
    if not isinstance(command, str) or not command.strip():
        return False
    if _tool_result_failed(result):
        return False
    lowered = command.lower()
    state_changing_patterns = (
        r"\bgit\s+(add|commit|push|tag)\b",
        r"\bgh\s+pr\s+(create|edit|ready|merge|close|comment)\b",
        r"\bgh\s+issue\s+(create|edit|close|comment)\b",
        r"\bmeshctl(?:\.py)?\s+task\s+(register|claim|review|changes|approve|merge|close|done|reopen|reset|abandon)\b",
        r"\bmeshctl(?:\.py)?\s+render\b",
        r"\bpython3?\s+\.mesh/tools/meshctl\.py\s+task\s+"
        r"(register|claim|review|changes|approve|merge|close|done|reopen|reset|abandon)\b",
        r"\bpython3?\s+\.mesh/tools/meshctl\.py\s+render\b",
    )
    return any(re.search(pattern, lowered) for pattern in state_changing_patterns)


def _tool_result_failed(result: Any) -> bool:
    if isinstance(result, str):
        parsed = safe_json_loads(result)
        if isinstance(parsed, dict):
            if parsed.get("error"):
                return True
            exit_code = parsed.get("exit_code")
            if exit_code is not None and exit_code != 0:
                return True
            if parsed.get("success") is False:
                return True
        elif result.startswith("Error"):
            return True
    return False


def _coerce_tool_call(tool_call: Any) -> _ToolCall:
    function = getattr(tool_call, "function", None)
    if function is not None:
        name = str(getattr(function, "name", "") or "")
        raw_args = getattr(function, "arguments", {}) or {}
    elif isinstance(tool_call, Mapping):
        function_data = tool_call.get("function")
        if isinstance(function_data, Mapping):
            name = str(function_data.get("name", "") or "")
            raw_args = function_data.get("arguments", {}) or {}
        else:
            name = str(tool_call.get("name", "") or "")
            raw_args = tool_call.get("arguments", {}) or {}
    else:
        return _ToolCall(name="")

    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except Exception:
            parsed = {}
    else:
        parsed = raw_args
    args = parsed if isinstance(parsed, Mapping) else {}
    return _ToolCall(name=name, args=args)


def _progress_outcome_message(tool_rounds: int, tool_calls: int) -> str:
    return (
        "Progress canary: this turn has run "
        f"{tool_rounds} tool rounds ({tool_calls} tool calls) without an "
        "observable work outcome. Before calling more tools in the same "
        "pattern, choose a concrete next step: make the edit, run one broad "
        "diagnostic and state the blocker with evidence, open/update the PR or "
        "task record, or explicitly change strategy."
    )


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return default


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default
