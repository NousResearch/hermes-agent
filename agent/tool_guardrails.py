"""Pure tool-call loop guardrail primitives.

The controller in this module is intentionally side-effect free: it tracks
per-turn tool-call observations and returns decisions. Runtime code owns whether
those decisions become warning guidance, synthetic tool results, or controlled
turn halts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from utils import safe_json_loads
from agent.tool_result_classification import file_mutation_result_landed


IDEMPOTENT_TOOL_NAMES = frozenset(
    {
        "read_file",
        "search_files",
        "web_search",
        "web_extract",
        "session_search",
        "browser_snapshot",
        "browser_console",
        "browser_get_images",
        "mcp_filesystem_read_file",
        "mcp_filesystem_read_text_file",
        "mcp_filesystem_read_multiple_files",
        "mcp_filesystem_list_directory",
        "mcp_filesystem_list_directory_with_sizes",
        "mcp_filesystem_directory_tree",
        "mcp_filesystem_get_file_info",
        "mcp_filesystem_search_files",
    }
)

MUTATING_TOOL_NAMES = frozenset(
    {
        "terminal",
        "execute_code",
        "write_file",
        "patch",
        "todo",
        "memory",
        "skill_manage",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_scroll",
        "browser_navigate",
        "send_message",
        "cronjob",
        "delegate_task",
        "process",
    }
)


@dataclass(frozen=True)
class ToolPrerequisiteRule:
    """A lightweight Forge-style prerequisite for tool calls.

    Rules are intentionally coarse and advisory by default. They nudge the
    model toward proven tool order without breaking existing sessions that
    legitimately carry prerequisite context from a previous user turn.
    """

    required_any: tuple[str, ...]
    message: str


DEFAULT_PREREQUISITE_RULES: dict[str, ToolPrerequisiteRule] = {
    "browser_back": ToolPrerequisiteRule(
        required_any=("browser_navigate",),
        message="Call browser_navigate before browser_back so there is browser history to go back through.",
    ),
    "browser_click": ToolPrerequisiteRule(
        required_any=("browser_navigate", "browser_snapshot", "browser_vision"),
        message="Get a fresh browser snapshot or navigate first so element refs are valid before clicking.",
    ),
    "browser_type": ToolPrerequisiteRule(
        required_any=("browser_navigate", "browser_snapshot", "browser_vision"),
        message="Get a fresh browser snapshot or navigate first so the input ref is valid before typing.",
    ),
    "browser_press": ToolPrerequisiteRule(
        required_any=("browser_navigate", "browser_snapshot", "browser_vision"),
        message="Navigate or inspect the page before sending keyboard input.",
    ),
    "browser_scroll": ToolPrerequisiteRule(
        required_any=("browser_navigate", "browser_snapshot", "browser_vision"),
        message="Navigate or inspect the page before scrolling it.",
    ),
    "browser_console": ToolPrerequisiteRule(
        required_any=("browser_navigate",),
        message="Open a browser page with browser_navigate before reading console output.",
    ),
    "browser_get_images": ToolPrerequisiteRule(
        required_any=("browser_navigate",),
        message="Open a browser page with browser_navigate before listing images.",
    ),
    "patch": ToolPrerequisiteRule(
        required_any=("read_file", "search_files"),
        message="Inspect the target with read_file or search_files before patching so the edit is grounded.",
    ),
}


@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """Thresholds for per-turn tool-call loop detection.

    Warnings are enabled by default and never prevent tool execution. Hard stops
    are explicit opt-in so interactive CLI/TUI sessions get a gentle nudge unless
    the user enables circuit-breaker behavior in config.yaml.
    """

    warnings_enabled: bool = True
    hard_stop_enabled: bool = False
    prerequisite_checks_enabled: bool = True
    prerequisite_hard_stop_enabled: bool = False
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 5
    idempotent_tools: frozenset[str] = field(default_factory=lambda: IDEMPOTENT_TOOL_NAMES)
    mutating_tools: frozenset[str] = field(default_factory=lambda: MUTATING_TOOL_NAMES)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ToolCallGuardrailConfig":
        """Build config from the `tool_loop_guardrails` config.yaml section."""
        if not isinstance(data, Mapping):
            return cls()

        warn_after = data.get("warn_after")
        if not isinstance(warn_after, Mapping):
            warn_after = {}
        hard_stop_after = data.get("hard_stop_after")
        if not isinstance(hard_stop_after, Mapping):
            hard_stop_after = {}

        defaults = cls()
        return cls(
            warnings_enabled=_as_bool(data.get("warnings_enabled"), defaults.warnings_enabled),
            hard_stop_enabled=_as_bool(data.get("hard_stop_enabled"), defaults.hard_stop_enabled),
            prerequisite_checks_enabled=_as_bool(
                data.get("prerequisite_checks_enabled"),
                defaults.prerequisite_checks_enabled,
            ),
            prerequisite_hard_stop_enabled=_as_bool(
                data.get("prerequisite_hard_stop_enabled"),
                defaults.prerequisite_hard_stop_enabled,
            ),
            exact_failure_warn_after=_positive_int(
                warn_after.get("exact_failure", data.get("exact_failure_warn_after")),
                defaults.exact_failure_warn_after,
            ),
            same_tool_failure_warn_after=_positive_int(
                warn_after.get("same_tool_failure", data.get("same_tool_failure_warn_after")),
                defaults.same_tool_failure_warn_after,
            ),
            no_progress_warn_after=_positive_int(
                warn_after.get("idempotent_no_progress", data.get("no_progress_warn_after")),
                defaults.no_progress_warn_after,
            ),
            exact_failure_block_after=_positive_int(
                hard_stop_after.get("exact_failure", data.get("exact_failure_block_after")),
                defaults.exact_failure_block_after,
            ),
            same_tool_failure_halt_after=_positive_int(
                hard_stop_after.get("same_tool_failure", data.get("same_tool_failure_halt_after")),
                defaults.same_tool_failure_halt_after,
            ),
            no_progress_block_after=_positive_int(
                hard_stop_after.get("idempotent_no_progress", data.get("no_progress_block_after")),
                defaults.no_progress_block_after,
            ),
        )


@dataclass(frozen=True)
class ToolCallSignature:
    """Stable, non-reversible identity for a tool name plus canonical args."""

    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Mapping[str, Any] | None) -> "ToolCallSignature":
        canonical = canonical_tool_args(args or {})
        return cls(tool_name=tool_name, args_hash=_sha256(canonical))

    def to_metadata(self) -> dict[str, str]:
        """Return public metadata without raw argument values."""
        return {"tool_name": self.tool_name, "args_hash": self.args_hash}


@dataclass(frozen=True)
class ToolGuardrailDecision:
    """Decision returned by the tool-call guardrail controller."""

    action: str = "allow"  # allow | warn | block | halt
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0
    signature: ToolCallSignature | None = None

    @property
    def allows_execution(self) -> bool:
        return self.action in {"allow", "warn"}

    @property
    def should_halt(self) -> bool:
        return self.action in {"block", "halt"}

    def to_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action": self.action,
            "code": self.code,
            "message": self.message,
            "tool_name": self.tool_name,
            "count": self.count,
        }
        if self.signature is not None:
            data["signature"] = self.signature.to_metadata()
        return data


def canonical_tool_args(args: Mapping[str, Any]) -> str:
    """Return sorted compact JSON for parsed tool arguments."""
    if not isinstance(args, Mapping):
        raise TypeError(f"tool args must be a mapping, got {type(args).__name__}")
    return json.dumps(
        args,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def classify_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Safety-fallback classifier used only when callers don't pass ``failed``.

    Mirrors ``agent.display._detect_tool_failure`` exactly so the guardrail
    never disagrees with the CLI's user-visible ``[error]`` tag. Production
    callers in ``run_agent.py`` always pass an explicit ``failed=`` derived
    from ``_detect_tool_failure``; this function exists so standalone callers
    (tests, tooling) still get consistent behavior.
    """
    if result is None:
        return False, ""
    if file_mutation_result_landed(tool_name, result):
        return False, ""

    if tool_name == "terminal":
        data = safe_json_loads(result)
        if isinstance(data, dict):
            exit_code = data.get("exit_code")
            if exit_code is not None and exit_code != 0:
                return True, f" [exit {exit_code}]"
        return False, ""

    if tool_name == "memory":
        data = safe_json_loads(result)
        if isinstance(data, dict):
            if data.get("success") is False and "exceed the limit" in data.get("error", ""):
                return True, " [full]"

    lower = result[:500].lower()
    if '"error"' in lower or '"failed"' in lower or result.startswith("Error"):
        return True, " [error]"

    return False, ""


class ToolCallGuardrailController:
    """Per-turn controller for repeated failed/non-progressing tool calls."""

    def __init__(self, config: ToolCallGuardrailConfig | None = None):
        self.config = config or ToolCallGuardrailConfig()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        self._successful_tools: set[str] = set()
        self._successful_tool_actions: set[tuple[str, str]] = set()
        self._halt_decision: ToolGuardrailDecision | None = None

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        return self._halt_decision

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        prereq_decision = self._check_prerequisite(tool_name, args, signature)

        # Existing hard-stop loop breakers must keep priority over new soft
        # prerequisite nudges. Otherwise a missing-prerequisite warning could
        # mask a repeated exact-failure block for the same call.
        if self.config.hard_stop_enabled:
            exact_count = self._exact_failure_counts.get(signature, 0)
            if exact_count >= self.config.exact_failure_block_after:
                decision = ToolGuardrailDecision(
                    action="block",
                    code="repeated_exact_failure_block",
                    message=(
                        f"Blocked {tool_name}: the same tool call failed {exact_count} "
                        "times with identical arguments. Stop retrying it unchanged; "
                        "change strategy or explain the blocker."
                    ),
                    tool_name=tool_name,
                    count=exact_count,
                    signature=signature,
                )
                self._halt_decision = decision
                return decision

            if self._is_idempotent(tool_name):
                record = self._no_progress.get(signature)
                if record is not None:
                    _result_hash, repeat_count = record
                    if repeat_count >= self.config.no_progress_block_after:
                        decision = ToolGuardrailDecision(
                            action="block",
                            code="idempotent_no_progress_block",
                            message=(
                                f"Blocked {tool_name}: this read-only call returned the same "
                                f"result {repeat_count} times. Stop repeating it unchanged; "
                                "use the result already provided or try a different query."
                            ),
                            tool_name=tool_name,
                            count=repeat_count,
                            signature=signature,
                        )
                        self._halt_decision = decision
                        return decision

        if prereq_decision is not None:
            if prereq_decision.should_halt:
                self._halt_decision = prereq_decision
            return prereq_decision

        return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

    def after_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        result: str | None,
        *,
        failed: bool | None = None,
    ) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        if failed is None:
            failed, _ = classify_tool_failure(tool_name, result)

        if failed:
            exact_count = self._exact_failure_counts.get(signature, 0) + 1
            self._exact_failure_counts[signature] = exact_count
            self._no_progress.pop(signature, None)

            same_count = self._same_tool_failure_counts.get(tool_name, 0) + 1
            self._same_tool_failure_counts[tool_name] = same_count

            if self.config.hard_stop_enabled and same_count >= self.config.same_tool_failure_halt_after:
                decision = ToolGuardrailDecision(
                    action="halt",
                    code="same_tool_failure_halt",
                    message=(
                        f"Stopped {tool_name}: it failed {same_count} times this turn. "
                        "Stop retrying the same failing tool path and choose a different approach."
                    ),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )
                self._halt_decision = decision
                return decision

            if self.config.warnings_enabled and exact_count >= self.config.exact_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="repeated_exact_failure_warning",
                    message=(
                        f"{tool_name} has failed {exact_count} times with identical arguments. "
                        "This looks like a loop; inspect the error and change strategy "
                        "instead of retrying it unchanged."
                    ),
                    tool_name=tool_name,
                    count=exact_count,
                    signature=signature,
                )

            if self.config.warnings_enabled and same_count >= self.config.same_tool_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="same_tool_failure_warning",
                    message=_tool_failure_recovery_hint(tool_name, same_count),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )

            return ToolGuardrailDecision(tool_name=tool_name, count=exact_count, signature=signature)

        self._record_success(tool_name, args)
        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)

        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        result_hash = _result_hash(result)
        previous = self._no_progress.get(signature)
        repeat_count = 1
        if previous is not None and previous[0] == result_hash:
            repeat_count = previous[1] + 1
        self._no_progress[signature] = (result_hash, repeat_count)

        if self.config.warnings_enabled and repeat_count >= self.config.no_progress_warn_after:
            return ToolGuardrailDecision(
                action="warn",
                code="idempotent_no_progress_warning",
                message=(
                    f"{tool_name} returned the same result {repeat_count} times. "
                    "Use the result already provided or change the query instead of "
                    "repeating it unchanged."
                ),
                tool_name=tool_name,
                count=repeat_count,
                signature=signature,
            )

        return ToolGuardrailDecision(tool_name=tool_name, count=repeat_count, signature=signature)

    def _is_idempotent(self, tool_name: str) -> bool:
        if tool_name in self.config.mutating_tools:
            return False
        return tool_name in self.config.idempotent_tools

    def _check_prerequisite(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        signature: ToolCallSignature,
    ) -> ToolGuardrailDecision | None:
        if not self.config.prerequisite_checks_enabled:
            return None

        if not self.config.prerequisite_hard_stop_enabled and not self.config.warnings_enabled:
            return None

        rule = _dynamic_prerequisite_rule(tool_name, args) or DEFAULT_PREREQUISITE_RULES.get(tool_name)
        if rule is None:
            return None
        if any(self._has_successful_requirement(required) for required in rule.required_any):
            return None

        action = "block" if self.config.prerequisite_hard_stop_enabled else "warn"
        code = (
            "missing_tool_prerequisite_block"
            if self.config.prerequisite_hard_stop_enabled
            else "missing_tool_prerequisite_warning"
        )
        required = ", ".join(rule.required_any)
        return ToolGuardrailDecision(
            action=action,
            code=code,
            message=(
                f"{tool_name} is being called before its recommended prerequisite "
                f"tool(s): {required}. {rule.message} If the prerequisite was "
                "already satisfied in earlier conversation context, proceed carefully; "
                "otherwise call the prerequisite first."
            ),
            tool_name=tool_name,
            count=0,
            signature=signature,
        )

    def _record_success(self, tool_name: str, args: Mapping[str, Any]) -> None:
        self._successful_tools.add(tool_name)
        action = args.get("action")
        if isinstance(action, str) and action:
            self._successful_tool_actions.add((tool_name, action))

    def _has_successful_requirement(self, requirement: str) -> bool:
        if ":" not in requirement:
            return requirement in self._successful_tools
        tool_name, action = requirement.split(":", 1)
        return (tool_name, action) in self._successful_tool_actions


def _dynamic_prerequisite_rule(
    tool_name: str,
    args: Mapping[str, Any],
) -> ToolPrerequisiteRule | None:
    """Return argument-sensitive prerequisite rules.

    These capture Hermes-specific workflow contracts from the tool docs without
    needing a separate workflow engine.
    """
    if tool_name == "send_message":
        action = str(args.get("action") or "send")
        target = str(args.get("target") or "")
        named_target = ":" in target or "#" in target
        if action == "send" and named_target:
            return ToolPrerequisiteRule(
                required_any=("send_message:list",),
                message="Use send_message(action='list') before sending to a named channel/person so the target is verified.",
            )

    if tool_name == "cronjob":
        action = str(args.get("action") or "")
        if action in {"update", "pause", "resume", "remove", "run"}:
            return ToolPrerequisiteRule(
                required_any=("cronjob:list",),
                message="Use cronjob(action='list') before managing an existing job so the job_id is verified.",
            )

    if tool_name == "process":
        action = str(args.get("action") or "")
        if action and action != "list":
            return ToolPrerequisiteRule(
                required_any=("terminal", "process"),
                message="Use process(action='list') or start/observe a terminal background process before acting on a process session_id.",
            )

    return None


def toolguard_synthetic_result(decision: ToolGuardrailDecision) -> str:
    """Build a synthetic role=tool content string for a blocked tool call."""
    return json.dumps(
        {
            "error": decision.message,
            "guardrail": decision.to_metadata(),
        },
        ensure_ascii=False,
    )


def append_toolguard_guidance(result: Any, decision: ToolGuardrailDecision) -> Any:
    """Append runtime guidance to the current tool result content.

    Most tool results are strings. Multimodal results may be dict/list payloads;
    leave them unchanged rather than corrupting the structured content.
    """
    if decision.action not in {"warn", "halt"} or not decision.message:
        return result
    if not isinstance(result, str):
        return result
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    suffix = (
        f"\n\n[{label}: "
        f"{decision.code}; count={decision.count}; {decision.message}]"
    )
    return (result or "") + suffix


def _tool_failure_recovery_hint(tool_name: str, count: int) -> str:
    """Action-oriented guidance for recovering from repeated tool failures."""
    common = (
        f"{tool_name} has failed {count} times this turn. This looks like a loop. "
        "Do not switch to text-only replies; keep using tools, but diagnose before retrying. "
        "First inspect the latest error/output and verify your assumptions. "
    )
    if tool_name == "terminal":
        return common + (
            "For terminal failures, run a small diagnostic such as `pwd && ls -la` "
            "in the same tool, then try an absolute path, a simpler command, a different "
            "working directory, or a different tool such as read_file/write_file/patch."
        )
    if tool_name == "patch":
        return common + (
            "For repeated patch failures, stop trying the same old_string. Re-read the "
            "target with read_file/search_files, choose a smaller exact context, or switch "
            "to write_file only when replacing the whole file is safer and scoped."
        )
    return common + (
        "Try different arguments, a narrower query/path, an absolute path when relevant, "
        "or a different tool that can make progress. If the blocker is external, report "
        "the blocker after one diagnostic attempt instead of repeating the same failing path."
    )


def _coerce_args(args: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return args if isinstance(args, Mapping) else {}


def _result_hash(result: str | None) -> str:
    parsed = safe_json_loads(result or "")
    if parsed is not None:
        try:
            canonical = json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except TypeError:
            canonical = str(parsed)
    else:
        canonical = result or ""
    return _sha256(canonical)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
