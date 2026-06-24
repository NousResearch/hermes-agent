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

from agent.tool_failure_detection import output_indicates_task_failure

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
class ToolCallGuardrailConfig:
    """Thresholds for per-turn tool-call loop detection.

    Warnings are enabled by default and never prevent tool execution. Hard stops
    are explicit opt-in so interactive CLI/TUI sessions get a gentle nudge unless
    the user enables circuit-breaker behavior in config.yaml.
    """

    warnings_enabled: bool = True
    hard_stop_enabled: bool = False
    exact_failure_warn_after: int = 2
    exact_failure_selfcheck_after: int = 3
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
            exact_failure_warn_after=_positive_int(
                warn_after.get("exact_failure", data.get("exact_failure_warn_after")),
                defaults.exact_failure_warn_after,
            ),
            exact_failure_selfcheck_after=_positive_int(
                warn_after.get("exact_failure_selfcheck", data.get("exact_failure_selfcheck_after")),
                defaults.exact_failure_selfcheck_after,
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
    # Optional structured context for self-check observations.
    last_tool_call_args: dict | None = None
    recent_failures: list[dict] | None = None

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


# Execution tools (terminal, execute_code) that can produce "fake success"
# — exit code 0 / status "success" while the actual task failed.  These are
# the only tools where we apply semantic output failure detection and
# no-progress tracking on mutating tools.
NO_PROGRESS_MUTATING_TOOL_NAMES = frozenset({"terminal", "execute_code"})


def _tracks_per_turn_no_progress(tool_name: str, config: ToolCallGuardrailConfig) -> bool:
    """Track repeated identical success within a single turn.

    Applies to idempotent (read-only) tools plus terminal/execute_code.
    """
    return tool_name in config.idempotent_tools or tool_name in NO_PROGRESS_MUTATING_TOOL_NAMES


def _tracks_cross_turn_no_progress(tool_name: str, config: ToolCallGuardrailConfig) -> bool:
    """Track repeated identical success across multiple turns.

    Only applies to terminal/execute_code — we do NOT want to block
    read-only tools (read_file, web_search, etc.) across turns because
    the user legitimately re-reads the same file or re-runs the same
    query in subsequent turns.
    """
    return tool_name in NO_PROGRESS_MUTATING_TOOL_NAMES


# Backward compat alias — used by tool_executor.py post-execution loop
def _tracks_no_progress(tool_name: str, config: ToolCallGuardrailConfig) -> bool:
    return _tracks_per_turn_no_progress(tool_name, config)


def classify_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Safety-fallback classifier used only when callers don't pass ``failed``.

    Mirrors ``agent.display._detect_tool_failure`` exactly so the guardrail
    never disagrees with the CLI's user-visible ``[error]`` tag. Production
    callers in ``run_agent.py`` always pass an explicit ``failed=`` derived
    from ``_detect_tool_failure``; this function exists so standalone callers
    (tests, tooling) still get consistent behavior.

    In addition to structural failure signals (exit code != 0, status="error"),
    also checks the output text of terminal and execute_code for task-level
    failure patterns (HTTP errors, tracebacks, etc.) even when the tool
    process itself succeeded.
    """
    if result is None:
        return False, ""
    if file_mutation_result_landed(tool_name, result):
        return False, ""

    data = safe_json_loads(result)

    if tool_name == "terminal":
        if isinstance(data, dict):
            exit_code = data.get("exit_code")
            if exit_code is not None and exit_code != 0:
                return True, f" [exit {exit_code}]"
            # exit_code == 0 but output contains task-level failure
            output = data.get("output", "")
            if output and output_indicates_task_failure(output):
                return True, " [task_failure]"
        return False, ""

    if tool_name == "execute_code":
        if isinstance(data, dict):
            status = data.get("status")
            if status == "error":
                return True, " [error]"
            # status == "success" but output contains task-level failure
            if status == "success":
                output = data.get("output", "")
                if output and output_indicates_task_failure(output):
                    return True, " [task_failure]"
        return False, ""

    if tool_name == "memory":
        if isinstance(data, dict):
            if data.get("success") is False and "exceed the limit" in data.get("error", ""):
                return True, " [full]"

    lower = result[:500].lower()
    if '"error"' in lower or '"failed"' in lower or result.startswith("Error"):
        return True, " [error]"

    return False, ""


class ToolCallGuardrailController:
    """Per-turn controller for repeated failed/non-progressing tool calls.

    Includes recovery budget tracking: after a guardrail block, the agent gets
    a limited number of recovery attempts (default: 2) to re-plan. If recovery
    also blocks, the turn is finally halted.
    """

    def __init__(self, config: ToolCallGuardrailConfig | None = None):
        self.config = config or ToolCallGuardrailConfig()
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        # Per-turn no-progress tracking (same-turn duplicates).
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        # Cross-turn no-progress tracking for terminal/execute_code so that
        # loops spanning multiple turns (one identical call per turn) are still
        # caught.  Value is (result_hash, repeat_count, last_turn).
        self._no_progress_cross_turn: dict[ToolCallSignature, tuple[str, int, int]] = {}
        self._turn_counter: int = 0
        self._halt_decision: ToolGuardrailDecision | None = None
        self._cross_turn_ttl: int = 10  # forget entries older than N turns
        # Recovery budget: after a guardrail block, allow N recovery attempts
        # before final halt. Prevents infinite recovery loops.
        self._recovery_attempts: int = 0
        self._max_recovery_attempts: int = 2
        self._blocked_signatures: set[ToolCallSignature] = set()  # signatures blocked this turn
        # Recent failure history per signature for structured self-check observations.
        # Value: list of dicts with {exit_code, output_tail, status} — most recent last.
        self._recent_failures: dict[ToolCallSignature, list[dict]] = {}

    def reset_for_turn(self) -> None:
        self._turn_counter += 1
        self._exact_failure_counts = {}
        self._same_tool_failure_counts = {}
        self._no_progress = {}
        self._halt_decision = None
        # Reset recovery budget for each new turn
        self._recovery_attempts = 0
        self._blocked_signatures = set()
        self._recent_failures = {}
        # Prune stale cross-turn entries so the dict doesn't grow unbounded.
        _cutoff = self._turn_counter - self._cross_turn_ttl
        self._no_progress_cross_turn = {
            sig: val
            for sig, val in self._no_progress_cross_turn.items()
            if val[2] >= _cutoff
        }

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        return self._halt_decision

    @property
    def recovery_attempts(self) -> int:
        """Number of recovery attempts used in the current turn."""
        return self._recovery_attempts

    @property
    def recovery_exhausted(self) -> bool:
        """True if the recovery budget is exhausted — must final halt."""
        return self._recovery_attempts >= self._max_recovery_attempts

    def record_recovery(self, signature: ToolCallSignature) -> None:
        """Record a recovery attempt after a guardrail block.

        Increments the recovery counter and tracks the blocked signature
        so the same signature cannot be retried during recovery.
        """
        self._recovery_attempts += 1
        self._blocked_signatures.add(signature)

    def is_signature_blocked(self, signature: ToolCallSignature) -> bool:
        """Check if a signature was previously blocked this turn."""
        return signature in self._blocked_signatures

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        signature = ToolCallSignature.from_call(tool_name, _coerce_args(args))
        if not self.config.hard_stop_enabled:
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        # Check if this signature was previously blocked this turn (recovery retry guard)
        if self.is_signature_blocked(signature):
            decision = ToolGuardrailDecision(
                action="block",
                code="recovery_retry_block",
                message=(
                    f"Blocked {tool_name}: this call was already blocked this turn. "
                    "Recovery requires a materially different strategy; do not retry "
                    "the same blocked call."
                ),
                tool_name=tool_name,
                count=self._recovery_attempts,
                signature=signature,
            )
            self._halt_decision = decision
            return decision

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

        # Per-turn no-progress check
        record = self._no_progress.get(signature) if _tracks_per_turn_no_progress(tool_name, self.config) else None
        if record is not None:
            _result_hash, repeat_count = record
            if repeat_count >= self.config.no_progress_block_after:
                decision = ToolGuardrailDecision(
                    action="block",
                    code="no_progress_block",
                    message=(
                        f"Blocked {tool_name}: this call returned the same "
                        f"result {repeat_count} times. Stop repeating it unchanged; "
                        "use the result already provided or try a different approach."
                    ),
                    tool_name=tool_name,
                    count=repeat_count,
                    signature=signature,
                )
                self._halt_decision = decision
                return decision

        # Cross-turn no-progress check (catches one-call-per-turn loops)
        if _tracks_cross_turn_no_progress(tool_name, self.config):
            cross = self._no_progress_cross_turn.get(signature)
            if cross is not None:
                _rh, _rc, _lt = cross
                if _rc >= self.config.no_progress_block_after:
                    decision = ToolGuardrailDecision(
                        action="block",
                        code="no_progress_cross_turn_block",
                        message=(
                            f"Blocked {tool_name}: this call returned the same "
                            f"result {_rc} times across multiple turns. Stop repeating it unchanged; "
                            "use the result already provided or try a different approach."
                        ),
                        tool_name=tool_name,
                        count=_rc,
                        signature=signature,
                    )
                    self._halt_decision = decision
                    return decision

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

            # Record failure history for structured self-check
            _record_failure(self._recent_failures, signature, tool_name, result)

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

            # Count >= exact_failure_block_after: handled by before_call block
            # (we don't block here, we just warn — before_call does the actual block)

            # Count >= selfcheck_after: structured self-check observation
            if self.config.warnings_enabled and exact_count >= self.config.exact_failure_selfcheck_after:
                return _build_selfcheck_decision(
                    tool_name, exact_count, signature, args,
                    self._recent_failures.get(signature, []),
                )

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

        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)

        if not _tracks_per_turn_no_progress(tool_name, self.config):
            self._no_progress.pop(signature, None)
            self._no_progress_cross_turn.pop(signature, None)
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        result_hash = _result_hash(result)
        previous = self._no_progress.get(signature)
        repeat_count = 1
        if previous is not None and previous[0] == result_hash:
            repeat_count = previous[1] + 1
        self._no_progress[signature] = (result_hash, repeat_count)

        # Cross-turn no-progress tracking (only for terminal/execute_code)
        if _tracks_cross_turn_no_progress(tool_name, self.config):
            cross_prev = self._no_progress_cross_turn.get(signature)
            if cross_prev is not None and cross_prev[0] == result_hash:
                self._no_progress_cross_turn[signature] = (result_hash, cross_prev[1] + 1, self._turn_counter)
            else:
                self._no_progress_cross_turn[signature] = (result_hash, 1, self._turn_counter)

        # Emit warning based on cross-turn count when per-turn count is 1
        # (i.e., the loop spans turns rather than repeating within one turn)
        if _tracks_cross_turn_no_progress(tool_name, self.config):
            _cross_count = self._no_progress_cross_turn[signature][1]
            if repeat_count == 1 and self.config.warnings_enabled and _cross_count >= self.config.no_progress_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="no_progress_cross_turn_warning",
                    message=(
                        f"{tool_name} returned the same result {_cross_count} times across turns. "
                        "Use the result already provided or change the query instead of "
                        "repeating it unchanged."
                    ),
                    tool_name=tool_name,
                    count=_cross_count,
                    signature=signature,
                )

        if self.config.warnings_enabled and repeat_count >= self.config.no_progress_warn_after:
            return ToolGuardrailDecision(
                action="warn",
                code="no_progress_warning",
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


def _record_failure(
    recent_failures: dict[ToolCallSignature, list[dict]],
    signature: ToolCallSignature,
    tool_name: str,
    result: str | None,
) -> None:
    """Record a structured failure entry for a tool call signature."""
    if signature not in recent_failures:
        recent_failures[signature] = []

    entry: dict[str, Any] = {}
    parsed = safe_json_loads(result or "")
    if isinstance(parsed, dict):
        if tool_name == "terminal":
            exit_code = parsed.get("exit_code")
            if exit_code is not None:
                entry["exit_code"] = exit_code
                meaning = _exit_code_meaning(exit_code)
                if meaning:
                    entry["exit_code_meaning"] = meaning
            output = parsed.get("output", "")
            if output:
                entry["output_tail"] = output[-300:] if len(output) > 300 else output
        elif tool_name == "execute_code":
            status = parsed.get("status")
            if status:
                entry["status"] = status
            output = parsed.get("output", "")
            if output:
                entry["output_tail"] = output[-300:] if len(output) > 300 else output
    else:
        # Non-JSON result — store tail
        if result:
            entry["output_tail"] = result[-300:] if len(result) > 300 else result

    recent_failures[signature].append(entry)
    # Keep last 5 entries to avoid unbounded growth
    if len(recent_failures[signature]) > 5:
        recent_failures[signature] = recent_failures[signature][-5:]


def _exit_code_meaning(code: int) -> str | None:
    """Return a human-readable meaning for common exit codes."""
    meanings = {
        1: "General error",
        2: "Misuse of shell builtins",
        7: "Failed to connect to host",
        124: "Command timed out",
        126: "Command invoked cannot execute",
        127: "Command not found",
        137: "Process killed (SIGKILL)",
        139: "Process segfaulted (SIGSEGV)",
        143: "Process terminated (SIGTERM)",
    }
    return meanings.get(code)


def _build_selfcheck_decision(
    tool_name: str,
    exact_count: int,
    signature: ToolCallSignature,
    args: Mapping[str, Any],
    recent_failures: list[dict],
) -> ToolGuardrailDecision:
    """Build a structured self-check observation for repeated failures.

    At count >= 3, this replaces the regular warning with a structured
    observation that forces the model to self-check before retrying.
    """
    # Build the self-check message based on count
    if exact_count >= 4:
        # Stronger warning at count 4+
        message = (
            f"{tool_name} has failed {exact_count} times with identical arguments. "
            "You have been warned but have not changed the arguments. "
            "This is a confirmed loop — you MUST change strategy before calling this tool again."
        )
    else:
        message = (
            f"{tool_name} has failed {exact_count} times with identical arguments. "
            "Before retrying, compare your intended fix with the actual tool arguments you are about to emit."
        )

    return ToolGuardrailDecision(
        action="warn",
        code="tool_call_self_check_required",
        message=message,
        tool_name=tool_name,
        count=exact_count,
        signature=signature,
        last_tool_call_args=dict(args),
        recent_failures=list(recent_failures),
    )


def toolguard_synthetic_result(decision: ToolGuardrailDecision) -> str:
    """Build a synthetic role=tool content string for a blocked tool call."""
    return json.dumps(
        {
            "error": decision.message,
            "guardrail": decision.to_metadata(),
        },
        ensure_ascii=False,
    )


def append_toolguard_guidance(result: str, decision: ToolGuardrailDecision) -> str:
    """Append runtime guidance to the current tool result content."""
    if decision.action not in {"warn", "halt"} or not decision.message:
        return result

    # For self-check observations, embed structured JSON data
    if decision.code == "tool_call_self_check_required":
        return _append_selfcheck_guidance(result, decision)

    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    suffix = (
        f"\n\n[{label}: "
        f"{decision.code}; count={decision.count}; {decision.message}]"
    )
    return (result or "") + suffix


def _append_selfcheck_guidance(result: str, decision: ToolGuardrailDecision) -> str:
    """Append structured self-check guidance for repeated tool failures.

    Embeds a JSON block with the failure history and required self-check steps
    so the model has structured evidence to self-correct.
    """
    # Build the structured self-check observation
    selfcheck = {
        "error": "Tool-call self-check required before retrying.",
        "guardrail": {
            "action": decision.action,
            "code": decision.code,
            "message": decision.message,
            "tool_name": decision.tool_name,
            "count": decision.count,
        },
        "last_tool_call": {
            "tool_name": decision.tool_name,
            "args": decision.last_tool_call_args or {},
        },
        "recent_failures": decision.recent_failures or [],
        "required_self_check": [
            "State the concrete change you will make before calling another tool.",
            "Verify the next tool arguments actually contain that change.",
            f"If you intend to run a long-lived service in background, verify args.background is true.",
            f"If you intend to change port/config/path/mode, verify the args reflect that exact change.",
            "Do not call the tool again if the args are unchanged.",
        ],
        "required_next_step": (
            "Either emit a materially different tool call, or explain the blocker to the user. "
            "Do not repeat the same args."
        ),
    }

    suffix = f"\n\n{json.dumps(selfcheck, ensure_ascii=False, indent=2)}"
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
