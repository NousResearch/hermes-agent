"""Pure tool-call loop guardrail primitives.

The controller in this module is intentionally side-effect free: it tracks
per-turn tool-call observations and returns decisions. Runtime code owns whether
those decisions become warning guidance, synthetic tool results, or controlled
turn halts.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
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
class ToolCallGuardrailConfig:
    """Thresholds for per-turn tool-call loop detection.

    Warnings are enabled by default. Hard stops remain explicit opt-in, but
    repeated low-information read-only calls may redirect the same tool for one
    turn so the model has to verify assumptions with a different tool path
    instead of burning the iteration budget on harmless-looking empty results.
    """

    warnings_enabled: bool = True
    hard_stop_enabled: bool = False
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 5
    low_information_warn_after: int = 3
    terminal_usage_error_warn_after: int = 2
    low_information_redirect_after: int = 4
    low_information_halt_after: int = 6
    terminal_usage_error_redirect_after: int = 3
    terminal_usage_error_halt_after: int = 5
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
            same_tool_failure_warn_after=_positive_int(
                warn_after.get("same_tool_failure", data.get("same_tool_failure_warn_after")),
                defaults.same_tool_failure_warn_after,
            ),
            no_progress_warn_after=_positive_int(
                warn_after.get("idempotent_no_progress", data.get("no_progress_warn_after")),
                defaults.no_progress_warn_after,
            ),
            low_information_warn_after=_positive_int(
                warn_after.get("low_information", data.get("low_information_warn_after")),
                defaults.low_information_warn_after,
            ),
            terminal_usage_error_warn_after=_positive_int(
                warn_after.get("terminal_usage_error", data.get("terminal_usage_error_warn_after")),
                defaults.terminal_usage_error_warn_after,
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
            low_information_redirect_after=_positive_int(
                hard_stop_after.get("low_information_redirect", data.get("low_information_redirect_after")),
                defaults.low_information_redirect_after,
            ),
            low_information_halt_after=_positive_int(
                hard_stop_after.get("low_information", data.get("low_information_halt_after")),
                defaults.low_information_halt_after,
            ),
            terminal_usage_error_redirect_after=_positive_int(
                hard_stop_after.get(
                    "terminal_usage_error_redirect",
                    data.get("terminal_usage_error_redirect_after"),
                ),
                defaults.terminal_usage_error_redirect_after,
            ),
            terminal_usage_error_halt_after=_positive_int(
                hard_stop_after.get("terminal_usage_error", data.get("terminal_usage_error_halt_after")),
                defaults.terminal_usage_error_halt_after,
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

    action: str = "allow"  # allow | warn | redirect | block | halt
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
        self._low_information_counts: dict[tuple[str, str], int] = {}
        self._tool_redirects: dict[str, ToolGuardrailDecision] = {}
        self._terminal_usage_error_counts: dict[str, int] = {}
        self._terminal_usage_redirects: dict[str, ToolGuardrailDecision] = {}
        self._halt_decision: ToolGuardrailDecision | None = None

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        return self._halt_decision

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        if tool_name == "terminal":
            usage_family = _terminal_command_family(args)
            if usage_family:
                usage_redirect = self._terminal_usage_redirects.get(usage_family)
                if usage_redirect is not None:
                    if _is_terminal_usage_diagnostic_command(args, usage_family):
                        return ToolGuardrailDecision(tool_name=tool_name, signature=signature)
                    redirected_count = usage_redirect.count + 1
                    action = (
                        "halt"
                        if self.config.hard_stop_enabled
                        and redirected_count >= self.config.terminal_usage_error_halt_after
                        else "redirect"
                    )
                    code = (
                        "terminal_usage_error_halt"
                        if action == "halt"
                        else "terminal_usage_error_redirect"
                    )
                    decision = ToolGuardrailDecision(
                        action=action,
                        code=code,
                        message=_terminal_usage_error_recovery_hint(usage_family, redirected_count),
                        tool_name=tool_name,
                        count=redirected_count,
                        signature=signature,
                    )
                    self._terminal_usage_redirects[usage_family] = decision
                    if decision.should_halt:
                        self._halt_decision = decision
                    return decision

        redirect = self._tool_redirects.get(tool_name)
        if redirect is not None:
            if tool_name == "terminal" and _terminal_probe_family(args) != "filter_probe":
                self._clear_low_information_state(tool_name)
                return ToolGuardrailDecision(tool_name=tool_name, signature=signature)
            redirected_count = redirect.count + 1
            action = (
                "halt"
                if self.config.hard_stop_enabled
                and redirected_count >= self.config.low_information_halt_after
                else "redirect"
            )
            code = (
                "low_information_tool_halt"
                if action == "halt"
                else "low_information_tool_redirect"
            )
            decision = ToolGuardrailDecision(
                action=action,
                code=code,
                message=_low_information_recovery_hint(tool_name, redirected_count),
                tool_name=tool_name,
                count=redirected_count,
                signature=signature,
            )
            self._tool_redirects[tool_name] = decision
            if decision.should_halt:
                self._halt_decision = decision
            return decision

        if not self.config.hard_stop_enabled:
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

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

        if self._is_idempotent_call(tool_name, args):
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

            tool_block = _tool_reported_loop_block(tool_name, result)
            if tool_block:
                decision = ToolGuardrailDecision(
                    action="halt",
                    code="tool_reported_loop_block",
                    message=(
                        f"Stopped {tool_name}: the tool reported a repeated-call "
                        f"loop block after {tool_block['count']} attempts. "
                        "Use the information already returned, narrow the query, "
                        "or switch to a different tool path instead of retrying "
                        "the same call."
                    ),
                    tool_name=tool_name,
                    count=tool_block["count"],
                    signature=signature,
                )
                self._halt_decision = decision
                return decision

            terminal_usage_family = _terminal_usage_error_family(tool_name, args, result)
            if terminal_usage_family:
                usage_count = self._terminal_usage_error_counts.get(terminal_usage_family, 0) + 1
                self._terminal_usage_error_counts[terminal_usage_family] = usage_count

                if (
                    self.config.hard_stop_enabled
                    and usage_count >= self.config.terminal_usage_error_halt_after
                ):
                    decision = ToolGuardrailDecision(
                        action="halt",
                        code="terminal_usage_error_halt",
                        message=_terminal_usage_error_recovery_hint(terminal_usage_family, usage_count),
                        tool_name=tool_name,
                        count=usage_count,
                        signature=signature,
                    )
                    self._terminal_usage_redirects[terminal_usage_family] = decision
                    self._halt_decision = decision
                    return decision

                if usage_count >= self.config.terminal_usage_error_redirect_after:
                    decision = ToolGuardrailDecision(
                        action="redirect",
                        code="terminal_usage_error_redirect",
                        message=_terminal_usage_error_recovery_hint(terminal_usage_family, usage_count),
                        tool_name=tool_name,
                        count=usage_count,
                        signature=signature,
                    )
                    self._terminal_usage_redirects[terminal_usage_family] = decision
                    return decision

                if (
                    self.config.warnings_enabled
                    and usage_count >= self.config.terminal_usage_error_warn_after
                ):
                    return ToolGuardrailDecision(
                        action="warn",
                        code="terminal_usage_error_warning",
                        message=_terminal_usage_error_recovery_hint(terminal_usage_family, usage_count),
                        tool_name=tool_name,
                        count=usage_count,
                        signature=signature,
                    )

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

        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)
        if tool_name == "terminal":
            terminal_family = _terminal_command_family(args)
            if terminal_family:
                clear_family, include_children = _terminal_usage_clear_target(args, terminal_family)
                self._clear_terminal_usage_state(
                    clear_family,
                    include_children=include_children,
                )

        if not self._is_idempotent_call(tool_name, args):
            self._no_progress.pop(signature, None)
            self._clear_low_information_state()
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        low_info_kind = _low_information_result(tool_name, args, result)
        if low_info_kind is not None:
            key = (tool_name, low_info_kind)
            low_info_count = self._low_information_counts.get(key, 0) + 1
            self._low_information_counts[key] = low_info_count
            if low_info_count >= self.config.low_information_redirect_after:
                redirect = ToolGuardrailDecision(
                    action="redirect",
                    code="low_information_tool_redirect",
                    message=_low_information_recovery_hint(tool_name, low_info_count),
                    tool_name=tool_name,
                    count=low_info_count,
                    signature=signature,
                )
                self._tool_redirects[tool_name] = redirect
                return ToolGuardrailDecision(
                    action="warn",
                    code="low_information_strategy_warning",
                    message=_low_information_recovery_hint(tool_name, low_info_count),
                    tool_name=tool_name,
                    count=low_info_count,
                    signature=signature,
                )
            if self.config.warnings_enabled and low_info_count >= self.config.low_information_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="low_information_strategy_warning",
                    message=_low_information_recovery_hint(tool_name, low_info_count),
                    tool_name=tool_name,
                    count=low_info_count,
                    signature=signature,
                )
        else:
            self._clear_low_information_state()

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

    def _is_idempotent_call(self, tool_name: str, args: Mapping[str, Any]) -> bool:
        if tool_name == "terminal" and _terminal_probe_family(args) is not None:
            return True
        return self._is_idempotent(tool_name)

    def _clear_low_information_state(self, tool_name: str | None = None) -> None:
        if tool_name is None:
            self._low_information_counts.clear()
            self._tool_redirects.clear()
            return
        for key in list(self._low_information_counts):
            if key[0] == tool_name:
                self._low_information_counts.pop(key, None)
        self._tool_redirects.pop(tool_name, None)

    def _clear_terminal_usage_state(self, family: str | None = None, *, include_children: bool = False) -> None:
        if family is None:
            self._terminal_usage_error_counts.clear()
            self._terminal_usage_redirects.clear()
            return
        families = {family}
        if include_children:
            prefix = f"{family} "
            families.update(
                candidate
                for candidate in set(self._terminal_usage_error_counts) | set(self._terminal_usage_redirects)
                if candidate.startswith(prefix)
            )
        for candidate in families:
            self._terminal_usage_error_counts.pop(candidate, None)
            self._terminal_usage_redirects.pop(candidate, None)


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
    if decision.action not in {"warn", "redirect", "halt"} or not decision.message:
        return result
    if decision.action == "halt":
        label = "Tool loop hard stop"
    elif decision.action == "redirect":
        label = "Tool strategy redirect"
    else:
        label = "Tool loop warning"
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
            "working directory, or a different tool such as read_file/write_file/patch. "
            "If a git commit or PR helper fails, verify the commit/PR state before "
            "claiming it landed; for MeshBoard CLI Python failures, try the repo's "
            "known Python such as python3.11 instead of repeating filtered probes."
        )
    return common + (
        "Try different arguments, a narrower query/path, an absolute path when relevant, "
        "or a different tool that can make progress. If the blocker is external, report "
        "the blocker after one diagnostic attempt instead of repeating the same failing path."
    )


def _terminal_usage_error_recovery_hint(family: str, count: int) -> str:
    """Guidance for repeated CLI parser/usage failures in one command family."""
    executable = family.split(" ", 1)[0]
    return (
        f"terminal has hit CLI usage errors for `{family}` {count} times this turn. "
        "Stop guessing new argument variants in the same command family. "
        f"Run a help or discovery command such as `{family} --help`, "
        f"`{executable} help`, `which {executable}`, or inspect the CLI source/docs "
        "before trying this family again. If the command surface is unavailable, "
        "switch tools or report the blocker instead of continuing the argument loop."
    )


def _low_information_recovery_hint(tool_name: str, count: int) -> str:
    """Guidance for repeated successful tool calls that returned no usable facts."""
    common = (
        f"{tool_name} has returned low-information results {count} times this turn. "
        "This is no longer an information-gathering phase; change strategy before "
        "calling the same tool again. "
    )
    if tool_name == "search_files":
        return common + (
            "Likely causes are the wrong search root, wrong target mode, or a query "
            "that is too abstract for grep-style search. First verify cwd/path with "
            "another tool such as terminal (`pwd`, `rg --files`, `rg -n`) or inspect "
            "a known candidate file with read_file; do not keep varying the same "
            "empty search."
        )
    if tool_name == "read_file":
        return common + (
            "The file content already in the conversation is still current. Use it "
            "to edit/respond, read a different range only if you need new lines, or "
            "switch to search_files/terminal to locate a different file."
        )
    if tool_name == "terminal":
        return common + (
            "Stop stacking filtered shell probes that return blank output. Run one "
            "broad diagnostic such as `pwd && ls -la`, inspect a concrete file, "
            "create a clean worktree from current main, or report the stale-state "
            "blocker instead of repeating `grep | head` / `git diff --name-only` "
            "variants."
        )
    return common + "Use a different tool path, make an edit, ask for clarification, or report the blocker."


def _low_information_result(tool_name: str, args: Mapping[str, Any], result: str | None) -> str | None:
    """Classify successful but non-progressing read-only results.

    Exact-repeat detection catches identical calls. This catches the systemic
    loop class where a model makes small query variations that all produce the
    same empty/unchanged payload, so exact-signature counters never fire.
    """
    parsed = safe_json_loads(result or "")
    if not isinstance(parsed, dict) or parsed.get("error"):
        return None

    if tool_name == "search_files":
        if (
            parsed.get("total_count") == 0
            and not parsed.get("matches")
            and not parsed.get("files")
            and not parsed.get("counts")
        ):
            return "empty_search_result"
        return None

    if tool_name == "read_file":
        if (
            parsed.get("status") == "unchanged"
            and parsed.get("dedup") is True
            and parsed.get("content_returned") is False
        ):
            return "unchanged_read_stub"

    if tool_name == "terminal":
        family = _terminal_probe_family(args)
        if family is None:
            return None
        exit_code = parsed.get("exit_code")
        if exit_code not in (None, 0):
            return None
        if not _terminal_result_text(parsed).strip():
            return f"empty_terminal_{family}"
        return None

    return None


def _terminal_result_text(parsed: Mapping[str, Any]) -> str:
    parts = []
    for key in ("stdout", "stderr", "output", "content", "error"):
        value = parsed.get(key)
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


def _terminal_usage_error_family(
    tool_name: str,
    args: Mapping[str, Any],
    result: str | None,
) -> str | None:
    if tool_name != "terminal":
        return None
    family = _terminal_command_family(args)
    if not family:
        return None

    parsed = safe_json_loads(result or "")
    exit_code: Any = None
    if isinstance(parsed, Mapping):
        exit_code = parsed.get("exit_code")
        text = _terminal_result_text(parsed)
    else:
        text = result or ""

    lowered = text[:5000].lower()
    has_usage = bool(re.search(r"(^|\n)\s*usage:\s", lowered))
    strong_usage_error = bool(
        re.search(
            r"\b("
            r"unrecognized arguments?|unknown option|unknown command|invalid command|"
            r"invalid choice|no such option|missing required|too few arguments|"
            r"too many arguments|subcommand required|bad option|illegal option|"
            r"requires at least"
            r")\b",
            lowered,
        )
    )
    help_hint = "--help" in lowered or " help" in lowered

    if exit_code in (2, 64) and (has_usage or strong_usage_error or help_hint):
        return family
    if strong_usage_error and (has_usage or help_hint or exit_code not in (None, 0)):
        return family
    return None


def _terminal_command_family(args: Mapping[str, Any]) -> str | None:
    command = args.get("command", args.get("cmd", ""))
    if not isinstance(command, str):
        return None
    segment = _terminal_primary_command_segment(command)
    if not segment:
        return None
    try:
        tokens = shlex.split(segment)
    except ValueError:
        tokens = segment.split()
    tokens = _strip_terminal_wrappers(tokens)
    if not tokens:
        return None

    base = tokens[0].rsplit("/", 1)[-1]
    rest = tokens[1:]
    if base.startswith("python"):
        python_family = _python_invoked_cli_family(rest)
        if python_family:
            return python_family
    if base.endswith(".py"):
        base = base[:-3]

    family = [base]
    for token in rest:
        if token.startswith("-") or "=" in token:
            break
        if token in {"2>&1", "1>&2"}:
            break
        family.append(token)
        break
    return " ".join(family)


def _terminal_primary_command_segment(command: str) -> str:
    command = _normalize_terminal_command(command)
    if not command:
        return ""
    chain_parts = [part.strip() for part in re.split(r"\s*(?:&&|;|\|\|)\s*", command) if part.strip()]
    segment = chain_parts[-1] if chain_parts else command
    segment = segment.split("|", 1)[0].strip()
    return segment


def _strip_terminal_wrappers(tokens: list[str]) -> list[str]:
    while tokens:
        token = tokens[0]
        if token == "env":
            tokens = tokens[1:]
            continue
        if "=" in token and not token.startswith("-") and token.split("=", 1)[0].isidentifier():
            tokens = tokens[1:]
            continue
        break
    return tokens


def _python_invoked_cli_family(tokens: list[str]) -> str | None:
    if not tokens:
        return None
    tokens = list(tokens)
    while tokens and tokens[0].startswith("-") and tokens[0] not in {"-m", "-c"}:
        tokens = tokens[1:]
    if len(tokens) >= 2 and tokens[0] == "-m":
        base = tokens[1].rsplit(".", 1)[-1]
        rest = tokens[2:]
    elif tokens and tokens[0].endswith(".py"):
        base = tokens[0].rsplit("/", 1)[-1][:-3]
        rest = tokens[1:]
    else:
        return None
    if base == "meshctl":
        base = "meshctl"
    family = [base]
    for token in rest:
        if token.startswith("-") or "=" in token:
            break
        family.append(token)
        break
    return " ".join(family)


def _is_terminal_usage_diagnostic_command(args: Mapping[str, Any], family: str) -> bool:
    command = args.get("command", args.get("cmd", ""))
    if not isinstance(command, str):
        return False
    if _terminal_availability_probe_target(command):
        return True
    lowered = command.lower()
    executable = re.escape(family.split(" ", 1)[0].lower())
    if re.search(r"(^|\s)(--help|-h|help)(\s|$)", lowered):
        return True
    if re.search(rf"\b(command\s+-v|which|type)\s+{executable}\b", lowered):
        return True
    if re.search(rf"\b{executable}\b[^\n;&|]*\b(version|doctor|diagnose|diagnostics)\b", lowered):
        return True
    return False


def _terminal_usage_clear_target(args: Mapping[str, Any], family: str) -> tuple[str, bool]:
    command = args.get("command", args.get("cmd", ""))
    if isinstance(command, str):
        target = _terminal_availability_probe_target(command)
        if target:
            return target, True
    return family, _is_terminal_usage_diagnostic_command(args, family)


def _terminal_availability_probe_target(command: str) -> str | None:
    segment = _terminal_primary_command_segment(command)
    if not segment:
        return None
    try:
        tokens = shlex.split(segment)
    except ValueError:
        tokens = segment.split()
    tokens = _strip_terminal_wrappers(tokens)
    if not tokens:
        return None
    if tokens[0] == "command" and len(tokens) >= 3 and tokens[1] == "-v":
        return _normalize_command_name(tokens[2])
    if tokens[0] in {"which", "type"} and len(tokens) >= 2 and not tokens[1].startswith("-"):
        return _normalize_command_name(tokens[1])
    return None


def _normalize_command_name(token: str) -> str:
    name = token.rsplit("/", 1)[-1]
    if name.endswith(".py"):
        return name[:-3]
    return name


def _terminal_probe_family(args: Mapping[str, Any]) -> str | None:
    command = args.get("command", args.get("cmd", ""))
    if not isinstance(command, str):
        return None
    command = _normalize_terminal_command(command)
    if not command or _has_mutating_shell_signal(command):
        return None
    if _is_filtered_terminal_probe(command):
        return "filter_probe"
    if _is_read_only_terminal_probe(command):
        return "read_probe"
    return None


def _normalize_terminal_command(command: str) -> str:
    command = command.strip()
    # Common harmless stderr redirections should not make a read probe look
    # mutating. Keep this conservative so arbitrary file writes still opt out.
    command = re.sub(r"\s+\d?>\s*/dev/null\b", "", command)
    command = command.replace(" 2>&1", "")
    command = re.sub(r"\bcd\s+[^;&|]+&&\s*", "", command)
    return command.strip()


def _has_mutating_shell_signal(command: str) -> bool:
    lowered = command.lower()
    mutating_command_re = (
        r"(^|[;&|]\s*)"
        r"(rm|mv|cp|mkdir|touch|chmod|chown|"
        r"git\s+(add|commit|push|checkout|switch|reset|clean|worktree\s+add)|"
        r"python\d?|python3|node|npm|pnpm|yarn|uv|pip|sed\s+-i|perl\s+-i)\b"
    )
    if re.search(mutating_command_re, lowered):
        return True
    if re.search(r"(^|[^0-9])>>?", command) and "/dev/null" not in command:
        return True
    if re.search(r"\btee\s+", lowered):
        return True
    return False


def _is_filtered_terminal_probe(command: str) -> bool:
    lowered = command.lower()
    if re.search(r"\|\s*(grep|head|tail|wc|sort|uniq)\b", lowered):
        return True
    if re.search(r"\bgrep\s+-r\b|\brg\s+.*\|\s*head\b", lowered):
        return True
    if "git diff" in lowered and ("--name-only" in lowered or "--stat" in lowered):
        return True
    return False


def _is_read_only_terminal_probe(command: str) -> bool:
    readonly_prefixes = (
        "pwd",
        "ls",
        "find",
        "rg",
        "grep",
        "cat",
        "sed -n",
        "head",
        "tail",
        "wc",
        "git log",
        "git show",
        "git diff",
        "git status",
        "git branch",
        "git remote",
        "git rev-parse",
        "git merge-base",
        "git worktree list",
        "meshctl task list",
        "meshctl task show",
        "meshctl worktree list",
    )
    lowered = command.lower().lstrip()
    return lowered.startswith(readonly_prefixes)


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


def _tool_reported_loop_block(tool_name: str, result: str | None) -> dict[str, int] | None:
    """Return metadata when a tool already enforced a repeated-call block.

    File/search tools emit explicit ``BLOCKED:`` JSON errors after a model
    repeats the exact same read/search enough times. Those are stronger than
    ordinary tool failures: the tool has already proven the next identical call
    cannot make progress, so the agent loop should halt even when the broader
    guardrail hard-stop mode is left at its conservative default.
    """
    parsed = safe_json_loads(result or "")
    if not isinstance(parsed, dict):
        return None

    error = parsed.get("error")
    if not isinstance(error, str) or not error.startswith("BLOCKED:"):
        return None

    count = parsed.get("already_searched", parsed.get("already_read"))
    if not isinstance(count, int) or count < 1:
        return None

    if (
        tool_name in {"search_files", "read_file", "mcp_filesystem_read_file"}
        or "exact search" in error
        or "exact file region" in error
        or "exact region" in error
    ):
        return {"count": count}
    return None


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
