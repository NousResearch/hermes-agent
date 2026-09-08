"""Pure tool-call loop guardrail primitives.

The controller is side-effect free: it tracks per-turn tool-call observations
and returns decisions. Runtime code decides whether a decision becomes warning
guidance, a synthetic tool result, or a controlled turn halt.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

from utils import safe_json_loads
from agent.tool_result_classification import file_mutation_result_landed


IDEMPOTENT_TOOL_NAMES = frozenset({
    "read_file", "search_files", "web_search", "web_extract", "session_search", "skill_view", "skills_list",
    "browser_snapshot", "browser_console", "browser_get_images", "mcp_filesystem_read_file",
    "mcp_filesystem_read_text_file", "mcp_filesystem_read_multiple_files", "mcp_filesystem_list_directory",
    "mcp_filesystem_list_directory_with_sizes", "mcp_filesystem_directory_tree", "mcp_filesystem_get_file_info",
    "mcp_filesystem_search_files",
})

MUTATING_TOOL_NAMES = frozenset({
    "terminal", "execute_code", "write_file", "patch", "todo_list", "memory", "skill_manage",
    "browser_click", "browser_type", "browser_press", "browser_scroll", "browser_navigate",
    "send_message", "cronjob_manage", "delegate_task", "process_manage",
})

# Pollers: legitimately re-invoked with identical args; the identical-call NOTICE never fires.
STALL_GUARD_REPEATABLE_TOOLS = frozenset({"process_manage"})
_STALL_GUARD_REPEATABLE_SUFFIXES = ("_get_result", "_poll")  # generated / MCP poller conventions
# Nth consecutive identical (tool, args, result) call that fires the notice; 3 tolerates one double-check.
STALL_GUARD_IDENTICAL_CALL_THRESHOLD = 3
# From the 2nd byte-identical repeat the duplicate payload becomes a reference stub; smaller results
# aren't worth it, errors never are. The args preview keeps WHAT was called if compression evicts the original.
IDENTICAL_RESULT_STUB_MIN_CHARS = 512
_RESULT_STUB_ARGS_PREVIEW_CHARS = 120

# Tools whose "failure" is normal work output (red test run, empty grep, page timeout).
# same_tool_failure (DIFFERENT commands) never halts these; only an exact-args replay with
# no intervening change, or an identical-result streak, can.
FAILURE_TOLERANT_TOOL_NAMES = frozenset({
    "terminal", "execute_code", "process_manage", "process", "browser_navigate", "web_extract",
})

# A successful call to one of these marks progress for every failing signature still counted
# this turn: the next retry is a new experiment (edit -> re-run), not a replay.
PROGRESS_RESET_TOOL_NAMES = frozenset({
    "write_file", "patch", "terminal", "execute_code", "browser_click", "browser_type", "browser_press",
    "browser_navigate", "process_manage", "process", "delegate_task", "send_message", "cronjob",
    "cronjob_manage", "todo", "todo_list", "memory", "skill_manage",
})

_BOOL_FIELDS = (
    "warnings_enabled",
    "hard_stop_enabled",
    "non_interactive_hard_stop_enabled",
    "finalize_on_complete_composite",
)
# Threshold field -> (nested section, nested key). The flat legacy key is the field name itself.
_THRESHOLD_SOURCES: dict[str, tuple[str, str]] = {
    "exact_failure_warn_after": ("warn_after", "exact_failure"),
    "same_tool_failure_warn_after": ("warn_after", "same_tool_failure"),
    "no_progress_warn_after": ("warn_after", "idempotent_no_progress"),
    "exact_failure_block_after": ("hard_stop_after", "exact_failure"),
    "same_tool_failure_halt_after": ("hard_stop_after", "same_tool_failure"),
    "no_progress_block_after": ("hard_stop_after", "idempotent_no_progress"),
}

# Per-turn caps on runaway-prone tools (counters reset in reset_for_turn).
_DEFAULT_MAX_WEB_SEARCHES_PER_TURN = 50
_DEFAULT_MAX_SUBAGENTS_PER_TURN = 50

# Interactive surfaces plus bounded supervised task loops (subagent stopped by its parent;
# api_server has a live client) doing real edit -> re-run work keep the warn-only default.
_ATTENDED_PLATFORMS = frozenset({"cli", "tui", "desktop", "acp", "subagent", "api_server"})


def is_stall_guard_repeatable(tool_name: str) -> bool:
    """Whether a tool is exempt from the identical-call loop notice."""
    return tool_name in STALL_GUARD_REPEATABLE_TOOLS or tool_name.endswith(_STALL_GUARD_REPEATABLE_SUFFIXES)


def _is_non_interactive_platform(platform: str | None) -> bool:
    """True for gateway/cron sessions where tool loops are unattended."""
    if not isinstance(platform, str) or not platform.strip():
        return False
    return platform.strip().lower() not in _ATTENDED_PLATFORMS


@dataclass(frozen=True)
class LoopCapConfig:
    """Per-turn hard ceilings on web_search calls / subagent spawns; count total calls (not
    repeats), fire regardless of ``hard_stop_enabled``; ``0`` disables a cap."""

    max_web_searches: int = _DEFAULT_MAX_WEB_SEARCHES_PER_TURN
    max_subagents: int = _DEFAULT_MAX_SUBAGENTS_PER_TURN

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "LoopCapConfig":
        """Build config from the ``tool_loop_guardrails.loop_caps`` section."""
        if not isinstance(data, Mapping):
            return cls()
        return cls(**{f.name: _int_at_least(data.get(f.name), f.default, 0) for f in fields(cls)})


@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """Thresholds for per-turn tool-call loop detection. Warnings never prevent execution; hard
    stops are opt-in on interactive platforms, default on for unattended gateway/cron platforms."""

    warnings_enabled: bool = True
    hard_stop_enabled: bool = False
    non_interactive_hard_stop_enabled: bool = True
    # Single-scope investigations benefit from an immediate tool-free synthesis
    # after a complete composite.  Longer orchestrations can disable only that
    # transition while retaining component coverage and exact-result reuse.
    finalize_on_complete_composite: bool = True
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 5
    idempotent_tools: frozenset[str] = field(default_factory=lambda: IDEMPOTENT_TOOL_NAMES)
    mutating_tools: frozenset[str] = field(default_factory=lambda: MUTATING_TOOL_NAMES)
    loop_caps: LoopCapConfig = field(default_factory=LoopCapConfig)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None, *, platform: str | None = None,
    ) -> "ToolCallGuardrailConfig":
        """Build config from `tool_loop_guardrails`; nested ``warn_after`` / ``hard_stop_after`` win over flat legacy keys."""
        if not isinstance(data, Mapping):
            data = {}
        d = cls()
        flags = {name: _as_bool(data.get(name), getattr(d, name)) for name in _BOOL_FIELDS}
        if flags["non_interactive_hard_stop_enabled"] and _is_non_interactive_platform(platform):
            flags["hard_stop_enabled"] = True

        def threshold(name: str, section_name: str, key: str) -> int:
            section = data.get(section_name)
            nested = section.get(key, data.get(name)) if isinstance(section, Mapping) else data.get(name)
            return _int_at_least(nested, getattr(d, name), 1)

        thresholds = {name: threshold(name, *src) for name, src in _THRESHOLD_SOURCES.items()}
        return cls(loop_caps=LoopCapConfig.from_mapping(data.get("loop_caps")), **flags, **thresholds)


@dataclass(frozen=True)
class IdenticalCallObservation:
    """``notice`` is appended after the result, ``stub`` replaces a byte-identical duplicate result."""

    notice: str | None = None
    stub: str | None = None


@dataclass(frozen=True)
class ToolCallSignature:
    """Stable, non-reversible identity for a tool name plus canonical args."""

    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Mapping[str, Any] | None) -> "ToolCallSignature":
        return cls(tool_name=tool_name, args_hash=_sha256(canonical_tool_args(args or {})))

    def to_metadata(self) -> dict[str, str]:
        """Return public metadata without raw argument values."""
        return asdict(self)


@dataclass(frozen=True)
class ToolGuardrailDecision:
    """Decision returned by the tool-call guardrail controller."""

    action: str = "allow"  # allow | warn | finalize | restrict | reuse | block | halt
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
        data = asdict(self)
        if data["signature"] is None:
            del data["signature"]
        return data


@dataclass(frozen=True)
class RecentRequestProvenance:
    """Origin of recent-request bounds carried from user text to dispatch.

    Tool arguments alone cannot distinguish a number copied from the request
    from one invented by the model.  Preserve that authority boundary before
    the model call and consult it when normalizing emitted MCP arguments.
    """

    is_recent: bool = False
    quantity_source: str = "model_or_default"
    range_source: str = "model_or_default"

    @property
    def is_vague(self) -> bool:
        return (
            self.is_recent
            and self.quantity_source != "user"
            and self.range_source != "user"
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def canonical_tool_args(args: Mapping[str, Any]) -> str:
    """Return sorted compact JSON for parsed tool arguments."""
    if not isinstance(args, Mapping):
        raise TypeError(f"tool args must be a mapping, got {type(args).__name__}")
    return _canonical_json(args)


def classify_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Fallback classifier used only when callers don't pass ``failed``; mirrors
    ``agent.display._detect_tool_failure`` so the guardrail never disagrees with the CLI's ``[error]`` tag."""
    if result is None or file_mutation_result_landed(tool_name, result):
        return False, ""

    if tool_name == "terminal":
        data = safe_json_loads(result)
        exit_code = data.get("exit_code") if isinstance(data, dict) else None
        return (True, f" [exit {exit_code}]") if exit_code is not None and exit_code != 0 else (False, "")

    if tool_name == "memory":
        data = safe_json_loads(result)
        if isinstance(data, dict) and data.get("success") is False and "exceed the limit" in data.get("error", ""):
            return True, " [full]"
    lower = result[:500].lower()
    return (True, " [error]") if '"error"' in lower or '"failed"' in lower or result.startswith("Error") else (False, "")


# Guardrail verdict text injected into the conversation, keyed by decision code.
# ``same_tool_failure_warning`` is built by _tool_failure_recovery_hint (tool-specific).
_DECISION_MESSAGES: dict[str, str] = {
    "repeated_exact_failure_block": (
        "Blocked {tool_name}: the same tool call failed {count} times with identical arguments. "
        "Stop retrying it unchanged; change strategy or explain the blocker."
    ),
    "idempotent_no_progress_block": (
        "Blocked {tool_name}: this read-only call returned the same result {count} times. "
        "Stop repeating it unchanged; use the result already provided or try a different query."
    ),
    "same_tool_failure_halt": (
        "Stopped {tool_name}: it failed {count} times this turn. "
        "Stop retrying the same failing tool path and choose a different approach."
    ),
    "repeated_exact_failure_warning": (
        "{tool_name} has failed {count} times with identical arguments. This looks like a loop; "
        "inspect the error and change strategy instead of retrying it unchanged."
    ),
    "idempotent_no_progress_warning": (
        "{tool_name} returned the same result {count} times. Use the result already provided "
        "or change the query instead of repeating it unchanged."
    ),
    "identical_call_streak_halt": (
        "Stopped {tool_name}: the same call with identical arguments returned the same result "
        "{count} times in a row. Stop repeating it unchanged; use the result already provided or change strategy."
    ),
    "loop_web_search_cap": (
        "Blocked web_search: this turn has already made {cap} web searches, the per-turn limit. "
        "This looks like a runaway search loop. Work with the results you already have and give the user your answer."
    ),
    "loop_subagent_cap": (
        "Blocked delegate_task: this turn has already spawned {count} subagents (limit {cap}). "
        "This looks like a runaway delegation loop. Finish the work with the results you have and answer the user."
    ),
}

_IDENTICAL_CALL_NOTICE = (
    "[hermes note: this is the {ordinal} consecutive identical call to "
    "{tool_name} with identical arguments returning the same result. "
    "Do not repeat it — change arguments, use a different tool, or "
    "proceed with what you have.]"
)

# tool -> (LoopCapConfig field, controller counter attribute, decision code)
_LOOP_CAPS: dict[str, tuple[str, str, str]] = {
    "web_search": ("max_web_searches", "_turn_web_search_count", "loop_web_search_cap"),
    "delegate_task": ("max_subagents", "_turn_subagent_count", "loop_subagent_cap"),
}


class ToolCallGuardrailController:
    """Per-turn controller for repeated failed/non-progressing tool calls."""

    def __init__(self, config: ToolCallGuardrailConfig | None = None):
        self.config = config or ToolCallGuardrailConfig()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        # signature -> a mutating call succeeded since its last failure
        self._progress_since_failure: dict[ToolCallSignature, bool] = {}
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        self._halt_decision: ToolGuardrailDecision | None = None
        # Identical-call streak: CONSECUTIVE identical (tool, args, result) calls; any different call or
        # result resets it, so re-reads after edits and varied polling are never flagged.
        # Identical-call loop-breaker state (agent.stall_guards): tracks the CONSECUTIVE streak of identical
        # (tool, canonical args) calls whose results were also identical. Per-turn, like everything else
        # here. NOTE: open PR #85352 (patrykkopycinski) tracks no-progress loops ACROSS turns via a
        # detection window — a different mechanism from this per-turn consecutive streak. Coordinate future
        # work there.
        self._identical_streak_sig: ToolCallSignature | None = None
        self._identical_streak_result_hash: str = ""
        self._identical_streak_count: int = 0
        self._identical_streak_first_call_id: str = ""
        # tool_call_id -> spillover path, so a stub referencing a persisted-output preview can't dangle.
        self._persisted_result_paths: dict[str, str] = {}
        # MCP servers can explicitly declare that an unchanged successful result
        # is reusable. Composite results can also declare which component reads
        # they already assembled. Both caches are per turn and store only hashed
        # call signatures or in-memory native identifiers.
        self._reusable_mcp_calls: set[ToolCallSignature] = set()
        self._mcp_composite_coverage: list[_CompositeCoverage] = []
        self._capability_only = False
        self._recent_request = RecentRequestProvenance()
        self._turn_web_search_count = 0
        self._turn_subagent_count = 0

    def set_capability_only(self, enabled: bool) -> None:
        """Restrict this turn to MCP capability/schema reads when explicitly requested."""
        self._capability_only = bool(enabled)

    def set_vague_recent(self, enabled: bool) -> None:
        """Compatibility shim for callers without request-bound provenance."""
        self._recent_request = RecentRequestProvenance(is_recent=bool(enabled))

    def set_recent_request_provenance(self, provenance: RecentRequestProvenance) -> None:
        """Carry user/default ownership of recent-request bounds into dispatch."""
        self._recent_request = (
            provenance if isinstance(provenance, RecentRequestProvenance)
            else RecentRequestProvenance()
        )

    def normalize_args(self, tool_name: str, args: Mapping[str, Any] | None) -> Mapping[str, Any]:
        """Bound vague-recent MCP reads before dispatch; explicit user ranges are excluded."""
        original = _coerce_args(args)
        server, _component = _mcp_tool_parts(tool_name)
        provenance = self._recent_request
        if not provenance.is_recent or server is None:
            return original
        normalized = dict(original)
        count = normalized.get("count")
        if (provenance.quantity_source != "user" and (
                (isinstance(count, (int, float)) and not isinstance(count, bool) and count > 10)
                or (isinstance(count, str) and count.strip().isdigit() and int(count) > 10))):
            normalized["count"] = 10
        start, end = normalized.get("start"), normalized.get("end")
        parsed_start, parsed_end = _parse_timestamp(start), _parse_timestamp(end)
        try:
            oversized_window = (
                parsed_start is not None and parsed_end is not None
                and parsed_end - parsed_start > timedelta(hours=24)
            )
        except TypeError:  # mixed naive/aware timestamps: leave untouched for schema validation
            oversized_window = False
        if oversized_window and provenance.range_source != "user":
            normalized["start"] = _format_timestamp_like(parsed_end - timedelta(hours=24), start)
        return normalized

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        return self._halt_decision

    def _decide(
        self, action: str, code: str, tool_name: str, count: int, signature: ToolCallSignature,
        *, message: str | None = None, **fmt: Any,
    ) -> ToolGuardrailDecision:
        """Build a warn/block/halt decision; block/halt is also recorded as the turn's halt decision."""
        if message is None:
            message = _DECISION_MESSAGES[code].format(tool_name=tool_name, count=count, **fmt)
        decision = ToolGuardrailDecision(action, code, message, tool_name, count, signature)
        if decision.should_halt:
            self._halt_decision = decision
        return decision

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        allow = ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        restricted_name = _underlying_tool_name(tool_name, args)
        restricted_server, restricted_component = _mcp_tool_parts(restricted_name)
        if (self._capability_only and restricted_server is not None
                and restricted_component not in {"get_capabilities", "get_agent_guide"}):
            return self._decide(
                "restrict", "capability_only_scope", tool_name, 1, signature,
                message=(
                    f"Skipped {restricted_name}: this turn explicitly requested capability "
                    "introspection without domain investigation. Synthesize the answer from "
                    "get_capabilities and, if already present, get_agent_guide."
                ),
            )

        if signature in self._reusable_mcp_calls:
            return self._decide(
                "reuse", "reusable_result_already_available", tool_name, 1, signature,
                message=(
                    f"Skipped {tool_name}: an identical successful MCP result is already "
                    "available earlier in this turn. Use that result and finish the answer."
                ),
            )
        covered = self._covered_by_composite(tool_name, args)
        if covered:
            return self._decide(
                "reuse", "covered_by_composite_result", tool_name, 1, signature,
                message=(
                    f"Skipped {tool_name}: the same native entity and scope are already "
                    "covered by an assembled MCP result that requires no component follow-up. "
                    "Use the existing composite evidence and finish the answer."
                ),
            )

        # Loop caps apply regardless of hard_stop_enabled (which only governs the detector).
        cap_block = self._check_loop_cap(tool_name, args, signature)
        if cap_block is not None or not self.config.hard_stop_enabled:
            return cap_block or allow
        # A mutation since this call last failed makes the retry a new experiment.
        exact_count = 0 if self._progress_since_failure.get(signature) else self._exact_failure_counts.get(signature, 0)
        if exact_count >= self.config.exact_failure_block_after:
            return self._decide("block", "repeated_exact_failure_block", tool_name, exact_count, signature)
        record = self._no_progress.get(signature) if self._is_idempotent(tool_name) else None
        if record is not None and record[1] >= self.config.no_progress_block_after:
            return self._decide("block", "idempotent_no_progress_block", tool_name, record[1], signature)
        return allow

    def after_call(
        self, tool_name: str, args: Mapping[str, Any] | None, result: str | None,
        *, failed: bool | None = None,
    ) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        if failed is None:
            failed, _ = classify_tool_failure(tool_name, result)
        warnings = self.config.warnings_enabled

        if failed:
            # An identical failing call is only a REPLAY if nothing landed in between;
            # a mutation since the last identical failure restarts the exact-args streak.
            if self._progress_since_failure.pop(signature, False):
                self._exact_failure_counts.pop(signature, None)
            exact_count = self._exact_failure_counts[signature] = self._exact_failure_counts.get(signature, 0) + 1
            same_count = self._same_tool_failure_counts[tool_name] = self._same_tool_failure_counts.get(tool_name, 0) + 1
            self._no_progress.pop(signature, None)
            # same_tool_failure counts DIFFERENT args on one tool; for failure-tolerant
            # tools a run of distinct red commands is diagnosis, not a loop — warn, never halt.
            if (
                # Hard-stop widening (#89069 / #100849 bundle): the per-turn no-progress BLOCK above only
                # covers tools in idempotent_tools, so a model replaying the same successful
                # `terminal`/`skill_view` call with a byte-identical result ran until the iteration budget.
                # The consecutive-identical streak is tool-agnostic; when hard stops are enabled, halt at
                # the same idempotent_no_progress threshold. Pollers stay exempt (an unchanged poll is
                # progress).
                self.config.hard_stop_enabled
                and tool_name not in FAILURE_TOLERANT_TOOL_NAMES
                and same_count >= self.config.same_tool_failure_halt_after
            ):
                return self._decide("halt", "same_tool_failure_halt", tool_name, same_count, signature)
            if warnings and exact_count >= self.config.exact_failure_warn_after:
                return self._decide("warn", "repeated_exact_failure_warning", tool_name, exact_count, signature)
            if warnings and same_count >= self.config.same_tool_failure_warn_after:
                return self._decide(
                    "warn", "same_tool_failure_warning", tool_name, same_count, signature,
                    message=_tool_failure_recovery_hint(tool_name, same_count),
                )
            return ToolGuardrailDecision(tool_name=tool_name, count=exact_count, signature=signature)

        assembled_complete = self._record_mcp_result_contract(tool_name, args, result, signature)
        _server, component = _mcp_tool_parts(_underlying_tool_name(tool_name, args))
        capability_complete = self._capability_only and component in {
            "get_capabilities", "get_agent_guide",
        }
        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)
        # A successful mutation is progress for every failing signature still counted
        # this turn. Pure loops never mutate between attempts, so the replay detector keeps its teeth.
        if tool_name in PROGRESS_RESET_TOOL_NAMES or file_mutation_result_landed(tool_name, result):
            self._progress_since_failure.update(dict.fromkeys(self._exact_failure_counts, True))
            self._same_tool_failure_counts.clear()
        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            if (
                assembled_complete and self.config.finalize_on_complete_composite
            ) or capability_complete:
                return self._decide(
                    "finalize",
                    "capability_evidence_collected" if capability_complete else "composite_evidence_assembled",
                    tool_name, 1, signature,
                    message=(
                        "The requested capability evidence is available. Synthesize the capability-only answer now."
                        if capability_complete else
                        "The MCP result is assembled and explicitly requires no component follow-up. "
                        "Synthesize the answer now from its included evidence."
                    ),
                )
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        result_hash = _result_hash(result)
        previous = self._no_progress.get(signature)
        repeat_count = previous[1] + 1 if previous is not None and previous[0] == result_hash else 1
        self._no_progress[signature] = (result_hash, repeat_count)
        if warnings and repeat_count >= self.config.no_progress_warn_after:
            return self._decide("warn", "idempotent_no_progress_warning", tool_name, repeat_count, signature)
        return ToolGuardrailDecision(tool_name=tool_name, count=repeat_count, signature=signature)

    def _is_idempotent(self, tool_name: str) -> bool:
        return tool_name not in self.config.mutating_tools and tool_name in self.config.idempotent_tools

    def _record_mcp_result_contract(
        self, tool_name: str, args: Mapping[str, Any], result: str | None,
        signature: ToolCallSignature,
    ) -> bool:
        """Remember only explicit structured MCP reuse/completeness contracts."""
        server, _ = _mcp_tool_parts(tool_name)
        structured = _mcp_structured_content(result)
        if server is None or structured is None:
            return False
        mappings = list(_iter_mappings(structured))
        if any(mapping.get("reuseResult") is True for mapping in mappings):
            self._reusable_mcp_calls.add(signature)
        for mapping in mappings:
            state = mapping.get("resultState")
            assembled = mapping.get("assembled") is True or (
                isinstance(state, str) and state.strip().lower() == "assembled"
            )
            includes = mapping.get("includes")
            if not assembled or mapping.get("componentFollowupNeeded") is not False or not isinstance(includes, Mapping):
                continue
            components = frozenset(
                _normalize_component_name(str(name))
                for name, included in includes.items()
                if included is True and _normalize_component_name(str(name))
            )
            if not components:
                continue
            # Associate component coverage only with IDs on the exact contract
            # object (plus the composite call's IDs). Recursing through every
            # evidence row would falsely treat a secondary device as the primary
            # entity whose detail the composite says it included.
            native_identities = frozenset({
                *_native_identities(mapping, recursive=False),
                *_native_identities(args, recursive=False),
            })
            scope_values = frozenset(_scope_values(args))
            self._mcp_composite_coverage.append(
                _CompositeCoverage(server, components, native_identities, scope_values)
            )
            return True
        return False

    def _covered_by_composite(self, tool_name: str, args: Mapping[str, Any]) -> bool:
        server, component = _mcp_tool_parts(tool_name)
        if server is None or component is None:
            return False
        normalized = _normalize_component_name(component)
        call_identities = set(_native_identities(args))
        if not call_identities:
            return False
        call_scope = set(_scope_values(args))
        for coverage in self._mcp_composite_coverage:
            if coverage.server != server or normalized not in coverage.components:
                continue
            if not call_identities.intersection(coverage.native_identities):
                continue
            # An explicitly changed time bound is a changed scope and remains allowed.
            if call_scope and not call_scope.issubset(coverage.scope_values):
                continue
            return True
        return False

    def observe_call(
        self, tool_name: str, args: Mapping[str, Any] | None, result: str | None,
        *, tool_call_id: str = "", failed: bool = False,
    ) -> IdenticalCallObservation:
        """Track consecutive identical calls; return notice + dedupe stub info.

        ``notice`` fires from the threshold-th consecutive identical (tool, args, result) call
        (observational, pollers exempt). ``stub`` replaces the CURRENT result from the 2nd byte-identical
        repeat — the tool still executed, only the context representation is deduplicated, so polling
        semantics survive; pollers are NOT exempt here since an unchanged poll is where it saves most.
        """
        is_plain_str = isinstance(result, str)
        signature = ToolCallSignature.from_call(tool_name, _coerce_args(args))
        result_hash = _result_hash(result) if is_plain_str else ""

        if is_plain_str and (signature, result_hash) == (self._identical_streak_sig, self._identical_streak_result_hash):
            self._identical_streak_count += 1
        else:
            # New streak; non-string (multimodal) results never form one.
            self._identical_streak_sig = signature if is_plain_str else None
            self._identical_streak_result_hash = result_hash
            self._identical_streak_count = 1 if is_plain_str else 0
            self._identical_streak_first_call_id = tool_call_id or ""
        count = self._identical_streak_count

        notice = None
        if not is_stall_guard_repeatable(tool_name) and count >= STALL_GUARD_IDENTICAL_CALL_THRESHOLD:
            notice = _IDENTICAL_CALL_NOTICE.format(ordinal=_ordinal(count), tool_name=tool_name)
            # The no-progress BLOCK in before_call only covers idempotent_tools; this streak
            # is tool-agnostic, so with hard stops on, halt at the same threshold (a model
            # replaying a successful `terminal` call otherwise runs to the budget).
            if self.config.hard_stop_enabled and count >= self.config.no_progress_block_after and self._halt_decision is None:
                self._decide("halt", "identical_call_streak_halt", tool_name, count, signature)

        stub = None
        if is_plain_str and count >= 2 and not failed and len(result) >= IDENTICAL_RESULT_STUB_MIN_CHARS:
            stub = self._build_result_reference_stub(tool_name, args)
        return IdenticalCallObservation(notice=notice, stub=stub)

    def record_persisted_result(self, tool_call_id: str, file_path: str) -> None:
        """Remember the spillover path a persisted result was saved to."""
        if tool_call_id and file_path:
            self._persisted_result_paths[tool_call_id] = file_path

    def _build_result_reference_stub(self, tool_name: str, args: Mapping[str, Any] | None) -> str:
        """Reference stub for a byte-identical duplicate result (tool + args preview)."""
        args_preview = canonical_tool_args(_coerce_args(args))
        if len(args_preview) > _RESULT_STUB_ARGS_PREVIEW_CHARS:
            args_preview = args_preview[:_RESULT_STUB_ARGS_PREVIEW_CHARS] + "…"
        first_id = self._identical_streak_first_call_id
        ref = f" (tool_call_id {first_id})" if first_id else ""
        stub = (
            f"[hermes note: this result is byte-identical to the {tool_name} "
            f"result earlier this turn{ref}. Refer to that result; it has not "
            f"changed. Args: {args_preview}]"
        )
        spill_path = self._persisted_result_paths.get(first_id) if first_id else None
        if spill_path:
            stub += f"\n[The referenced result was persisted to: {spill_path} — page through it with read_file if you need the full content.]"
        return stub

    def _check_loop_cap(
        self, tool_name: str, args: Mapping[str, Any], signature: ToolCallSignature,
    ) -> ToolGuardrailDecision | None:
        """Block once a per-turn cap is reached (BEFORE the call, so the (cap+1)-th is refused), else advance
        the counter and return None. delegate_task control actions spawn nothing and keep working after the cap."""
        spec = _LOOP_CAPS.get(tool_name)
        if spec is None:
            return None
        cap_field, count_attr, code = spec
        cap, count = getattr(self.config.loop_caps, cap_field), getattr(self, count_attr)
        increment = 1 if tool_name == "web_search" else (_subagent_spawn_count(args) if cap else 0)
        if increment and cap and count >= cap:
            return self._decide("block", code, tool_name, count, signature, cap=cap)
        setattr(self, count_attr, count + increment)
        return None


def toolguard_synthetic_result(decision: ToolGuardrailDecision) -> str:
    """Build a synthetic role=tool content string for a blocked tool call."""
    if decision.action in {"reuse", "restrict"}:
        return json.dumps({
            "result": decision.message,
            "reused": decision.action == "reuse",
            "blockedByTurnPolicy": decision.action == "restrict",
            "guardrail": decision.to_metadata(),
        }, ensure_ascii=False)
    return json.dumps({"error": decision.message, "guardrail": decision.to_metadata()}, ensure_ascii=False)


def append_toolguard_guidance(result: str, decision: ToolGuardrailDecision) -> str:
    """Append runtime guidance to the current tool result content."""
    if decision.action not in {"warn", "halt"} or not decision.message:
        return result
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    return (result or "") + f"\n\n[{label}: {decision.code}; count={decision.count}; {decision.message}]"


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


def _ordinal(count: int) -> str:
    return f"{count}{'th' if 11 <= count % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(count % 10, 'th')}"


def _coerce_args(args: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return args if isinstance(args, Mapping) else {}


def _result_hash(result: str | None) -> str:
    parsed = safe_json_loads(result or "")
    return _sha256(_canonical_json(parsed) if parsed is not None else (result or ""))


@dataclass(frozen=True)
class _CompositeCoverage:
    server: str
    components: frozenset[str]
    native_identities: frozenset[tuple[str, str]]
    scope_values: frozenset[str]


def _mcp_tool_parts(tool_name: str) -> tuple[str | None, str | None]:
    """Return the generated MCP server/tool components without guessing names."""
    if not isinstance(tool_name, str) or not tool_name.startswith("mcp_"):
        return None, None
    server, separator, component = tool_name[4:].partition("__")
    return (server, component) if separator and server and component else (None, None)


def _underlying_tool_name(tool_name: str, args: Mapping[str, Any]) -> str:
    """Return a bridge target when present, otherwise the directly emitted tool name."""
    if tool_name == "tool_call":
        target = args.get("name")
        if isinstance(target, str) and target.strip():
            return target.strip()
    return tool_name


def is_capability_only_request(value: Any) -> bool:
    """Detect an explicit capability ask that also forbids operational investigation.

    This intentionally requires both halves. A general question about what a system can
    do is not restricted unless the user also says not to investigate findings.
    """
    text = _request_text(value)
    if not text:
        return False
    normalized = " ".join(text.casefold().split())
    capability_markers = (
        "capability", "capabilities", "capability-only", "available tools",
        "investigation capabilities", "untersuchungsmöglich", "welche untersuch",
    )
    no_investigation_markers = (
        "do not investigate", "without investigating", "no investigation",
        "keine untersuchung", "keine findings", "keine investigation",
        "führe keine untersuchung", "fuehre keine untersuchung",
    )
    return any(marker in normalized for marker in capability_markers) and any(
        marker in normalized for marker in no_investigation_markers
    )


def is_vague_recent_request(value: Any) -> bool:
    """True for a recent-time ask only when the user supplied no concrete bound."""
    return recent_request_provenance(value).is_vague


def recent_request_provenance(value: Any) -> RecentRequestProvenance:
    """Classify recent bounds while retaining whether the user supplied them."""
    text = _request_text(value)
    if not text:
        return RecentRequestProvenance()
    normalized = " ".join(text.casefold().split())
    recent_markers = (
        "recent", "lately", "latest", "newest", "in letzter zeit", "kürzlich",
        "kuerzlich", "aktuell", "neuest", "letzten",
    )
    if not any(marker in normalized for marker in recent_markers):
        return RecentRequestProvenance()
    explicit_quantity = re.search(
        r"\bcount\s*[=:]?\s*\d+\b|"
        r"\b(?:show|list|display|return|get|fetch)\s+(?:me\s+)?(?:the\s+)?\d+\b|"
        r"\bgive\s+me\s+(?:the\s+)?\d+\b|"
        r"\b(?:top|latest|newest|recent|most\s+recent)\s+\d+\b|"
        r"\b\d+\s+(?:most\s+recent|recent|latest|newest|neuest\w*|aktuell\w*|letzte\w*)\b|"
        r"\b\d+\s+(?:findings?|items?|results?|events?|records?|alerts?|devices?)\b",
        normalized,
    )
    explicit_range = re.search(
        r"\b\d{4}-\d{2}-\d{2}\b|"
        r"\b\d+(?:[.,]\d+)?\s*(?:h|hours?|stunden?|d|days?|tage?n?|weeks?|wochen?|months?|monate?n?)\b|"
        r"\b(?:today|yesterday|heute|gestern)\b|"
        r"\b(?:since|until|from|between|seit|bis|zwischen)\b",
        normalized,
    )
    return RecentRequestProvenance(
        is_recent=True,
        quantity_source="user" if explicit_quantity else "model_or_default",
        range_source="user" if explicit_range else "model_or_default",
    )


def _request_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(
            str(part.get("text") or "") for part in value if isinstance(part, Mapping)
        )
    return ""


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_timestamp_like(value: datetime, original: Any) -> str:
    rendered = value.isoformat()
    if isinstance(original, str) and original.strip().endswith("Z"):
        rendered = rendered.replace("+00:00", "Z")
    return rendered


def _mcp_structured_content(result: str | None) -> Any | None:
    parsed = safe_json_loads(result or "")
    if not isinstance(parsed, Mapping):
        return None
    structured = parsed.get("structuredContent")
    if isinstance(structured, (Mapping, list)):
        return structured
    # MCP results with no usable text put structuredContent directly in `result`.
    nested = parsed.get("result")
    return nested if isinstance(nested, (Mapping, list)) else None


def _iter_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _iter_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_mappings(child)


def _normalize_component_name(value: str) -> str:
    normalized = "".join(ch for ch in value.lower() if ch.isalnum())
    for prefix in ("get", "read", "fetch", "list"):
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            return normalized[len(prefix):]
    return normalized


def _native_identities(value: Any, *, recursive: bool = True):
    """Yield normalized ``(field namespace, value)`` native identities."""
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        raw_key = str(key)
        lowered = raw_key.casefold()
        normalized = "".join(ch for ch in lowered if ch.isalnum())
        is_id_key = (
            normalized in {"id", "pbid", "did", "pid", "uuid"}
            or lowered.endswith(("_id", "-id"))
            or (len(raw_key) > 2 and raw_key.endswith(("Id", "ID")))
        )
        if is_id_key and isinstance(child, (str, int)) and not isinstance(child, bool):
            yield normalized, str(child)
        if recursive and isinstance(child, Mapping):
            yield from _native_identities(child, recursive=True)
        elif recursive and isinstance(child, list):
            for item in child:
                if isinstance(item, Mapping):
                    yield from _native_identities(item, recursive=True)


def _scope_values(args: Mapping[str, Any]):
    """Return time-bound argument values used to distinguish a changed read scope."""
    exact_names = {
        "start", "end", "from", "to", "since", "until",
        "starttime", "endtime", "fromtime", "totime",
        "starttimestamp", "endtimestamp",
    }
    for key, value in args.items():
        normalized = "".join(ch for ch in str(key).lower() if ch.isalnum())
        is_bound = normalized in exact_names or normalized.endswith(
            ("starttime", "endtime", "starttimestamp", "endtimestamp", "since", "until")
        )
        if is_bound:
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                yield str(value)


_BOOL_WORDS = {w: True for w in ("1", "true", "yes", "on", "enabled")} | {w: False for w in ("0", "false", "no", "off", "disabled")}


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, (bool, int, float)):
        return bool(value)
    if isinstance(value, str):
        return _BOOL_WORDS.get(value.strip().lower(), default)
    return default


def _int_at_least(value: Any, default: int, minimum: int) -> int:
    """junk/None/below-minimum fall back to default (caps use minimum 0 so 0 = disabled)."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _subagent_spawn_count(args: Mapping[str, Any]) -> int:
    """Subagents one delegate_task call spawns: ``len(tasks)`` for a non-empty batch, else 1; control actions 0."""
    if str(args.get("action") or "").strip().lower() in ("list", "steer", "stop"):
        return 0
    tasks = args.get("tasks")
    return len(tasks) if isinstance(tasks, list) and tasks else 1


def _sha256(value: str) -> str:
    # surrogatepass: web-scraped results can carry unpaired UTF-16 surrogates; a
    # strict encode would raise and take down the conversation loop.
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()
