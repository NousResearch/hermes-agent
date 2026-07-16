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
class ToolCallGuardrailConfig:
    """Thresholds for per-turn tool-call loop detection.

    Warnings are enabled by default and never prevent tool execution. Hard stops
    are explicit opt-in so interactive CLI/TUI sessions get a gentle nudge unless
    the user enables circuit-breaker behavior in config.yaml.
    """

    warnings_enabled: bool = True
    hard_stop_enabled: bool = False
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 5
    repeated_result_warn_after: int = 3
    repeated_result_halt_after: int = 5
    repeated_result_min_chars: int = 200
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
            repeated_result_warn_after=_positive_int(
                warn_after.get("repeated_result", data.get("repeated_result_warn_after")),
                defaults.repeated_result_warn_after,
            ),
            repeated_result_halt_after=_positive_int(
                hard_stop_after.get("repeated_result", data.get("repeated_result_halt_after")),
                defaults.repeated_result_halt_after,
            ),
            repeated_result_min_chars=_positive_int(
                data.get("repeated_result_min_chars"),
                defaults.repeated_result_min_chars,
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
        self._last_result_hash: str | None = None
        self._result_streak: int = 0
        self._halt_decision: ToolGuardrailDecision | None = None

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        return self._halt_decision

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        signature = ToolCallSignature.from_call(tool_name, _coerce_args(args))
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

        return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

    def after_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        result: str | Mapping[str, Any] | None,
        *,
        failed: bool | None = None,
    ) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        if failed is None:
            failed, _ = classify_tool_failure(tool_name, result)

        # Result-repetition loop detection. This is deliberately independent of
        # the tool name, its argument signature, and whether the call "failed":
        # the worst observed loops are *successful* calls with *varying* args
        # (e.g. execute_code wrapping web fetches) that keep returning the same
        # blocked/404 body. The failure- and signature-keyed counters below all
        # miss that shape, so we key purely on the result content here.
        repeat_decision = self._track_result_repetition(tool_name, result, signature)
        if repeat_decision is not None and repeat_decision.action == "halt":
            self._halt_decision = repeat_decision
            return repeat_decision

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

            return repeat_decision or ToolGuardrailDecision(
                tool_name=tool_name, count=exact_count, signature=signature
            )

        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)

        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return repeat_decision or ToolGuardrailDecision(
                tool_name=tool_name, signature=signature
            )

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

        return repeat_decision or ToolGuardrailDecision(
            tool_name=tool_name, count=repeat_count, signature=signature
        )

    def _track_result_repetition(
        self,
        tool_name: str,
        result: str | Mapping[str, Any] | None,
        signature: ToolCallSignature,
    ) -> ToolGuardrailDecision | None:
        """Detect a consecutive streak of identical tool results regardless of args/outcome.

        Returns a halt decision once the same result has recurred on
        ``repeated_result_halt_after`` consecutive calls this turn, a warn
        decision at the warn threshold, or ``None`` otherwise. Any different or
        untracked result breaks the streak, so interleaved sequences (A/B/A/C/A)
        never fire and the "last N calls" wording in the messages stays literal.

        Short results are ignored so trivial outputs (``"[]"``, ``"OK"``, empty
        strings) never trip it — with one exception: an empty-success envelope
        (``{"success": true, "results": [], ...}``) from a non-mutating tool
        counts, because a read-style tool returning the same "nothing found"
        payload for ever-changing queries is exactly the loop shape from
        issue #60084, and repeated empty successes from mutating tools are
        legitimate (each call did its work).
        """
        if not result:
            self._last_result_hash = None
            self._result_streak = 0
            return None
        text = _repetition_text(result)
        if len(text) < self.config.repeated_result_min_chars and not (
            tool_name not in self.config.mutating_tools
            and _is_empty_success_envelope(result)
        ):
            self._last_result_hash = None
            self._result_streak = 0
            return None

        result_hash = _result_hash(text)
        if result_hash == self._last_result_hash:
            self._result_streak += 1
        else:
            self._last_result_hash = result_hash
            self._result_streak = 1
        count = self._result_streak

        if self.config.hard_stop_enabled and count >= self.config.repeated_result_halt_after:
            return ToolGuardrailDecision(
                action="halt",
                code="repeated_result_halt",
                message=(
                    f"Stopped: the last {count} tool calls returned the same result even "
                    "though the arguments changed. You are looping without making progress "
                    "(the data source is unavailable/blocking). Stop retrying variations; "
                    "report what you already have and the blocker, or ask how to proceed."
                ),
                tool_name=tool_name,
                count=count,
                signature=signature,
            )

        if self.config.warnings_enabled and count >= self.config.repeated_result_warn_after:
            return ToolGuardrailDecision(
                action="warn",
                code="repeated_result_warning",
                message=(
                    f"The last {count} tool calls returned an identical result despite "
                    "changed arguments. This is a no-progress loop — the target likely "
                    "isn't reachable. Use what you have or report the blocker instead of "
                    "trying more variations of the same approach."
                ),
                tool_name=tool_name,
                count=count,
                signature=signature,
            )

        return None

    def _is_idempotent(self, tool_name: str) -> bool:
        if tool_name in self.config.mutating_tools:
            return False
        return tool_name in self.config.idempotent_tools


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

    Plain string results get the guidance appended in place. Multimodal results
    (``{"_multimodal": True, "content": [...]}``) keep their dict shape — the
    guidance is added as a trailing text part (and mirrored into
    ``text_summary`` when present) so providers and the text-only fallback both
    surface it; string concatenation would raise on the dict.
    """
    if decision.action not in {"warn", "halt"} or not decision.message:
        return result
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    guidance = (
        f"[{label}: "
        f"{decision.code}; count={decision.count}; {decision.message}]"
    )
    if isinstance(result, Mapping) and result.get("_multimodal"):
        updated = dict(result)
        updated["content"] = list(updated.get("content") or []) + [
            {"type": "text", "text": guidance}
        ]
        if updated.get("text_summary"):
            updated["text_summary"] = f"{updated['text_summary']}\n\n{guidance}"
        return updated
    return (result or "") + f"\n\n{guidance}"


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


def _repetition_text(result: Any) -> str:
    """Normalize a tool result for repetition tracking.

    Multimodal results (``{"_multimodal": True, "content": [...]}``, e.g.
    vision_analyze) embed base64 image payloads, so ``str(result)`` is unique
    per image and identical *text* parts can never be seen as repetition —
    the guard would otherwise be blind to a repeated "Image loaded into your
    context" placeholder. Represent them as their text parts plus a short
    digest of each non-text payload: re-loading the *same* image repeatedly
    still counts as repetition, while distinct images (legitimate page-scroll
    screenshots) still count as progress.
    """
    if isinstance(result, Mapping) and result.get("_multimodal"):
        parts: list[str] = []
        for part in result.get("content") or []:
            if not isinstance(part, Mapping):
                parts.append(str(part))
                continue
            if part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            else:
                payload = part.get("image_url")
                if isinstance(payload, Mapping):
                    payload = payload.get("url")
                parts.append(f"[media:{_sha256(str(payload or ''))[:16]}]")
        return "\n".join(parts)
    return result if isinstance(result, str) else str(result)


_EMPTY_PAYLOAD_KEYS = ("results", "items", "matches", "entries", "sessions", "data")


def _is_empty_success_envelope(result: Any) -> bool:
    """True for a JSON success payload that carries no content.

    The shape from issue #60084: ``{"success": true, "results": [], "count": 0}``
    — too short for ``repeated_result_min_chars`` but a loop signal all the
    same when it keeps coming back for different queries. Requires an explicit
    emptiness marker (``count == 0`` or an empty payload list), so a bare
    ``{"success": true}`` acknowledgment never qualifies, and any non-empty
    payload key disqualifies.
    """
    if not isinstance(result, str):
        return False
    parsed = safe_json_loads(result)
    if not isinstance(parsed, Mapping) or parsed.get("success") is not True:
        return False
    empty = parsed.get("count") == 0
    for key in _EMPTY_PAYLOAD_KEYS:
        if key not in parsed:
            continue
        if parsed[key] in (None, [], {}, ""):
            empty = True
        else:
            return False
    return empty


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
