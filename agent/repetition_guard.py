"""Assistant-turn repetition guard.

Complements ``agent.tool_guardrails``, which keys on tool *results* (repeated
failures, identical read-only results). That guardrail cannot see the loop
where every tool call *succeeds* but the assistant keeps emitting the same
message with the same tool calls — e.g. re-screenshotting a page that changes
by a few pixels each frame while narrating the identical "verifying the
scene" line, indefinitely.

This guard keys on the assistant turn itself: a signature over the visible
content plus the tool-call names/arguments. When the same signature keeps
recurring, the runtime injects a corrective user message (nudge); if the
model still repeats after being told and hard aborts are enabled, the runtime
ends the turn cleanly instead of burning the iteration budget.

Like ``tool_guardrails``, this module is side-effect free: it tracks
observations and returns decisions; the conversation loop owns what those
decisions become (a synthetic user message, a controlled turn halt).
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

from agent.message_content import flatten_message_text


REPETITION_NUDGE_MESSAGE = (
    "[Repetition guard] Your recent turns repeated the same message and the "
    "same tool calls without making progress. Do not repeat this step again. "
    "Take a materially different action — a different tool, different "
    "arguments, or a different approach — or state clearly why you are "
    "blocked and stop."
)


@dataclass(frozen=True)
class RepetitionGuardConfig:
    """Thresholds for assistant-turn repetition detection.

    Nudges are enabled by default and never prevent tool execution — they only
    append a corrective user message after the repeated turn's tool results.
    Hard aborts are explicit opt-in (mirroring ``tool_loop_guardrails``'s
    ``hard_stop_enabled``) because legitimate workflows can repeat identical
    turns for a while (e.g. polling a build), and a nudged model that keeps
    polling on purpose should be allowed to.
    """

    enabled: bool = True
    window: int = 5           # how many recent turns the nudge check looks at
    nudge_after: int = 3      # identical turns within the window → nudge
    abort_enabled: bool = False
    abort_after: int = 2      # identical turns AFTER the nudge → abort

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RepetitionGuardConfig":
        """Build config from the ``repetition_guard`` config.yaml section."""
        if not isinstance(data, Mapping):
            return cls()
        defaults = cls()
        return cls(
            enabled=_as_bool(data.get("enabled"), defaults.enabled),
            window=_positive_int(data.get("window"), defaults.window),
            nudge_after=_positive_int(data.get("nudge_after"), defaults.nudge_after),
            abort_enabled=_as_bool(data.get("abort_enabled"), defaults.abort_enabled),
            abort_after=_positive_int(data.get("abort_after"), defaults.abort_after),
        )


def assistant_turn_signature(assistant_msg: Mapping[str, Any]) -> str:
    """Stable identity for an assistant turn: visible text + tool calls.

    Tool-call ids are excluded — they are freshly generated per API call, so
    including them would make every turn unique. Reasoning is also excluded:
    two turns that produce the same visible action are the same step from the
    outside, regardless of how the model talked itself into it.
    """
    content_text = flatten_message_text(assistant_msg.get("content")).strip()
    calls = []
    for tc in assistant_msg.get("tool_calls") or []:
        if not isinstance(tc, Mapping):
            continue
        fn = tc.get("function") or {}
        if not isinstance(fn, Mapping):
            fn = {}
        calls.append((str(fn.get("name") or ""), str(fn.get("arguments") or "")))
    canonical = json.dumps(
        {"content": content_text, "tool_calls": calls},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AssistantRepetitionGuard:
    """Tracks recent assistant-turn signatures and decides nudge/abort."""

    def __init__(self, config: RepetitionGuardConfig | None = None):
        self.config = config or RepetitionGuardConfig()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        self._recent: deque[str] = deque(maxlen=max(1, self.config.window))
        self._nudged_sig: str | None = None
        self._post_nudge_repeats = 0

    def observe(self, assistant_msg: Mapping[str, Any]) -> str:
        """Record one assistant turn; return ``"ok"``, ``"nudge"``, or ``"abort"``.

        ``"nudge"`` may be returned repeatedly for the same signature — every
        post-nudge repeat re-injects the corrective message so the pressure
        does not decay as the transcript grows.
        """
        if not self.config.enabled:
            return "ok"

        sig = assistant_turn_signature(assistant_msg)
        self._recent.append(sig)

        if self._nudged_sig == sig:
            self._post_nudge_repeats += 1
            if self.config.abort_enabled and self._post_nudge_repeats >= self.config.abort_after:
                return "abort"
            return "nudge"

        if list(self._recent).count(sig) >= self.config.nudge_after:
            self._nudged_sig = sig
            self._post_nudge_repeats = 0
            return "nudge"

        return "ok"


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
