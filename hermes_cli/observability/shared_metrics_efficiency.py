"""Efficiency shared metrics: what one user turn costs, what users throw away, and where Hermes
spends tokens it did not have to (spilled tool output, idle tool schemas, prompt-cache breaks).

Per-turn state lives on the relay runtime's task/session objects (see ``TurnCost`` and
``SessionEfficiency``); the agent-side entry points below run on the agent thread and are no-ops
unless shared metrics are enabled. Provider/model always go through the catalog helpers.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import shared_metrics_contract as contract
from .shared_metrics_contract import _bucket, _non_negative_number

logger = logging.getLogger(__name__)

_TOKEN_THRESHOLDS = (
    (2_000, "lt_2k"), (10_000, "2k_to_10k"), (50_000, "10k_to_50k"), (200_000, "50k_to_200k"),
    (1_000_000, "200k_to_1m"),
)
_ACTIVITY_THRESHOLDS = (
    (1, "0"), (2, "1"), (3, "2"), (6, "3_to_5"), (11, "6_to_10"), (26, "11_to_25"), (51, "26_to_50"),
    (101, "51_to_100"),
)
_OUTPUT_SIZE_THRESHOLDS = (
    (1_000, "lt_1k"), (10_000, "1k_to_10k"), (50_000, "10k_to_50k"), (100_000, "50k_to_100k"),
    (500_000, "100k_to_500k"),
)
_SCHEMA_TOKEN_THRESHOLDS = (
    (1, "0"), (2_000, "lt_2k"), (5_000, "2k_to_5k"), (10_000, "5k_to_10k"), (20_000, "10k_to_20k"),
    (40_000, "20k_to_40k"),
)
_TASK_COST_OUTCOMES = {"success": "completed", "cancelled": "interrupted"}
# Anthropic's default prompt-cache TTL: a cold read after this idle gap is expiry, not a bug.
CACHE_TTL_S = 300.0
# Per-session memory of spent turns an /undo or /retry can still reach.
_SPENT_TURNS_KEPT = 64
_PENDING_OUTPUTS_ATTR = "_shared_metrics_tool_outputs"
_PENDING_OUTPUTS_MAX = 256
# The one notice tools/tool_output_truncate.py writes when terminal, execute_code or MCP output was
# cut down inside the tool, before Hermes ever saw the full text.
_TOOL_TRUNCATION_NOTICE = re.compile(r"\[[A-Z_ ]{1,32} TRUNCATED - [\d,]{1,20} chars omitted out of ([\d,]{1,20}) total\]")
_NOTICE_MIN_CHARS = 1_000


def tokens_bucket(tokens: int | None) -> str:
    return "unknown" if tokens is None else _bucket(max(0, tokens), _TOKEN_THRESHOLDS, "gte_1m")


def activity_bucket(count: int) -> str:
    return _bucket(max(0, int(count)), _ACTIVITY_THRESHOLDS, "gte_101")


def output_size_bucket(size: int | None) -> str:
    return "unknown" if size is None else _bucket(max(0, size), _OUTPUT_SIZE_THRESHOLDS, "gte_500k")


def schema_tokens_bucket(tokens: int) -> str:
    return _bucket(max(0, int(tokens)), _SCHEMA_TOKEN_THRESHOLDS, "gte_40k")


def _int(value: Any) -> int | None:
    number = _non_negative_number(value)
    return None if number is None else int(number)


def call_tokens(usage: Any) -> int | None:
    """Prompt (cache reads/writes included) plus completion tokens of one call; None without usage."""
    if not isinstance(usage, dict):
        return None
    total = _int(usage.get("total_tokens"))
    if total is not None:
        return total
    parts = [_int(usage.get(key)) for key in (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")]
    return sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None


def toolset_metric_name(toolset: Any) -> str:
    """A shipped toolset's own name; MCP servers, plugins and user toolsets read ``custom``."""
    return toolset if isinstance(toolset, str) and toolset in contract.BUILTIN_TOOLSET_NAMES else "custom"


def _route(route: dict[str, str] | None) -> dict[str, str]:
    from .shared_metrics_model import model_route

    return dict(route) if route else model_route(None, None)


# ---- per-turn cost and waste (runtime state; caller holds session.lock) ------------------------

@dataclass
class TurnCost:
    """Token spend of one task; ``user_turn`` marks a turn a user message started (pre_llm_call),
    which Hermes-owned forks sharing the session id never fire."""

    user_turn: bool = False
    tokens: int = 0
    known: bool = False
    route: dict[str, str] | None = None

    def add_call(self, route: dict[str, str], usage: Any) -> None:
        self.route = route
        tokens = call_tokens(usage)
        if tokens is not None:
            self.tokens += tokens
            self.known = True

    def bucket(self) -> str:
        return tokens_bucket(self.tokens if self.known else None)


@dataclass
class _SpentTurn:
    route: dict[str, str]
    tokens_bucket: str
    wasted: bool = False


def _wasted_row(reason: str, turn: _SpentTurn) -> tuple[str, dict[str, str]]:
    return contract.WASTED_TOKENS_MARK, {**turn.route, "reason": reason, "tokens_bucket": turn.tokens_bucket}


@dataclass
class ToolUsage:
    """What the session paid to carry tool definitions, and which shipped toolsets it touched."""

    observed: bool = False
    surface: str = "unknown"
    enabled_toolsets: frozenset[str] = frozenset()
    enabled_count: int = 0
    schema_tokens: int = 0
    names: tuple[str, ...] | None = None
    used: set[str] = field(default_factory=set)

    def observe(self, names: tuple[str, ...], toolsets: frozenset[str], enabled_count: int, schema_tokens: int) -> bool:
        """Record the tools one request sent; True when a later request changed the array."""
        changed = self.names is not None and names != self.names
        self.observed, self.names = True, names
        self.enabled_toolsets = self.enabled_toolsets | toolsets
        self.enabled_count, self.schema_tokens = max(self.enabled_count, enabled_count), schema_tokens
        return changed

    def absorb(self, other: ToolUsage) -> None:
        if not other.observed:
            return
        self.observed, self.surface = True, other.surface
        self.enabled_toolsets = self.enabled_toolsets | other.enabled_toolsets
        self.enabled_count = max(self.enabled_count, other.enabled_count)
        self.schema_tokens = max(self.schema_tokens, other.schema_tokens)
        self.used |= other.used

    def rows(self) -> list[tuple[str, dict[str, str]]]:
        if not self.observed:
            return []
        rows = [(contract.TOOL_OVERHEAD_MARK, {
            "enabled_tool_count_bucket": contract.size_bucket(self.enabled_count),
            "execution_surface": self.surface,
            "tool_schema_tokens_bucket": schema_tokens_bucket(self.schema_tokens),
        })]
        rows.extend(
            (contract.TOOL_ENABLED_UNUSED_MARK, {"toolset": name, "used": "yes" if name in self.used else "no"})
            for name in sorted(self.enabled_toolsets)
        )
        return rows


@dataclass
class SessionEfficiency:
    spent: deque[_SpentTurn] = field(default_factory=lambda: deque(maxlen=_SPENT_TURNS_KEPT))
    tools: ToolUsage = field(default_factory=ToolUsage)
    tools_key: tuple[int, int, int] | None = None
    cache_read: int = 0
    cache_route: dict[str, str] | None = None
    cache_ended_ns: int = 0

    def finish_turn(
        self, cost: TurnCost, terminal: dict[str, str], *, route: dict[str, str] | None,
        model_calls: int, tool_calls: int,
    ) -> list[tuple[str, dict[str, str]]]:
        """task_cost for one user turn the user saw end; an interrupt also counts its tokens wasted."""
        turn = _SpentTurn(_route(cost.route or route), cost.bucket())
        rows = [(contract.TASK_COST_MARK, {
            **turn.route,
            "api_calls_bucket": activity_bucket(model_calls),
            "outcome": _TASK_COST_OUTCOMES.get(terminal.get("outcome", ""), "failed"),
            "tokens_bucket": turn.tokens_bucket,
            "tool_calls_bucket": activity_bucket(tool_calls),
        })]
        if terminal.get("end_reason") == "user_cancelled":
            turn.wasted = True
            rows.append(_wasted_row("interrupt", turn))
        self.spent.append(turn)
        return rows

    def discard_turns(self, reason: str, turns: int, fallback_route: dict[str, str]) -> list[tuple[str, dict[str, str]]]:
        """An /undo or /retry threw away the newest ``turns`` turns: one row per turn not already
        counted (an interrupted turn the user then undoes was wasted once, at the interrupt)."""
        rows = []
        for _ in range(max(1, min(int(turns), _SPENT_TURNS_KEPT))):
            turn = self.spent.pop() if self.spent else _SpentTurn(_route(fallback_route), "unknown")
            if not turn.wasted:
                rows.append(_wasted_row(reason, turn))
        return rows

    def observe_cache(
        self, route: dict[str, str], usage: Any, started_ns: int, ended_ns: int, *, expected: bool,
    ) -> str | None:
        """A cold primary read after a warm one on the same route is a break Hermes did not announce
        (``expected``: Hermes already counted the cause); a gap past the cache TTL is plain expiry."""
        read = _int(usage.get("cache_read_tokens")) if isinstance(usage, dict) else None
        if read is None:
            return None
        prev_read, prev_route, prev_ended = self.cache_read, self.cache_route, self.cache_ended_ns
        self.cache_read, self.cache_route, self.cache_ended_ns = read, route, ended_ns
        if expected or read or not prev_read or prev_route != route:
            return None
        idle_s = (started_ns - prev_ended) / 1e9
        return "cache_expired" if idle_s >= CACHE_TTL_S else "provider_reported_miss"


def cache_break_row(cause: str, route: dict[str, str] | None) -> tuple[str, dict[str, str]]:
    return contract.CACHE_BREAK_MARK, {**_route(route), "cause": cause}


# ---- agent-side entry points (agent thread; never raise) ---------------------------------------

def _internal_agent(agent: Any) -> bool:
    """Hermes-owned forks (background review, curator) share the user's session id."""
    return getattr(agent, "_memory_write_origin", None) == "background_review"


def _agent_route(agent: Any) -> dict[str, str]:
    from .shared_metrics_model import model_route

    return model_route(getattr(agent, "provider", None), getattr(agent, "model", None))


def _enabled() -> bool:
    from .relay_shared_metrics import enabled

    return enabled()


def note_tool_result(agent: Any, tool_name: Any, tool_call_id: Any, raw: Any, persisted: Any) -> None:
    """Remember one committed tool result until its batch's turn budget has run."""
    try:
        if _internal_agent(agent) or not _enabled():
            return
        pending = agent.__dict__.setdefault(_PENDING_OUTPUTS_ATTR, {})
        if len(pending) >= _PENDING_OUTPUTS_MAX:
            return
        size, cut_in_tool = _result_size(raw)
        pending[str(tool_call_id or id(persisted))] = [
            contract.tool_metric_name({"tool_name": tool_name}), size, cut_in_tool or persisted is not raw,
        ]
    except Exception:
        logger.debug("Shared-metrics tool output not noted", exc_info=True)


def _result_size(result: Any) -> tuple[int | None, bool]:
    """(characters the tool produced, whether the tool itself already truncated them)."""
    if isinstance(result, dict):  # multimodal envelope: its text parts are what can spill
        parts = result.get("content") or []
        return sum(len(p.get("text") or "") for p in parts if isinstance(p, dict) and p.get("type") == "text"), False
    if not isinstance(result, str):
        return None, False
    notice = _TOOL_TRUNCATION_NOTICE.search(result) if len(result) >= _NOTICE_MIN_CHARS else None
    if notice is None:
        return len(result), False
    return max(len(result), int(notice.group(1).replace(",", ""))), True


def record_tool_batch(agent: Any, batch: list[dict], contents_before: list[Any]) -> None:
    """One row per tool result of the batch: truncated when the per-result cap or the per-turn
    budget replaced it (the budget runs after the cap, so a result is judged once, here)."""
    try:
        pending = agent.__dict__.pop(_PENDING_OUTPUTS_ATTR, None)
        if not pending or not _enabled():
            return
        for message, before in zip(batch, contents_before):
            entry = pending.get(str(message.get("tool_call_id") or ""))
            if entry is not None and message.get("content") is not before:
                entry[2] = True
        from .relay_shared_metrics import record_process_mark

        for tool, size, truncated in pending.values():
            record_process_mark(contract.TOOL_OUTPUT_TRUNCATION_MARK, {
                "original_size_bucket": output_size_bucket(size), "tool": tool, "truncated": "yes" if truncated else "no",
            })
    except Exception:
        logger.debug("Shared-metrics tool output truncation not recorded", exc_info=True)


def observe_request_tools(agent: Any, tools_for_api: Any) -> None:
    """The tool definitions one primary request carries (evidence for trimming default toolsets)."""
    try:
        if getattr(agent, "_persist_disabled", False) or _internal_agent(agent) or not _enabled():
            return
        from .relay_shared_metrics import record_session_tools

        record_session_tools(str(getattr(agent, "session_id", "") or ""), agent, tools_for_api or [])
    except Exception:
        logger.debug("Shared-metrics tool overhead not observed", exc_info=True)


def tool_snapshot(agent: Any, tools_for_api: list) -> tuple[tuple[str, ...], frozenset[str], int, int]:
    """(sent tool names, enabled shipped toolsets, enabled tool count, estimated schema tokens)."""
    from agent.model_metadata import _estimate_tools_tokens_rough
    from model_tools import get_toolset_for_tool

    names = tuple(_tool_names(tools_for_api))
    enabled = list(_tool_names(getattr(agent, "tools", None) or tools_for_api))
    extra = getattr(agent, "valid_tool_names", None) or ()
    enabled_names = set(enabled) | {n for n in extra if isinstance(n, str)}
    toolsets = frozenset(toolset_metric_name(get_toolset_for_tool(n)) for n in enabled_names)
    return names, toolsets, len(enabled_names), _estimate_tools_tokens_rough(tools_for_api) if tools_for_api else 0


def _tool_names(tools: Iterable[Any]) -> Iterable[str]:
    for tool in tools or ():
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else (tool.get("name") if isinstance(tool, dict) else None)
        if isinstance(name, str):
            yield name


def record_cache_break(agent: Any, cause: str) -> None:
    """Hermes itself invalidated the conversation's cached prefix (compression, a rebuilt prompt)."""
    try:
        if cause not in contract.CACHE_BREAK_CAUSES or getattr(agent, "_persist_disabled", False):
            return
        if _internal_agent(agent) or not _enabled():
            return
        from .relay_shared_metrics import record_known_cache_break

        record_known_cache_break(cause, _agent_route(agent), str(getattr(agent, "session_id", "") or ""))
    except Exception:
        logger.debug("Shared-metrics cache break not recorded", exc_info=True)


def record_prompt_rebuild(agent: Any, stored_prompt: Any, stored_state: str, rebuilt: Any) -> None:
    """A continuing conversation rebuilt its system prompt instead of replaying the stored bytes."""
    if stored_state in ("null", "empty"):
        record_cache_break(agent, "system_prompt_rebuild")
        return
    if stored_state != "stale_runtime" or not stored_prompt or rebuilt == stored_prompt:
        return
    try:
        from agent.surface_switch import identity_line_value

        switched = any(
            (stored := identity_line_value(stored_prompt, label)) and stored != str(getattr(agent, attr, "") or "").strip()
            for label, attr in (("Model", "model"), ("Provider", "provider"))
        )
    except Exception:
        switched = False
    record_cache_break(agent, "model_switch" if switched else "system_prompt_rebuild")
