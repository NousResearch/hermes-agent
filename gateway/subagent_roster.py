"""Pure state + rendering for the live delegate_task subagent roster bubble.

Gateway-free and import-cycle-free (only depends on the tiny shared
``gateway.duration_format`` util, never on ``gateway.run``). The gateway
consumer owns one instance per turn, feeds it lifecycle sentinels via
``apply_event``, and renders with ``fold`` + ``format_subagent_roster``.

Design contract (see plan ~/.hermes/plans/subagent-roster-bubble.md):
- SINGLE-WRITER per-turn state. Only the loop-bound gateway consumer mutates an
  instance. NOT thread-safe and deliberately lock-free: the worker-thread
  progress callback only ENQUEUES sentinels, it never touches this object.
- Membership + terminal status come from subagent.start/.complete EVENTS
  (the active-subagent registry deletes a child the instant it finishes, so a
  poll can never observe done/errored/timed-out). Live elapsed for RUNNING rows
  comes from a periodic poll of the registry, passed into ``fold`` as
  ``active_by_id``.
- Rows NEVER reorder: first-seen order is stable, a done row stays in place.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Throttle for roster bubble edits, in seconds. Module constant so it is defined
# once and imported by the consumer (never re-declared locally). Set to 10s:
# Telegram enforces a per-chat edit flood ceiling, and a busy chat with other
# bubbles can trip "Flood control exceeded" — which froze the live timer. A
# 10s cadence stays well under the ceiling; the final collapse (force) bypasses
# this throttle so the terminal state always lands. This is the DEFAULT; the
# effective value is per-platform configurable via
# display.platforms.<platform>.subagent_roster_interval (see resolve_roster_interval).
ROSTER_EDIT_INTERVAL = 10.0

# Hard floor so a misconfigured tiny interval can never flood the platform's
# edit rate limiter. Mirrors the clamp in gateway.display_config._normalise.
ROSTER_EDIT_INTERVAL_FLOOR = 1.0


def resolve_roster_interval(user_config: Any, platform_key: str) -> float:
    """Resolve the effective roster edit interval (seconds) for a platform.

    Reads ``display.platforms.<platform>.subagent_roster_interval`` (falling
    back through the global setting and the built-in default) and clamps to a
    1.0s floor. Best-effort: any error returns the default ``ROSTER_EDIT_INTERVAL``.
    """
    try:
        from gateway.display_config import resolve_display_setting

        raw = resolve_display_setting(
            user_config,
            platform_key,
            "subagent_roster_interval",
            ROSTER_EDIT_INTERVAL,
        )
        return max(ROSTER_EDIT_INTERVAL_FLOOR, float(raw))
    except Exception:
        return ROSTER_EDIT_INTERVAL


def is_flood_error(result: Any) -> bool:
    """True if a failed adapter send/edit result is flood-control / rate-limit.

    Thin re-export of the canonical predicate (gateway.flood_detect) so the
    roster seed path and the stream consumer share ONE token set. A flood/rate
    reject is known-not-delivered, so re-seeding after one cannot duplicate;
    an ambiguous failure must latch instead. Kept here as a name the roster
    seed paths already import (gateway/run.py).
    """
    from gateway.flood_detect import is_flood_error as _is_flood_error
    return _is_flood_error(result)


_LABEL_CAP = 60
_MAX_ROWS = 10


def format_elapsed(seconds: float) -> str:
    """Human elapsed for a roster row: ``3m 9s`` / ``45s`` / ``1h 2m``.

    Distinct from ``gateway.duration_format.format_duration`` (clock-style
    ``M:SS``), which is shared with media/audio durations and must stay
    clock-style. Here seconds are NOT zero-padded (``3m 9s``, not ``3m 09s``)
    and a trailing zero unit is dropped (``1m``, not ``1m 0s``). Clamps < 0.
    """
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        total = 0
    if total < 0:
        total = 0
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        out = f"{hours}h {minutes}m"
        return f"{out} {secs}s" if secs else out
    if minutes:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    return f"{secs}s"


def shorten_model(model: Optional[str]) -> str:
    """Compact a model id for a roster row.

    Strips a leading region/provider dotted prefix so a row stays readable:
    ``us.anthropic.claude-opus-4-8`` -> ``opus-4-8``,
    ``claude-sonnet-4-6`` -> ``sonnet-4-6``. Model ids whose only dots are a
    version number (``gpt-5.5``) are left intact. Best-effort: unknown shapes
    return unchanged.
    """
    text = str(model or "").strip()
    if not text:
        return ""
    # Only treat dots as a provider/region prefix separator when EVERY segment
    # before the last is a bare alpha token (us, anthropic, openai, ...). This
    # avoids mangling a version dot like "gpt-5.5" (segment "5" is not alpha).
    if "." in text:
        segs = text.split(".")
        if all(s.isalpha() for s in segs[:-1]):
            text = segs[-1]
    # Strip a redundant vendor token so "claude-opus-4-8" -> "opus-4-8".
    for vendor in ("claude-", "anthropic-"):
        if text.startswith(vendor):
            text = text[len(vendor):]
            break
    return text


def reasoning_tag(reasoning: Any) -> str:
    """Render a reasoning_config dict/string into a short effort tag.

    ``{"enabled": True, "effort": "high"}`` -> ``high``;
    ``{"enabled": False}`` -> ``""`` (no reasoning, nothing to show);
    a bare string ``"max"`` -> ``max``. Anything else -> ``""``.
    """
    if isinstance(reasoning, dict):
        if not reasoning.get("enabled", True):
            return ""
        return str(reasoning.get("effort") or "").strip().lower()
    if isinstance(reasoning, str):
        return reasoning.strip().lower()
    return ""


def _inline_code(raw: Any) -> str:
    """Sanitise a short scan cell and render it as a Markdown inline-code span."""
    text = " ".join(str(raw or "").replace("`", "").split())
    return f"`{text}`" if text else ""


def _model_suffix(row: Dict[str, Any]) -> str:
    """Model/reasoning suffix as plain text, or '' when unknown."""
    model = shorten_model(row.get("model"))
    if not model:
        return ""
    tag = reasoning_tag(row.get("reasoning"))
    rendered = f"{model} {tag}".rstrip() if tag else model
    rendered = " ".join(rendered.replace("`", "").split())
    return f" · {rendered}" if rendered else ""


def _profile_suffix(row: Dict[str, Any]) -> str:
    """`· <profile>` suffix for a row, or '' when no profile is known.

    The profile (delegation lane, e.g. ``reviewer-codex``) is the audit cell
    Alfredo wants kept VISIBLE after the dispatched-card seed frame morphs into
    the live roster: the model id alone (``gpt-5.5``) does not say WHICH lane
    ran. Shown on running, pending, AND finished rows so it never vanishes.
    Backticks/whitespace are normalised like the card cell. Missing/empty -> ''.
    """
    raw = row.get("profile")
    if not raw:
        return ""
    profile = " ".join(str(raw).replace("`", "").split())
    if not profile:
        return ""
    return f" · {_inline_code(profile)}"


def _tools_suffix(row: Dict[str, Any]) -> str:
    """`· N tool(s)` suffix for a row, or '' when the count is 0/missing.

    Shown for BOTH running and finished rows — a done child keeps the count of
    tools it actually ran (Alfredo asked to keep it after the agent finishes).
    """
    try:
        n = int(row.get("tools") or 0)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    return f" · {n} tool" + ("s" if n != 1 else "")


def _format_cost(cost: float) -> str:
    """Adaptive USD format: 2dp at/above $1 ($1.23), 4dp below ($0.0123).

    Big multi-agent totals read clean; small per-subagent costs keep precision.
    """
    return f"${cost:.2f}" if cost >= 1 else f"${cost:.4f}"


def _cost_suffix(row: Dict[str, Any]) -> str:
    """`· $0.0123` / `· $1.23` suffix for a row, or '' when cost is unknown.

    Cost is a COMPLETION-only number (the live registry has no running cost),
    so this is only ever non-empty on a finished row. Missing/None/<=0 -> '' so
    a provider with no pricing shows no cost cell instead of a misleading $0.00.
    """
    raw = row.get("cost_usd")
    if raw is None:
        return ""
    try:
        cost = float(raw)
    except (TypeError, ValueError):
        return ""
    if cost <= 0:
        return ""
    return f" · {_inline_code(_format_cost(cost))}"

# delegate_tool subagent.complete status string -> (glyph, display bucket).
# Vocabulary verified in tools/delegate_tool.py: completed | failed |
# interrupted | timeout | error. Unknown -> fail-CLOSED to errored (never render
# an unrecognised terminal state as success).
STATUS_GLYPH: Dict[str, Tuple[str, str]] = {
    "completed": ("✓", "done"),
    "failed": ("✗", "errored"),
    "error": ("✗", "errored"),
    "timeout": ("⏱", "timed-out"),
    "interrupted": ("⏹", "interrupted"),
}
_UNKNOWN_GLYPH = ("?", "errored")


def roster_label(goal: Optional[str]) -> str:
    """Collapse whitespace/newlines and hard-cap a child goal for one row.

    Backticks are stripped: the label is rendered inside an inline code span
    (`` `label` ``) so a stray backtick would break the span on Telegram.
    """
    text = " ".join(str(goal or "").replace("`", "").split())
    if not text:
        return "subagent"
    if len(text) > _LABEL_CAP:
        return text[: _LABEL_CAP - 1] + "…"
    return text


@dataclass
class SubagentRosterState:
    """Single-writer per-turn roster state. Lock-free by design."""

    # sid -> {"goal", "task_index", "started_at"}
    meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # sid -> {"status", "duration"}
    terminal: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # first-seen order; rows never reorder
    seen_order: List[str] = field(default_factory=list)

    def has_records(self) -> bool:
        return bool(self.seen_order)

    def start(
        self,
        sid: str,
        goal: Optional[str] = None,
        task_index: int = 0,
        started_at: float = 0.0,
        model: Optional[str] = None,
        reasoning: Any = None,
        context_available: bool = False,
        context_child_session_id: Optional[str] = None,
    ) -> None:
        if not sid:
            return
        if sid not in self.meta:
            self.seen_order.append(sid)
        self.meta[sid] = {
            "goal": goal or "",
            "task_index": int(task_index or 0),
            "started_at": float(started_at or 0.0),
            "model": model or "",
            "reasoning": reasoning,
            "context_available": bool(context_available),
            "context_child_session_id": str(context_child_session_id or ""),
        }
        # A re-run/restart of the same sid clears any prior terminal state.
        self.terminal.pop(sid, None)

    def complete(
        self,
        sid: str,
        status: str = "completed",
        duration: float = 0.0,
        started_at: float = 0.0,
        tools: int = 0,
    ) -> None:
        if not sid:
            return
        if sid not in self.meta:
            # subagent.complete with no prior start (fast child): synthesize a
            # minimal meta so the row still renders.
            self.seen_order.append(sid)
            self.meta[sid] = {
                "goal": "",
                "task_index": 0,
                "started_at": float(started_at or 0.0),
                "model": "",
                "reasoning": None,
                "context_available": False,
                "context_child_session_id": "",
            }
        try:
            _tools = int(tools or 0)
        except (TypeError, ValueError):
            _tools = 0
        self.terminal[sid] = {
            "status": str(status or "completed").lower(),
            "duration": float(duration or 0.0),
            # Final tool count: the registry drops the live entry on completion,
            # so the count is carried on the complete event and kept here.
            "tools": _tools,
        }

    def apply_event(self, raw: Tuple[Any, ...]) -> None:
        """Mutate from a dequeued sentinel tuple (loop thread only).

        ``("__roster_start__", sid, goal, task_index, started_at[, model, reasoning])``
        ``("__roster_complete__", sid, status, duration[, tool_count])``

        The start tuple's model/reasoning tail and the complete tuple's
        tool_count tail are optional so older producers (and replayed queues)
        without them still apply cleanly.
        """
        if not raw:
            return
        kind = raw[0]
        if kind == "__roster_start__":
            sid = raw[1] if len(raw) > 1 else ""
            goal = raw[2] if len(raw) > 2 else ""
            task_index = raw[3] if len(raw) > 3 else 0
            started_at = raw[4] if len(raw) > 4 else 0.0
            model = raw[5] if len(raw) > 5 else None
            reasoning = raw[6] if len(raw) > 6 else None
            context_available = raw[7] if len(raw) > 7 else False
            context_child_session_id = raw[8] if len(raw) > 8 else ""
            self.start(
                sid,
                goal,
                task_index,
                started_at,
                model,
                reasoning,
                context_available=context_available,
                context_child_session_id=context_child_session_id,
            )
        elif kind == "__roster_complete__":
            sid = raw[1] if len(raw) > 1 else ""
            status = raw[2] if len(raw) > 2 else "completed"
            duration = raw[3] if len(raw) > 3 else 0.0
            tools = raw[4] if len(raw) > 4 else 0
            self.complete(sid, status, duration, tools=tools)

    def fold(self, active_by_id: Dict[str, Dict[str, Any]], now: float) -> List[Dict[str, Any]]:
        """Build ordered display rows from current state + a registry snapshot.

        ``active_by_id``: sid -> live registry record (started_at, tool_count),
        already filtered to this run's sids. Used only for RUNNING rows.
        """
        rows: List[Dict[str, Any]] = []
        for sid in self.seen_order:
            m = self.meta.get(sid) or {}
            label = roster_label(m.get("goal"))
            model = m.get("model") or ""
            reasoning = m.get("reasoning")
            if sid in self.terminal:
                t = self.terminal[sid]
                glyph, _bucket = STATUS_GLYPH.get(
                    str(t.get("status") or "").lower(), _UNKNOWN_GLYPH
                )
                row = {
                    "glyph": glyph,
                    "label": label,
                    "elapsed": float(t.get("duration") or 0.0),
                    "running": False,
                    "tools": int(t.get("tools") or 0),
                    "model": model,
                    "reasoning": reasoning,
                }
                if m.get("context_available"):
                    row["context_available"] = True
                    row["context_child_session_id"] = str(m.get("context_child_session_id") or "")
                rows.append(row)
            else:
                rec = active_by_id.get(sid) or {}
                started = rec.get("started_at") or m.get("started_at") or now
                row = {
                    "glyph": "▶",
                    "label": label,
                    "elapsed": max(0.0, now - float(started)),
                    "running": True,
                    "tools": int(rec.get("tool_count") or 0),
                    "model": model,
                    "reasoning": reasoning,
                }
                if m.get("context_available"):
                    row["context_available"] = True
                    row["context_child_session_id"] = str(m.get("context_child_session_id") or "")
                rows.append(row)
        return rows


def _bucket_of(row: Dict[str, Any]) -> str:
    explicit = str(row.get("bucket") or "").strip().lower()
    if explicit:
        return explicit
    if row["running"]:
        return "running"
    glyph = row["glyph"]
    for _status, (g, bucket) in STATUS_GLYPH.items():
        if g == glyph:
            return bucket
    return "errored"


def format_subagent_roster(
    rows: List[Dict[str, Any]],
    *,
    collapsed: bool = False,
    wall_clock: Optional[float] = None,
) -> Optional[str]:
    """Render roster rows into a bubble string. None when there are no rows.

    ``wall_clock`` is the real elapsed time of the delegate_task call (the time
    the user actually waited). When given and usable it is the header total;
    otherwise the header falls back to the SLOWEST child's elapsed (parallel-
    safe). We NEVER sum child elapsed: children run concurrently, so a sum
    overcounts (3 parallel 3s children finish in ~3s wall-clock, not 9s).
    """
    if not rows:
        return None

    # Header time = authoritative wall-clock when usable, else the SLOWEST
    # child (parallel-safe). NEVER sum. Reject non-finite/overflow: format_elapsed
    # only catches TypeError/ValueError, so an int(round(inf)) -> OverflowError
    # would crash the render and skip the max fallback.
    def _usable(w: Any) -> Optional[float]:
        try:
            w = float(w)
        except (TypeError, ValueError, OverflowError):
            return None
        return w if (math.isfinite(w) and w >= 0) else None

    _wc = _usable(wall_clock)
    header_elapsed = (
        _wc if _wc is not None else max((r["elapsed"] for r in rows), default=0.0)
    )

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(_bucket_of(row), []).append(row)

    running = buckets.get("running", [])
    pending = buckets.get("pending", [])
    done = buckets.get("done", [])
    errored = buckets.get("errored", [])
    timed_out = buckets.get("timed-out", [])
    interrupted = buckets.get("interrupted", [])
    failed_total = errored + timed_out + interrupted

    if collapsed:
        # Final render. Keep the per-child breakdown (each child marked with its
        # terminal glyph) instead of collapsing to a bare one-liner — the user
        # wants to see WHICH children did what, not just a count. The summary
        # line becomes a header above the rows.
        # Header elapsed is the WALL-CLOCK of the delegate_task call when known
        # (computed once above as header_elapsed), else the slowest child. NOT
        # the sum: children run in parallel, so summing overcounts the time the
        # user actually waited.
        total = header_elapsed
        # Clear "finished" indicator on the header, replacing the 🤖 robot:
        #   ✅  every child finished and NONE failed/timed-out/interrupted
        #   ⚠️  finished but at least one child failed (a green check would lie)
        #   🤖  defensive: something is still running/pending at collapse time
        if running or pending:
            lead = "🤖"
        elif failed_total:
            lead = "⚠️"
        else:
            lead = "✅"
        head_parts = [f"{lead} {len(rows)} subagent" + ("s" if len(rows) != 1 else "")]
        if pending:
            head_parts.append(f"{len(pending)} pending")
        if done:
            head_parts.append(f"{len(done)} ✓")
        if failed_total:
            head_parts.append(f"{len(failed_total)} ✗")
        head_parts.append(_inline_code(format_elapsed(total)))
        # Total cost across all children (adaptive format), when known. Sums only
        # rows that carry a numeric cost_usd; absent costs contribute nothing.
        total_cost = 0.0
        for r in rows:
            try:
                total_cost += float(r.get("cost_usd") or 0.0)
            except (TypeError, ValueError):
                pass
        if total_cost > 0:
            head_parts.append(_inline_code(_format_cost(total_cost)))
        head = " · ".join(head_parts)

        lines = [head]
        shown = rows[:_MAX_ROWS]
        for r in shown:
            # On the final render a running row (shouldn't normally happen) is
            # shown with its live glyph; terminal rows keep ✓/✗/⏱/⏹. Tool count
            # is kept on done rows too, not dropped when running flips to False.
            line = (
                f"{r['glyph']} `{r['label']}`{_profile_suffix(r)}{_model_suffix(r)}"
                f" · {_inline_code(format_elapsed(r['elapsed']))}{_tools_suffix(r)}{_cost_suffix(r)}"
            )
            lines.append(line)
        extra = len(rows) - len(shown)
        if extra > 0:
            lines.append(f"… +{extra} more")
        return "\n".join(lines)

    head = f"🤖 Subagents — {len(running)} running"
    if pending:
        head += f", {len(pending)} pending"
    if done:
        head += f", {len(done)} done"
    if failed_total:
        head += f", {len(failed_total)} failed"
    # Live header elapsed is the WALL-CLOCK since dispatch when known (computed
    # once above as header_elapsed), else the slowest running child. NOT the
    # sum: children run in parallel, so a ticking sum overcounts wall-clock.
    live_total = header_elapsed
    if live_total > 0:
        head += f" · {_inline_code(format_elapsed(live_total))}"

    lines = [head]
    shown = rows[:_MAX_ROWS]
    for r in shown:
        line = (
            f"{r['glyph']} `{r['label']}`{_profile_suffix(r)}{_model_suffix(r)}"
            f" · {_inline_code(format_elapsed(r['elapsed']))}{_tools_suffix(r)}{_cost_suffix(r)}"
        )
        lines.append(line)
    extra = len(rows) - len(shown)
    if extra > 0:
        lines.append(f"… +{extra} more")
    return "\n".join(lines)
