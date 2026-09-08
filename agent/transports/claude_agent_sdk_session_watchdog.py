"""Turn-lifetime watchdog for the claude-agent-sdk session.

The activity-aware budget/quiet rules (``_TurnWatch``), their defaults, the
stream-end sentinel and the future-result swallowers used by interrupt/steer.
Extracted from ``claude_agent_sdk_session.py``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional
from agent.transports.claude_agent_sdk_session_availability import (
    _safe_sdk_error_text,
)

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.transports.claude_agent_sdk_session")


# ---------- turn-lifetime defaults ----------
# The soft turn budget. It was a hard wall-clock over the whole turn since the
# provider's birth (transplanted verbatim from the codex twin, where a
# post-tool quiet watchdog compensates); production forensics on six 600s
# kills showed four were actively-working turns (tool loops, human approval
# taps) — so the budget is now evaluated together with activity evidence
# (see _TurnWatch) instead of alone.
_DEFAULT_TURN_TIMEOUT = 600.0


# Post-tool quiet watchdog default WHEN streaming is on (codex parity: its
# post_tool_quiet_timeout=90 mirrors openclaw's #81697 watchdog). With
# streaming OFF there is no liveness signal between a tool result and the
# next complete AssistantMessage — thinking is indistinguishable from wedged
# — so the watchdog defaults to DISABLED there (operator opt-in).
_DEFAULT_POST_TOOL_QUIET_STREAMING = 90.0


# The budget rule only fires when the turn has ALSO been quiet this long
# (capped at the budget itself so tiny test budgets keep tripping at the
# budget): an actively-producing turn is never killed mid-sentence.
_BUDGET_GRACE_IDLE = 30.0


# After a watchdog trip we interrupt the CLI and give _consume_turn this long
# to unwind on the interrupt-ack ResultMessage — a clean unwind preserves the
# partial transcript and the resumable session id; only expiry hard-cancels.
_TURN_ABORT_GRACE = 15.0


# An inter-poll gap this many times the poll interval means the PROCESS was
# stalled (swap/OOM descheduling on a memory-constrained host), not the turn
# — re-baseline instead of tripping on time nobody was actually waiting.
_POLL_STALL_FACTOR = 5.0


# Upper bound on how long a compaction may suspend the watchdog. compact_boundary
# is NOT guaranteed -- measured 2026-08-16, a compaction that started at 03:57:17
# never produced one -- so an unbounded gate would trade a killed turn for a hung
# one. 600s is an order of magnitude above the ~90-125s compactions observed in
# production while still landing well inside the gateway's 1800s ceiling.
_COMPACTION_MAX_SUSPEND = 600.0


class _TurnWatch:
    """Activity evidence for one in-flight turn.

    Threading contract: mutating calls happen on the session's loop thread
    (message drain, projections, approval bridge) with ONE sanctioned
    exception — rebaseline(), called from the run_turn poll thread, also
    writes last_activity. That dual-writer race is benign by construction:
    float stores are GIL-atomic (never torn), and a lost update leaves
    last_activity merely STALE, which the stall detector re-baselines and
    the two-poll debounce absorbs before any verdict — trips can only be
    DELAYED by it, never wrongly fired. Everything else is single-writer;
    the poll thread otherwise only READS. No lock, deliberately: a lock
    shared with the loop thread would risk stalling the SDK stream drain.

    Evidence gate: a turn with tool calls outstanding (issued ToolUseBlocks
    minus resolved ToolResultBlocks — server tools never enter the count,
    they resolve inside their own assistant message) or an approval prompt
    awaiting a human tap is PROVABLY working/waiting and is never tripped.
    If the CLI never resolves an issued tool id (interrupted mid-tool), the
    suspension persists and the gateway's 1800s inactivity ceiling remains
    the backstop — documented, deliberate."""

    def __init__(self) -> None:
        now = time.monotonic()
        self.started = now
        self.last_activity = now
        self.post_tool_armed = False
        self.outstanding_tools = 0
        self.approvals_pending = 0
        self.compaction_active = 0
        self.compaction_started = 0.0

    # -- loop-thread writers --

    def tick(self) -> None:
        self.last_activity = time.monotonic()

    def note_tools_issued(self, count: int) -> None:
        if count > 0:
            self.outstanding_tools += count

    def note_tools_resolved(self, count: int) -> None:
        if count > 0:
            self.outstanding_tools = max(0, self.outstanding_tools - count)

    def arm_post_tool(self) -> None:
        self.post_tool_armed = True

    def disarm_post_tool(self) -> None:
        self.post_tool_armed = False

    def approval_begin(self) -> None:
        self.approvals_pending += 1
        self.tick()

    def approval_end(self) -> None:
        self.approvals_pending = max(0, self.approvals_pending - 1)
        self.tick()

    def compaction_begin(self) -> None:
        """PreCompact fired: the CLI is about to go silent, legitimately.

        Earliest start wins -- a re-entrant PreCompact must not restart the
        bounding clock, or a pathological loop could extend the suspension
        indefinitely, which is exactly what the bound exists to prevent.
        """
        if self.compaction_active == 0:
            self.compaction_started = time.monotonic()
        self.compaction_active += 1
        self.tick()

    def compaction_end(self) -> None:
        """compact_boundary arrived: resume normal watchdog rules.

        tick() is load-bearing, not hygiene. Without it a turn that compacted
        for 91s would resume already 91s idle and trip on the very next poll --
        the same kill, one poll later.
        """
        self.compaction_active = max(0, self.compaction_active - 1)
        self.tick()

    # -- caller-thread readers --

    def rebaseline(self) -> None:
        """After a detected process stall: the elapsed gap was spent
        descheduled, not waiting — restamp so neither rule fires on it."""
        self.last_activity = time.monotonic()

    def check(self, *, budget: float, quiet: float) -> Optional[str]:
        """Returns None (keep waiting), "post_tool_quiet", or "budget"."""
        if self.outstanding_tools > 0 or self.approvals_pending > 0:
            return None
        now = time.monotonic()
        # A compacting CLI is indistinguishable from a wedged one: between
        # PreCompact and compact_boundary it emits nothing at all. Without this
        # gate the post_tool_quiet rule reads that silence as a wedge and
        # interrupts the CLI mid-compaction, so the terminal ResultMessage never
        # arrives -- surfacing next turn as "discarding N stale unsolicited
        # text(s)" and, to the user, as a turn that simply died. Bounded, so a
        # boundary that never arrives cannot hang the turn instead.
        if (
            self.compaction_active > 0
            and (now - self.compaction_started) < _COMPACTION_MAX_SUSPEND
        ):
            return None
        idle = now - self.last_activity
        if quiet > 0 and self.post_tool_armed and idle >= quiet:
            return "post_tool_quiet"
        elapsed = now - self.started
        if elapsed >= budget and idle >= min(_BUDGET_GRACE_IDLE, budget):
            return "budget"
        return None


def _swallow_interrupt_result(future: Any) -> None:
    """Done-callback for the fire-and-forget client.interrupt() future: the
    SDK's control request times out after 60s on a wedged CLI and an
    unretrieved exception would log 'Future exception was never retrieved'
    at teardown — retrieve and demote it."""
    try:
        future.result()
    except Exception as exc:
        logger.debug(
            "SDK interrupt control request failed: %s",
            _safe_sdk_error_text(exc),
        )


def _swallow_steer_result(future: Any) -> None:
    """Same contract as _swallow_interrupt_result, for the fire-and-forget
    steer query(). The caller has already returned True by the time this
    resolves, so a failure here can only be logged, not surfaced."""
    try:
        future.result()
    except Exception:
        logger.debug("SDK steer query failed after scheduling", exc_info=True)


class _StreamEnd:
    """Reader-loop sentinel: the SDK message stream ended (CLI exited or the
    transport tore down). Routed to the in-flight turn so it fails fast and
    retires cleanly instead of waiting out its full turn_timeout on a dead
    stream."""

    def __init__(self, error: Optional[str] = None) -> None:
        self.error = error
