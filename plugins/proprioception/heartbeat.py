"""Delta heartbeat: decide *whether* to speak and *what* to say.

Design constraints (these are load-bearing, not style — several came out
of an adversarial review):

* **Ephemeral tail injection only.** The returned text goes through the
  ``pre_llm_call`` → ``plugin_user_context`` path, which appends it to the
  current turn's user message at API-call time and never persists it. At
  most ONE heartbeat is visible in context at any moment, so repeated
  near-identical blocks cannot accumulate and self-condition the model
  (the failure mode behind the tool-loop degeneration work, #41490).
* **Every emission has a cache price.** The injected tokens make the
  current user message diverge from what history replays next turn, so
  the previous turn's tail gets re-prefilled once per emission. Bounded
  (one turn of lag), but it means emissions must be RARE: no all-green
  baselines, no content-free changes, no flap storms.
* **Fenced and attributed.** The text rides inside the user message, so
  a smaller model needs an explicit fence + system note to know it is
  telemetry about the HOST machine, not the user talking (mirrors the
  ``<memory-context>`` convention in agent/memory_manager.py).
* **Renderer/fingerprint parity.** Everything the change-fingerprint
  tracks has a renderer; if nothing renders, nothing is emitted. A
  fingerprint delta can never produce an empty change message.
* **Perception, not policy.** The text reports state; it never instructs
  the model to act, retry, or route around anything.
* **Never break the turn.** Any internal failure returns ``None``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from plugins.proprioception import collector
from plugins.proprioception.collector import ATTENTION_STATES

logger = logging.getLogger(__name__)

# Context-fill thresholds (fractions of the configured window). Upward
# crossings are reported with a small hysteresis margin; downward moves
# (compression just ran — self-evident) silently re-bucket.
_CTX_BUCKETS = (0.50, 0.70, 0.85)
_CTX_HYSTERESIS = 0.02

# A (system, attention-state) edge may bypass the rate limit only once
# per this window; a flapping system degrades to normal rate-limited
# reporting instead of emitting on every turn.
_BYPASS_MEMORY_SECONDS = 600.0

_FENCE_OPEN = "<host-telemetry>"
_FENCE_CLOSE = "</host-telemetry>"
_SYSTEM_NOTE = (
    "[System note: host status for this assistant's own machine — not the user's "
    "words, not a feeling to perform. Reason with it; never narrate your own state. "
    "Mention it only if the user asks about system status, or a component their "
    "request needs is down (then say what it means for their task, not the infra).]"
)

# Rotating sentence frames for change messages. With ephemeral injection
# only one is ever visible, so rotation is belt-and-suspenders against
# repetition — cheap, deterministic (per-session counter), testable.
_CHANGE_FRAMES = (
    "Host status change since the last reading:",
    "Since the previous reading, host status shifted:",
    "Host telemetry update:",
    "A change in the host machine's status:",
)

_STATE_WORDS = {
    "ok": "ok",
    "info": "info",
    "warn": "needs attention",
    "warning": "needs attention",
    "down": "DOWN",
    "error": "ERROR",
    "crit": "CRITICAL",
    "critical": "CRITICAL",
    "absent": "no longer tracked",
}


class _SessionState:
    __slots__ = (
        "lock",
        "fingerprint",
        "ctx_bucket",
        "last_emit",
        "emit_count",
        "last_dash_snapshot",
        "prev_sensors_down",
        "prev_gateway_state",
        "prev_verdict",
        "prev_on_primary",
        "bypass_memory",
        "last_turn_wall",
    )

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.fingerprint: Optional[Tuple] = None
        self.ctx_bucket: Optional[int] = None
        self.last_emit: float = 0.0
        self.emit_count: int = 0
        # Last snapshot that actually CARRIED dashboard data — kept
        # separately so a sensor outage between readings can't swallow a
        # system transition (diffing against a dashboard-less snapshot
        # returns no transitions).
        self.last_dash_snapshot: Optional[collector.Snapshot] = None
        self.prev_sensors_down: Tuple[str, ...] = ()
        self.prev_gateway_state: str = ""
        self.prev_verdict: str = ""
        # Whether the previous turn ran on the primary model runtime. Starts
        # True (assume home); flips when last_turn telemetry reports a fallback.
        self.prev_on_primary: bool = True
        self.bypass_memory: Dict[Tuple[str, str], float] = {}
        # Wall-clock stamp of this session's previous turn. Wall, not
        # monotonic, on purpose: we want real elapsed time across a
        # suspension/idle gap, including any time the machine spent asleep.
        # None until the first turn records one.
        self.last_turn_wall: Optional[float] = None


_SESSIONS_LOCK = threading.Lock()
_SESSIONS: "OrderedDict[str, _SessionState]" = OrderedDict()
_MAX_SESSIONS = 1024


def _session_state(session_id: str) -> _SessionState:
    with _SESSIONS_LOCK:
        state = _SESSIONS.get(session_id)
        if state is None:
            state = _SessionState()
            _SESSIONS[session_id] = state
        _SESSIONS.move_to_end(session_id)
        while len(_SESSIONS) > _MAX_SESSIONS:
            evicted_id, _ = _SESSIONS.popitem(last=False)
            logger.info("proprioception: evicted session state for %s", evicted_id)
        return state


def _ctx_bucket(tokens: int, window: int, *, hysteresis: bool = False) -> int:
    frac = tokens / max(1, window)
    margin = _CTX_HYSTERESIS if hysteresis else 0.0
    bucket = 0
    for threshold in _CTX_BUCKETS:
        if frac >= threshold + margin:
            bucket += 1
    return bucket


def _estimate_context_tokens(conversation_history: Optional[List[Dict[str, Any]]]) -> int:
    if not conversation_history:
        return 0
    from agent.model_metadata import estimate_messages_tokens_rough

    return estimate_messages_tokens_rough(conversation_history)


def _fmt_tokens(tokens: int) -> str:
    if tokens >= 1000:
        return f"~{tokens / 1000:.0f}k"
    return f"~{tokens}"


def _humanize_gap(seconds: float) -> str:
    """Coarse human duration for a suspension gap: 45s / 20 min / 2.5 h / 3.1 days."""
    if seconds < 90:
        return f"{int(seconds)}s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = minutes / 60.0
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24.0:.1f} days"


def _wrap(body: str, settings: Dict[str, Any]) -> str:
    text = f"{_FENCE_OPEN}\n{_SYSTEM_NOTE}\n\n{body}\n{_FENCE_CLOSE}"
    limit = int(settings["max_chars"])
    if len(text) <= limit:
        return text
    # Truncate the body, keep the fence intact.
    overhead = len(_FENCE_OPEN) + len(_FENCE_CLOSE) + len(_SYSTEM_NOTE) + 4
    body = body[: max(20, limit - overhead - 1)] + "…"
    return f"{_FENCE_OPEN}\n{_SYSTEM_NOTE}\n\n{body}\n{_FENCE_CLOSE}"


def _ctx_phrase(ctx_tokens: int, window: int) -> str:
    pct = 100.0 * ctx_tokens / max(1, window)
    return (
        f"conversation {_fmt_tokens(ctx_tokens)} tokens of the "
        f"{_fmt_tokens(window)}-token window (rough count, excludes system prompt)"
    )


def _baseline_body(
    snap: collector.Snapshot, ctx_tokens: int, window: int, bucket: int
) -> Tuple[str, bool]:
    """Return (body, has_signal). All-green baselines carry no signal."""
    parts: List[str] = []
    signal = False
    if snap.dashboard is not None:
        systems = [s for s in (snap.dashboard.get("systems") or []) if isinstance(s, dict)]
        not_ok = [s for s in systems if str(s.get("state")) in ATTENTION_STATES]
        if not_ok:
            signal = True
            names = "; ".join(
                f"{s.get('label', s.get('id', '?'))}: "
                f"{_STATE_WORDS.get(str(s.get('state')), s.get('state'))}"
                for s in not_ok[:4]
            )
            parts.append(
                f"{len(systems)} host systems tracked, attention on {len(not_ok)} ({names})"
            )
        else:
            parts.append(f"all {len(systems)} tracked host systems ok")
        verdict = str(snap.dashboard.get("verdict", ""))
        if verdict and verdict != "ok":
            signal = True
            parts.append(f"overall verdict: {verdict}")
    else:
        # A cold-start fetch miss is usually the dashboard's own collection
        # cycle briefly blocking its single-threaded listener — with no prior
        # reading to compare against, announcing "feed unreachable" would be
        # a likely false alarm. Stay silent; a real outage is reported as a
        # sensor-loss transition on a later turn (where history exists).
        parts.append("host status feed unreachable (no external readings available)")
    if snap.gateway is not None and snap.gateway_state not in ("running", "ok"):
        signal = True
        parts.append(f"gateway {snap.gateway_state}")
    if bucket > 0:
        signal = True
        parts.append(_ctx_phrase(ctx_tokens, window))
    body = "First status reading this session: " + "; ".join(parts) + "."
    return body, signal


def build_heartbeat(
    *,
    session_id: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    settings: Dict[str, Any],
    last_turn: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return heartbeat text for this turn, or ``None`` to stay silent.

    ``last_turn`` is the core turn-telemetry record (agent/turn_telemetry.py)
    describing the PREVIOUS turn's outcome — used to notice when the agent
    silently ran off its primary model runtime.
    """
    mode = settings["heartbeat"]
    if mode == "off":
        return None

    # Collect BEFORE taking the per-session lock: get_snapshot serves
    # stale-while-revalidating, so this is cheap for everyone except the
    # single refreshing thread.
    snap = collector.get_snapshot(settings)
    fp = collector.fingerprint(snap)
    ctx_tokens = _estimate_context_tokens(conversation_history)
    window = int(settings["context_window"])

    state = _session_state(session_id)
    with state.lock:
        return _decide_locked(state, snap, fp, ctx_tokens, window, mode, settings, last_turn)


def _decide_locked(
    state: _SessionState,
    snap: collector.Snapshot,
    fp: Tuple,
    ctx_tokens: int,
    window: int,
    mode: str,
    settings: Dict[str, Any],
    last_turn: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    now = time.monotonic()
    wall_now = time.time()

    # Suspension/continuity sense: real wall-clock time elapsed since this
    # session's previous turn. Wall clock (not monotonic) so a machine that
    # slept counts the sleep. Reported once, when the gap clears the configured
    # threshold; the stamp advances every turn so it cannot repeat. A backward
    # clock (skew/NTP) yields a non-positive delta and is ignored.
    gap_seconds: Optional[float] = None
    if state.last_turn_wall is not None:
        delta = wall_now - state.last_turn_wall
        if delta > 0 and delta >= float(settings["gap_report_seconds"]):
            gap_seconds = delta

    # Optional always-on clock: opt-in temporal grounding. When enabled, a
    # compact "time + delta since last turn" line is emitted EVERY turn (forcing
    # emission, bypassing the rate limit) so the model is never time-blind — it
    # reasons about elapsed time accurately when handed a stamp, and asserts
    # "0 seconds, confidence 1.0" without one. Trades prefix-cache reuse for the
    # grounding, hence off by default. (A persisted-in-history stamp would be
    # cache-optimal; this ephemeral line is the simple first version.)
    clock_line: Optional[str] = None
    if settings.get("clock"):
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(wall_now))
        if state.last_turn_wall is not None and wall_now > state.last_turn_wall:
            clock_line = (f"clock {ts}, +{_humanize_gap(wall_now - state.last_turn_wall)} "
                          "since your last turn")
        else:
            clock_line = f"clock {ts} (first turn this session)"

    # Whether the PREVIOUS turn ran on the primary model runtime. Only trust a
    # record that actually carried data (has_data); otherwise assume unchanged.
    lt = last_turn or {}
    if lt.get("has_data"):
        cur_on_primary = not bool(lt.get("was_fallback"))
    else:
        cur_on_primary = state.prev_on_primary

    # Bucket update rule: climb only past threshold+hysteresis (and report
    # it); fall silently on any drop below a threshold (compression just
    # ran — self-evident); inside the hysteresis band, keep the old bucket
    # so a slow creep can't silently swallow a crossing.
    bucket_up = _ctx_bucket(ctx_tokens, window, hysteresis=True)
    bucket_down = _ctx_bucket(ctx_tokens, window)
    prev_bucket = state.ctx_bucket if state.ctx_bucket is not None else 0
    if bucket_up > prev_bucket:
        new_bucket, ctx_crossed_up = bucket_up, True
    elif bucket_down < prev_bucket:
        new_bucket, ctx_crossed_up = bucket_down, False
    else:
        new_bucket, ctx_crossed_up = prev_bucket, False

    def _record(emitted: bool) -> None:
        state.fingerprint = fp
        state.ctx_bucket = new_bucket
        if snap.dashboard is not None:
            state.last_dash_snapshot = snap
        state.prev_sensors_down = snap.sensors_down
        state.prev_gateway_state = snap.gateway_state
        state.prev_verdict = (
            str(snap.dashboard.get("verdict", "")) if snap.dashboard is not None else ""
        )
        state.prev_on_primary = cur_on_primary
        state.last_turn_wall = wall_now
        if emitted:
            state.last_emit = now
            state.emit_count += 1

    # --- first reading for this session ---
    if state.fingerprint is None:
        body, signal = _baseline_body(snap, ctx_tokens, window, bucket_down)
        if clock_line:
            body = clock_line + ". " + body
        _record(emitted=signal or bool(clock_line))
        # Keep "home" as the fallback reference across the baseline: if the very
        # first turn already ran off-primary, we want the next reading to see a
        # True->False transition and report it — not silently adopt the degraded
        # state as normal.
        state.prev_on_primary = True
        if signal or clock_line or mode == "always":
            return _wrap(body, settings)
        return None  # all green: silence is the baseline

    # --- diff against the previous reading ---
    lines: List[str] = []

    transitions = collector.diff_systems(state.last_dash_snapshot, snap)
    for label, old, new in transitions[:6]:
        lines.append(
            f"{label}: {_STATE_WORDS.get(old, old)} -> {_STATE_WORDS.get(new, new)}"
        )
    if len(transitions) > 6:
        lines.append(f"(+{len(transitions) - 6} more transitions)")

    prev_down, cur_down = set(state.prev_sensors_down), set(snap.sensors_down)
    sensor_lost = bool(cur_down - prev_down)
    for sensor in sorted(cur_down - prev_down):
        lines.append(f"lost the {sensor} status feed (unreachable)")
    for sensor in sorted(prev_down - cur_down):
        lines.append(f"{sensor} status feed back online")

    # Fallback awareness: did the previous turn silently run off the primary
    # model? Report the TRANSITION only (not every degraded turn). Framed as
    # "off primary", never "cloud" — the output firewall + system-note keep it
    # from reaching a user; this line is for the agent's own reasoning.
    fallback_transition = False
    if state.prev_on_primary and not cur_on_primary:
        fallback_transition = True
        primary = str(lt.get("primary_model") or lt.get("primary_provider") or "your primary model")
        served = str(lt.get("provider") or "a fallback runtime")
        lines.append(
            f"last turn was served by a fallback runtime ({served}), not your primary "
            f"({primary}) — the primary was unreachable"
        )
    elif (not state.prev_on_primary) and cur_on_primary:
        fallback_transition = True
        lines.append("back on your primary model runtime")

    if snap.gateway is not None and state.prev_gateway_state not in ("", snap.gateway_state):
        lines.append(f"gateway: {state.prev_gateway_state} -> {snap.gateway_state}")

    cur_verdict = (
        str(snap.dashboard.get("verdict", "")) if snap.dashboard is not None else ""
    )
    # Verdict changes usually ride along with a system transition; only
    # render it alone when nothing else explains the shift.
    if not transitions and cur_verdict and state.prev_verdict not in ("", cur_verdict):
        lines.append(f"overall verdict: {state.prev_verdict} -> {cur_verdict}")

    if ctx_crossed_up:
        boundary = _CTX_BUCKETS[new_bucket - 1]
        pct = 100.0 * ctx_tokens / max(1, window)
        lines.append(
            f"context fill climbed past {boundary:.0%} "
            f"(now ~{pct:.0f}% of {_fmt_tokens(window)})"
        )

    if gap_seconds is not None:
        lines.append(
            f"~{_humanize_gap(gap_seconds)} elapsed since your last turn "
            "(idle/suspended between turns) — time-sensitive state (system "
            "health, running jobs, the clock) may have moved on since"
        )

    if mode == "always":
        frame = _CHANGE_FRAMES[state.emit_count % len(_CHANGE_FRAMES)]
        if lines:
            body = f"{frame} " + "; ".join(lines) + ". Status only — act on it only if it matters to the user's request."
        else:
            body, _ = _baseline_body(snap, ctx_tokens, window, bucket_down)
            body = body.replace("First status reading this session:", f"Periodic reading — {frame.rstrip(':').lower()}", 1)
        if clock_line:
            body = clock_line + ". " + body
        _record(emitted=True)
        return _wrap(body, settings)

    if not lines and not clock_line:
        # Includes the fingerprint-changed-but-nothing-renders case (by
        # construction there shouldn't be one) and downward ctx moves:
        # update silently so the same non-event can't re-trigger.
        _record(emitted=False)
        return None

    # Rate limit with a degradation bypass — but each (system, bad-state)
    # edge may bypass only once per window, so a flapping system can't
    # emit on every turn.
    min_interval = float(settings["min_interval_seconds"])
    degraded_edges = [
        (label, new)
        for label, _old, new in transitions
        if new in ATTENTION_STATES
    ]
    fresh_degradation = False
    for edge in degraded_edges:
        if now - state.bypass_memory.get(edge, -_BYPASS_MEMORY_SECONDS) >= _BYPASS_MEMORY_SECONDS:
            fresh_degradation = True
    # A fallback transition or a long suspension gap is a discrete, important
    # event — emit promptly (each is naturally one-shot: the stamp/flag advances).
    bypass = (fresh_degradation or sensor_lost or fallback_transition
              or (gap_seconds is not None) or (clock_line is not None))

    if not bypass and (now - state.last_emit) < min_interval:
        # Suppressed: deliberately do NOT update state, so the change is
        # still reported once the window opens (diff old-vs-newest).
        return None

    for edge in degraded_edges:
        state.bypass_memory[edge] = now
    if len(state.bypass_memory) > 64:
        cutoff = now - _BYPASS_MEMORY_SECONDS
        state.bypass_memory = {
            k: v for k, v in state.bypass_memory.items() if v >= cutoff
        }

    frame = _CHANGE_FRAMES[state.emit_count % len(_CHANGE_FRAMES)]
    _record(emitted=True)
    change_body = ""
    if lines:
        change_body = (f"{frame} " + "; ".join(lines)
                       + ". Status only — act on it only if it matters to the user's request.")
    if clock_line and change_body:
        body = clock_line + ". " + change_body
    elif clock_line:
        body = clock_line + "."
    else:
        body = change_body
    return _wrap(body, settings)


def reset_for_tests() -> None:
    with _SESSIONS_LOCK:
        _SESSIONS.clear()
