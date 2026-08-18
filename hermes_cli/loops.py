"""Recurring in-session wakeups — the /loop command (Claude Code parity).

``/loop [interval] <prompt>`` re-runs a prompt (or a slash command) on a
recurring cadence INSIDE the current session. Each tick is a real agent
turn: the wakeup prompt is injected through the exact same input path as
a typed user message, so the agent always sees current state (latest CI
result, newest queue depth, the file as it is now).

Two cadence modes, mirroring Claude Code's ``/loop``:

- **Fixed interval** — ``/loop 5m check the deploy`` fires every 5 minutes.
- **Self-paced** — ``/loop keep refining the failing test until green``
  (no interval token) lets the loop set its own rhythm: it starts fast and
  backs off exponentially while the agent's replies stop changing, then
  snaps back to the floor as soon as a reply differs. Zero extra LLM cost —
  change detection is a local digest comparison.

Stop conditions (any of):

- The agent ends a wakeup reply with ``LOOP_COMPLETE`` on its own line
  (the wakeup prompt teaches it to do so when the task is done/moot).
- ``--times N`` — stop after N ticks.
- ``--until <condition>`` — an evidence-based stop judged by the same
  auxiliary judge that powers /goal (fail-open: a broken judge never
  wedges the loop; the tick budget is the backstop).
- ``/loop stop`` / ``/loop clear`` — user control.
- ``loops.max_ticks`` config backstop (default 100, 0 = unlimited).

Design notes / invariants (same contract as ``hermes_cli/goals.py``):

- A wakeup is just a normal user-role message appended via the surface's
  ordinary input path. No system-prompt mutation, no toolset swap —
  prompt caching stays intact and role alternation is preserved.
- Wakeups only fire while the session is IDLE. A real user message always
  wins; the tick just re-arms and fires at the next idle boundary.
- State is persisted in SessionDB's ``state_meta`` table keyed by
  ``loop:<session_id>`` so ``/resume`` picks the loop back up.
- /goal mixing: an active /goal takes priority. When the goal loop has a
  continuation queued (or the goal judge is mid-flight), the /loop tick
  defers to the next interval instead of racing a second synthetic turn.
  Goal-continuation turns never count as loop ticks and vice versa.
- This module has zero hard dependency on ``cli.HermesCLI``, the gateway
  runner, or the TUI gateway — all three drive the same ``LoopManager``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────

# Floor for fixed intervals. Claude Code allows 30s; anything tighter is
# almost always an accident that burns tokens polling state that hasn't
# changed. Overridable via loops.min_interval_seconds (still clamped ≥ 5).
DEFAULT_MIN_INTERVAL_SECONDS = 30

# Backstop tick budget so an unattended loop can't run forever by default.
# 0 = unlimited (Claude Code behavior); config loops.max_ticks.
DEFAULT_MAX_TICKS = 100

# Self-paced mode: start at the floor, double while replies are unchanged,
# cap at the ceiling, snap back to the floor on any change.
DEFAULT_SELF_PACED_FLOOR_SECONDS = 60
DEFAULT_SELF_PACED_CEILING_SECONDS = 15 * 60

# The completion sentinel the wakeup prompt teaches the agent to emit when
# the loop's task is finished or no longer applicable.
LOOP_COMPLETE_MARKER = "LOOP_COMPLETE"

# Matches the marker on its own line (possibly with surrounding whitespace
# or trailing punctuation the model added despite instructions).
_LOOP_COMPLETE_RE = re.compile(
    r"(?im)^\s*" + re.escape(LOOP_COMPLETE_MARKER) + r"\s*[.!]?\s*$"
)

# Interval token: 30s / 5m / 2h / 1h30m (compound units allowed, at least one).
_INTERVAL_TOKEN_RE = re.compile(
    r"^(?=\d)(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", re.IGNORECASE
)


WAKEUP_PROMPT_TEMPLATE = (
    "[/loop wakeup #{tick}{cadence}]\n"
    "Recurring task: {prompt}\n\n"
    "This is an automatic wakeup from the /loop the user set. Perform the "
    "task now against the CURRENT state (re-check files, processes, or "
    "services fresh — do not assume anything from earlier iterations still "
    "holds). Report concisely what you found or did this iteration.\n"
    "If the task is now complete, no longer applicable, or the thing you "
    "were watching has finished, say so and end your reply with "
    f"{LOOP_COMPLETE_MARKER} on its own line — that stops the loop."
)

WAKEUP_PROMPT_WITH_UNTIL_TEMPLATE = (
    "[/loop wakeup #{tick}{cadence}]\n"
    "Recurring task: {prompt}\n\n"
    "Stop condition: {until}\n\n"
    "This is an automatic wakeup from the /loop the user set. Perform the "
    "task now against the CURRENT state (re-check files, processes, or "
    "services fresh — do not assume anything from earlier iterations still "
    "holds). Report concisely what you found or did this iteration, and "
    "show concrete evidence of the stop condition's status.\n"
    "If the stop condition is met, or the task is no longer applicable, say "
    f"so and end your reply with {LOOP_COMPLETE_MARKER} on its own line — "
    "that stops the loop."
)


# ──────────────────────────────────────────────────────────────────────
# Interval parsing
# ──────────────────────────────────────────────────────────────────────


def parse_interval_token(token: str) -> Optional[int]:
    """Parse a compact interval token (``30s``/``5m``/``2h``/``1h30m``).

    Returns total seconds, or None when the token is not an interval.
    A bare number is NOT an interval (too easy to collide with prompt
    text like ``/loop 3 things to check``) — units are required.
    """
    if not token:
        return None
    m = _INTERVAL_TOKEN_RE.match(token.strip())
    if not m:
        return None
    h, mnt, s = (int(g) if g else 0 for g in m.groups())
    total = h * 3600 + mnt * 60 + s
    return total if total > 0 else None


def parse_loop_args(text: str) -> Dict[str, Any]:
    """Parse the argument string of ``/loop [interval] <prompt> [flags]``.

    Recognized shapes::

        /loop 5m check the deploy status
        /loop every 10m /babysit-prs
        /loop keep fixing the failing test until the suite passes
        /loop 2m poll CI --times 30
        /loop 5m watch the queue --until queue depth reaches zero

    Returns ``{"interval_seconds": int|None, "prompt": str, "times": int,
    "until": str, "error": str|None}``. ``interval_seconds`` None means
    self-paced. ``error`` is set for unusable input (empty prompt,
    interval-only, bad --times).
    """
    raw = (text or "").strip()
    result: Dict[str, Any] = {
        "interval_seconds": None,
        "prompt": "",
        "times": 0,
        "until": "",
        "error": None,
    }
    if not raw:
        result["error"] = "empty"
        return result

    # Pull trailing flags first so an interval-looking token inside the
    # --until clause can't confuse the front parse. Flags may appear in
    # either order at the end of the line; --until consumes to end-of-line
    # (or to a following --times).
    times = 0
    until = ""

    m_times = re.search(r"\s--times\s+(\S+)", raw)
    if m_times:
        try:
            times = int(m_times.group(1))
            if times < 1:
                raise ValueError
        except ValueError:
            result["error"] = f"--times expects a positive integer, got {m_times.group(1)!r}"
            return result
        raw = (raw[: m_times.start()] + raw[m_times.end():]).strip()

    m_until = re.search(r"\s--until\s+(.+)$", raw, re.DOTALL)
    if m_until:
        until = m_until.group(1).strip()
        raw = raw[: m_until.start()].strip()

    # Leading "every" sugar: /loop every 5m <prompt>
    tokens = raw.split(None, 1)
    if tokens and tokens[0].lower() == "every" and len(tokens) > 1:
        raw = tokens[1]
        tokens = raw.split(None, 1)

    interval: Optional[int] = None
    if tokens:
        maybe = parse_interval_token(tokens[0])
        if maybe is not None:
            interval = maybe
            raw = tokens[1].strip() if len(tokens) > 1 else ""

    if not raw:
        result["error"] = "missing prompt (usage: /loop [interval] <prompt>)"
        return result

    result["interval_seconds"] = interval
    result["prompt"] = raw
    result["times"] = times
    result["until"] = until
    return result


def format_interval(seconds: float) -> str:
    """Render seconds as a compact human interval (``90`` → ``1m30s``)."""
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s or not parts:
        parts.append(f"{s}s")
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────


def _loops_config() -> Dict[str, Any]:
    """Read the ``loops:`` config section (cached load_config underneath)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        section = cfg.get("loops") or {}
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def min_interval_seconds() -> int:
    try:
        value = int(_loops_config().get("min_interval_seconds", DEFAULT_MIN_INTERVAL_SECONDS))
        return max(5, value)
    except Exception:
        return DEFAULT_MIN_INTERVAL_SECONDS


def max_ticks_default() -> int:
    try:
        value = int(_loops_config().get("max_ticks", DEFAULT_MAX_TICKS))
        return max(0, value)
    except Exception:
        return DEFAULT_MAX_TICKS


def self_paced_floor_seconds() -> int:
    try:
        value = int(_loops_config().get("self_paced_floor_seconds", DEFAULT_SELF_PACED_FLOOR_SECONDS))
        return max(10, value)
    except Exception:
        return DEFAULT_SELF_PACED_FLOOR_SECONDS


def self_paced_ceiling_seconds() -> int:
    floor = self_paced_floor_seconds()
    try:
        value = int(_loops_config().get("self_paced_ceiling_seconds", DEFAULT_SELF_PACED_CEILING_SECONDS))
        return max(floor, value)
    except Exception:
        return max(floor, DEFAULT_SELF_PACED_CEILING_SECONDS)


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class LoopState:
    """Serializable /loop state stored per session."""

    prompt: str
    status: str = "active"            # active | paused | done | cleared
    mode: str = "interval"            # interval | self_paced
    interval_seconds: float = 0.0     # fixed cadence (mode == "interval")
    current_delay: float = 0.0        # live cadence (self-paced backoff)
    times: int = 0                    # user cap (--times N); 0 = none
    until: str = ""                   # judged stop condition; "" = none
    max_ticks: int = DEFAULT_MAX_TICKS  # config backstop; 0 = unlimited
    ticks_fired: int = 0
    created_at: float = 0.0
    last_fired_at: float = 0.0
    next_due_at: float = 0.0
    # True between "wakeup injected" and "that turn's response evaluated".
    # Keeps a tick from double-firing while its turn is still running and
    # tells the post-turn hook that the turn that just ended was ours.
    awaiting_response: bool = False
    # Self-paced change detection: digest of the previous wakeup's reply.
    last_response_digest: str = ""
    paused_reason: Optional[str] = None
    last_stop_reason: Optional[str] = None
    # Gateway routing captured at creation time (platform / chat_id /
    # chat_type / thread_id) so the idle wakeup watcher can inject the
    # tick back into the right chat. Empty for CLI / TUI sessions, which
    # drive ticks from their own session-local schedulers.
    route: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "LoopState":
        data = json.loads(raw)
        route = data.get("route")
        return cls(
            prompt=data.get("prompt", ""),
            status=data.get("status", "active"),
            mode=data.get("mode", "interval"),
            interval_seconds=float(data.get("interval_seconds", 0.0) or 0.0),
            current_delay=float(data.get("current_delay", 0.0) or 0.0),
            times=int(data.get("times", 0) or 0),
            until=str(data.get("until", "") or ""),
            max_ticks=int(data.get("max_ticks", DEFAULT_MAX_TICKS) or 0),
            ticks_fired=int(data.get("ticks_fired", 0) or 0),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            last_fired_at=float(data.get("last_fired_at", 0.0) or 0.0),
            next_due_at=float(data.get("next_due_at", 0.0) or 0.0),
            awaiting_response=bool(data.get("awaiting_response", False)),
            last_response_digest=str(data.get("last_response_digest", "") or ""),
            paused_reason=data.get("paused_reason"),
            last_stop_reason=data.get("last_stop_reason"),
            route=route if isinstance(route, dict) else {},
        )

    # --- helpers -------------------------------------------------------

    def cadence_label(self) -> str:
        if self.mode == "self_paced":
            live = f", currently {format_interval(self.current_delay)}" if self.current_delay else ""
            return f"self-paced{live}"
        return f"every {format_interval(self.interval_seconds)}"

    def remaining_label(self) -> str:
        if self.status != "active":
            return ""
        remaining = self.next_due_at - time.time()
        if remaining <= 0:
            return "due now"
        return f"next in {format_interval(remaining)}"


# ──────────────────────────────────────────────────────────────────────
# Persistence (SessionDB state_meta)
# ──────────────────────────────────────────────────────────────────────

_META_PREFIX = "loop:"


def _meta_key(session_id: str) -> str:
    return f"{_META_PREFIX}{session_id}"


def _get_session_db() -> Optional[Any]:
    """One SessionDB per HERMES_HOME.

    Delegates to the goals module's cached SessionDB so goals, loops,
    and heartbeats share one connection (same pattern as
    ``hermes_cli/heartbeat.py``). The delegation also inherits the
    off-loop bootstrap and the window logic: a cold cache on the loop
    thread never runs ``SessionDB()`` inline. The previous copy here
    did, which froze the loop for the init duration and dropped the
    first ``loop:*`` write (the /goal bug class, #88965).
    """
    try:
        from hermes_cli.goals import _get_session_db as _goals_db
    except Exception as exc:  # pragma: no cover
        logger.debug("LoopManager: SessionDB bootstrap failed (%s)", exc)
        return None
    return _goals_db()


def load_loop(session_id: str) -> Optional[LoopState]:
    """Load the loop for a session, or None if none exists."""
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        logger.debug("LoopManager: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        return LoopState.from_json(raw)
    except Exception as exc:
        logger.warning("LoopManager: could not parse stored loop for %s: %s", session_id, exc)
        return None


def save_loop(session_id: str, state: LoopState) -> None:
    """Persist a loop to SessionDB. No-op if DB unavailable."""
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        logger.warning(
            "LoopManager: loop for %s not persisted — session DB "
            "unavailable (bootstrap window exceeded, in-memory state "
            "still active)",
            session_id,
        )
        return
    try:
        db.set_meta(_meta_key(session_id), state.to_json())
    except Exception as exc:
        logger.debug("LoopManager: set_meta failed: %s", exc)


def clear_loop(session_id: str) -> None:
    """Mark a loop cleared in the DB (preserved for audit, status=cleared)."""
    state = load_loop(session_id)
    if state is None:
        return
    state.status = "cleared"
    save_loop(session_id, state)


def list_active_loops() -> List[Tuple[str, LoopState]]:
    """Return ``[(session_id, LoopState), ...]`` for every ACTIVE loop.

    Used by the gateway's idle wakeup watcher, which has no per-session
    scheduler and instead scans for due loops on a coarse tick. Best-effort:
    any DB error yields ``[]``.
    """
    db = _get_session_db()
    if db is None:
        return []
    try:
        rows = db.list_meta_prefix(_META_PREFIX)
    except Exception as exc:
        logger.debug("LoopManager: list_meta_prefix failed: %s", exc)
        return []
    out: List[Tuple[str, LoopState]] = []
    for key, raw in rows:
        session_id = key[len(_META_PREFIX):]
        if not session_id or not raw:
            continue
        try:
            state = LoopState.from_json(raw)
        except Exception:
            continue
        if state.status == "active":
            out.append((session_id, state))
    return out


def migrate_loop_to_session(old_session_id: str, new_session_id: str, *, reason: str = "") -> bool:
    """Carry a persistent /loop from a parent session to its continuation.

    Context compression rotates ``session_id`` to a fresh child session;
    without this the loop silently dies at the compaction boundary (the
    same hazard /goal hit in #33618). Copies the loop onto the new session
    and archives the old row as ``cleared`` so exactly one active loop row
    exists per logical conversation. Best-effort and never raises.
    """
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        state = load_loop(old_session_id)
        if state is None or state.status == "cleared":
            return False
        if load_loop(new_session_id) is not None:
            return False
        save_loop(new_session_id, state)
        clear_loop(old_session_id)
        logger.debug(
            "LoopManager: migrated loop %s -> %s (%s)",
            old_session_id, new_session_id, reason or "rotation",
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("LoopManager: loop migration failed: %s", exc)
        return False


# ──────────────────────────────────────────────────────────────────────
# Response evaluation helpers
# ──────────────────────────────────────────────────────────────────────


def response_signals_complete(response: str) -> bool:
    """True when the agent ended its reply with the LOOP_COMPLETE marker."""
    if not response:
        return False
    return _LOOP_COMPLETE_RE.search(response) is not None


def _digest_response(response: str) -> str:
    """Stable digest for self-paced change detection.

    Normalizes whitespace and strips volatile timestamp-ish tokens so a
    reply that differs only by 'checked at 14:02:33' doesn't defeat the
    backoff.
    """
    text = (response or "").strip().lower()
    # Drop clock/timestamp tokens (14:02:33, 2026-07-26, 1500s, 25m ago...).
    text = re.sub(r"\d{1,2}:\d{2}(:\d{2})?", "", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(s|sec|secs|seconds|m|min|mins|minutes|h|hr|hrs|hours)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# LoopManager — the orchestration surface CLI + gateway + TUI talk to
# ──────────────────────────────────────────────────────────────────────


class LoopManager:
    """Per-session /loop state + tick decisions.

    Drivers (CLI process_loop, gateway wakeup watcher, TUI ticker) call:

    - ``set(...)`` / ``pause()`` / ``resume()`` / ``clear()`` — user controls.
    - ``is_due()`` — should a wakeup fire now? (cheap, in-memory)
    - ``fire_tick()`` — claim the tick; returns the wakeup message to inject.
    - ``complete_tick(last_response)`` — evaluate the finished wakeup turn:
      detect LOOP_COMPLETE, judge --until, apply --times / max_ticks caps,
      schedule the next tick (with self-paced backoff when applicable).
    - ``status_line()`` — printable one-liner.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._state: Optional[LoopState] = load_loop(session_id)

    # --- introspection ------------------------------------------------

    @property
    def state(self) -> Optional[LoopState]:
        return self._state

    def refresh(self) -> None:
        """Re-read state from the DB (cross-process safety for the gateway)."""
        self._state = load_loop(self.session_id)

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_loop(self) -> bool:
        return self._state is not None and self._state.status in {"active", "paused"}

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status == "cleared":
            return "No loop set. Start one with /loop [interval] <prompt>."
        fired = f"{s.ticks_fired} tick{'s' if s.ticks_fired != 1 else ''}"
        caps = []
        if s.times:
            caps.append(f"{s.ticks_fired}/{s.times} runs")
        elif s.max_ticks:
            caps.append(f"{s.ticks_fired}/{s.max_ticks} budget")
        else:
            caps.append(fired)
        if s.until:
            caps.append(f"until: {s.until}")
        meta = f"{s.cadence_label()}, {', '.join(caps)}"
        if s.status == "active":
            remaining = s.remaining_label()
            tail = f", {remaining}" if remaining else ""
            if s.awaiting_response:
                tail = ", wakeup running"
            return f"↻ Loop (active, {meta}{tail}): {s.prompt}"
        if s.status == "paused":
            extra = f" — {s.paused_reason}" if s.paused_reason else ""
            return f"⏸ Loop (paused, {meta}{extra}): {s.prompt}"
        if s.status == "done":
            extra = f" — {s.last_stop_reason}" if s.last_stop_reason else ""
            return f"✓ Loop finished ({fired}{extra}): {s.prompt}"
        return f"Loop ({s.status}, {meta}): {s.prompt}"

    # --- mutation -----------------------------------------------------

    def set(
        self,
        prompt: str,
        *,
        interval_seconds: Optional[int] = None,
        times: int = 0,
        until: str = "",
        route: Optional[Dict[str, str]] = None,
    ) -> LoopState:
        """Start a new loop (replaces any existing one for the session)."""
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("loop prompt is empty")

        now = time.time()
        if interval_seconds is not None:
            interval = max(int(interval_seconds), min_interval_seconds())
            state = LoopState(
                prompt=prompt,
                mode="interval",
                interval_seconds=float(interval),
                current_delay=float(interval),
                next_due_at=now + interval,
            )
        else:
            floor = self_paced_floor_seconds()
            state = LoopState(
                prompt=prompt,
                mode="self_paced",
                interval_seconds=0.0,
                current_delay=float(floor),
                next_due_at=now + floor,
            )
        state.times = max(0, int(times or 0))
        state.until = (until or "").strip()
        state.max_ticks = max_ticks_default()
        state.created_at = now
        state.route = dict(route or {})
        self._state = state
        save_loop(self.session_id, state)
        return state

    def pause(self, reason: str = "user-paused") -> Optional[LoopState]:
        if not self._state or self._state.status not in {"active", "paused"}:
            return None
        self._state.status = "paused"
        self._state.paused_reason = reason
        self._state.awaiting_response = False
        save_loop(self.session_id, self._state)
        return self._state

    def resume(self) -> Optional[LoopState]:
        if not self._state or self._state.status == "cleared":
            return None
        self._state.status = "active"
        self._state.paused_reason = None
        self._state.awaiting_response = False
        # Re-arm relative to now so a long pause doesn't fire instantly N times.
        delay = self._state.current_delay or self._state.interval_seconds or self_paced_floor_seconds()
        self._state.next_due_at = time.time() + min(delay, 5.0)
        save_loop(self.session_id, self._state)
        return self._state

    def clear(self) -> bool:
        if self._state is None or self._state.status == "cleared":
            return False
        self._state.status = "cleared"
        save_loop(self.session_id, self._state)
        self._state = None
        return True

    def mark_done(self, reason: str) -> None:
        if not self._state:
            return
        self._state.status = "done"
        self._state.last_stop_reason = reason
        self._state.awaiting_response = False
        save_loop(self.session_id, self._state)

    # --- tick lifecycle -------------------------------------------------

    def is_due(self, now: Optional[float] = None) -> bool:
        """Cheap check: active, not mid-wakeup, and the clock has passed."""
        s = self._state
        if s is None or s.status != "active" or s.awaiting_response:
            return False
        return (now if now is not None else time.time()) >= s.next_due_at

    def fire_tick(self) -> Optional[str]:
        """Claim a due tick. Returns the message to inject, or None.

        The returned text is either the wakeup-framed prompt or — when the
        loop's prompt is itself a slash command (``/loop 10m /recap``) —
        the raw command so the surface's normal slash dispatch handles it.
        Marks ``awaiting_response`` so the tick can't double-fire while its
        turn runs; drivers MUST follow up with ``complete_tick`` (or
        ``abandon_tick`` on injection failure).
        """
        s = self._state
        if s is None or not self.is_due():
            return None
        s.ticks_fired += 1
        s.last_fired_at = time.time()
        s.awaiting_response = True
        # Provisionally schedule the next tick from NOW; complete_tick
        # reschedules from turn end (so a 10-minute turn doesn't cause an
        # instant re-fire), but if the process dies mid-turn the provisional
        # schedule keeps the persisted loop from being 'due' in a tight loop.
        delay = s.current_delay or s.interval_seconds or self_paced_floor_seconds()
        s.next_due_at = s.last_fired_at + delay
        save_loop(self.session_id, s)

        if s.prompt.lstrip().startswith("/"):
            return s.prompt.strip()
        cadence = f", {s.cadence_label()}" if s.mode == "interval" else ", self-paced"
        template = WAKEUP_PROMPT_WITH_UNTIL_TEMPLATE if s.until else WAKEUP_PROMPT_TEMPLATE
        return template.format(tick=s.ticks_fired, cadence=cadence, prompt=s.prompt, until=s.until)

    def abandon_tick(self) -> None:
        """Roll back a fired tick whose injection failed (nothing ran)."""
        s = self._state
        if s is None or not s.awaiting_response:
            return
        s.awaiting_response = False
        s.ticks_fired = max(0, s.ticks_fired - 1)
        save_loop(self.session_id, s)

    def complete_tick(self, last_response: str) -> Dict[str, Any]:
        """Evaluate the finished wakeup turn and schedule what's next.

        Returns a decision dict::

            {"status": "active|done|paused", "stopped": bool,
             "reason": str, "message": str}

        ``message`` is a user-visible one-liner ("" when nothing worth
        saying — the common still-looping case stays quiet).
        """
        s = self._state
        if s is None or not s.awaiting_response:
            return {"status": s.status if s else None, "stopped": False, "reason": "no tick in flight", "message": ""}
        s.awaiting_response = False
        now = time.time()

        # 1. Agent self-stop marker.
        if response_signals_complete(last_response):
            s.status = "done"
            s.last_stop_reason = "agent signaled the task is complete"
            save_loop(self.session_id, s)
            return {
                "status": "done",
                "stopped": True,
                "reason": s.last_stop_reason,
                "message": f"✓ Loop finished after {s.ticks_fired} tick{'s' if s.ticks_fired != 1 else ''} — task complete.",
            }

        # 2. Evidence-based --until judge (reuses the /goal judge; fail-open).
        if s.until and (last_response or "").strip():
            try:
                from hermes_cli.goals import judge_goal

                verdict, reason, _pf, _wait, _tf = judge_goal(s.until, last_response)
            except Exception as exc:
                verdict, reason = "continue", f"judge unavailable: {type(exc).__name__}"
            if verdict == "done":
                s.status = "done"
                s.last_stop_reason = f"stop condition met: {reason}"
                save_loop(self.session_id, s)
                return {
                    "status": "done",
                    "stopped": True,
                    "reason": s.last_stop_reason,
                    "message": f"✓ Loop finished after {s.ticks_fired} tick{'s' if s.ticks_fired != 1 else ''} — {reason}",
                }

        # 3. --times user cap.
        if s.times and s.ticks_fired >= s.times:
            s.status = "done"
            s.last_stop_reason = f"completed the requested {s.times} runs"
            save_loop(self.session_id, s)
            return {
                "status": "done",
                "stopped": True,
                "reason": s.last_stop_reason,
                "message": f"✓ Loop finished — ran {s.times}/{s.times} times.",
            }

        # 4. Config backstop budget → pause (recoverable), not done.
        if s.max_ticks and s.ticks_fired >= s.max_ticks:
            s.status = "paused"
            s.paused_reason = f"tick budget exhausted ({s.ticks_fired}/{s.max_ticks})"
            save_loop(self.session_id, s)
            return {
                "status": "paused",
                "stopped": True,
                "reason": s.paused_reason,
                "message": (
                    f"⏸ Loop paused — {s.ticks_fired}/{s.max_ticks} ticks used "
                    "(loops.max_ticks). /loop resume to keep going, /loop stop to end it."
                ),
            }

        # 5. Still looping — schedule the next tick from turn end.
        if s.mode == "self_paced":
            digest = _digest_response(last_response)
            floor = self_paced_floor_seconds()
            ceiling = self_paced_ceiling_seconds()
            if digest and digest == s.last_response_digest:
                # Nothing changed — back off.
                s.current_delay = min(max(s.current_delay, floor) * 2, ceiling)
            else:
                s.current_delay = float(floor)
            s.last_response_digest = digest
        else:
            s.current_delay = s.interval_seconds
        s.next_due_at = now + s.current_delay
        save_loop(self.session_id, s)
        return {
            "status": "active",
            "stopped": False,
            "reason": "loop continues",
            "message": "",
        }


# ──────────────────────────────────────────────────────────────────────
# /goal mixing
# ──────────────────────────────────────────────────────────────────────


def goal_blocks_loop_tick(session_id: str) -> bool:
    """True when an ACTIVE /goal should defer this session's /loop tick.

    Both features inject synthetic continuation turns at idle boundaries.
    When a goal is actively driving the session (status ``active`` and not
    parked on a wait barrier), its judge-driven continuations own the idle
    boundary — firing a loop wakeup in between would interleave two
    synthetic conversations and burn the goal's turn budget on loop chatter.
    A goal that is parked (waiting on a pid/session/deadline), paused, or
    done does NOT block the loop.
    """
    try:
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id=session_id)
        if not mgr.is_active():
            return False
        # Parked goal → the loop may use the idle time.
        return not mgr.is_waiting()
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# Shared slash-command dispatch (CLI + gateway + TUI use the same logic)
# ──────────────────────────────────────────────────────────────────────


def dispatch_loop_command(
    mgr: "LoopManager",
    args: str,
    *,
    route: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Surface-agnostic handler for ``/loop <args>``.

    Returns ``{"output": str, "created": bool}``. ``output`` is ready to
    print/send verbatim; each surface only decorates it (dim colors on the
    CLI, plain text on messaging platforms). ``route`` is stored on newly
    created loops so the gateway's idle watcher can inject wakeups back
    into the right chat; CLI/TUI pass None.
    """
    arg = (args or "").strip()
    lower = arg.lower()

    if not arg or lower == "status":
        return {"output": mgr.status_line(), "created": False}

    if lower == "pause":
        state = mgr.pause(reason="user-paused")
        if state is None:
            return {"output": "No loop set.", "created": False}
        return {"output": f"⏸ Loop paused: {state.prompt}\nUse /loop resume to continue.", "created": False}

    if lower == "resume":
        state = mgr.resume()
        if state is None:
            return {"output": "No loop to resume.", "created": False}
        return {
            "output": f"▶ Loop resumed ({state.cadence_label()}): {state.prompt}",
            "created": False,
        }

    if lower in {"stop", "clear", "cancel"}:
        had = mgr.clear()
        return {"output": "✓ Loop stopped." if had else "No active loop.", "created": False}

    if lower in {"help", "--help", "-h"}:
        return {
            "output": (
                "Usage: /loop [interval] <prompt> [--times N] [--until <condition>]\n"
                "  /loop 5m check the deploy status      — fixed cadence\n"
                "  /loop every 10m /recap                — loop a slash command\n"
                "  /loop keep fixing tests until green   — self-paced (backs off while output is unchanged)\n"
                "  /loop 2m poll CI --times 30           — stop after 30 runs\n"
                "  /loop 5m watch the queue --until queue is empty\n"
                "Controls: /loop status · /loop pause · /loop resume · /loop stop\n"
                "The loop also stops itself when the agent replies with "
                f"{LOOP_COMPLETE_MARKER}."
            ),
            "created": False,
        }

    parsed = parse_loop_args(arg)
    if parsed["error"]:
        if parsed["error"] == "empty":
            return {"output": "Usage: /loop [interval] <prompt> — see /loop help.", "created": False}
        return {"output": f"/loop: {parsed['error']}", "created": False}

    replacing = mgr.has_loop()
    try:
        state = mgr.set(
            parsed["prompt"],
            interval_seconds=parsed["interval_seconds"],
            times=parsed["times"],
            until=parsed["until"],
            route=route,
        )
    except ValueError as exc:
        return {"output": f"/loop: {exc}", "created": False}

    lines = [f"↻ Loop set ({state.cadence_label()}): {state.prompt}"]
    if parsed["interval_seconds"] is not None and parsed["interval_seconds"] < state.interval_seconds:
        lines.append(
            f"(interval raised to the {format_interval(state.interval_seconds)} minimum — "
            "loops.min_interval_seconds)"
        )
    if state.mode == "self_paced":
        lines.append(
            f"Self-paced: first check in {format_interval(state.current_delay)}; "
            f"backs off up to {format_interval(self_paced_ceiling_seconds())} while nothing changes."
        )
    if state.times:
        lines.append(f"Runs {state.times} time{'s' if state.times != 1 else ''}, then stops.")
    if state.until:
        lines.append(f"Stops when: {state.until}")
    if not state.times and state.max_ticks:
        lines.append(f"Backstop budget: {state.max_ticks} ticks (loops.max_ticks; 0 = unlimited).")
    lines.append(f"First wakeup {state.remaining_label()}. Controls: /loop status · pause · resume · stop.")
    if replacing:
        lines.insert(1, "(replaced the previous loop for this session)")
    return {"output": "\n".join(lines), "created": True}


__all__ = [
    "LoopState",
    "LoopManager",
    "parse_loop_args",
    "parse_interval_token",
    "format_interval",
    "response_signals_complete",
    "goal_blocks_loop_tick",
    "load_loop",
    "save_loop",
    "clear_loop",
    "list_active_loops",
    "migrate_loop_to_session",
    "dispatch_loop_command",
    "LOOP_COMPLETE_MARKER",
    "WAKEUP_PROMPT_TEMPLATE",
    "WAKEUP_PROMPT_WITH_UNTIL_TEMPLATE",
    "DEFAULT_MIN_INTERVAL_SECONDS",
    "DEFAULT_MAX_TICKS",
]
