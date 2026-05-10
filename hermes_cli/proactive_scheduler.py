"""Proactive Communication Loop scheduler — flow-aware synthesis timing.

This module does two things:

1. FLOW ANALYSIS — studies the user's conversation history to find their
   peak creative window: the time of day when they are most talkative,
   most likely to be in deep work, and most receptive to a surprising insight.

   Three signals combined:
   - Message frequency by hour (when are they most active?)
   - Message depth by hour (long messages = deep work, not quick check-ins)
   - Session continuity (sustained hours, not brief pings)

   The result is a "peak flow hour" (0–23, local time) stored per session
   and refreshed weekly.

2. SCHEDULED SYNTHESIS — fires run_synthesis() once per day at the peak
   flow hour for each active session.

   The loop does NOT send a message every day. run_synthesis() has a high
   bar: BartokGraph connections required, scoring threshold, daily rate limit.
   Most days it will stay silent. But when it does find something worth
   saying, it arrives when the user is already in flow.

Integration (gateway/run.py _start_cron_ticker):

    from hermes_cli.proactive_scheduler import ProactiveScheduler
    _proactive_scheduler = ProactiveScheduler(adapters=adapters, loop=loop)

    if tick_count % PROACTIVE_CHECK_EVERY == 0:
        _proactive_scheduler.tick()

Config keys:
    proactive_communication.enabled           true/false (default: false)
    proactive_communication.peak_flow_hour    int 0-23 (optional override)
    proactive_communication.threshold         conservative/balanced/eager
    proactive_communication.max_per_day       int (default: 1)
    proactive_communication.bartokgraph.*     (see bartokgraph_adapter.py)
    timezone_offset_hours                     float UTC offset (default: 0)
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

_FLOW_ANALYSIS_WINDOW_DAYS = 30
_MIN_MESSAGES_FOR_ANALYSIS = 20
_DEFAULT_PEAK_HOUR = 9          # 9 AM — sensible morning default
_PEAK_HOUR_WINDOW_MINUTES = 15  # ±15 min around peak hour


def _safe_load_config() -> Dict[str, Any]:
    """Load hermes config dict. Returns empty dict on any failure."""
    try:
        from hermes_cli.config import load_config, cfg_get  # noqa: F401
        return load_config()
    except Exception:
        return {}


def _cfg_get(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a dotted key from a config dict. e.g. 'proactive_communication.enabled'."""
    try:
        from hermes_cli.config import cfg_get
        parts = dotted_key.split(".")
        return cfg_get(cfg, *parts, default=default)
    except Exception:
        return default


# ──────────────────────────────────────────────────────────────────────
# Flow analysis
# ──────────────────────────────────────────────────────────────────────

class FlowProfile:
    """The user's peak creative window, derived from message history.

    Attributes:
        peak_hour:   Hour of day (0–23, local time) with highest flow score.
        confidence:  0–1. Low = default used, not enough history.
        scores:      Dict[int, float] — flow score per hour (for inspection).
        analyzed_at: Unix timestamp when this was last computed.
    """

    def __init__(
        self,
        peak_hour: int,
        confidence: float,
        scores: Dict[int, float],
        analyzed_at: float,
    ) -> None:
        self.peak_hour = peak_hour
        self.confidence = confidence
        self.scores = scores
        self.analyzed_at = analyzed_at

    def is_stale(self, max_age_days: int = 7) -> bool:
        return (time.time() - self.analyzed_at) > max_age_days * 86400

    def __repr__(self) -> str:
        age_h = int((time.time() - self.analyzed_at) / 3600)
        return (
            f"FlowProfile(peak_hour={self.peak_hour}, "
            f"confidence={self.confidence:.2f}, age={age_h}h)"
        )


def analyze_flow(messages: List[Dict[str, Any]], tz_offset_hours: float = 0.0) -> FlowProfile:
    """Derive the user's peak creative window from message history.

    Scoring (three signals):
      30% — message frequency per hour
      40% — average message length per hour (depth signal)
      30% — session continuity (hours with adjacent active windows)

    Args:
        messages: List of dicts with at minimum {'role', 'ts', 'content'}.
        tz_offset_hours: User's UTC offset (e.g. -4.0 for EDT).

    Returns:
        FlowProfile with peak_hour in local time.
    """
    user_msgs = [m for m in messages if m.get("role") == "user" and m.get("ts")]

    if len(user_msgs) < _MIN_MESSAGES_FOR_ANALYSIS:
        return FlowProfile(_DEFAULT_PEAK_HOUR, 0.0, {}, time.time())

    tz_offset_secs = tz_offset_hours * 3600

    freq: Dict[int, int] = defaultdict(int)
    depth: Dict[int, List[int]] = defaultdict(list)
    active_hours_by_day: Dict[Tuple[int, int], set] = defaultdict(set)

    for msg in user_msgs:
        try:
            ts = float(msg["ts"])
        except (TypeError, ValueError):
            continue
        local_ts = ts + tz_offset_secs
        dt = datetime.fromtimestamp(local_ts, tz=timezone.utc)
        hour = dt.hour
        day_key = (dt.year, dt.timetuple().tm_yday)
        freq[hour] += 1
        depth[hour].append(len(str(msg.get("content", ""))))
        active_hours_by_day[day_key].add(hour)

    if not freq:
        return FlowProfile(_DEFAULT_PEAK_HOUR, 0.0, {}, time.time())

    # 1. Frequency score
    max_freq = max(freq.values())
    freq_score = {h: v / max_freq for h, v in freq.items()}

    # 2. Depth score
    avg_depth = {h: sum(lens) / len(lens) for h, lens in depth.items()}
    max_depth = max(avg_depth.values()) if avg_depth else 1.0
    depth_score = {h: v / max_depth for h, v in avg_depth.items()}

    # 3. Continuity score
    continuity: Dict[int, float] = defaultdict(float)
    for day_hours in active_hours_by_day.values():
        for hour in day_hours:
            adjacent = sum(1 for adj in [(hour - 1) % 24, (hour + 1) % 24] if adj in day_hours)
            continuity[hour] += adjacent
    max_cont = max(continuity.values()) if continuity else 0.0
    cont_score = {h: v / max_cont for h, v in continuity.items()} if max_cont > 0 else {h: 0.0 for h in continuity}

    # Combined
    all_hours = set(freq_score) | set(depth_score) | set(cont_score)
    combined: Dict[int, float] = {
        h: (
            0.30 * freq_score.get(h, 0.0) +
            0.40 * depth_score.get(h, 0.0) +
            0.30 * cont_score.get(h, 0.0)
        )
        for h in all_hours
    }

    if not combined:
        return FlowProfile(_DEFAULT_PEAK_HOUR, 0.0, {}, time.time())

    peak_hour = max(combined, key=combined.get)
    peak_score = combined[peak_hour]
    mean_score = sum(combined.values()) / len(combined)
    variance = sum((v - mean_score) ** 2 for v in combined.values()) / len(combined)
    std = math.sqrt(variance) if variance > 0 else 0.0

    z_score = (peak_score - mean_score) / std if std > 1e-6 else 0.0
    confidence = min(1.0, max(0.0, z_score / 3.0))

    logger.debug(
        "FlowAnalysis: peak_hour=%d confidence=%.2f (z=%.2f) from %d messages",
        peak_hour, confidence, z_score, len(user_msgs),
    )
    return FlowProfile(peak_hour=peak_hour, confidence=confidence, scores=combined, analyzed_at=time.time())


# ──────────────────────────────────────────────────────────────────────
# Scheduler
# ──────────────────────────────────────────────────────────────────────

class ProactiveScheduler:
    """Manages per-session flow profiles and fires synthesis at the right moment.

    One instance lives in the gateway cron ticker thread. Thread-safe:
    tick() runs in the ticker thread; synthesis is dispatched to the
    gateway event loop via asyncio.run_coroutine_threadsafe.
    """

    def __init__(self, adapters=None, loop=None) -> None:
        self._adapters = adapters
        self._loop = loop
        self._flow_profiles: Dict[str, FlowProfile] = {}
        self._last_synthesis_date: Dict[str, str] = {}

    def tick(self) -> None:
        """Called once per minute from the cron ticker. Never raises."""
        try:
            self._tick_inner()
        except Exception as exc:
            logger.debug("ProactiveScheduler: tick error: %s", exc)

    def _tick_inner(self) -> None:
        cfg = _safe_load_config()
        enabled = _cfg_get(cfg, "proactive_communication.enabled", False)
        if not enabled:
            return

        active_sessions = self._get_active_sessions()
        for session_id in active_sessions:
            try:
                self._maybe_synthesize(session_id, cfg=cfg)
            except Exception as exc:
                logger.debug("ProactiveScheduler: error on %s: %s", session_id, exc)

    def _maybe_synthesize(self, session_id: str, cfg: Optional[Dict] = None) -> None:
        if cfg is None:
            cfg = _safe_load_config()

        profile = self._get_or_compute_profile(session_id, cfg=cfg)
        peak_hour = self._resolve_peak_hour(profile, cfg=cfg)

        now_local = self._local_now(cfg=cfg)
        peak_minute_of_day = peak_hour * 60
        current_minute_of_day = now_local.hour * 60 + now_local.minute
        delta = abs(current_minute_of_day - peak_minute_of_day)
        delta = min(delta, 1440 - delta)  # handle midnight wrap

        if delta > _PEAK_HOUR_WINDOW_MINUTES:
            return

        today_str = now_local.strftime("%Y-%m-%d")
        if self._last_synthesis_date.get(session_id) == today_str:
            return

        self._last_synthesis_date[session_id] = today_str

        logger.info(
            "ProactiveScheduler: firing synthesis for %s at peak hour %d (confidence=%.2f)",
            session_id, peak_hour, profile.confidence,
        )
        self._fire_synthesis(session_id)

    def _fire_synthesis(self, session_id: str) -> None:
        if self._loop is None or self._adapters is None:
            logger.debug("ProactiveScheduler: no loop/adapters — skipping")
            return

        async def _run() -> None:
            try:
                from hermes_state import SessionDB
                from hermes_cli.config import load_config
                from hermes_cli.proactive_communication_loop import ProactiveCommunicationLoop

                class _CfgWrapper:
                    """Wrap config dict so .get(dotted_key, default) works."""
                    def __init__(self, cfg: dict) -> None:
                        self._cfg = cfg

                    def get(self, dotted_key: str, default: Any = None) -> Any:
                        return _cfg_get(self._cfg, dotted_key, default)

                db = SessionDB()
                cfg_wrapper = _CfgWrapper(load_config())

                loop_obj = ProactiveCommunicationLoop(session_db=db, config=cfg_wrapper)
                result = await loop_obj.run_synthesis(session_id)

                if result.should_send and result.message:
                    await self._deliver(session_id, result.message)
                    await loop_obj.record_sent(session_id, result)
                    logger.info(
                        "ProactiveScheduler: delivered message for %s (type=%s score=%.2f)",
                        session_id, result.connection_type, result.combined_score,
                    )
                else:
                    logger.debug(
                        "ProactiveScheduler: silent for %s — %s",
                        session_id, result.reasoning,
                    )
            except Exception as exc:
                logger.warning("ProactiveScheduler: synthesis failed for %s: %s", session_id, exc)

        future = asyncio.run_coroutine_threadsafe(_run(), self._loop)
        future.add_done_callback(
            lambda f: logger.debug(
                "ProactiveScheduler: future done: %s",
                f.exception() or "ok",
            )
        )

    async def _deliver(self, session_id: str, message: str) -> None:
        """Deliver message to the session's origin channel.

        Uses the same delivery path as cron jobs: looks up the session's
        origin (the platform/chat_id where it started) and delivers via
        cron.scheduler._deliver_result with a synthetic job dict.
        """
        try:
            from hermes_state import SessionDB
            db = SessionDB()
            session = db.get_session(session_id)
            if not session:
                logger.debug("ProactiveScheduler: session not found: %s", session_id)
                return

            # Build a minimal synthetic job dict that _deliver_result can route
            origin = session.get("origin") or session.get("source")
            if not origin:
                logger.debug("ProactiveScheduler: no origin for session %s", session_id)
                return

            # Parse origin — stored as 'platform:chat_id' or as a dict
            if isinstance(origin, str) and ":" in origin:
                parts = origin.split(":", 1)
                job = {
                    "id": f"proactive:{session_id[:8]}",
                    "name": "Proactive Communication",
                    "origin": {"platform": parts[0], "chat_id": parts[1]},
                    "deliver": "origin",
                    "wrap_response": False,  # no cron header — deliver message as-is
                }
            elif isinstance(origin, dict):
                job = {
                    "id": f"proactive:{session_id[:8]}",
                    "name": "Proactive Communication",
                    "origin": origin,
                    "deliver": "origin",
                    "wrap_response": False,
                }
            else:
                logger.debug("ProactiveScheduler: unrecognised origin format for %s: %r", session_id, origin)
                return

            from cron.scheduler import _deliver_result
            err = _deliver_result(job, message, adapters=self._adapters, loop=self._loop)
            if err:
                logger.warning("ProactiveScheduler: delivery error for %s: %s", session_id, err)

        except Exception as exc:
            logger.debug("ProactiveScheduler: delivery failed for %s: %s", session_id, exc)

    def _get_or_compute_profile(self, session_id: str, cfg: Optional[Dict] = None) -> FlowProfile:
        existing = self._flow_profiles.get(session_id)
        if existing and not existing.is_stale(max_age_days=7):
            return existing

        try:
            from hermes_state import SessionDB
            db = SessionDB()
            # get_messages returns all messages; filter by timestamp in analyze_flow
            cutoff = time.time() - _FLOW_ANALYSIS_WINDOW_DAYS * 86400
            all_messages = db.get_messages(session_id)
            # Normalize: state_db uses 'timestamp' key, flow analysis uses 'ts'
            messages = [
                {**m, "ts": float(m.get("timestamp") or 0)}
                for m in all_messages
                if float(m.get("timestamp") or 0) >= cutoff
            ]
            tz_offset = float(_cfg_get(cfg or {}, "timezone_offset_hours", 0.0))
            profile = analyze_flow(messages, tz_offset_hours=tz_offset)
            self._flow_profiles[session_id] = profile
            return profile
        except Exception as exc:
            logger.debug("ProactiveScheduler: flow analysis failed for %s: %s", session_id, exc)
            return FlowProfile(_DEFAULT_PEAK_HOUR, 0.0, {}, time.time())

    def _resolve_peak_hour(self, profile: FlowProfile, cfg: Optional[Dict] = None) -> int:
        override = _cfg_get(cfg or {}, "proactive_communication.peak_flow_hour", None)
        if override is not None:
            try:
                return int(override)
            except (TypeError, ValueError):
                pass
        return profile.peak_hour

    def _local_now(self, cfg: Optional[Dict] = None) -> datetime:
        tz_offset = float(_cfg_get(cfg or {}, "timezone_offset_hours", 0.0))
        local_ts = time.time() + tz_offset * 3600
        return datetime.fromtimestamp(local_ts, tz=timezone.utc)

    def _get_active_sessions(self) -> List[str]:
        """Get session IDs with recent message activity (last 7 days)."""
        try:
            from hermes_state import SessionDB
            db = SessionDB()
            cutoff_ts = time.time() - 7 * 24 * 3600
            # list_sessions_rich with order_by_last_active returns last_active timestamp
            sessions = db.list_sessions_rich(
                order_by_last_active=True,
                limit=50,
                include_children=False,
            )
            return [
                s["id"] for s in sessions
                if float(s.get("last_active") or 0) >= cutoff_ts
            ]
        except Exception as exc:
            logger.debug("ProactiveScheduler: session list failed: %s", exc)
            return []
