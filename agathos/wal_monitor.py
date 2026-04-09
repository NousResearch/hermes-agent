#!/usr/bin/env python3
"""
ToolCallMonitor — Real-time entropy detection via state.db polling.

Follows the same pattern as mcp_serve.EventBridge:
  - mtime gate on state.db (skip work when unchanged)
  - Per-session timestamp tracking (know what's new)
  - Background daemon thread (non-blocking)
  - In-memory queue with cursor (structured events)

Consumed by Agathos for real-time entropy detection.

Event types produced:
- tool_call: New tool execution detected
- repeat_detected: Same tool called repeatedly
- stuck_loop_detected: Repeating pattern detected
"""

import os
import sys
import json
import time
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

# === PATH RESOLUTION ===
_HERMES_AGENT = os.path.expanduser("~/.hermes/hermes-agent")
if os.path.isdir(_HERMES_AGENT) and _HERMES_AGENT not in sys.path:
    sys.path.insert(0, _HERMES_AGENT)

_HERMES_INTERNALS_AVAILABLE = False
SessionDB = None  # Will be set if import succeeds
try:
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB, DEFAULT_DB_PATH

    _HERMES_INTERNALS_AVAILABLE = True
    _HERMES_HOME = get_hermes_home()
except (ImportError, TypeError) as _e:
    DEFAULT_DB_PATH = str(Path.home() / ".hermes" / "state.db")
    _HERMES_HOME = Path.home() / ".hermes"

logger = logging.getLogger("agathos.wal")


@dataclass
class ToolCallEvent:
    """A detected tool call or entropy pattern from WAL monitoring.

    Represents a single event detected by ToolCallMonitor during state.db
    polling. Events are queued in-memory and consumed by Agathos daemon.

    Attributes:
        cursor: Event sequence number (for cursor-based consumption)
        session_id: Hermes session ID where event occurred
        event_type: Type of event detected:
            - "tool_call": New tool execution
            - "repeat_detected": Repeated tool calls (entropy)
            - "stuck_loop_detected": Repeating pattern (entropy)
        tool_name: Name of the tool called (e.g., "read_file")
        tool_args: JSON string of tool arguments
        timestamp: Unix timestamp when event was detected
        details: Dict with additional context (e.g., repeat count)
    """

    cursor: int
    session_id: str
    event_type: str  # "tool_call", "repeat_detected", "stuck_loop_detected"
    tool_name: str = ""
    tool_args: str = ""
    timestamp: float = 0.0
    details: dict = field(default_factory=dict)


class ToolCallMonitor:
    """Polls state.db for new tool calls, detects entropy patterns in real-time.

    Same architecture as mcp_serve.EventBridge:
      - mtime gate (~1μs check, skip when DB unchanged)
      - Per-session timestamp tracking
      - Background daemon thread at configurable interval
      - Thread-safe event queue with cursor

    The monitor runs continuously in the background, populating an in-memory
    event queue. The Agathos daemon consumes events on each poll cycle.

    Usage:
        monitor = ToolCallMonitor()
        monitor.start()

        # On each Agathos poll cycle:
        events = monitor.get_events()
        for e in events:
            if e.event_type == 'repeat_detected':
                # Trigger Agathos action

        monitor.stop()

    Attributes:
        db_path: Path to state.db being monitored
        poll_interval: Seconds between polls (default: 5.0)
        _last_mtime: Last state.db mtime (for mtime gate optimization)
        _session_positions: Dict mapping session_id -> last processed timestamp
        _events: Thread-safe deque of ToolCallEvent objects
        _cursor: Monotonic event counter
        _running: Boolean indicating if monitor thread is active
        _thread: Background daemon thread handle
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        poll_interval: float = 2.0,
        repeat_threshold: int = 3,
        loop_pattern_length: int = 3,
    ):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.poll_interval = poll_interval
        self.repeat_threshold = repeat_threshold
        self.loop_pattern_length = loop_pattern_length

        # State
        self._queue: List[ToolCallEvent] = []
        self._cursor = 0
        self._lock = threading.Lock()
        self._new_event = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # mtime gate — skip work when state.db hasn't changed
        self._state_db_mtime: float = 0.0

        # Per-session tracking
        self._last_poll_timestamps: Dict[str, float] = {}
        # Per-session recent tool call history (for pattern detection)
        self._recent_tool_calls: Dict[str, List[str]] = {}  # session_id -> [tool_names]

    def start(self):
        """Start the background polling thread."""
        if self._running:
            return
        if not _HERMES_INTERNALS_AVAILABLE:
            logger.warning(
                "Hermes internals unavailable — ToolCallMonitor cannot start"
            )
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("ToolCallMonitor started (poll_interval=%.1fs)", self.poll_interval)

    def stop(self):
        """Stop the background polling thread."""
        self._running = False
        self._new_event.set()  # Wake any waiters
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("ToolCallMonitor stopped")

    def get_events(self, limit: int = 50) -> List[ToolCallEvent]:
        """Drain the event queue. Called by ARGUS on each poll cycle."""
        with self._lock:
            events = self._queue[:limit]
            self._queue = self._queue[limit:]
        return events

    def get_session_tool_history(self, session_id: str) -> List[str]:
        """Get recent tool call history for a session (for entropy analysis)."""
        with self._lock:
            return list(self._recent_tool_calls.get(session_id, []))

    def get_entropy_summary(self) -> Dict:
        """Get entropy summary across all monitored sessions."""
        with self._lock:
            summary = {}
            for session_id, history in self._recent_tool_calls.items():
                if len(history) < 3:
                    continue
                counts = Counter(history)
                repeats = {
                    t: c for t, c in counts.items() if c >= self.repeat_threshold
                }
                if repeats:
                    summary[session_id] = {
                        "total_calls": len(history),
                        "repeat_tools": repeats,
                        "last_5": history[-5:],
                    }
            return summary

    def _poll_loop(self):
        """Background loop: poll state.db for new tool calls."""
        db = None
        consecutive_errors = 0

        while self._running:
            try:
                # Reconnect if needed
                if db is None:
                    db = SessionDB(DEFAULT_DB_PATH)
                    consecutive_errors = 0

                self._poll_once(db)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.warning(
                    "ToolCallMonitor poll error (%d): %s", consecutive_errors, e
                )
                # Close and reconnect on repeated errors
                if db is not None:
                    try:
                        db.close()
                    except Exception:
                        pass
                    db = None
                # Back off on persistent errors
                if consecutive_errors > 3:
                    time.sleep(self.poll_interval * 3)

            time.sleep(self.poll_interval)

        if db is not None:
            try:
                db.close()
            except Exception:
                pass

    def _poll_once(self, db):
        """Check for new tool calls across all sessions.

        Uses mtime check on state.db to skip work when nothing changed
        — makes polling essentially free when DB is idle.
        """
        # mtime gate (~1μs) — skip if state.db hasn't been written to
        db_file = Path(self.db_path)
        try:
            db_mtime = db_file.stat().st_mtime if db_file.exists() else 0.0
        except OSError:
            db_mtime = 0.0

        if db_mtime == self._state_db_mtime:
            return  # Nothing changed — skip entirely

        self._state_db_mtime = db_mtime

        # Get all sessions with recent activity
        try:
            sessions = db.list_sessions_rich(limit=50)
        except Exception:
            return

        for session in sessions:
            if session is None:
                continue
            session_id = session.get("id", "")
            if not session_id:
                continue

            self._process_session(db, session_id)

    def _process_session(self, db, session_id: str):
        """Check a single session for new tool calls."""
        last_seen = self._last_poll_timestamps.get(session_id, 0.0)

        try:
            messages = db.get_messages(session_id)
        except Exception:
            return

        if not messages:
            return

        # Find assistant messages with tool_calls newer than last_seen
        new_tool_calls = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            ts = self._ts_float(msg.get("timestamp", 0))
            if ts <= last_seen:
                continue

            tool_calls_raw = msg.get("tool_calls")
            if not tool_calls_raw:
                continue

            # Parse tool_calls JSON
            try:
                if isinstance(tool_calls_raw, str):
                    tool_calls_raw = json.loads(tool_calls_raw)
                for tc in tool_calls_raw:
                    name = tc.get("function", {}).get("name") or tc.get("name", "")
                    if name:
                        new_tool_calls.append((name, ts))
            except (json.JSONDecodeError, TypeError):
                continue

        if not new_tool_calls:
            # Still update timestamp even if no tool calls
            all_ts = [self._ts_float(m.get("timestamp", 0)) for m in messages]
            if all_ts:
                latest = max(all_ts)
                if latest > last_seen:
                    self._last_poll_timestamps[session_id] = latest
            return

        # Process new tool calls
        with self._lock:
            history = self._recent_tool_calls.setdefault(session_id, [])

            for tool_name, ts in new_tool_calls:
                history.append(tool_name)

                # Emit tool_call event
                self._enqueue(
                    ToolCallEvent(
                        cursor=0,
                        session_id=session_id,
                        event_type="tool_call",
                        tool_name=tool_name,
                        timestamp=ts,
                    )
                )

            # Check for repeat patterns (3+ same tool consecutively)
            if len(history) >= self.repeat_threshold:
                recent = history[-self.repeat_threshold :]
                if len(set(recent)) == 1:  # All the same
                    self._enqueue(
                        ToolCallEvent(
                            cursor=0,
                            session_id=session_id,
                            event_type="repeat_detected",
                            tool_name=recent[0],
                            details={"count": len(recent), "consecutive": True},
                        )
                    )
                    logger.warning(
                        "Repeat detected: %s called %d+ times in session %s",
                        recent[0],
                        len(recent),
                        session_id[:15],
                    )

            # Check for stuck loop patterns (A,B,C,A,B,C)
            if len(history) >= self.loop_pattern_length * 2:
                pattern = history[
                    -self.loop_pattern_length * 2 : -self.loop_pattern_length
                ]
                next_pattern = history[-self.loop_pattern_length :]
                if pattern == next_pattern:
                    self._enqueue(
                        ToolCallEvent(
                            cursor=0,
                            session_id=session_id,
                            event_type="stuck_loop_detected",
                            details={
                                "pattern": pattern,
                                "pattern_length": self.loop_pattern_length,
                            },
                        )
                    )
                    logger.warning(
                        "Stuck loop detected in session %s: %s repeating",
                        session_id[:15],
                        pattern,
                    )

            # Trim history to last 50 calls
            if len(history) > 50:
                self._recent_tool_calls[session_id] = history[-50:]

        # Evict stale sessions (no activity in 30 minutes)
        now = time.time()
        stale_cutoff = now - 1800  # 30 minutes
        stale_sessions = [
            sid for sid, ts in self._last_poll_timestamps.items() if ts < stale_cutoff
        ]
        for sid in stale_sessions:
            self._last_poll_timestamps.pop(sid, None)
            self._recent_tool_calls.pop(sid, None)
        if stale_sessions:
            logger.debug(
                "Evicted %d stale sessions from WAL monitor", len(stale_sessions)
            )

        # Update last seen timestamp
        all_ts = [self._ts_float(m.get("timestamp", 0)) for m in messages]
        if all_ts:
            latest = max(all_ts)
            if latest > last_seen:
                self._last_poll_timestamps[session_id] = latest

    def _enqueue(self, event: ToolCallEvent):
        """Add an event to the queue."""
        self._cursor += 1
        event.cursor = self._cursor
        self._queue.append(event)
        # Trim queue
        while len(self._queue) > 1000:
            self._queue.pop(0)
        self._new_event.set()

    @staticmethod
    def _ts_float(ts) -> float:
        """Normalize timestamp to float."""
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str) and ts:
            try:
                return float(ts)
            except ValueError:
                try:
                    from datetime import datetime

                    return datetime.fromisoformat(ts).timestamp()
                except Exception:
                    return 0.0
        return 0.0


# === Convenience: quick entropy check without the daemon ===


def check_session_entropy(
    session_id: str,
    db_path: Optional[str] = None,
    repeat_threshold: int = 3,
) -> Dict:
    """One-shot entropy check on a session (no daemon needed).

    Standalone function for ad-hoc entropy analysis without running the
    full ToolCallMonitor daemon. Queries state.db directly, analyzes
    message history for tool patterns.

    Args:
        session_id: Hermes session to analyze
        db_path: Path to state.db (defaults to ~/.hermes/state.db)
        repeat_threshold: Minimum repeats to flag as entropy (default: 3)

    Returns:
        Dict with keys:
        - session_id: Session analyzed
        - tool_calls: Total tool call count
        - entropy: "none", "repeat_detected", or "stuck_loop_detected"
        - counts: Dict mapping tool_name -> call count
        - repeated_tools: List of tools exceeding repeat_threshold
        - loop_detected: Boolean if stuck loop pattern found

    Returns {"error": "hermes internals unavailable"} if SessionDB cannot be imported.

    Side effects:
        - Opens/closes SessionDB connection
        - Queries all messages for session from state.db
    """
    if not _HERMES_INTERNALS_AVAILABLE:
        return {"error": "hermes internals unavailable"}

    db = SessionDB(db_path or DEFAULT_DB_PATH)
    messages = db.get_messages(session_id)
    db.close()

    # Extract all tool calls
    tool_names = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tc_raw = msg.get("tool_calls")
        if not tc_raw:
            continue
        try:
            if isinstance(tc_raw, str):
                tc_raw = json.loads(tc_raw)
            for tc in tc_raw:
                name = tc.get("function", {}).get("name") or tc.get("name", "")
                if name:
                    tool_names.append(name)
        except (json.JSONDecodeError, TypeError):
            continue

    if not tool_names:
        return {"session_id": session_id, "tool_calls": 0, "entropy": "none"}

    # Analyze
    counts = Counter(tool_names)
    repeats = {t: c for t, c in counts.items() if c >= repeat_threshold}

    # Check consecutive repeats
    consecutive_repeats = 0
    for i in range(len(tool_names) - (repeat_threshold - 1)):
        window = tool_names[i : i + repeat_threshold]
        if len(set(window)) == 1:
            consecutive_repeats += 1

    # Check stuck loops
    stuck_loops = 0
    for pattern_len in range(2, 4):
        if len(tool_names) >= pattern_len * 2:
            for i in range(len(tool_names) - pattern_len * 2 + 1):
                p1 = tool_names[i : i + pattern_len]
                p2 = tool_names[i + pattern_len : i + pattern_len * 2]
                if p1 == p2:
                    stuck_loops += 1

    return {
        "session_id": session_id,
        "tool_calls": len(tool_names),
        "unique_tools": len(counts),
        "top_tools": counts.most_common(5),
        "repeat_tools": repeats,
        "consecutive_repeats": consecutive_repeats,
        "stuck_loops": stuck_loops,
        "entropy_level": "critical"
        if stuck_loops > 0 or consecutive_repeats > 2
        else "warning"
        if consecutive_repeats > 0
        else "none",
    }
