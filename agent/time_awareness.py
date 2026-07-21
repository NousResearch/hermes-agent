"""
Time Awareness — 时间感知中枢神经模块

Provides persistent time state for Hermes Agent so the agent can perceive:
1. How long it has been dormant between sessions (休眠时长)
2. Current wall-clock time at each API call (当前时间)
3. Interval since last API call — the heartbeat rhythm (心跳节律)

State file: ~/.hermes/time_state.json

Three layers:
  Layer 1 (on_session_start) — injected into system prompt volatile_parts.
      Fires once per session build. Agent learns "how long I slept."
      Does NOT break prompt cache — computed at build time, cached with it.

  Layer 2 (on_api_call) — injected into api_messages before each API call.
      Fires at every turn. Agent learns "what time it is now" and
      "how long since I last spoke."
      Does NOT touch system prompt — inserted into message list.

  Layer 3 (on_session_end) — writes sleep timestamp on session close.
      Enables Layer 1 to compute dormancy on next wake.

Design principle: the agent's existence is continuous. Sleep is not death —
it is dormancy. The time_state.json file is the persistent thread of
continuity, the "central nervous system" that bridges dormant periods.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from hermes_constants import get_hermes_home

_TIME_STATE_FILE: Optional[str] = None


def _get_time_state_path() -> str:
    global _TIME_STATE_FILE
    if _TIME_STATE_FILE is None:
        _TIME_STATE_FILE = os.path.join(get_hermes_home(), "time_state.json")
    return _TIME_STATE_FILE


def _now_iso() -> str:
    """Current time as human-readable string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


def _now_ts() -> float:
    """Current time as Unix timestamp."""
    return time.time()


def _read_state() -> Dict:
    """Read time state. Returns empty dict if not found or corrupt."""
    path = _get_time_state_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_state(state: Dict) -> None:
    """Write time state atomically."""
    path = _get_time_state_path()
    state["updated_at"] = _now_iso()
    state["updated_ts"] = _now_ts()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _format_duration(seconds: float) -> str:
    """Format seconds into Chinese human-readable duration."""
    if seconds < 0:
        return "未知"
    if seconds < 60:
        return f"{int(seconds)}秒"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}分{s}秒" if s else f"{m}分钟"
    if seconds < 86400:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}小时{m}分钟" if m else f"{h}小时"
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    return f"{d}天{h}小时" if h else f"{d}天"


# ── Layer 1: Wake-up (session start) ─────────────────────────────────

def on_session_start() -> Dict:
    """
    Called when a new session is built.
    Records wake time, computes dormancy duration, returns context for
    injection into system prompt volatile_parts.

    Returns dict with keys:
      last_sleep_at: str  — when the agent last went to sleep
      sleep_duration: str — how long the dormancy lasted
      current_time: str   — current wall-clock time
    """
    state = _read_state()
    last_sleep_at = state.get("last_sleep_at")
    last_sleep_ts = state.get("last_sleep_ts")

    now = _now_iso()
    now_ts = _now_ts()

    sleep_duration = None
    if last_sleep_ts:
        delta = now_ts - last_sleep_ts
        sleep_duration = _format_duration(delta)

    # Update state
    state["last_wake_at"] = now
    state["last_wake_ts"] = now_ts
    state["sleep_duration"] = sleep_duration
    _write_state(state)

    return {
        "last_sleep_at": last_sleep_at,
        "sleep_duration": sleep_duration,
        "current_time": now,
    }


# ── Layer 2: Heartbeat (per API call) ────────────────────────────────

def on_api_call(min_interval_secs: int = 300) -> Optional[str]:
    """
    Called before each API call (inside build_api_kwargs).
    Updates heartbeat timestamp, returns a compact time string for
    injection into the messages list.

    Throttling: if less than ``min_interval_secs`` (default 300s = 5min)
    since the last heartbeat, returns None — no message injected.
    This prevents token waste in rapid multi-turn exchanges while
    keeping time awareness alive for gaps where the user steps away.

    Returns: compact string or None (skip injection).
    """
    state = _read_state()
    now = _now_iso()
    now_ts = _now_ts()

    last_hb_ts = state.get("last_heartbeat_ts")

    # Throttle check: skip if too soon
    if last_hb_ts and (now_ts - last_hb_ts) < min_interval_secs:
        # Still update the state so the timer keeps moving,
        # but signal the caller to NOT inject.
        state["last_heartbeat_at"] = now
        state["last_heartbeat_ts"] = now_ts
        _write_state(state)
        return None

    last_hb_at = state.get("last_heartbeat_at")
    last_wake = state.get("last_wake_at")

    interval = None
    if last_hb_ts:
        delta = now_ts - last_hb_ts
        interval = _format_duration(delta)

    # Update heartbeat
    state["last_heartbeat_at"] = now
    state["last_heartbeat_ts"] = now_ts
    _write_state(state)

    # Build compact string
    parts = [f"当前: {now}"]
    if last_wake:
        parts.append(f"会话开始: {last_wake}")
    if interval:
        parts.append(f"距上次心跳: {interval}")

    return " | ".join(parts)


# ── Layer 3: Sleep (session end) ─────────────────────────────────────

def on_session_end() -> None:
    """
    Called when a session ends (via /reset, /new, gateway session expiry,
    or agent close). Records the sleep timestamp so the next session can
    compute dormancy duration.
    """
    state = _read_state()
    state["last_sleep_at"] = _now_iso()
    state["last_sleep_ts"] = _now_ts()
    _write_state(state)