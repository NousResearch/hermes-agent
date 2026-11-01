"""Provider-neutral task timing lifecycle and human-readable ETA status.

The host owns observation and display. Providers may persist only closed-domain
metadata supplied here; raw user text never crosses the timing hooks.
"""
from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, Optional

_SUBJECTS = (
    ("development", ("code", "debug", "test", "deploy", "git", "pull request", "코드", "개발", "버그", "배포", "테스트")),
    ("scheduling", ("calendar", "schedule", "meeting", "일정", "캘린더", "미팅")),
    ("health", ("health", "sleep", "workout", "doctor", "건강", "수면", "운동", "병원")),
    ("research", ("research", "paper", "compare", "investigate", "리서치", "연구", "논문", "조사")),
    ("operations", ("install", "configure", "server", "restart", "monitor", "설치", "설정", "서버", "재시작", "모니터")),
)


def classify_timing_subject(message: Any) -> str:
    """Collapse raw input immediately into a small allowlisted task class."""
    text = str(message or "").lower()
    for subject, terms in _SUBJECTS:
        for term in terms:
            if re.search(r"(?<![\w가-힣]){}(?![\w가-힣])".format(re.escape(term)), text):
                return subject
    return "general"


def _format_duration(milliseconds: int) -> str:
    seconds = max(1, int(math.ceil(milliseconds / 1000)))
    if seconds < 60:
        return "{} sec".format(seconds)
    minutes = int(math.ceil(seconds / 60))
    if minutes < 60:
        return "{} min".format(minutes)
    hours = minutes / 60
    return "{:.1f} hr".format(hours).rstrip("0").rstrip(".")


def _safe_emit(agent: Any, text: str) -> None:
    try:
        agent._emit_status(text)
    except Exception:
        pass


def start_turn_timing(
    agent: Any,
    user_message: Any,
    *,
    conversation_history: Optional[list] = None,
) -> Optional[Dict[str, Any]]:
    manager = getattr(agent, "_memory_manager", None)
    if manager is None:
        return None
    history_turns = sum(
        1 for item in (conversation_history or [])
        if isinstance(item, dict) and item.get("role") == "user"
    )
    turn_number = max(int(getattr(agent, "_user_turn_count", 0) or 0), history_turns) + 1
    subject = classify_timing_subject(user_message)
    platform = str(getattr(agent, "platform", None) or "cli")
    state: Dict[str, Any] = {
        "turn_number": turn_number,
        "subject": subject,
        "platform": platform,
        "started_monotonic": time.monotonic(),
        "estimate": None,
        "last_remaining_announcement": None,
    }
    agent._turn_timing = state
    try:
        manager.on_turn_timing_start(
            turn_number,
            session_id=str(getattr(agent, "session_id", None) or ""),
            platform=platform,
            subject=subject,
        )
        estimate = manager.estimate_turn(platform=platform, subject=subject)
        if isinstance(estimate, dict) and int(estimate.get("sample_count", 0) or 0) >= 5:
            recommended = int(estimate.get("recommended_ms") or estimate.get("p80_ms") or 0)
            if recommended > 0:
                state["estimate"] = dict(estimate, recommended_ms=recommended)
                _safe_emit(agent, "⏱ Estimated total time: about {}".format(_format_duration(recommended)))
    except Exception:
        pass
    return state


def progress_turn_timing(
    agent: Any,
    phase: str,
    *,
    iteration: int = 0,
    announce_remaining: bool = False,
) -> None:
    state = getattr(agent, "_turn_timing", None)
    manager = getattr(agent, "_memory_manager", None)
    if not isinstance(state, dict) or manager is None:
        return
    try:
        manager.on_turn_progress(
            state["turn_number"],
            session_id=str(getattr(agent, "session_id", None) or ""),
            phase=phase,
            iteration=max(0, int(iteration)),
        )
    except Exception:
        pass
    estimate = state.get("estimate")
    if not announce_remaining or not isinstance(estimate, dict):
        return
    elapsed_ms = max(0, int((time.monotonic() - state["started_monotonic"]) * 1000))
    remaining_ms = max(0, int(estimate["recommended_ms"]) - elapsed_ms)
    rendered = _format_duration(remaining_ms)
    if rendered == state.get("last_remaining_announcement"):
        return
    state["last_remaining_announcement"] = rendered
    _safe_emit(agent, "⏱ Estimated time remaining: about {}".format(rendered))


def _outcome(result: Any, escaped: Optional[BaseException], agent: Any) -> str:
    if escaped is not None:
        if isinstance(escaped, (KeyboardInterrupt, SystemExit, InterruptedError)) or type(escaped).__name__ == "CancelledError":
            return "interrupted"
        return "failed"
    if not isinstance(result, dict):
        return "incomplete"
    if result.get("interrupted") or getattr(agent, "_interrupt_requested", False):
        return "interrupted"
    if result.get("failed") or result.get("error"):
        return "failed"
    if result.get("completed") is True:
        return "completed"
    return "incomplete"


def finish_turn_timing(
    agent: Any,
    result: Any = None,
    escaped: Optional[BaseException] = None,
) -> None:
    state = getattr(agent, "_turn_timing", None)
    manager = getattr(agent, "_memory_manager", None)
    try:
        if isinstance(state, dict) and manager is not None:
            manager.on_turn_finish(
                state["turn_number"],
                session_id=str(getattr(agent, "session_id", None) or ""),
                outcome=_outcome(result, escaped, agent),
            )
    except Exception:
        pass
    finally:
        agent._turn_timing = None


__all__ = [
    "classify_timing_subject",
    "finish_turn_timing",
    "progress_turn_timing",
    "start_turn_timing",
]
