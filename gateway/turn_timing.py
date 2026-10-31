"""Content-free timing receipt at response readiness, before delivery."""
import time
from dataclasses import dataclass, field
from datetime import datetime

from agent.turn_timing import valid_seconds


@dataclass
class TurnTiming:
    handler_start: float | None = None
    hygiene_s: float | None = None
    agent_start: float | None = None
    agent_end: float | None = None
    agent_fields: dict = field(default_factory=dict)
    finished: float | None = None
    since_msg_s: float | None = None


def format_turn_timing_line(timing, *, platform, session_key, outcome, api_calls) -> str:
    """Format frozen measurements; absent durations never become zeroes."""
    fields = {}
    if timing.finished is not None and timing.handler_start is not None:
        fields["handler"] = timing.finished - timing.handler_start
    if timing.agent_start is not None and timing.handler_start is not None and timing.hygiene_s is not None:
        fields["setup"] = max(0.0, timing.agent_start - timing.handler_start - timing.hygiene_s)
    fields["hygiene"] = timing.hygiene_s
    if timing.agent_end is not None and timing.agent_start is not None:
        fields["agent"] = timing.agent_end - timing.agent_start
    for name in ("tools", "compaction", "model", "ttfb"):
        fields[name] = timing.agent_fields.get(name + "_s")
    fields["since_msg"] = timing.since_msg_s
    parts = [f"turn timing: platform={platform} session={session_key} outcome={outcome}"]
    parts.extend(f"{key}={value:.1f}s" for key, value in fields.items() if valid_seconds(value))
    if api_calls is not None:
        parts.append(f"api_calls={api_calls}")
    return " ".join(parts)


def since_message_seconds(timestamp, now):
    """Wall-clock age only; provider sentinels, clock skew and stale replay are unknown."""
    if not isinstance(timestamp, datetime) or (timestamp.utcoffset() is None) != (now.utcoffset() is None):
        return None
    seconds = (now - timestamp).total_seconds()
    return seconds if 0 <= seconds <= 86400 else None


def start_handler_timing(event) -> None:
    event._turn_timing = TurnTiming(handler_start=time.monotonic())


def finish_agent_timing(event, result, ended) -> None:
    timing = getattr(event, "_turn_timing", None)
    if timing is not None:
        timing.agent_end = ended
        timing.agent_fields = result.get("turn_timing") or {}


async def timed_hygiene(event, call, *args):
    started = time.monotonic()
    try:
        return await call(*args)
    finally:
        timing = getattr(event, "_turn_timing", None)
        if timing is not None:
            timing.hygiene_s = time.monotonic() - started


def log_response_timing(timing, timestamp, result, platform, session_key, logger):
    if timing is None:
        return
    timing.finished = time.monotonic()
    now = datetime.now(tz=timestamp.tzinfo if isinstance(timestamp, datetime) else None)
    timing.since_msg_s = since_message_seconds(timestamp, now)
    outcome = next((name for name in ("failed", "interrupted", "partial") if result.get(name)), "ok")
    logger.info(format_turn_timing_line(timing, platform=platform, session_key=session_key,
                                       outcome=outcome, api_calls=result.get("api_calls")))
