"""Voice benchmark telemetry for Hermes gateway voice turns."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any


DEFAULT_LIMIT = 5
MAX_EVENTS = 500
TEXT_PREVIEW_LIMIT = 160
_LOGGER = logging.getLogger(__name__)
_LAST_READ_ERROR: str | None = None
_LAST_WRITE_ERROR: str | None = None

_SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization)\s*[:=]\s*"
        r"([^\s,;]{4,})"
    ),
    re.compile(r"\b(?:sk|ghp|gho|ghu|ghs|glpat|xox[abprs])-[-A-Za-z0-9_]{8,}\b"),
    re.compile(r"(?i)\bbearer\s+[-A-Za-z0-9._~+/=]{8,}\b"),
)


def new_turn_id() -> str:
    return f"voice-{int(time.time())}-{uuid.uuid4().hex[:8]}"


def bench_path() -> Path:
    raw = os.environ.get("HERMES_VOICE_BENCH_PATH")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".hermes" / "voice_bench.jsonl"


def _redact_text(value: Any, *, limit: int = TEXT_PREVIEW_LIMIT) -> str:
    text = str(value or "").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            text = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _safe_event(event: dict[str, Any]) -> dict[str, Any]:
    safe = dict(event)
    for raw_key, preview_key in (
        ("transcript", "transcript_preview"),
        ("response", "response_preview"),
    ):
        if raw_key in safe:
            raw_value = safe.pop(raw_key)
            safe[f"{raw_key}_chars"] = len(str(raw_value or ""))
            preview = _redact_text(raw_value)
            if preview:
                safe[preview_key] = preview
    if "error" in safe:
        safe["error"] = _redact_text(safe.get("error"), limit=240)
    return safe


def append_event(event: dict[str, Any]) -> None:
    global _LAST_WRITE_ERROR
    payload = {
        "ts": time.time(),
        **_safe_event(event),
    }
    path = bench_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        _LAST_WRITE_ERROR = None
    except OSError as exc:
        _LAST_WRITE_ERROR = str(exc)
        _LOGGER.warning("Voice bench telemetry write failed: %s", exc)
        return


def recent_events(*, platform: str | None = None, chat_id: str | None = None, max_events: int = MAX_EVENTS) -> list[dict[str, Any]]:
    global _LAST_READ_ERROR
    path = bench_path()
    if not path.exists():
        _LAST_READ_ERROR = None
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_events:]
        _LAST_READ_ERROR = None
    except OSError as exc:
        _LAST_READ_ERROR = str(exc)
        _LOGGER.warning("Voice bench telemetry read failed: %s", exc)
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        if platform and str(item.get("platform") or "") != platform:
            continue
        if chat_id and str(item.get("chat_id") or "") != str(chat_id):
            continue
        events.append(item)
    return events


def grouped_turns(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for item in events:
        turn_id = str(item.get("turn_id") or "").strip()
        if not turn_id:
            continue
        turn = turns.setdefault(
            turn_id,
            {
                "turn_id": turn_id,
                "ts": item.get("ts"),
                "platform": item.get("platform"),
                "chat_id": item.get("chat_id"),
                "message_id": item.get("message_id"),
                "stages": {},
            },
        )
        turn["ts"] = item.get("ts") or turn.get("ts")
        turn["platform"] = item.get("platform") or turn.get("platform")
        turn["chat_id"] = item.get("chat_id") or turn.get("chat_id")
        turn["message_id"] = item.get("message_id") or turn.get("message_id")
        stage = str(item.get("stage") or "").strip()
        if stage:
            turn["stages"][stage] = item
    return list(turns.values())


def _ms(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.0f}ms"
    return "?"


def format_recent(platform: str | None = None, chat_id: str | None = None, *, limit: int = DEFAULT_LIMIT) -> str:
    events = recent_events(platform=platform, chat_id=chat_id)
    if _LAST_READ_ERROR:
        return "Voice bench unavailable: telemetry read failed."
    turns = grouped_turns(events)[-max(1, min(limit, 20)):]
    if not turns:
        return "Voice bench: no measured voice turns yet."

    lines = ["Voice bench recent turns:"]
    for turn in reversed(turns):
        stages = turn.get("stages") or {}
        stt = stages.get("stt") or {}
        agent = stages.get("agent") or {}
        tts = stages.get("tts") or {}
        delivery = stages.get("delivery") or {}
        stage_values = [
            stage.get("elapsed_ms")
            for stage in (stt, agent, tts, delivery)
            if isinstance(stage.get("elapsed_ms"), (int, float))
        ]
        total_ms = sum(stage_values) if stage_values else None
        status = "ok" if not any((s.get("error") for s in stages.values() if isinstance(s, dict))) else "warn"
        lines.append(
            f"- {status} total={_ms(total_ms)} "
            f"stt={_ms(stt.get('elapsed_ms'))} "
            f"agent={_ms(agent.get('elapsed_ms'))} "
            f"tts={_ms(tts.get('elapsed_ms'))} "
            f"send={_ms(delivery.get('elapsed_ms'))}"
        )
        transcript = str(stt.get("transcript_preview") or stt.get("transcript") or "").strip()
        if transcript:
            lines.append(f"  heard: {transcript[:120]}")
        response = str(agent.get("response_preview") or agent.get("response") or "").strip()
        if response:
            lines.append(f"  reply: {response[:120]}")
    return "\n".join(lines)
