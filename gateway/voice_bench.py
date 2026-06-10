"""Voice benchmark telemetry for Hermes gateway voice turns."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import stat
import atexit
import threading
import time
import uuid
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any


DEFAULT_LIMIT = 5
MAX_EVENTS = 500
TEXT_PREVIEW_LIMIT = 160
MAX_FILE_BYTES = 1_000_000
_LOGGER = logging.getLogger(__name__)
_LAST_READ_ERROR: str | None = None
_LAST_WRITE_ERROR: str | None = None
_WRITE_QUEUE: queue.Queue[tuple[Path, dict[str, Any]] | None] = queue.Queue(maxsize=1000)
_WRITE_WORKER_STARTED = False
_WRITE_WORKER_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False

_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[-A-Za-z0-9._~+/=]{8,}\b"),
    re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization)\s*[:=]\s*"
        r"([^\s,;]{4,})"
    ),
    re.compile(r"\b(?:sk|glpat|xox[abprs])-[-A-Za-z0-9_]{8,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{8,}\b"),
)
_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|passwd|authorization|auth|credential)"
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
    for key, value in list(safe.items()):
        safe[key] = _safe_value(key, value)
    return safe


def _safe_value(key: str, value: Any) -> Any:
    if _SECRET_KEY_RE.search(str(key)):
        return "[REDACTED]"
    if isinstance(value, str):
        return _redact_text(value, limit=240)
    if isinstance(value, dict):
        return {str(k): _safe_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_value(key, item) for item in value]
    return value


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    global _LAST_WRITE_ERROR
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            )
        _LAST_WRITE_ERROR = None
        _compact_if_needed(path)
    except Exception as exc:
        _LAST_WRITE_ERROR = str(exc)
        _LOGGER.warning("Voice bench telemetry write failed: %s", exc)
        return


def _compact_if_needed(path: Path, *, max_events: int | None = None) -> None:
    try:
        keep_events = MAX_EVENTS if max_events is None else max_events
        if path.stat().st_size <= MAX_FILE_BYTES:
            return
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = list(deque(fh, maxlen=keep_events))
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
        tmp_path.replace(path)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as exc:
        _LOGGER.warning("Voice bench telemetry compaction failed: %s", exc)


def _writer_loop() -> None:
    while True:
        item = _WRITE_QUEUE.get()
        try:
            if item is None:
                return
            path, payload = item
            try:
                _write_payload(path, payload)
            except Exception as exc:
                global _LAST_WRITE_ERROR
                _LAST_WRITE_ERROR = str(exc)
                _LOGGER.warning("Voice bench telemetry worker failed: %s", exc)
        finally:
            _WRITE_QUEUE.task_done()


def _ensure_write_worker() -> None:
    global _ATEXIT_REGISTERED, _WRITE_WORKER_STARTED
    if _WRITE_WORKER_STARTED:
        return
    with _WRITE_WORKER_LOCK:
        if _WRITE_WORKER_STARTED:
            return
        thread = threading.Thread(
            target=_writer_loop,
            name="hermes-voice-bench-writer",
            daemon=True,
        )
        thread.start()
        _WRITE_WORKER_STARTED = True
        if not _ATEXIT_REGISTERED:
            atexit.register(flush_events)
            _ATEXIT_REGISTERED = True


def append_event(event: dict[str, Any]) -> None:
    global _LAST_WRITE_ERROR
    payload = {
        "ts": time.time(),
        **_safe_event(event),
    }
    path = bench_path()
    if os.environ.get("HERMES_VOICE_BENCH_SYNC", "").lower() in {"1", "true", "yes"}:
        _write_payload(path, payload)
        return

    _ensure_write_worker()
    try:
        _WRITE_QUEUE.put_nowait((path, payload))
    except queue.Full:
        _LAST_WRITE_ERROR = "write queue full"
        _LOGGER.warning("Voice bench telemetry write queue full; dropping event")


def flush_events(timeout: float = 2.0) -> bool:
    """Wait until queued telemetry writes are persisted."""
    if not _WRITE_WORKER_STARTED:
        return True
    deadline = time.monotonic() + max(0.0, timeout)
    while getattr(_WRITE_QUEUE, "unfinished_tasks", 0):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def recent_events(*, platform: str | None = None, chat_id: str | None = None, max_events: int = MAX_EVENTS) -> list[dict[str, Any]]:
    global _LAST_READ_ERROR
    path = bench_path()
    if not path.exists():
        _LAST_READ_ERROR = None
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = list(deque(fh, maxlen=max_events))
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
                "mode": item.get("mode") or item.get("pipeline_mode"),
                "realtime_provider": item.get("realtime_provider"),
                "stt_provider": item.get("stt_provider"),
                "tts_provider": item.get("tts_provider"),
                "realtime_model": item.get("realtime_model"),
                "stt_model": item.get("stt_model") or item.get("deepgram_model"),
                "tts_model": item.get("tts_model") or item.get("cartesia_model"),
                "stages": {},
            },
        )
        turn["ts"] = item.get("ts") or turn.get("ts")
        turn["platform"] = item.get("platform") or turn.get("platform")
        turn["chat_id"] = item.get("chat_id") or turn.get("chat_id")
        turn["message_id"] = item.get("message_id") or turn.get("message_id")
        for key in ("mode", "pipeline_mode", "realtime_provider", "stt_provider", "tts_provider", "realtime_model", "stt_model", "tts_model", "deepgram_model", "cartesia_model"):
            value = item.get(key)
            if value and key == "pipeline_mode":
                turn["mode"] = value
            elif value and key == "deepgram_model":
                turn["stt_model"] = value
            elif value and key == "cartesia_model":
                turn["tts_model"] = value
            elif value:
                turn[key] = value
        stage = str(item.get("stage") or "").strip()
        if stage:
            existing = turn["stages"].get(stage)
            if existing is None:
                turn["stages"][stage] = item
            elif isinstance(existing, list):
                existing.append(item)
            else:
                turn["stages"][stage] = [existing, item]
    return list(turns.values())


def _ms(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.0f}ms"
    return "?"


def _stage_item(stage: Any) -> dict[str, Any]:
    if isinstance(stage, list):
        result: dict[str, Any] = {}
        elapsed = 0.0
        has_elapsed = False
        errors = []
        for item in stage:
            if not isinstance(item, dict):
                continue
            result.update(item)
            value = item.get("elapsed_ms")
            if isinstance(value, (int, float)):
                elapsed += float(value)
                has_elapsed = True
            if item.get("error"):
                errors.append(str(item.get("error")))
        if has_elapsed:
            result["elapsed_ms"] = elapsed
        if errors:
            result["error"] = "; ".join(errors)
        return result
    return stage if isinstance(stage, dict) else {}


def format_recent(platform: str | None = None, chat_id: str | None = None, *, limit: int = DEFAULT_LIMIT) -> str:
    if not flush_events():
        return "Voice bench unavailable: telemetry flush timed out."
    if _LAST_WRITE_ERROR:
        return "Voice bench unavailable: telemetry write failed."
    events = recent_events(platform=platform, chat_id=chat_id)
    if _LAST_READ_ERROR:
        return "Voice bench unavailable: telemetry read failed."
    turns = grouped_turns(events)[-max(1, min(limit, 20)):]
    if not turns:
        return "Voice bench: no measured voice turns yet."

    lines = ["Voice bench recent turns:"]
    for turn in reversed(turns):
        stages = turn.get("stages") or {}
        stt = _stage_item(stages.get("stt"))
        agent = _stage_item(stages.get("agent"))
        tts = _stage_item(stages.get("tts"))
        delivery = _stage_item(stages.get("delivery"))
        brain = _stage_item(stages.get("brain"))
        first_audio_ms = delivery.get("first_audio_ms") or tts.get("first_audio_ms")
        stage_values = [
            stage.get("elapsed_ms")
            for stage in (stt, brain, agent, tts, delivery)
            if isinstance(stage.get("elapsed_ms"), (int, float))
        ]
        total_ms = turn.get("total_ms") if isinstance(turn.get("total_ms"), (int, float)) else (sum(stage_values) if stage_values else None)
        status = "ok" if not any(
            (_stage_item(s).get("error") for s in stages.values())
        ) else "warn"
        labels = []
        if turn.get("mode"):
            labels.append(f"mode={turn.get('mode')}")
        if turn.get("realtime_provider"):
            labels.append(f"realtime={turn.get('realtime_provider')}")
        if turn.get("stt_provider"):
            labels.append(f"stt_provider={turn.get('stt_provider')}")
        if turn.get("tts_provider"):
            labels.append(f"tts_provider={turn.get('tts_provider')}")
        lines.append(
            f"- {status} total={_ms(total_ms)} "
            f"first_audio={_ms(first_audio_ms)} "
            f"stt={_ms(stt.get('elapsed_ms'))} "
            f"brain={_ms(brain.get('elapsed_ms'))} "
            f"agent={_ms(agent.get('elapsed_ms'))} "
            f"tts={_ms(tts.get('elapsed_ms'))} "
            f"send={_ms(delivery.get('elapsed_ms'))} "
            + " ".join(labels)
        )
        transcript = str(
            stt.get("transcript_preview") or _redact_text(stt.get("transcript"))
        ).strip()
        if transcript:
            lines.append(f"  heard: {transcript[:120]}")
        response = str(
            agent.get("response_preview") or _redact_text(agent.get("response"))
        ).strip()
        if response:
            lines.append(f"  reply: {response[:120]}")
    return "\n".join(lines)
