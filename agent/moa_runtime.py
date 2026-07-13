"""Runtime resilience и telemetry без содержимого для Mixture of Agents.

Модуль намеренно хранит только метки provider/model и числовые runtime-метрики.
Prompts, outputs, credentials, тексты исключений и сессий никогда не попадают
в telemetry или feedback-файлы.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CIRCUIT_LOCK = threading.Lock()
_CIRCUITS: dict[str, dict[str, Any]] = {}
_FILE_LOCK = threading.Lock()

_DEFAULT_COOLDOWNS = {
    "rate_limit": 60.0,
    "quota": 900.0,
    "timeout": 30.0,
    "transient": 30.0,
    "auth": 60.0,
}
_TELEMETRY_MAX_BYTES = 2 * 1024 * 1024
_TELEMETRY_BACKUPS = 3


def slot_label(slot: dict[str, Any]) -> str:
    return f"{slot.get('provider', '')}:{slot.get('model', '')}"


def classify_moa_error(exc: Exception) -> str:
    """Классифицировать исключение provider в очищенную failover-категорию."""
    try:
        from agent.auxiliary_client import (
            _is_payment_error,
            _is_rate_limit_error,
            _is_timeout_error,
            _is_transient_transport_error,
        )

        if _is_payment_error(exc):
            return "quota"
        if _is_rate_limit_error(exc):
            return "rate_limit"
        if _is_timeout_error(exc):
            return "timeout"
        if _is_transient_transport_error(exc):
            return "transient"
    except Exception:
        pass

    status = getattr(exc, "status_code", None)
    message = str(exc).casefold()
    if status in {500, 502, 503, 504}:
        return "transient"
    if status in {401, 403} and any(
        token in message
        for token in ("oauth", "token expired", "temporarily unavailable", "refresh")
    ):
        return "auth"
    return "fatal"


def should_failover_moa_error(exc: Exception) -> bool:
    return classify_moa_error(exc) != "fatal"


def circuit_status(slot: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    label = slot_label(slot)
    current = time.time() if now is None else now
    with _CIRCUIT_LOCK:
        state = dict(_CIRCUITS.get(label) or {})
        until = float(state.get("until") or 0.0)
        if until <= current:
            if label in _CIRCUITS:
                _CIRCUITS.pop(label, None)
            return {"active": False, "retry_after_seconds": 0, "reason": "", "failures": 0}
    return {
        "active": True,
        "retry_after_seconds": max(0, int(until - current + 0.999)),
        "reason": str(state.get("reason") or "transient"),
        "failures": int(state.get("failures") or 1),
    }


def mark_slot_failure(
    slot: dict[str, Any],
    exc: Exception,
    *,
    cooldown_seconds: float | None = None,
    quota_cooldown_seconds: float | None = None,
) -> dict[str, Any]:
    kind = classify_moa_error(exc)
    if kind == "fatal":
        return {"active": False, "reason": kind, "retry_after_seconds": 0, "failures": 0}
    base = (
        quota_cooldown_seconds
        if kind == "quota" and quota_cooldown_seconds is not None
        else cooldown_seconds
    )
    duration = max(1.0, float(base if base is not None else _DEFAULT_COOLDOWNS[kind]))
    label = slot_label(slot)
    now = time.time()
    with _CIRCUIT_LOCK:
        previous = _CIRCUITS.get(label) or {}
        failures = int(previous.get("failures") or 0) + 1
        # Повторные сбои продлевают cooldown, но не создают постоянную блокировку.
        duration = min(duration * (2 ** min(failures - 1, 3)), 3600.0)
        _CIRCUITS[label] = {
            "until": now + duration,
            "reason": kind,
            "failures": failures,
        }
    return circuit_status(slot, now=now)


def mark_slot_success(slot: dict[str, Any]) -> None:
    with _CIRCUIT_LOCK:
        _CIRCUITS.pop(slot_label(slot), None)


def reset_runtime_state_for_tests() -> None:
    with _CIRCUIT_LOCK:
        _CIRCUITS.clear()


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _telemetry_path() -> Path:
    return _hermes_home() / "logs" / "moa_runtime.jsonl"


def _feedback_path() -> Path:
    return _hermes_home() / "state" / "moa_router_feedback.json"


def _rotate_jsonl(path: Path) -> None:
    if not path.exists() or path.stat().st_size < _TELEMETRY_MAX_BYTES:
        return
    oldest = path.with_suffix(path.suffix + f".{_TELEMETRY_BACKUPS}")
    if oldest.exists():
        oldest.unlink()
    for idx in range(_TELEMETRY_BACKUPS - 1, 0, -1):
        source = path.with_suffix(path.suffix + f".{idx}")
        if source.exists():
            source.replace(path.with_suffix(path.suffix + f".{idx + 1}"))
    path.replace(path.with_suffix(path.suffix + ".1"))


def record_moa_telemetry(event: dict[str, Any]) -> None:
    """Добавить ограниченное runtime-событие без содержимого задачи."""
    allowed = {
        "preset",
        "status",
        "aggregator",
        "fallback_used",
        "attempts",
        "latency_ms",
        "reference_count",
        "reference_cost_usd",
        "budget_exceeded",
        "failure_kind",
    }
    clean = {key: event[key] for key in allowed if key in event}
    for key in ("attempts", "latency_ms", "reference_count"):
        if key in clean and clean[key] is not None:
            clean[key] = int(clean[key])
    if clean.get("reference_cost_usd") is not None:
        clean["reference_cost_usd"] = float(clean["reference_cost_usd"])
    for key in ("fallback_used", "budget_exceeded"):
        if key in clean:
            clean[key] = bool(clean[key])
    clean["timestamp"] = int(time.time())
    path = _telemetry_path()
    try:
        with _FILE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_jsonl(path)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(clean, ensure_ascii=True, separators=(",", ":")) + "\n")
    except Exception as exc:  # pragma: no cover - telemetry never breaks a turn
        logger.debug("MoA telemetry write skipped: %s", type(exc).__name__)


def read_moa_telemetry(limit: int = 100) -> dict[str, Any]:
    """Вернуть компактный агрегат без prompts и текстов исключений."""
    path = _telemetry_path()
    if not path.exists():
        return {"events": 0, "presets": {}}
    try:
        with _FILE_LOCK:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 256 * 1024))
                raw = handle.read().decode("utf-8", "replace")
        rows: list[dict[str, Any]] = []
        for line in raw.splitlines()[-max(1, min(limit, 500)):]:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("preset"):
                rows.append(value)
        presets: dict[str, dict[str, Any]] = {}
        for row in rows:
            name = str(row["preset"])
            item = presets.setdefault(name, {
                "turns": 0,
                "failures": 0,
                "fallbacks": 0,
                "average_latency_ms": 0,
                "known_reference_cost_usd": 0.0,
                "last": {},
            })
            item["turns"] += 1
            item["failures"] += int(row.get("status") == "failed")
            item["fallbacks"] += int(bool(row.get("fallback_used")))
            item["average_latency_ms"] += int(row.get("latency_ms") or 0)
            if row.get("reference_cost_usd") is not None:
                item["known_reference_cost_usd"] += float(row["reference_cost_usd"])
            item["last"] = row
        for item in presets.values():
            turns = max(1, int(item["turns"]))
            item["average_latency_ms"] = int(item["average_latency_ms"] / turns)
            item["known_reference_cost_usd"] = round(item["known_reference_cost_usd"], 6)
        return {"events": len(rows), "presets": presets}
    except Exception as exc:
        logger.debug("MoA telemetry read skipped: %s", type(exc).__name__)
        return {"events": 0, "presets": {}}


def _load_feedback() -> dict[str, Any]:
    path = _feedback_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def record_route_feedback(suggested_route: str, chosen_route: str) -> dict[str, Any]:
    valid = {"fast", "balanced", "research", "code_heavy", "max"}
    suggested = str(suggested_route or "").strip()
    chosen = str(chosen_route or "").strip()
    if suggested not in valid or chosen not in valid:
        raise ValueError("unsupported MoA route feedback")
    path = _feedback_path()
    with _FILE_LOCK:
        data = _load_feedback()
        routes = data.setdefault("routes", {})
        choices = routes.setdefault(suggested, {})
        choices[chosen] = int(choices.get(chosen) or 0) + 1
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    return route_feedback_summary()


def apply_route_feedback(suggested_route: str) -> str:
    suggested = str(suggested_route or "").strip()
    choices = ((_load_feedback().get("routes") or {}).get(suggested) or {})
    if not isinstance(choices, dict):
        return suggested
    total = sum(max(0, int(value or 0)) for value in choices.values())
    if total < 3:
        return suggested
    chosen, count = max(choices.items(), key=lambda item: int(item[1] or 0))
    return str(chosen) if int(count or 0) / total >= 0.67 else suggested


def route_feedback_summary() -> dict[str, Any]:
    routes = _load_feedback().get("routes") or {}
    summary: dict[str, Any] = {}
    for suggested, choices in routes.items():
        if not isinstance(choices, dict):
            continue
        total = sum(max(0, int(value or 0)) for value in choices.values())
        summary[str(suggested)] = {
            "samples": total,
            "effective_route": apply_route_feedback(str(suggested)),
        }
    return summary
