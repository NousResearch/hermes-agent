"""Hermes-native machine-readable context telemetry.

This module emits a small local JSON artifact for external operator UIs that
need current context-window occupancy without scraping terminal text. It never
includes prompt/message content or credentials.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home

_SCHEMA_VERSION = 1
_LAST_WRITE: dict[str, tuple[float, str]] = {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _number(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _profile_name() -> str:
    home = get_hermes_home()
    parts = home.parts
    if len(parts) >= 2 and parts[-2] == "profiles":
        return parts[-1]
    return os.environ.get("HERMES_PROFILE") or "default"


def _telemetry_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config() or {}
        except Exception:
            config = {}
    context_cfg = config.get("context", {}) if isinstance(config, Mapping) else {}
    telemetry = context_cfg.get("telemetry", {}) if isinstance(context_cfg, Mapping) else {}
    if not isinstance(telemetry, Mapping):
        return {}
    return dict(telemetry)


def telemetry_enabled(config: Mapping[str, Any] | None = None) -> bool:
    cfg = _telemetry_config(config)
    return _truthy(cfg.get("enabled")) and bool(str(cfg.get("path") or "").strip())


def build_context_telemetry_payload(
    agent: Any,
    *,
    usage: Mapping[str, Any] | None = None,
    snapshot: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a content-free JSON-serializable context telemetry payload."""
    cfg = _telemetry_config(config)
    model = str(
        (snapshot or {}).get("model_name")
        or (usage or {}).get("model")
        or getattr(agent, "model", "")
        or ""
    )
    provider = str(getattr(agent, "provider", "") or "")
    session_id = str(getattr(agent, "session_id", "") or "")

    used = _number((snapshot or {}).get("context_tokens"), 0)
    max_tokens = _number((snapshot or {}).get("context_length"), 0)
    percent_value: Any = (snapshot or {}).get("context_percent")
    token_source = "unknown"

    if usage:
        used = _number(usage.get("context_used"), used)
        max_tokens = _number(usage.get("context_max"), max_tokens)
        percent_value = usage.get("context_percent", percent_value)

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        raw_prompt = getattr(compressor, "last_prompt_tokens", None)
        raw_context_len = getattr(compressor, "context_length", None)
        prompt_tokens = _number(raw_prompt, 0)
        context_length = _number(raw_context_len, 0)
        if prompt_tokens < 0:
            prompt_tokens = 0
        if prompt_tokens and context_length:
            used = prompt_tokens
            max_tokens = context_length
            percent_value = max(0, min(100, round((used / max_tokens) * 100)))
            token_source = "actual"
        elif context_length and not max_tokens:
            max_tokens = context_length

    if token_source == "unknown" and used and max_tokens:
        token_source = "actual"
    percent: int | None
    if token_source == "unknown" or not max_tokens:
        percent = None
    else:
        percent = _number(percent_value, max(0, min(100, round((used / max_tokens) * 100))))
        percent = max(0, min(100, percent))

    session_usage = {
        "input_tokens": _number((snapshot or {}).get("session_input_tokens"), _number((usage or {}).get("input"), _number(getattr(agent, "session_input_tokens", 0)))),
        "output_tokens": _number((snapshot or {}).get("session_output_tokens"), _number((usage or {}).get("output"), _number(getattr(agent, "session_output_tokens", 0)))),
        "reasoning_tokens": _number((usage or {}).get("reasoning"), _number(getattr(agent, "session_reasoning_tokens", 0))),
        "prompt_tokens": _number((snapshot or {}).get("session_prompt_tokens"), _number((usage or {}).get("prompt"), _number(getattr(agent, "session_prompt_tokens", 0)))),
        "completion_tokens": _number((snapshot or {}).get("session_completion_tokens"), _number((usage or {}).get("completion"), _number(getattr(agent, "session_completion_tokens", 0)))),
        "total_tokens": _number((snapshot or {}).get("session_total_tokens"), _number((usage or {}).get("total"), _number(getattr(agent, "session_total_tokens", 0)))),
        "api_calls": _number((snapshot or {}).get("session_api_calls"), _number((usage or {}).get("calls"), _number(getattr(agent, "session_api_calls", 0)))),
        "compressions": _number((snapshot or {}).get("compressions"), _number((usage or {}).get("compressions"), _number(getattr(compressor, "compression_count", 0) if compressor else 0))),
    }

    if now is None:
        now = datetime.now(timezone.utc)
    thresholds = {
        "smart_zone_tokens": _number(cfg.get("smart_zone_tokens"), 120000),
        "smart_zone_context_pct": _number(cfg.get("smart_zone_context_pct"), 45),
        "compression_threshold": _float(cfg.get("compression_threshold"), _float((config or {}).get("compression", {}).get("threshold") if isinstance((config or {}).get("compression"), Mapping) else None, 0.85)),
    }
    return {
        "schema_version": _SCHEMA_VERSION,
        "source": "hermes_runtime",
        "source_kind": "native_context_telemetry",
        "profile": _profile_name(),
        "session_id": session_id,
        "model": model,
        "provider": provider,
        "pid": os.getpid(),
        "updated_at": now.isoformat().replace("+00:00", "Z"),
        "context": {
            "used_tokens": used if token_source != "unknown" else None,
            "max_tokens": max_tokens or None,
            "percent": percent,
            "token_source": token_source,
        },
        "session_usage": session_usage,
        "thresholds": thresholds,
    }


def write_context_telemetry(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    min_write_interval_seconds: float = 1.0,
    force: bool = False,
) -> bool:
    """Atomically write telemetry JSON when changed or the interval elapsed."""
    target = Path(path).expanduser()
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    now = time.monotonic()
    key = str(target)
    last = _LAST_WRITE.get(key)
    if not force and last is not None:
        last_at, last_blob = last
        if blob == last_blob and now - last_at < max(0.0, float(min_write_interval_seconds)):
            return False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    pretty = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp.write_text(pretty, encoding="utf-8")
    os.replace(tmp, target)
    _LAST_WRITE[key] = (now, blob)
    return True


def emit_context_telemetry(
    agent: Any,
    *,
    usage: Mapping[str, Any] | None = None,
    snapshot: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    force: bool = False,
) -> bool:
    """Best-effort telemetry emission; never raises into the UI/runtime path."""
    cfg = _telemetry_config(config)
    if not (_truthy(cfg.get("enabled")) and str(cfg.get("path") or "").strip()):
        return False
    try:
        payload = build_context_telemetry_payload(agent, usage=usage, snapshot=snapshot, config=config)
        return write_context_telemetry(
            payload,
            cfg["path"],
            min_write_interval_seconds=_float(cfg.get("min_write_interval_seconds"), 1.0),
            force=force,
        )
    except Exception:
        return False
