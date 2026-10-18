"""Optional local Laya shadow router for comparison with Jev.

Laya is an auxiliary typed-decision model. This module deliberately mirrors
agent.jev_router: it runs asynchronously, never changes the active Hermes
model, and writes auditable JSONL records keyed by the same task_id.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

ROUTE_POLICIES: dict[str, dict[str, Any]] = {
    "coding": {"primary": "openai-codex/gpt-5.5", "fallback": "zai/glm-5.3"},
    "multimodal": {"primary": "kimi-k3", "fallback": ""},
    "long_context": {"primary": "kimi-k3", "fallback": ""},
    "light_chat": {"primary": "deepseek/deepseek-v4-flash", "fallback": ""},
    "research": {"primary": "current-model+web", "fallback": ""},
    "local_tools": {"primary": "current-model", "fallback": ""},
    "complex_reasoning": {"primary": "current-frontier", "fallback": ""},
    "ask_user": {"primary": "none", "fallback": ""},
}
ROUTE_OPTIONS = tuple(ROUTE_POLICIES)
_ROUTER: Any = None
_ROUTER_LOCK = threading.RLock()


def _config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        value = load_config()
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _settings() -> dict[str, Any]:
    value = _config().get("laya_router", {})
    return value if isinstance(value, dict) else {}


def _enabled() -> bool:
    settings = _settings()
    raw = os.environ.get("HERMES_LAYA_ROUTER_ENABLED")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(settings.get("enabled", False))


def _is_multimodal(message: Any) -> bool:
    if isinstance(message, list):
        return any(_is_multimodal(item) for item in message)
    if isinstance(message, dict):
        kind = str(message.get("type", "")).lower()
        return kind in {"image", "image_url", "input_image", "audio", "video"} or any(
            _is_multimodal(v) for v in message.values()
        )
    return False


def _text(message: Any) -> str:
    if isinstance(message, str):
        return message
    try:
        return json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(message)


def build_state(agent: Any, user_message: Any) -> dict[str, Any]:
    text = _text(user_message)
    return {
        "user_message": text[:8000],
        "message_truncated": len(text) > 8000,
        "input_chars": len(text),
        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "has_multimodal_content": _is_multimodal(user_message),
        "current_provider": str(getattr(agent, "provider", "") or ""),
        "current_model": str(getattr(agent, "model", "") or ""),
        "platform": str(getattr(agent, "platform", "") or ""),
        "route_policies": ROUTE_POLICIES,
    }


def _questions() -> dict[str, dict[str, Any]]:
    return {
        "task_route": {
            "type": "choice",
            "instructions": (
                "Choose the single best route for the user's task. "
                "Choose multimodal when image, audio, or video understanding is needed. "
                "Choose coding for repository/code changes. Choose ask_user when the task "
                "is ambiguous or high-risk."
            ),
            "criteria": {
                "coding": "Code changes, debugging, tests, repositories, or software engineering.",
                "multimodal": "The task requires understanding image, audio, or video content.",
                "long_context": "The task requires unusually long context or many documents.",
                "light_chat": "Simple conversation, explanation, rewriting, or lightweight help.",
                "research": "Current external facts or source-backed web research are needed.",
                "local_tools": "The main work is local files, shell commands, or deterministic tools.",
                "complex_reasoning": "Difficult analysis or planning that is not primarily coding.",
                "ask_user": "Required information is missing or the action is too ambiguous/high-risk.",
            },
        }
    }


def _router(settings: dict[str, Any]) -> Any:
    global _ROUTER
    with _ROUTER_LOCK:
        if _ROUTER is None:
            from laya import Router
            _ROUTER = Router(
                device=str(settings.get("device", "auto")) if settings.get("device") not in (None, "", "auto") else None,
                max_loaded=max(1, int(settings.get("max_loaded", 2))),
                default=str(settings.get("default", "english")),
                preload=bool(settings.get("preload", False)),
            )
        return _ROUTER


def _request(state: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    router = _router(settings)
    with _ROUTER_LOCK:
        return router.predict(state, _questions())


def _write_record(record: dict[str, Any], settings: dict[str, Any]) -> None:
    configured = str(settings.get("log_path", "")).strip()
    path = Path(configured).expanduser() if configured else get_hermes_home() / "logs" / "laya-router.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _run(agent: Any, user_message: Any, task_id: str | None) -> None:
    settings = _settings()
    state = build_state(agent, user_message)
    started = time.perf_counter()
    record: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id or "",
        "session_id": str(getattr(agent, "session_id", "") or ""),
        "mode": "shadow",
        "actual_provider": str(getattr(agent, "provider", "") or ""),
        "actual_model": str(getattr(agent, "model", "") or ""),
        "input_chars": state["input_chars"],
        "input_sha256": state["input_sha256"],
        "has_multimodal_content": state["has_multimodal_content"],
    }
    try:
        result = _request(state, settings)
        answer = ((result.get("answers") or {}).get("task_route") or {}) if isinstance(result, dict) else {}
        choice = answer.get("choice")
        record.update({
            "ok": True,
            "laya_model": result.get("model") if isinstance(result, dict) else None,
            "laya_routing": result.get("routing") if isinstance(result, dict) else None,
            "choice": choice,
            "confidence": answer.get("confidence"),
            "probabilities": answer.get("probabilities"),
            "recommended_policy": ROUTE_POLICIES.get(choice if isinstance(choice, str) else ""),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        })
    except Exception as exc:
        record.update({
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "error_type": type(exc).__name__,
            "error": f"{type(exc).__name__}: {exc}",
        })
        logger.debug("Laya shadow routing failed", exc_info=True)
    try:
        _write_record(record, settings)
    except Exception:
        logger.debug("Could not write Laya shadow routing record", exc_info=True)


def record_outcome(*, task_id: str, session_id: str = "", platform: str = "", result: dict[str, Any] | None = None,
                   error: BaseException | None = None, duration_ms: float | None = None,
                   retry_count: int = 0, tool_error_count: int = 0) -> None:
    if not _enabled():
        return
    terminal = result if isinstance(result, dict) else {}
    if error is not None:
        outcome = "interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "error"
        error_type = type(error).__name__
    elif terminal.get("interrupted") is True:
        outcome, error_type = "interrupted", None
    elif terminal.get("failed") is True:
        outcome, error_type = "failed", None
    elif terminal.get("completed") is True:
        outcome, error_type = "completed", None
    else:
        outcome, error_type = "returned", None
    settings = _settings()
    configured = str(settings.get("outcome_log_path", "")).strip()
    path = Path(configured).expanduser() if configured else get_hermes_home() / "logs" / "laya-outcomes.jsonl"
    record = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "task_id": task_id,
              "session_id": session_id, "platform": platform, "outcome": outcome, "error_type": error_type,
              "duration_ms": duration_ms, "retry_count": retry_count, "tool_error_count": tool_error_count}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        logger.debug("Could not write Laya task outcome", exc_info=True)


def shadow_route(agent: Any, user_message: Any, task_id: str | None = None) -> None:
    """Start one non-blocking Laya classification, if explicitly enabled."""
    if not _enabled() or str(getattr(agent, "platform", "") or "") == "subagent":
        return
    threading.Thread(target=_run, args=(agent, user_message, task_id),
                     name="hermes-laya-shadow-router", daemon=True).start()


__all__ = ["ROUTE_POLICIES", "build_state", "record_outcome", "shadow_route"]
