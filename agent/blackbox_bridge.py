"""Best-effort bridge from Hermes session persistence to TheWon Blackbox.

TheWon treats blackbox raw logs as the non-lossy audit layer that must exist
*before* any context compression/summary can safely discard live context.  This
module intentionally sits below the agent loop, at SessionDB persistence, so CLI,
gateway, cron mirrors, and future frontends all flow through one append-only
recorder without adding model tools or changing prompts.

The bridge is config-gated and fail-open: if TheWon is absent or the recorder
raises, Hermes session persistence must continue unaffected.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_recorder: Any = None
_recorder_key: tuple[str, str] | None = None
_disabled_reason: str | None = None


def _safe_text(value: Any, *, limit: int = 200_000) -> str:
    """Return a searchable, JSON-safe text representation for blackbox data."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") in (None, "text", "input_text"):
                    parts.append(str(item.get("text") or ""))
                elif item.get("type") in {"image", "image_url", "input_image"}:
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, default=str))
            else:
                parts.append(str(item))
        text = "\n".join(p for p in parts if p)
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    if len(text) > limit:
        return text[:limit] + f"\n[blackbox_bridge truncated at {limit} chars]"
    return text


def _load_cfg() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    except Exception as exc:
        logger.debug("blackbox_bridge: config load failed: %s", exc)
        return {}
    section = cfg.get("blackbox") if isinstance(cfg, dict) else {}
    return section if isinstance(section, dict) else {}


def _enabled(cfg: dict[str, Any]) -> bool:
    value = cfg.get("enabled", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_thewon_system(cfg: dict[str, Any]) -> Path:
    raw = cfg.get("thewon_system") or os.getenv("THEWON_SYSTEM") or "/Users/elroy/TheWon/System"
    return Path(str(raw)).expanduser().resolve()


def _get_recorder(cfg: dict[str, Any]) -> Any:
    """Return a cached BlackboxRecorder or None when disabled/unavailable."""
    global _recorder, _recorder_key, _disabled_reason
    if not _enabled(cfg):
        return None

    agent_id = str(cfg.get("agent_id") or "VN").strip() or "VN"
    thewon_system = _resolve_thewon_system(cfg)
    key = (agent_id, str(thewon_system))

    with _lock:
        if _recorder is not None and _recorder_key == key:
            return _recorder
        try:
            system_root = thewon_system.parent / "00_System"
            if str(system_root) not in sys.path:
                sys.path.insert(0, str(system_root))
            os.environ.setdefault("THEWON_SYSTEM", str(thewon_system))
            from shared.blackbox_recorder import BlackboxRecorder  # type: ignore[import-not-found]

            _recorder = BlackboxRecorder(agent_id)
            _recorder_key = key
            _disabled_reason = None
            return _recorder
        except Exception as exc:
            _recorder = None
            _recorder_key = None
            _disabled_reason = str(exc)
            logger.warning("TheWon blackbox bridge unavailable: %s", exc)
            return None


def record_session_message(
    *,
    session_id: str,
    message_id: Optional[int],
    role: str,
    content: Any = None,
    tool_name: Optional[str] = None,
    tool_calls: Any = None,
    tool_call_id: Optional[str] = None,
    token_count: Optional[int] = None,
    finish_reason: Optional[str] = None,
    platform_message_id: Optional[str] = None,
    observed: bool = False,
    timestamp: Any = None,
) -> None:
    """Append one Hermes message to TheWon raw blackbox, best-effort."""
    cfg = _load_cfg()
    rec = _get_recorder(cfg)
    if rec is None:
        return

    role = str(role or "unknown")
    content_text = _safe_text(content)
    data = {
        "source": "hermes_state",
        "message_role": role,
        "content_text": content_text,
        "message_id": message_id,
        "platform_message_id": platform_message_id,
        "tool_name": tool_name,
        "tool_calls": tool_calls,
        "tool_call_id": tool_call_id,
        "token_count": token_count,
        "finish_reason": finish_reason,
        "observed": bool(observed),
    }
    if timestamp is not None:
        data["message_timestamp"] = str(timestamp)

    # Keep events semantically queryable while preserving exact fields in data.
    if role == "user":
        event_type = "input"
    elif role == "assistant":
        event_type = "output"
    elif role == "tool" or tool_name:
        event_type = "tool_call"
    else:
        event_type = "message"

    try:
        rec.record(
            {
                "type": event_type,
                "session": session_id,
                "project": cfg.get("project"),
                "tags": ["hermes", "session_db", "raw"],
                "severity": "info",
                "data": data,
            },
            session_id=session_id,
            request_id=str(platform_message_id or message_id or session_id),
        )
    except Exception as exc:
        logger.debug("TheWon blackbox message record failed: %s", exc)


def record_transcript_event(
    *,
    session_id: str,
    event_type: str,
    messages: list[dict[str, Any]] | None = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Record transcript-level events such as compression/rewrite boundaries."""
    cfg = _load_cfg()
    rec = _get_recorder(cfg)
    if rec is None:
        return

    messages = messages or []
    preview_parts: list[str] = []
    for msg in messages[:20]:
        role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
        content = msg.get("content") if isinstance(msg, dict) else msg
        text = _safe_text(content, limit=8_000)
        if text:
            preview_parts.append(f"[{role}] {text}")
    content_text = "\n\n".join(preview_parts)
    data = {
        "source": "hermes_state",
        "event": event_type,
        "message_count": len(messages),
        "content_text": content_text,
    }
    if extra:
        data.update(extra)
    try:
        rec.record(
            {
                "type": event_type,
                "session": session_id,
                "project": cfg.get("project"),
                "tags": ["hermes", "session_db", "compression" if "compact" in event_type or "compress" in event_type else "rewrite"],
                "severity": "info",
                "data": data,
            },
            session_id=session_id,
            request_id=f"{session_id}:{event_type}",
        )
    except Exception as exc:
        logger.debug("TheWon blackbox transcript event failed: %s", exc)


def status() -> dict[str, Any]:
    """Small diagnostic surface for tests/manual smoke checks."""
    cfg = _load_cfg()
    return {
        "enabled": _enabled(cfg),
        "agent_id": cfg.get("agent_id") or "VN",
        "thewon_system": str(_resolve_thewon_system(cfg)),
        "recorder_ready": _get_recorder(cfg) is not None if _enabled(cfg) else False,
        "disabled_reason": _disabled_reason,
    }
