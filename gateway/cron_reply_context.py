"""Short-lived context for replies to cron deliveries."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_STORE_PATH = get_hermes_home() / "gateway" / "cron_reply_contexts.json"
_MAX_CONTENT_CHARS = 4000


def _normalize_chat_id(platform: str, chat_id: str) -> str:
    chat = str(chat_id)
    if platform.lower() == "teams" and ";messageid=" in chat:
        return chat.split(";messageid=", 1)[0]
    return chat


def _record_key(platform: str, chat_id: str, thread_id: Optional[str]) -> str:
    platform_name = platform.lower()
    chat = _normalize_chat_id(platform_name, chat_id)
    return f"{platform_name}::{chat}::{thread_id or ''}"


def _load_records() -> dict[str, dict[str, Any]]:
    try:
        if not _STORE_PATH.exists():
            return {}
        raw = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
        records = raw.get("contexts", raw) if isinstance(raw, dict) else {}
        if not isinstance(records, dict):
            return {}
        return {
            str(key): value
            for key, value in records.items()
            if isinstance(value, dict)
        }
    except Exception:
        logger.debug("Failed to load cron reply context store", exc_info=True)
        return {}


def _write_records(records: dict[str, dict[str, Any]]) -> None:
    payload = {
        "version": 1,
        "contexts": records,
    }
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _STORE_PATH.with_suffix(_STORE_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(_STORE_PATH)


def record_cron_reply_context(
    platform: str,
    chat_id: str,
    content: str,
    *,
    thread_id: Optional[str] = None,
    message_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> None:
    """Persist cron delivery content for a future reply in the same chat/thread."""
    text = str(content or "").strip()
    if not text:
        return
    platform_name = str(platform).lower()
    chat = _normalize_chat_id(platform_name, str(chat_id))
    thread = str(thread_id) if thread_id else None
    record = {
        "platform": platform_name,
        "chat_id": chat,
        "thread_id": thread,
        "message_id": str(message_id) if message_id else None,
        "job_id": str(job_id) if job_id else None,
        "content": text[:_MAX_CONTENT_CHARS],
        "updated_at": time.time(),
    }
    records = _load_records()
    records[_record_key(platform_name, chat, thread)] = record
    _write_records(records)


def find_cron_reply_context(
    platform: str,
    chat_id: str,
    *,
    thread_id: Optional[str] = None,
    max_age_seconds: int = 6 * 60 * 60,
) -> Optional[dict[str, Any]]:
    """Return recent cron context for an incoming message, if any."""
    platform_name = str(platform).lower()
    chat = _normalize_chat_id(platform_name, str(chat_id))
    thread = str(thread_id) if thread_id else None
    records = _load_records()
    now = time.time()

    def _fresh(record: dict[str, Any]) -> bool:
        try:
            return now - float(record.get("updated_at", 0)) <= max_age_seconds
        except (TypeError, ValueError):
            return False

    exact = records.get(_record_key(platform_name, chat, thread))
    if exact and _fresh(exact):
        return exact
    if thread is not None:
        return None

    candidates = [
        record
        for record in records.values()
        if record.get("platform") == platform_name
        and str(record.get("chat_id")) == chat
        and _fresh(record)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda record: float(record.get("updated_at", 0) or 0))
