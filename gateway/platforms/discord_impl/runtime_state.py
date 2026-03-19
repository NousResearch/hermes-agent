"""Persisted Discord runtime state for thread bindings and activation overrides."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

RUNTIME_STATE_VERSION = 1
DEFAULT_THREAD_BINDING_IDLE_MINUTES = 24 * 60
DEFAULT_THREAD_BINDING_MAX_AGE_MINUTES = 0


@dataclass
class DiscordThreadBinding:
    """Persisted thread-focus binding for Discord thread UX controls."""

    thread_id: str
    session_key: str
    chat_id: str
    parent_chat_id: Optional[str] = None
    chat_name: str = ""
    bound_by: str = ""
    bound_at: str = ""
    last_activity_at: str = ""
    idle_timeout_minutes: int = DEFAULT_THREAD_BINDING_IDLE_MINUTES
    max_age_minutes: int = DEFAULT_THREAD_BINDING_MAX_AGE_MINUTES


def runtime_state_path():
    """Return the persisted Discord runtime state path."""
    return get_hermes_home() / "discord_runtime_state.json"


def _parse_datetime(raw: str) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def load_runtime_state() -> tuple[dict[str, DiscordThreadBinding], dict[str, str]]:
    """Load persisted Discord thread bindings and activation overrides."""
    path = runtime_state_path()
    if not path.exists():
        return {}, {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Could not load Discord runtime state: %s", exc)
        return {}, {}

    bindings_payload = payload.get("thread_bindings") or {}
    bindings: dict[str, DiscordThreadBinding] = {}
    for thread_id, raw in bindings_payload.items():
        if not isinstance(raw, dict):
            continue
        try:
            binding = DiscordThreadBinding(
                thread_id=str(raw.get("thread_id") or thread_id),
                session_key=str(raw.get("session_key") or ""),
                chat_id=str(raw.get("chat_id") or raw.get("thread_id") or thread_id),
                parent_chat_id=str(raw.get("parent_chat_id") or "").strip() or None,
                chat_name=str(raw.get("chat_name") or ""),
                bound_by=str(raw.get("bound_by") or ""),
                bound_at=str(raw.get("bound_at") or ""),
                last_activity_at=str(raw.get("last_activity_at") or raw.get("bound_at") or ""),
                idle_timeout_minutes=int(raw.get("idle_timeout_minutes") or DEFAULT_THREAD_BINDING_IDLE_MINUTES),
                max_age_minutes=int(raw.get("max_age_minutes") or DEFAULT_THREAD_BINDING_MAX_AGE_MINUTES),
            )
        except Exception:
            continue
        if binding.thread_id and binding.session_key:
            bindings[binding.thread_id] = binding

    activation_payload = payload.get("activation_overrides") or {}
    activation_overrides = {
        str(chat_id): str(mode).strip().lower()
        for chat_id, mode in activation_payload.items()
        if str(chat_id).strip() and str(mode).strip().lower() in {"mention", "always"}
    }
    return bindings, activation_overrides


def save_runtime_state(
    bindings: dict[str, DiscordThreadBinding],
    activation_overrides: dict[str, str],
) -> None:
    """Persist Discord thread bindings and activation overrides."""
    path = runtime_state_path()
    payload = {
        "version": RUNTIME_STATE_VERSION,
        "thread_bindings": {
            thread_id: asdict(binding)
            for thread_id, binding in sorted(bindings.items())
        },
        "activation_overrides": {
            str(chat_id): str(mode)
            for chat_id, mode in sorted(activation_overrides.items())
            if str(mode).strip().lower() in {"mention", "always"}
        },
    }
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        logger.debug("Could not save Discord runtime state: %s", exc)


def binding_expiration_reason(
    binding: DiscordThreadBinding,
    *,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Return ``idle`` or ``max-age`` when the binding has expired."""
    now = now or datetime.now()
    last_activity_at = _parse_datetime(binding.last_activity_at) or _parse_datetime(binding.bound_at) or now
    bound_at = _parse_datetime(binding.bound_at) or last_activity_at

    if binding.max_age_minutes and now >= bound_at + timedelta(minutes=binding.max_age_minutes):
        return "max-age"
    if binding.idle_timeout_minutes and now >= last_activity_at + timedelta(minutes=binding.idle_timeout_minutes):
        return "idle"
    return None


def touch_binding(
    binding: DiscordThreadBinding,
    *,
    now: Optional[datetime] = None,
) -> DiscordThreadBinding:
    """Return a copy of ``binding`` with refreshed activity time."""
    now = now or datetime.now()
    return DiscordThreadBinding(
        thread_id=binding.thread_id,
        session_key=binding.session_key,
        chat_id=binding.chat_id,
        parent_chat_id=binding.parent_chat_id,
        chat_name=binding.chat_name,
        bound_by=binding.bound_by,
        bound_at=binding.bound_at,
        last_activity_at=now.isoformat(),
        idle_timeout_minutes=binding.idle_timeout_minutes,
        max_age_minutes=binding.max_age_minutes,
    )
