"""Shared runtime helpers for gateway auto-vision enrichment."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from gateway.config import Platform
from gateway.platforms.base import MessageType


def should_prefer_remote_auto_vision_source(image_ref: str) -> bool:
    """Return True when auto-vision should preserve the remote image ref."""
    try:
        from tools.vision_tools import should_prefer_remote_vision_source

        return should_prefer_remote_vision_source(image_ref)
    except Exception:
        return False


def media_ref_suffix(ref: str) -> str:
    value = str(ref or "").strip()
    if not value:
        return ""
    try:
        parsed = urlsplit(value)
        if parsed.scheme:
            value = parsed.path
    except Exception:
        pass
    return Path(value).suffix.lower()


def should_skip_auto_vision_media(
    *,
    path: str,
    media_type: str,
    preferred_source: str = "",
) -> bool:
    """Drop low-signal animated media before it reaches the vision model."""
    normalized_type = str(media_type or "").strip().lower()
    if normalized_type == "image/gif":
        return True
    for ref in (preferred_source, path):
        if media_ref_suffix(ref) == ".gif":
            return True
    return False


def image_vision_inputs_from_event(
    event: Any,
    *,
    remote_source_selector: Callable[[str], bool] = should_prefer_remote_auto_vision_source,
) -> list[str]:
    """Return image analysis inputs for a message event."""
    image_inputs: list[str] = []
    source = getattr(event, "source", None)
    prefer_local = getattr(source, "platform", None) == Platform.QQ_NAPCAT

    for attachment in event.ensure_attachments():
        mtype = str(attachment.mime_type or "").strip()
        if attachment.kind != "image" and not (
            event.message_type == MessageType.PHOTO and mtype.startswith("image/")
        ):
            continue
        path = str(attachment.local_path or "").strip()
        preferred = str(attachment.analysis_ref or attachment.remote_url or "").strip()
        if should_skip_auto_vision_media(
            path=path,
            media_type=mtype,
            preferred_source=preferred,
        ):
            continue
        if prefer_local and preferred and remote_source_selector(preferred):
            image_inputs.append(preferred)
            continue
        image_inputs.append(path if prefer_local and path else (preferred or path))

    return image_inputs


def ensure_auto_vision_state(owner: Any) -> None:
    """Initialize auto-vision state containers on a partially built runner."""
    if not hasattr(owner, "_background_tasks") or not isinstance(owner._background_tasks, set):
        owner._background_tasks = set()
    if not hasattr(owner, "_auto_vision_cache") or not isinstance(owner._auto_vision_cache, dict):
        owner._auto_vision_cache = {}
    if not hasattr(owner, "_auto_vision_tasks") or not isinstance(owner._auto_vision_tasks, dict):
        owner._auto_vision_tasks = {}
    if not hasattr(owner, "_auto_vision_unhealthy_until"):
        owner._auto_vision_unhealthy_until = 0.0
    if not hasattr(owner, "_auto_vision_unhealthy_reason"):
        owner._auto_vision_unhealthy_reason = ""


def prune_auto_vision_state(
    owner: Any,
    *,
    now_ts: float | None = None,
    max_cache_entries: int,
) -> None:
    ensure_auto_vision_state(owner)
    now = time.time() if now_ts is None else float(now_ts)

    for cache_key, task in list(owner._auto_vision_tasks.items()):
        try:
            done = bool(task.done())
        except Exception:
            done = True
        if done:
            owner._auto_vision_tasks.pop(cache_key, None)

    for cache_key, entry in list(owner._auto_vision_cache.items()):
        if not isinstance(entry, dict):
            owner._auto_vision_cache.pop(cache_key, None)
            continue
        expires_at = entry.get("expires_at")
        try:
            expires_at_value = float(expires_at) if expires_at is not None else 0.0
        except (TypeError, ValueError):
            expires_at_value = 0.0
        if expires_at_value and expires_at_value <= now:
            owner._auto_vision_cache.pop(cache_key, None)

    overflow = len(owner._auto_vision_cache) - int(max_cache_entries)
    if overflow > 0:
        ranked_keys = sorted(
            owner._auto_vision_cache,
            key=lambda key: float(
                (owner._auto_vision_cache.get(key) or {}).get("updated_at") or 0.0
            ),
        )
        for cache_key in ranked_keys[:overflow]:
            owner._auto_vision_cache.pop(cache_key, None)


def auto_vision_cache_key(path: str) -> str:
    candidate = Path(os.path.expanduser(str(path or "")))
    try:
        resolved = candidate.resolve(strict=False)
    except Exception:
        resolved = candidate
    try:
        stat = resolved.stat()
        return f"{resolved}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return str(resolved)


def get_auto_vision_cache_entry(
    owner: Any,
    cache_key: str,
    *,
    now_ts: float | None = None,
    max_cache_entries: int,
) -> dict[str, Any] | None:
    prune_auto_vision_state(owner, now_ts=now_ts, max_cache_entries=max_cache_entries)
    entry = owner._auto_vision_cache.get(cache_key)
    if not isinstance(entry, dict):
        return None
    current_now = time.time() if now_ts is None else float(now_ts)
    expires_at = entry.get("expires_at")
    try:
        expires_at_value = float(expires_at) if expires_at is not None else 0.0
    except (TypeError, ValueError):
        expires_at_value = 0.0
    if expires_at_value and expires_at_value <= current_now:
        owner._auto_vision_cache.pop(cache_key, None)
        return None
    return entry


def auto_vision_degraded_note(path: str, *, pending: bool) -> str:
    del path
    if pending:
        return "[Image attached; no verified image description is available yet.]"
    return "[Image attached; no verified image description is available for this turn.]"


def classify_auto_vision_failure(error_text: str) -> str:
    lowered = str(error_text or "").strip().lower()
    if not lowered:
        return "none"
    deterministic_markers = (
        "payment required",
        "insufficient",
        "credits",
        "billing",
        "does not support",
        "not support image",
        "content_policy",
        "unauthorized",
        "blocked",
        "forbidden",
        "no available channel",
        "no available accounts for this model tier",
    )
    transient_markers = (
        "timed out",
        "timeout",
        "empty content",
        "http 5",
        "httpx",
        "remoteprotocolerror",
        "connection",
        "provider",
    )
    if any(marker in lowered for marker in deterministic_markers):
        return "deterministic"
    if any(marker in lowered for marker in transient_markers):
        return "transient"
    return "none"


def auto_vision_cooldown_remaining(owner: Any, *, now_ts: float | None = None) -> tuple[float, str]:
    current_now = time.time() if now_ts is None else float(now_ts)
    until = float(getattr(owner, "_auto_vision_unhealthy_until", 0.0) or 0.0)
    reason = str(getattr(owner, "_auto_vision_unhealthy_reason", "") or "").strip()
    remaining = until - current_now
    if remaining <= 0:
        owner._auto_vision_unhealthy_until = 0.0
        owner._auto_vision_unhealthy_reason = ""
        return 0.0, ""
    return remaining, reason


def mark_auto_vision_cooldown(
    owner: Any,
    *,
    reason: str,
    seconds: float,
    now_ts: float | None = None,
) -> None:
    try:
        seconds_value = float(seconds)
    except (TypeError, ValueError):
        return
    if seconds_value <= 0:
        return
    current_now = time.time() if now_ts is None else float(now_ts)
    owner._auto_vision_unhealthy_until = current_now + seconds_value
    owner._auto_vision_unhealthy_reason = str(reason or "").strip()


def clear_auto_vision_cooldown(owner: Any) -> None:
    owner._auto_vision_unhealthy_until = 0.0
    owner._auto_vision_unhealthy_reason = ""
