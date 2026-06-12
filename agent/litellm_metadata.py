"""LiteLLM request metadata helpers for source-level spend attribution."""

from __future__ import annotations

import hashlib
import os
from typing import Any
from urllib.parse import urlparse


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _clean(value: Any, *, limit: int = 160) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # Metadata is for aggregation keys only; keep it compact and printable.
    text = "".join(ch if 32 <= ord(ch) < 127 else "_" for ch in text)
    return text[:limit]


def _digest(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def is_litellm_proxy(base_url: Any) -> bool:
    """Return True when the target endpoint is a LiteLLM proxy.

    The explicit env override is useful for private hostnames that do not include
    ``litellm`` but still expose LiteLLM's metadata-aware OpenAI-compatible API.
    """
    if _truthy(os.getenv("HERMES_LITELLM_METADATA")):
        return True
    try:
        host = urlparse(str(base_url or "")).hostname or ""
    except Exception:
        host = ""
    return "litellm" in host.lower()


def _session_env(name: str, default: str = "") -> str:
    try:
        from gateway.session_context import get_session_env

        return get_session_env(name, default)
    except Exception:
        return os.getenv(name, default)


def _context_env(name: str, default: str = "") -> str:
    """Read only task-local contextvars, avoiding stale process env fallback."""
    try:
        from gateway.session_context import get_context_env

        return get_context_env(name, default)
    except Exception:
        return default


def build_litellm_request_metadata(
    agent: Any, *, caller: str = "main"
) -> dict[str, str] | None:
    """Build safe LiteLLM metadata tags for request/spend attribution.

    Returns ``None`` for non-LiteLLM endpoints so strict direct providers do not
    receive extra request fields they might reject.
    """
    if not is_litellm_proxy(getattr(agent, "base_url", "")):
        return None

    platform = _clean(
        getattr(agent, "platform", "") or _session_env("HERMES_SESSION_PLATFORM")
    )
    cron_job_id = _context_env("HERMES_CRON_JOB_ID")
    cron_job_hash = _digest(cron_job_id)
    parent_session_hash = _digest(getattr(agent, "_parent_session_id", ""))
    cron_active = bool(cron_job_hash)

    if cron_active:
        source = "cron"
        source_tag = f"cron:{cron_job_hash}" if cron_job_hash else "cron"
    else:
        source = platform or _clean(os.getenv("HERMES_SESSION_SOURCE")) or "cli"
        if parent_session_hash:
            source_tag = f"delegate:{source}"
        elif source in {
            "slack",
            "discord",
            "telegram",
            "matrix",
            "signal",
            "whatsapp",
            "yuanbao",
        }:
            source_tag = f"interactive:{source}"
        else:
            source_tag = source

    metadata: dict[str, str] = {
        "hermes_app": "hermes-agent",
        "hermes_caller": _clean(caller, limit=80) or "main",
        "hermes_source": source,
        "hermes_source_tag": source_tag,
    }

    for key, value in {
        "hermes_session_hash": _digest(getattr(agent, "session_id", "")),
        "hermes_parent_session_hash": parent_session_hash,
        "hermes_provider": getattr(agent, "provider", ""),
        "hermes_profile": os.getenv("HERMES_PROFILE", ""),
        "hermes_cron_job_hash": cron_job_hash,
        "hermes_gateway_session_hash": _digest(
            getattr(agent, "_gateway_session_key", "")
        ),
    }.items():
        cleaned = _clean(value)
        if cleaned:
            metadata[key] = cleaned

    return metadata
