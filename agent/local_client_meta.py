"""Identity metadata for loopback OpenAI-compatible model servers.

The fields in this module are deliberately emitted only to loopback endpoints.
They are Turbohaul scheduling/cache hints, not portable provider API fields.
"""

from __future__ import annotations

import ipaddress
from typing import Any, Mapping
from urllib.parse import urlparse


def is_loopback_base_url(base_url: object) -> bool:
    """Return True only for HTTP(S) URLs whose host is loopback."""
    if not isinstance(base_url, str) or not base_url.strip():
        return False
    try:
        parsed = urlparse(base_url.strip())
        if parsed.scheme not in {"http", "https"}:
            return False
        host = (parsed.hostname or "").strip().lower()
        if host == "localhost":
            return True
        return bool(host and ipaddress.ip_address(host).is_loopback)
    except (ValueError, TypeError):
        return False


def agent_role_metadata(agent: Any) -> dict[str, Any]:
    """Classify an interactive agent as main or delegated work."""
    try:
        depth = int(getattr(agent, "_delegate_depth", 0) or 0)
    except (TypeError, ValueError):
        depth = 0
    if depth > 0:
        meta: dict[str, Any] = {"is_sub_agent": True, "role": "sub_agent"}
        delegate_role = str(getattr(agent, "_delegate_role", "") or "").strip()
        if delegate_role:
            meta["delegate_role"] = delegate_role
        return meta
    return {"is_main": True, "role": "main"}


def build_local_client_meta(
    *,
    session_id: object = None,
    role_metadata: Mapping[str, Any] | None = None,
    task: object = None,
) -> dict[str, Any]:
    """Build Turbohaul identity metadata with auxiliary-task precedence."""
    meta: dict[str, Any] = {}
    sid = str(session_id or "").strip()
    if sid:
        meta["session_id"] = sid

    task_name = str(task or "").strip().lower()
    if task_name == "compression":
        meta.update({"is_compression": True, "role": "compression"})
    elif task_name == "curator":
        meta.update({"is_curator": True, "role": "curator"})
    elif task_name:
        meta.update(
            {
                "is_sub_agent": True,
                "role": "sub_agent",
                "auxiliary_task": task_name,
            }
        )
    elif role_metadata:
        for key in (
            "is_main",
            "is_sub_agent",
            "is_curator",
            "is_compression",
            "role",
            "delegate_role",
        ):
            value = role_metadata.get(key)
            if value not in (None, "", False):
                meta[key] = value
    return meta


def local_client_meta_extra_body(
    *,
    base_url: object,
    session_id: object = None,
    role_metadata: Mapping[str, Any] | None = None,
    task: object = None,
) -> dict[str, Any]:
    """Return an extra-body fragment only for a loopback destination."""
    if not is_loopback_base_url(base_url):
        return {}
    meta = build_local_client_meta(
        session_id=session_id,
        role_metadata=role_metadata,
        task=task,
    )
    return {"client_meta": meta} if meta else {}
