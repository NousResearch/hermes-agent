"""Optional HTTP approval gateway before mutating chain calls."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def approval_gateway_allow(
    *,
    tool_name: str,
    preview: Dict[str, Any],
    timeout_sec: float = 5.0,
) -> bool:
    """
    If WEB3_APPROVAL_GATEWAY_URL is set, POST JSON {tool, preview}.
    Returns True only on HTTP 2xx with body {\"allow\": true}.
    If env unset, returns True (no gateway).
    On timeout / network error: deny when WEB3_APPROVAL_DENY_ON_ERROR=1 else allow.
    """
    url = os.environ.get("WEB3_APPROVAL_GATEWAY_URL", "").strip()
    if not url:
        return True
    deny_on_err = os.environ.get("WEB3_APPROVAL_DENY_ON_ERROR", "1").strip() in (
        "1",
        "true",
        "yes",
    )
    try:
        import httpx
    except ImportError:
        logger.warning("httpx missing; denying send while gateway URL is set")
        return False

    payload = {"tool": tool_name, "preview": preview}
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            r = await client.post(url, json=payload)
    except Exception as exc:  # noqa: BLE001 — boundary: fail closed/open by flag
        logger.warning("approval gateway error: %s", exc)
        return not deny_on_err

    if r.status_code < 200 or r.status_code >= 300:
        logger.warning("approval gateway HTTP %s", r.status_code)
        return False
    try:
        data = r.json()
    except json.JSONDecodeError:
        return False
    return bool(data.get("allow"))


def sync_preview_redact(preview: Dict[str, Any]) -> Dict[str, Any]:
    """Strip long raw payloads from logs (copy)."""
    out = dict(preview)
    for k in ("raw_transaction", "signed_bytes"):
        if k in out and isinstance(out[k], str) and len(out[k]) > 32:
            out[k] = out[k][:16] + "…(redacted)"
    return out
