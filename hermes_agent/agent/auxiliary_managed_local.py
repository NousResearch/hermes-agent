"""Which endpoint is the llama-server this Hermes manages (sibling of ``auxiliary_client``)."""

from __future__ import annotations

import json
import time
from typing import Optional
from urllib.parse import urlparse

_MANAGED_LOCAL_STATE_TTL_S = 15.0
_managed_local_cache: tuple[float, str] = (0.0, "")


def _managed_local_netloc() -> str:
    """host:port of the managed local llama-server ("" when none), read with a short TTL from
    the supervisor state file provider resolution also uses (exact match)."""
    global _managed_local_cache
    now = time.monotonic()
    ts, cached = _managed_local_cache
    if now - ts < _MANAGED_LOCAL_STATE_TTL_S:
        return cached
    try:
        from hermes_cli.local_runtime.supervisor import state_path

        raw = state_path().read_text(encoding="utf-8-sig")
        base = str((json.loads(raw) or {}).get("base_url", ""))
        netloc = urlparse(base).netloc.lower()
    except (OSError, ValueError, AttributeError):  # no state file, bad JSON, or not an object
        netloc = ""
    _managed_local_cache = (now, netloc)
    return netloc


def _is_managed_local_endpoint(base_url: Optional[str]) -> bool:
    """True when *base_url* targets the llama-server this Hermes manages."""
    if not base_url:
        return False
    managed = _managed_local_netloc()
    if not managed:
        return False
    try:
        return urlparse(str(base_url)).netloc.lower() == managed
    except ValueError:
        return False
