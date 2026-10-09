"""Read-time compatibility for existing authentication JSON schemas."""

from __future__ import annotations
import logging
from typing import Any, Dict, FrozenSet
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
DEFAULT_NOUS_PORTAL_URL = "https://portal.nousresearch.com"


_NOUS_STALE_PORTAL_HOSTS: FrozenSet[str] = frozenset({"api.nousresearch.com"})


def _migrate_stale_nous_portal_url(providers: Dict[str, Any]) -> None:
    nous = providers.get("nous")
    if not isinstance(nous, dict):
        return
    stored = (nous.get("portal_base_url") or "").strip()
    if stored and urlparse(stored).hostname in _NOUS_STALE_PORTAL_HOSTS:
        logger.warning(
            "auth: migrating stale nous portal_base_url %s -> %s",
            stored,
            DEFAULT_NOUS_PORTAL_URL,
        )
        nous["portal_base_url"] = DEFAULT_NOUS_PORTAL_URL
