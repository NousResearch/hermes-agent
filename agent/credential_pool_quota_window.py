"""Spent subscription windows, read from Anthropic response headers.

Anthropic subscription (OAuth) traffic carries ``anthropic-ratelimit-unified-*`` on every
response, 200 and 429 alike. A window reporting ``rejected`` is a spent plan quota (the 5h
session or the 7d week): account-wide, not per-model, and it stays spent until its own
``-reset`` instant. Recording that on the pool entry lets every session and process skip the
credential at selection until then, instead of each one spending a request to rediscover it.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_UNIFIED_PREFIX = "anthropic-ratelimit-unified-"
QUOTA_EXHAUSTED_REASON = "quota_exhausted"


def spent_quota_reset_at(headers: Any) -> Optional[float]:
    """Epoch at which a spent Anthropic subscription window reopens, else ``None``.

    Only ``status: rejected`` counts: an account with overage enabled keeps serving past 100%
    utilization, so utilization alone would bench a working credential. ``overage-status`` is
    skipped because it reads ``rejected`` on every account whose org disables overage and says
    nothing about quota; usage windows are the ones that also report ``-utilization``.
    """
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    from agent.credential_pool import _parse_absolute_timestamp

    def _future(name: str) -> Optional[float]:
        reset_at = _parse_absolute_timestamp(getter(_UNIFIED_PREFIX + name))
        return reset_at if reset_at is not None and reset_at > time.time() else None

    def _rejected(name: str) -> bool:
        return str(getter(_UNIFIED_PREFIX + name) or "").strip().lower() == "rejected"

    if _rejected("status"):
        overall = _future("reset")
        if overall is not None:
            return overall
    keys = headers.keys() if callable(getattr(headers, "keys", None)) else ()
    resets = []
    for key in keys:
        name = str(key).lower()
        if not name.startswith(_UNIFIED_PREFIX) or not name.endswith("-status") or name == _UNIFIED_PREFIX + "status":
            continue
        window = name[len(_UNIFIED_PREFIX):-len("-status")]
        if _rejected(f"{window}-status") and getter(f"{_UNIFIED_PREFIX}{window}-utilization") is not None:
            reset_at = _future(f"{window}-reset")
            if reset_at is not None:
                resets.append(reset_at)
    return max(resets) if resets else None


class CredentialPoolQuotaWindowMixin:
    def mark_quota_exhausted(
        self, *, reset_at: float, credential_id: Optional[str] = None, api_key_hint: Optional[str] = None,
    ) -> bool:
        """Bench the entry that served a spent-window response until *reset_at* (persisted).

        No rotation: the request that carried the headers already succeeded. The next request
        selects past the bench, and a 429 on this entry meanwhile takes the pre-exhausted fast
        path. Returns True when the entry's state changed.
        """
        from agent.credential_pool import STATUS_EXHAUSTED, _parse_absolute_timestamp

        with self._lock:
            entry = self._identify_failed_entry(credential_id, api_key_hint)
            if entry is None:
                return False
            if entry.last_status == STATUS_EXHAUSTED and _parse_absolute_timestamp(entry.last_error_reset_at) == reset_at:
                return False
            self._mark_exhausted(entry, 429, {"reason": QUOTA_EXHAUSTED_REASON, "reset_at": reset_at})
            logger.info(
                "credential pool: %s subscription window spent; benched until %s",
                entry.label or entry.id[:8], time.strftime("%Y-%m-%d %H:%M", time.localtime(reset_at)),
            )
            return True
