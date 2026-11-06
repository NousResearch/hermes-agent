"""credential-scope-example — minimal consumer of the credential-scope contract.

Records per-scope API attempt outcomes (keyed by
``(provider_id, credential_scope_id)``) from ``pre_api_request`` /
``post_api_request`` / ``api_request_error``. It never reads credentials,
makes no network calls, and degrades to a no-op when the
``llm.credential_scope`` capability is not granted (the host strips the
``credential_scope`` field from this plugin's payloads in that case).

The counters are in-memory and illustrative: a real consumer (quota tracker,
health monitor) would persist them keyed the same way and treat
``unknown``/``unavailable`` scopes as provider-wide, not per-account.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_counters: dict[str, dict[str, int]] = {}
_lock = threading.Lock()


def _scope_key(credential_scope: Any) -> str:
    if not isinstance(credential_scope, dict):
        return "<no-scope>"
    provider = credential_scope.get("provider_id") or "?"
    scope_id = credential_scope.get("credential_scope_id")
    status = credential_scope.get("identity_status") or "unknown"
    return f"{provider}:{scope_id if scope_id else status}"


def _record(outcome: str, kwargs: dict[str, Any]) -> None:
    key = _scope_key(kwargs.get("credential_scope"))
    with _lock:
        bucket = _counters.setdefault(key, {"attempts": 0, "ok": 0, "error": 0})
        bucket["attempts"] += 1
        if outcome in bucket:
            bucket[outcome] += 1
    logger.debug("credential-scope-example %s %s", outcome, key)


def on_pre_api_request(**kwargs: Any) -> None:
    _record("attempt", kwargs)


def on_post_api_request(**kwargs: Any) -> None:
    _record("ok", kwargs)


def on_api_request_error(**kwargs: Any) -> None:
    _record("error", kwargs)


def register(ctx) -> None:
    if not ctx.has_capability("llm.credential_scope"):
        logger.info("credential-scope-example: llm.credential_scope not granted; staying idle")
        return
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("api_request_error", on_api_request_error)
