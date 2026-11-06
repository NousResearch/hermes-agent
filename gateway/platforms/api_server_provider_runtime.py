"""Per-request provider/model resolution for the API server.

An API caller may name ``provider`` (and ``model``) in the request body. Those resolve through
the same authenticated provider catalog as the gateway default, without touching config.yaml.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _resolve_request_runtime_agent_kwargs(
    provider: str, target_model: Optional[str] = None, *, allow_fallback: bool = False,
) -> dict[str, Any]:
    """gateway.run._resolve_runtime_agent_kwargs() for an explicit provider/model, so an API
    caller uses the same authenticated provider catalog without mutating config.yaml.

    With ``allow_fallback`` an ``AuthError`` from the requested provider (no credentials, an
    expired login, a pool benched for *target_model*) walks the configured fallback chain
    through ``resolve_runtime_with_fallback`` — the same resolution-time fallback the gateway
    default gets — and the result carries the entry's ``model``. Misconfiguration (an unknown
    provider) propagates unchanged either way; a chain with no usable entry re-raises the
    primary error, so the caller's controlled error response is untouched.
    """
    from hermes_cli.runtime_provider import (
        format_runtime_provider_error, resolve_runtime_provider, resolve_runtime_with_fallback)
    try:
        if allow_fallback:
            from gateway.run import _load_gateway_config
            runtime, entry = resolve_runtime_with_fallback(
                _load_gateway_config(), requested=provider, target_model=target_model or None)
        else:
            runtime, entry = resolve_runtime_provider(requested=provider, target_model=target_model), None
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc

    kwargs: dict[str, Any] = {
        **{k: runtime.get(k) for k in ("api_key", "base_url", "provider", "api_mode", "command")},
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool")}
    if entry is not None:
        kwargs["model"] = entry["model"]
        logger.info(
            "api_server: requested %s/%s is unavailable at resolution time; this turn runs on %s/%s",
            provider, target_model or "", kwargs.get("provider") or entry.get("provider") or "?", kwargs["model"])
    return kwargs
