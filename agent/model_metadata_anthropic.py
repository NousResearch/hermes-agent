"""Live Anthropic context windows: ``max_input_tokens`` from ``/v1/models`` for the caller's credential.

The listing is per credential (an API key, or a Claude login's OAuth token), so every model the
account can use gets its published window. That includes families with no row in
``agent.model_metadata.DEFAULT_CONTEXT_LENGTHS`` yet. ``get_model_context_length`` calls
``query_anthropic_context_length`` as its Anthropic step.
"""

import hashlib
import logging
import time
from typing import Any, Optional

from agent import model_metadata_http
from agent.model_metadata_memo_bounds import MEMO_LOCK
from utils import base_url_hostname

logger = logging.getLogger(__name__)

# Catalog windows memoized per (endpoint, credential digest). A model switch re-reads /v1/models at
# most once per TTL, and a failed read is retried after the short failure window, not on every lookup.
# Entitlements are per account, so a different credential is a different catalog.
_ANTHROPIC_CONTEXT_LIMITS_TTL = 600.0
_ANTHROPIC_CONTEXT_LIMITS_FAILURE_TTL = 60.0
_anthropic_context_limits: dict[tuple[str, str], tuple[float, Optional[dict[str, int]]]] = {}


def _anthropic_context_credential(base_url: str, api_key: Any) -> Optional[str]:
    """The credential the window probe may send to *base_url*, or None to skip the probe.

    A metadata probe never mints a ``key_cmd`` callable token source (#114967). An OAuth token, or
    the native credential resolved because the caller passed none, only ever goes to Anthropic's own
    host.
    """
    if callable(api_key):
        return None
    native = base_url_hostname(base_url) == "api.anthropic.com"
    token = api_key.strip() if isinstance(api_key, str) else ""
    if not token and native:
        try:
            from agent.anthropic_credentials import native_anthropic_fallback_token
            token = native_anthropic_fallback_token(base_url).strip()
        except Exception:  # a credential-store failure must not break the context lookup
            logger.debug("Anthropic credential resolution for the window probe failed", exc_info=True)
            token = ""
    if not token or (token.startswith("sk-ant-oat") and not native):
        return None
    return token


def _anthropic_live_context_limits(base_url: str, token: str) -> Optional[dict[str, int]]:
    """``{model id: max_input_tokens}`` from Anthropic's /v1/models for *token*, or None."""
    base = base_url.rstrip("/").removesuffix("/v1")
    key = (base, hashlib.sha256(token.encode()).hexdigest()[:16])
    now = time.monotonic()
    with MEMO_LOCK:
        hit = _anthropic_context_limits.get(key)
    if hit is not None and hit[0] > now:
        return hit[1]
    headers = {"anthropic-version": "2023-06-01"}
    if token.startswith("sk-ant-oat"):
        # A Claude login authenticates as an OAuth bearer, which is accepted only with the OAuth
        # beta (x-api-key answers 401 for these tokens).
        from agent.anthropic_adapter import _OAUTH_ONLY_BETAS
        headers["Authorization"] = f"Bearer {token}"
        headers["anthropic-beta"] = ",".join(_OAUTH_ONLY_BETAS)
    else:
        headers["x-api-key"] = token
    limits: Optional[dict[str, int]] = None
    try:
        resp = model_metadata_http.get(f"{base}/v1/models?limit=1000", headers=headers, timeout=(5, 10), verify=model_metadata_http.resolve_verify(base_url))
        if resp.status_code == 200:
            limits = {
                m["id"]: m["max_input_tokens"] for m in resp.json().get("data", [])
                if isinstance(m, dict) and isinstance(m.get("id"), str)
                and type(m.get("max_input_tokens")) is int and m["max_input_tokens"] > 0
            } or None
        else:
            logger.debug("Anthropic /v1/models window probe returned HTTP %s", resp.status_code)
    except Exception:  # any probe failure falls through to the static tables
        logger.debug("Anthropic /v1/models window probe failed", exc_info=True)
    ttl = _ANTHROPIC_CONTEXT_LIMITS_TTL if limits else _ANTHROPIC_CONTEXT_LIMITS_FAILURE_TTL
    with MEMO_LOCK:
        for stale in [k for k, (expires, _) in _anthropic_context_limits.items() if expires <= now]:
            del _anthropic_context_limits[stale]
        _anthropic_context_limits[key] = (now + ttl, limits)
    return limits


def query_anthropic_context_length(model: str, base_url: str, api_key: Any) -> Optional[int]:
    """Live ``max_input_tokens`` for *model* from Anthropic's /v1/models (API key or Claude login).

    Returns None when no credential may be sent, the endpoint fails, or the listing omits the model
    or its window. The caller then falls through to the static tables.
    """
    token = _anthropic_context_credential(base_url, api_key)
    if token is None:
        return None
    limits = _anthropic_live_context_limits(base_url, token)
    return (limits or {}).get(model)
