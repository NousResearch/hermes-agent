"""xAI OAuth upstream adapter for the Hermes proxy.

Uses resolve_xai_http_credentials() from tools/xai_http.py for OAuth-aware
credential resolution (handles runtime provider, PKCE refresh, and env fallback).
Only forwards /v1/chat/completions; model allowlist and stream rejection are
enforced at the proxy server layer for this provider.
"""

from __future__ import annotations

import logging
from typing import FrozenSet

from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

_ALLOWED_PATHS: FrozenSet[str] = frozenset(
    {
        "/chat/completions",
    }
)


class XaiOAuthAdapter(UpstreamAdapter):
    """Proxy upstream for xAI (api.x.ai) via Hermes-managed OAuth."""

    def __init__(self) -> None:
        # resolve_xai_http_credentials handles its own refresh + locking.
        pass

    @property
    def name(self) -> str:
        return "xai-oauth"

    @property
    def display_name(self) -> str:
        return "xAI OAuth"

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return _ALLOWED_PATHS

    def is_authenticated(self) -> bool:
        try:
            from tools.xai_http import resolve_xai_http_credentials

            creds = resolve_xai_http_credentials()
            return bool(creds.get("api_key"))
        except Exception:
            return False

    def get_credential(self) -> UpstreamCredential:
        from tools.xai_http import resolve_xai_http_credentials

        try:
            creds = resolve_xai_http_credentials()
            api_key = str(creds.get("api_key") or "").strip()
            if not api_key:
                raise RuntimeError(
                    "No xAI OAuth credentials available. "
                    "Run `hermes login xai` or set XAI_API_KEY."
                )
            base_url = str(creds.get("base_url") or "https://api.x.ai/v1").strip().rstrip("/")
            return UpstreamCredential(
                bearer=api_key,
                base_url=base_url,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to resolve xAI credentials: {exc}") from exc

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
    ) -> UpstreamCredential | None:
        # Per spec: on 401 we have already attempted one resolve (which does refresh).
        # Returning None lets the caller synthesize 503 instead of leaking 401.
        if status_code == 401:
            logger.info("proxy: xAI returned 401; synthesizing 503 for retryable failover")
            return None
        return None


__all__ = ["XaiOAuthAdapter"]
