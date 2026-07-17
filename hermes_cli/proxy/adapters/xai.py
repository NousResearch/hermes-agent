"""xAI Grok OAuth upstream adapter."""

from __future__ import annotations

import logging
import threading
from typing import Dict, FrozenSet, Optional

from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

_POOL_PROVIDER = "xai-oauth"

# xAI's public API is OpenAI-compatible for the endpoints Hermes commonly
# uses. The Responses endpoint is included because Hermes' native xAI runtime
# uses codex_responses mode.
_ALLOWED_PATHS: FrozenSet[str] = frozenset(
    {
        "/responses",
        "/chat/completions",
        "/completions",
        "/embeddings",
        "/models",
    }
)


class XAIGrokAdapter(UpstreamAdapter):
    """Proxy upstream for xAI Grok via Hermes-managed OAuth credentials."""

    auth_hint = "hermes auth add xai-oauth --type oauth"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool: Optional[CredentialPool] = None

    @property
    def name(self) -> str:
        return "xai"

    @property
    def display_name(self) -> str:
        return "xAI Grok OAuth"

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return _ALLOWED_PATHS

    def is_authenticated(self) -> bool:
        pool = self._load_pool()
        return bool(pool and pool.has_available())

    def get_credential(self) -> UpstreamCredential:
        with self._lock:
            pool = self._load_pool()
            if pool is None or not pool.has_credentials():
                raise RuntimeError(
                    "No xAI OAuth credentials found. Run "
                    "`hermes auth add xai-oauth --type oauth` first."
                )

            entry = pool.select()
            if entry is None:
                raise RuntimeError(
                    "No available xAI OAuth credentials found. Run "
                    "`hermes auth reset xai-oauth` or re-authenticate with "
                    "`hermes auth add xai-oauth --type oauth`."
                )

            self._pool = pool
            return self._credential_from_entry(entry)

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
        error_context: Optional[Dict[str, object]] = None,
    ) -> Optional[UpstreamCredential]:
        if status_code not in {401, 429}:
            return None

        with self._lock:
            pool = self._pool or self._load_pool()
            if pool is None:
                return None

            if status_code == 429:
                # Mark the rate-limited key with its 1-hour cooldown and rotate
                # to the next available credential. Returns None when the pool
                # has no other key to offer — the 429 will flow back to the client.
                refreshed = pool.mark_exhausted_and_rotate(
                    status_code=status_code,
                    error_context=error_context,
                    api_key_hint=failed_credential.bearer,
                )
            else:
                refreshed = pool.try_refresh_current()
                if refreshed is None:
                    refreshed = pool.mark_exhausted_and_rotate(
                        status_code=status_code,
                        error_context=error_context,
                        api_key_hint=failed_credential.bearer,
                    )
            if refreshed is None:
                return None

            retry_cred = self._credential_from_entry(refreshed)
            if retry_cred.bearer == failed_credential.bearer:
                return None
            logger.info(
                "proxy: xAI upstream returned %s; retrying with rotated pool credential",
                status_code,
            )
            return retry_cred

    def _load_pool(self) -> Optional[CredentialPool]:
        try:
            return load_pool(_POOL_PROVIDER)
        except Exception as exc:
            logger.warning("proxy: failed to load xAI OAuth credential pool: %s", exc)
            return None

    def _credential_from_entry(self, entry: PooledCredential) -> UpstreamCredential:
        bearer = (
            getattr(entry, "runtime_api_key", None)
            or getattr(entry, "access_token", "")
            or ""
        )
        bearer = str(bearer).strip()
        if not bearer:
            raise RuntimeError(
                "xAI OAuth credential pool entry did not contain an access token. "
                "Re-authenticate with `hermes auth add xai-oauth --type oauth`."
            )

        base_url = (
            getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", None)
            or DEFAULT_XAI_OAUTH_BASE_URL
        )
        base_url = str(base_url or DEFAULT_XAI_OAUTH_BASE_URL).strip().rstrip("/")

        return UpstreamCredential(
            bearer=bearer,
            base_url=base_url or DEFAULT_XAI_OAUTH_BASE_URL,
            expires_at=getattr(entry, "expires_at", None),
        )


__all__ = ["XAIGrokAdapter"]
