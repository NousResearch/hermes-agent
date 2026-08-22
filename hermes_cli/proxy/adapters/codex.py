"""OpenAI Codex OAuth upstream adapter for the loopback Hermes proxy."""

from __future__ import annotations

import logging
import threading
from typing import FrozenSet, Optional

from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from hermes_cli.auth import DEFAULT_CODEX_BASE_URL
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

_POOL_PROVIDER = "openai-codex"
_ALLOWED_PATHS: FrozenSet[str] = frozenset({"/responses", "/models"})
_OWNED_HEADERS: FrozenSet[str] = frozenset(
    {"User-Agent", "originator", "ChatGPT-Account-ID"}
)


class OpenAICodexAdapter(UpstreamAdapter):
    """Attach the main Hermes Codex OAuth credential to Responses requests."""

    auth_hint = "hermes auth add openai-codex --type oauth"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool: Optional[CredentialPool] = None

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex OAuth"

    @property
    def loopback_only(self) -> bool:
        return True

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return _ALLOWED_PATHS

    def is_authenticated(self) -> bool:
        """Check local pool state only; never call the network."""
        pool = self._load_pool()
        return bool(pool and pool.has_available())

    def get_credential(self) -> UpstreamCredential:
        with self._lock:
            pool = self._load_pool()
            if pool is None or not pool.has_credentials():
                raise RuntimeError(
                    "No OpenAI Codex OAuth credentials found. Run "
                    "`hermes auth add openai-codex --type oauth` first."
                )
            entry = pool.select()
            if entry is None:
                raise RuntimeError(
                    "No available OpenAI Codex OAuth credentials. Reset the "
                    "pool cooldown or re-authenticate."
                )
            self._pool = pool
            return self._credential_from_entry(entry)

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
    ) -> Optional[UpstreamCredential]:
        if status_code not in {401, 429}:
            return None

        with self._lock:
            pool = self._pool or self._load_pool()
            if pool is None:
                return None

            if status_code == 401:
                refreshed = pool.try_refresh_matching(
                    api_key_hint=failed_credential.bearer
                )
                if refreshed is None:
                    # A concurrent request may have queued behind the adapter
                    # lock with the same now-stale bearer. If the first request
                    # already refreshed/rotated the pool, coalesce onto that
                    # current credential instead of failing or exhausting the
                    # obsolete token a second time.
                    current = pool.select()
                    if (
                        current is not None
                        and str(current.runtime_api_key or "").strip()
                        != failed_credential.bearer
                    ):
                        refreshed = current
                if refreshed is None:
                    refreshed = pool.mark_exhausted_and_rotate(
                        status_code=401,
                        api_key_hint=failed_credential.bearer,
                    )
            else:
                refreshed = pool.mark_exhausted_and_rotate(
                    status_code=429,
                    api_key_hint=failed_credential.bearer,
                )

            if refreshed is None:
                return None
            retry = self._credential_from_entry(refreshed)
            if retry.bearer == failed_credential.bearer:
                return None
            logger.info(
                "proxy: Codex upstream returned %s; retrying with refreshed/rotated credential",
                status_code,
            )
            return retry

    def get_owned_upstream_header_names(self) -> frozenset[str]:
        return _OWNED_HEADERS

    def get_upstream_headers(
        self,
        credential: UpstreamCredential,
    ) -> dict[str, str]:
        # Reuse the native Codex request identity contract: originator and
        # User-Agent avoid Cloudflare challenges on VPS traffic, while the
        # account header is extracted from the JWT when present.
        from agent.auxiliary_client import _codex_cloudflare_headers

        return _codex_cloudflare_headers(credential.bearer)

    def _load_pool(self) -> Optional[CredentialPool]:
        try:
            return load_pool(_POOL_PROVIDER)
        except Exception:
            logger.warning("proxy: failed to load Codex OAuth credential pool", exc_info=True)
            return None

    def _credential_from_entry(self, entry: PooledCredential) -> UpstreamCredential:
        bearer = str(entry.runtime_api_key or "").strip()
        if not bearer:
            raise RuntimeError(
                "Codex OAuth pool entry has no access token. Re-authenticate "
                "with `hermes auth add openai-codex --type oauth`."
            )
        base_url = str(
            entry.runtime_base_url or entry.base_url or DEFAULT_CODEX_BASE_URL
        ).strip().rstrip("/")
        trusted_base = DEFAULT_CODEX_BASE_URL.rstrip("/")
        if base_url != trusted_base:
            raise RuntimeError(
                "Refusing to attach OpenAI Codex OAuth credentials to untrusted "
                f"upstream {base_url!r}; expected {trusted_base!r}."
            )
        return UpstreamCredential(
            bearer=bearer,
            base_url=trusted_base,
            expires_at=entry.expires_at,
        )


__all__ = ["OpenAICodexAdapter"]
