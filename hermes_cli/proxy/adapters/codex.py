"""OpenAI Codex (ChatGPT OAuth) upstream adapter.

This adapter lets ``hermes proxy`` act as the single local owner of the
ChatGPT/Codex OAuth token chain. Downstream Hermes/OpenClaw clients can point at
``http://127.0.0.1:8645/v1`` with a dummy bearer; the proxy resolves the current
Codex token from the shared Hermes credential pool, attaches the Cloudflare-safe
Codex headers, and refreshes through the existing pool machinery on a 401.
"""

from __future__ import annotations

import logging
import threading
from typing import FrozenSet, Optional

from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from agent.auxiliary_client import _codex_cloudflare_headers
from hermes_cli.auth import DEFAULT_CODEX_BASE_URL
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

_POOL_PROVIDER = "openai-codex"

_ALLOWED_PATHS: FrozenSet[str] = frozenset(
    {
        "/responses",
        "/chat/completions",
        "/completions",
        "/embeddings",
        "/models",
    }
)


class OpenAICodexAdapter(UpstreamAdapter):
    """Proxy upstream for ChatGPT-account Codex OAuth credentials."""

    auth_hint = "hermes auth add openai-codex"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool: Optional[CredentialPool] = None

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex"

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
                    "No OpenAI Codex OAuth credentials found. Run "
                    "`hermes auth add openai-codex` first."
                )

            entry = pool.select()
            if entry is None:
                raise RuntimeError(
                    "No available OpenAI Codex OAuth credentials found. Run "
                    "`hermes auth reset openai-codex` or re-authenticate with "
                    "`hermes auth add openai-codex`."
                )
            self._pool = pool
            return self._credential_from_entry(entry)

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
    ) -> Optional[UpstreamCredential]:
        if status_code != 401:
            return None

        with self._lock:
            pool = self._pool or self._load_pool()
            if pool is None:
                return None
            refreshed = pool.try_refresh_current()
            if refreshed is None:
                return None
            retry_cred = self._credential_from_entry(refreshed)
            if retry_cred.bearer == failed_credential.bearer:
                return None
            logger.info("proxy: Codex upstream returned 401; retrying with refreshed token")
            return retry_cred

    def _load_pool(self) -> Optional[CredentialPool]:
        try:
            return load_pool(_POOL_PROVIDER)
        except Exception as exc:
            logger.warning("proxy: failed to load OpenAI Codex credential pool: %s", exc)
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
                "OpenAI Codex credential pool entry did not contain an access token. "
                "Re-authenticate with `hermes auth add openai-codex`."
            )

        # Always route Codex through the ChatGPT Codex backend. Pool entries may
        # contain historical or user-overridden base URLs; the broker's contract
        # is to be the stable local façade in front of this upstream.
        base_url = str(DEFAULT_CODEX_BASE_URL).strip().rstrip("/")

        return UpstreamCredential(
            bearer=bearer,
            base_url=base_url,
            expires_at=getattr(entry, "expires_at", None),
            extra_headers=_codex_cloudflare_headers(bearer),
        )


__all__ = ["OpenAICodexAdapter"]
