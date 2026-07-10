"""OpenAI Codex subscription upstream backed by Hermes' OAuth pool."""

from __future__ import annotations

import logging
import threading
from typing import Dict, FrozenSet, Optional

from agent.auxiliary_client import _CODEX_AUX_BASE_URL, _codex_cloudflare_headers
from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)
_POOL_PROVIDER = "openai-codex"
_ALLOWED_PATHS: FrozenSet[str] = frozenset({"/responses", "/models"})


class OpenAICodexAdapter(UpstreamAdapter):
    """Attach centrally-managed ChatGPT Codex OAuth credentials to Responses calls."""

    auth_hint = "hermes auth add openai-codex"
    fail_closed_on_exhaustion = True
    upstream_base_url = _CODEX_AUX_BASE_URL

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool: Optional[CredentialPool] = None

    @property
    def name(self) -> str:
        return _POOL_PROVIDER

    @property
    def display_name(self) -> str:
        return "OpenAI Codex OAuth"

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return _ALLOWED_PATHS

    def _load_pool(self) -> Optional[CredentialPool]:
        try:
            return load_pool(_POOL_PROVIDER)
        except Exception as exc:
            logger.warning("proxy: failed to load OpenAI Codex credential pool: %s", exc)
            return None

    def is_authenticated(self) -> bool:
        pool = self._load_pool()
        return bool(pool and pool.has_available())

    def get_credential(self) -> UpstreamCredential:
        with self._lock:
            # Keep one authoritative in-process pool so least_used counters and
            # exhaustion state coordinate every isolated downstream client.
            pool = self._pool or self._load_pool()
            if pool is None or not pool.has_credentials():
                raise RuntimeError(
                    "No OpenAI Codex OAuth credentials found. Run "
                    "`hermes auth add openai-codex` first."
                )
            entry = pool.select()
            if entry is None:
                raise RuntimeError(
                    "All OpenAI Codex credentials are exhausted. Run "
                    "`hermes auth reset openai-codex` after their usage window resets."
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
            if status_code == 401:
                # Reuse the pool's serialized, single-use-token-safe refresh path.
                refreshed_entry = pool.try_refresh_current()
                if refreshed_entry is not None:
                    refreshed = self._credential_from_entry(refreshed_entry)
                    if refreshed.bearer != failed_credential.bearer:
                        return refreshed
            next_entry = pool.mark_exhausted_and_rotate(
                status_code=status_code,
                error_context=error_context,
                api_key_hint=failed_credential.bearer,
            )
            if next_entry is None:
                return None
            rotated = self._credential_from_entry(next_entry)
            return rotated if rotated.bearer != failed_credential.bearer else None

    def _credential_from_entry(self, entry: PooledCredential) -> UpstreamCredential:
        bearer = str(entry.runtime_api_key or "").strip()
        if not bearer:
            raise RuntimeError("OpenAI Codex pool entry has no access token; re-authenticate it.")
        return UpstreamCredential(
            bearer=bearer,
            base_url=str(self.upstream_base_url).rstrip("/"),
            expires_at=entry.expires_at,
            headers=_codex_cloudflare_headers(bearer),
        )


__all__ = ["OpenAICodexAdapter"]
