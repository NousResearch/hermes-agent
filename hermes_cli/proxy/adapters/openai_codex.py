"""OpenAI Codex subscription upstream adapter."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import FrozenSet, Iterator, Mapping, Optional

from agent.codex_headers import codex_cloudflare_headers
from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential, load_pool
from hermes_cli.auth import DEFAULT_CODEX_BASE_URL, resolve_codex_runtime_credentials
from hermes_cli.profiles import _get_default_hermes_home
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

logger = logging.getLogger(__name__)

_POOL_PROVIDER = "openai-codex"
_ALLOWED_PATHS: FrozenSet[str] = frozenset({"/responses", "/models"})
_REPLACED_REQUEST_HEADERS: FrozenSet[str] = frozenset({
    "user-agent",
    "originator",
    "chatgpt-account-id",
    "x-openai-internal-codex-residency",
})


class OpenAICodexAdapter(UpstreamAdapter):
    """Proxy Codex Responses API calls through the default profile's OAuth pool.

    The explicit default-home binding is the sharing boundary: named profiles can use the
    loopback proxy, but neither read nor receive the default profile's rotating tokens.
    """

    auth_hint = "hermes auth add openai-codex"

    def __init__(self) -> None:
        self._default_home = _get_default_hermes_home()
        self._lock = threading.Lock()
        self._pool: Optional[CredentialPool] = None

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex Subscription"

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return _ALLOWED_PATHS

    @property
    def replaced_request_headers(self) -> FrozenSet[str]:
        return _REPLACED_REQUEST_HEADERS

    def is_authenticated(self) -> bool:
        with self._default_scope():
            pool = self._load_pool()
            return bool(pool and pool.has_available())

    def get_credential(self) -> UpstreamCredential:
        with self._lock, self._default_scope():
            resolver_error: Optional[Exception] = None
            try:
                # Refresh/synchronize the singleton through the canonical Codex resolver before
                # pool selection. load_pool() then applies cooldown and multi-account rotation.
                resolve_codex_runtime_credentials()
            except Exception as exc:
                resolver_error = exc

            pool = self._load_pool()
            entry = pool.select() if pool is not None else None
            if entry is None:
                message = (
                    "No available Codex OAuth credentials in the default Hermes profile. "
                    "Run `hermes auth add openai-codex` without `--profile` first."
                )
                raise RuntimeError(message) from resolver_error
            self._pool = pool
            return self._credential_from_entry(entry)

    def get_retry_credential(
        self, *, failed_credential: UpstreamCredential, status_code: int
    ) -> Optional[UpstreamCredential]:
        if status_code not in {401, 429}:
            return None
        with self._lock, self._default_scope():
            pool = self._pool or self._load_pool()
            if pool is None:
                return None
            entry = None
            if status_code == 401:
                entry = pool.try_refresh_matching(api_key_hint=failed_credential.bearer)
            if entry is None or entry.runtime_api_key == failed_credential.bearer:
                entry = pool.mark_exhausted_and_rotate(
                    status_code=status_code,
                    api_key_hint=failed_credential.bearer,
                )
            if entry is None or entry.runtime_api_key == failed_credential.bearer:
                return None
            logger.info(
                "proxy: Codex upstream returned %s; retrying with refreshed/rotated pool credential",
                status_code,
            )
            return self._credential_from_entry(entry)

    def get_upstream_headers(self, credential: UpstreamCredential) -> Mapping[str, str]:
        return codex_cloudflare_headers(credential.bearer, base_url=DEFAULT_CODEX_BASE_URL)

    def _load_pool(self) -> Optional[CredentialPool]:
        try:
            return load_pool(_POOL_PROVIDER)
        except Exception as exc:
            logger.warning("proxy: failed to load default-profile Codex OAuth pool: %s", exc)
            return None

    @staticmethod
    def _credential_from_entry(entry: PooledCredential) -> UpstreamCredential:
        if entry.auth_type != AUTH_TYPE_OAUTH:
            raise RuntimeError(
                "The default profile's openai-codex pool selected a non-OAuth credential. "
                "Re-authenticate with `hermes auth add openai-codex`."
            )
        bearer = str(entry.runtime_api_key or entry.access_token or "").strip()
        if not bearer:
            raise RuntimeError(
                "Codex OAuth pool entry did not contain an access token. "
                "Run `hermes auth add openai-codex` without `--profile` to re-authenticate."
            )
        return UpstreamCredential(
            bearer=bearer,
            base_url=DEFAULT_CODEX_BASE_URL,
            expires_at=entry.expires_at,
        )

    @contextmanager
    def _default_scope(self) -> Iterator[Path]:
        token = set_hermes_home_override(self._default_home)
        try:
            yield self._default_home
        finally:
            reset_hermes_home_override(token)


__all__ = ["OpenAICodexAdapter"]
