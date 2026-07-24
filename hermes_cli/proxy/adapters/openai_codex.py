"""OpenAI Codex OAuth upstream adapter."""

from __future__ import annotations

import logging
import threading
from typing import FrozenSet, Optional

from hermes_cli.auth import resolve_codex_runtime_credentials
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
_ALLOWED_PATHS: FrozenSet[str] = frozenset({"/responses", "/models"})


class OpenAICodexAdapter(UpstreamAdapter):
    """Proxy upstream for ChatGPT Codex via Hermes-managed OAuth credentials."""

    auth_hint = "hermes auth add openai-codex --type oauth"

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex OAuth"

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return _ALLOWED_PATHS

    def is_authenticated(self) -> bool:
        try:
            resolved = resolve_codex_runtime_credentials(refresh_if_expiring=False)
        except Exception:
            return False
        return bool(resolved and resolved.get("api_key"))

    def get_credential(self) -> UpstreamCredential:
        with self._lock:
            return self._resolve_credential(force_refresh=False)

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
    ) -> Optional[UpstreamCredential]:
        if status_code != 401:
            return None
        with self._lock:
            refreshed = self._resolve_credential(force_refresh=True)
        if refreshed.bearer == failed_credential.bearer:
            return None
        logger.info("proxy: Codex upstream returned 401; retrying with refreshed OAuth token")
        return refreshed

    @staticmethod
    def _resolve_credential(*, force_refresh: bool) -> UpstreamCredential:
        resolved = resolve_codex_runtime_credentials(
            force_refresh=force_refresh,
            refresh_if_expiring=True,
        )
        bearer = str((resolved or {}).get("api_key") or "").strip()
        if not bearer:
            raise RuntimeError(
                "No OpenAI Codex OAuth credentials found. Run "
                "`hermes auth add openai-codex --type oauth` first."
            )
        base_url = str((resolved or {}).get("base_url") or _DEFAULT_BASE_URL).strip()
        # Codex's Cloudflare edge requires first-party originator headers; the
        # helper also extracts ChatGPT-Account-ID from the OAuth JWT when present.
        from agent.auxiliary_client import _codex_cloudflare_headers

        return UpstreamCredential(
            bearer=bearer,
            base_url=base_url.rstrip("/") or _DEFAULT_BASE_URL,
            extra_headers=_codex_cloudflare_headers(bearer),
        )


__all__ = ["OpenAICodexAdapter"]
