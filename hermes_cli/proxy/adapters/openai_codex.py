"""OpenAI Codex subscription adapter using Hermes-managed ChatGPT OAuth."""

from __future__ import annotations

import logging
import threading
from typing import FrozenSet, Optional

from agent.auxiliary_client import _codex_cloudflare_headers
from hermes_cli.auth import get_codex_auth_status, resolve_codex_runtime_credentials
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

_ALLOWED_PATHS: FrozenSet[str] = frozenset({"/responses"})


class OpenAICodexAdapter(UpstreamAdapter):
    """Raw Responses proxy for an OpenAI Codex subscription."""

    auth_hint = "hermes auth add openai-codex --type oauth"

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex"

    @property
    def health_contract(self) -> str:
        return "hermes-codex-responses-v1"

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return _ALLOWED_PATHS

    def is_authenticated(self) -> bool:
        try:
            status = get_codex_auth_status()
            return bool(status.get("logged_in") and not status.get("rate_limited"))
        except Exception:
            return False

    def get_credential(self) -> UpstreamCredential:
        return self._get_credential(force_refresh=False)

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
    ) -> Optional[UpstreamCredential]:
        _ = failed_credential
        if status_code != 401:
            return None
        logger.info("proxy: Codex upstream rejected bearer; refreshing once")
        return self._get_credential(force_refresh=True)

    def _get_credential(self, *, force_refresh: bool) -> UpstreamCredential:
        with self._lock:
            try:
                resolved = resolve_codex_runtime_credentials(
                    force_refresh=force_refresh
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to resolve OpenAI Codex credentials: {exc}"
                ) from exc

            bearer = str(resolved.get("api_key") or "").strip()
            base_url = str(resolved.get("base_url") or "").strip().rstrip("/")
            if not bearer or not base_url:
                raise RuntimeError(
                    "OpenAI Codex authentication is unavailable. Run "
                    "`hermes auth add openai-codex --type oauth` first."
                )

            return UpstreamCredential(
                bearer=bearer,
                base_url=base_url,
                headers=tuple(_codex_cloudflare_headers(bearer).items()),
            )


__all__ = ["OpenAICodexAdapter"]
