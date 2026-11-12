"""OpenAI Codex subscription adapter for the local model proxy."""

from __future__ import annotations

import logging

from agent.codex_headers import codex_cloudflare_headers, is_official_codex_base_url
from hermes_cli.auth_codex import resolve_codex_runtime_credentials
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)


class CodexAdapter(UpstreamAdapter):
    """Use the existing Codex subscription credentials without file refresh."""

    @property
    def name(self) -> str:
        return "codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex (ChatGPT subscription)"

    @property
    def allowed_paths(self) -> frozenset[str]:
        return frozenset({"/responses"})

    def is_authenticated(self) -> bool:
        """Check an existing subscription token without reading a refresh token."""
        try:
            result = resolve_codex_runtime_credentials(read_only=True)
        except Exception:  # an auth-store failure means "not authenticated", never a crash
            logger.debug("codex auth probe failed", exc_info=True)
            return False
        return bool(result.get("api_key") and is_official_codex_base_url(result.get("base_url", "")))

    def get_credential(self) -> UpstreamCredential:
        """Resolve the official Codex endpoint with read-only Hermes auth access.

        The parent process must not let Pi's HOME see ``.hermes`` or the Codex
        credential store. A refresh is deliberately not allowed here: this
        adapter runs in the model proxy, not in the sandboxed Pi process.
        """
        result = resolve_codex_runtime_credentials(read_only=True)
        api_key = result.get("api_key", "")
        base_url = result.get("base_url", "")
        if not api_key:
            raise RuntimeError("Codex auth is not configured for this Hermes account")
        if not is_official_codex_base_url(base_url):
            raise RuntimeError("refusing to forward the Codex token to a non-official Codex endpoint")
        return UpstreamCredential(
            bearer=api_key,
            base_url=base_url,
            headers=codex_cloudflare_headers(api_key, base_url=base_url),
            expires_at=result.get("last_refresh"),
        )
