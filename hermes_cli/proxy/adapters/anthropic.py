"""Anthropic Claude OAuth upstream adapter.

Forwards Anthropic Messages API requests using Hermes-managed Anthropic OAuth
credentials (Claude Pro/Max subscription), so external apps can hit Claude
through the local proxy with any bearer token.

Unlike the OpenAI-compatible providers (nous, xai), Anthropic's OAuth endpoint
requires more than a bearer swap:
  - specific beta + version + user-agent headers (see `extra_headers`)
  - the request body's system prompt must lead with the Claude Code identity
    block, or OAuth traffic is intermittently rejected/misrouted
    (see `transform_body`). This mirrors agent/anthropic_adapter.py.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Dict, FrozenSet, Optional

from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

_POOL_PROVIDER = "anthropic"
_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"

# Only the Messages API is forwarded. (Anthropic OAuth does not serve the
# OpenAI-compatible /chat/completions surface.)
_ALLOWED_PATHS: FrozenSet[str] = frozenset({"/messages"})

# Beta headers required for OAuth/subscription auth. Matches what Claude Code
# sends and what agent/anthropic_adapter.py uses for OAuth traffic.
_OAUTH_BETAS = "claude-code-20250219,oauth-2025-04-20"
_ANTHROPIC_VERSION = "2023-06-01"
_CLAUDE_CODE_VERSION_FALLBACK = "2.1.74"

# The Claude Code system identity block. Anthropic's OAuth infra expects the
# system prompt to lead with this or it misbehaves on subscription tokens.
_CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."

_cc_version_cache: Optional[str] = None


def _claude_code_version() -> str:
    global _cc_version_cache
    if _cc_version_cache is None:
        import subprocess as _sp

        _cc_version_cache = _CLAUDE_CODE_VERSION_FALLBACK
        for cmd in ("claude", "claude-code"):
            try:
                r = _sp.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
                if r.returncode == 0 and r.stdout.strip():
                    v = r.stdout.strip().split()[0]
                    if v and v[0].isdigit():
                        _cc_version_cache = v
                        break
            except Exception:
                pass
    return _cc_version_cache


class AnthropicOAuthAdapter(UpstreamAdapter):
    """Proxy upstream for Anthropic Claude via Hermes-managed OAuth credentials."""

    auth_hint = "hermes auth add anthropic --type oauth"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool: Optional[CredentialPool] = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def display_name(self) -> str:
        return "Anthropic Claude OAuth"

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
                    "No Anthropic OAuth credentials found. Run "
                    "`hermes auth add anthropic --type oauth` first."
                )
            entry = pool.select()
            if entry is None:
                raise RuntimeError(
                    "No available Anthropic OAuth credentials found. Run "
                    "`hermes auth reset anthropic` or re-authenticate with "
                    "`hermes auth add anthropic --type oauth`."
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
            if status_code == 429:
                refreshed = pool.mark_exhausted_and_rotate(status_code=status_code)
            else:
                refreshed = pool.try_refresh_current()
                if refreshed is None:
                    refreshed = pool.mark_exhausted_and_rotate(status_code=status_code)
            if refreshed is None:
                return None
            retry_cred = self._credential_from_entry(refreshed)
            if retry_cred.bearer == failed_credential.bearer:
                return None
            logger.info(
                "proxy: Anthropic upstream returned %s; retrying with rotated pool credential",
                status_code,
            )
            return retry_cred

    # --- optional server hooks (see server.py) ---

    def extra_headers(self, cred: UpstreamCredential) -> Dict[str, str]:
        """Headers Anthropic OAuth requires on every Messages request."""
        return {
            "anthropic-version": _ANTHROPIC_VERSION,
            "anthropic-beta": _OAUTH_BETAS,
            "user-agent": f"claude-cli/{_claude_code_version()} (external, cli)",
        }

    def transform_body(self, rel_path: str, body: bytes) -> bytes:
        """Prepend the Claude Code system identity block to the request body.

        Anthropic's OAuth endpoint expects the system prompt to lead with the
        Claude Code identity; without it, subscription tokens intermittently
        get rejected. Non-JSON or unexpected shapes pass through unchanged.
        """
        if rel_path != "/messages" or not body:
            return body
        try:
            payload = json.loads(body)
        except Exception:
            return body
        if not isinstance(payload, dict):
            return body

        cc_block = {"type": "text", "text": _CLAUDE_CODE_SYSTEM_PREFIX}
        system = payload.get("system")
        if isinstance(system, list):
            # Already-prefixed requests must not double-prefix.
            if not (system and isinstance(system[0], dict)
                    and system[0].get("text") == _CLAUDE_CODE_SYSTEM_PREFIX):
                payload["system"] = [cc_block] + system
        elif isinstance(system, str) and system:
            if system != _CLAUDE_CODE_SYSTEM_PREFIX:
                payload["system"] = [cc_block, {"type": "text", "text": system}]
        else:
            payload["system"] = [cc_block]

        return json.dumps(payload).encode("utf-8")

    # --- internals (mirror xai adapter) ---

    def _load_pool(self) -> Optional[CredentialPool]:
        try:
            return load_pool(_POOL_PROVIDER)
        except Exception as exc:
            logger.warning("proxy: failed to load Anthropic OAuth credential pool: %s", exc)
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
                "Anthropic OAuth credential pool entry did not contain an access token. "
                "Re-authenticate with `hermes auth add anthropic --type oauth`."
            )
        base_url = (
            getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", None)
            or _DEFAULT_BASE_URL
        )
        base_url = str(base_url or _DEFAULT_BASE_URL).strip().rstrip("/")
        return UpstreamCredential(
            bearer=bearer,
            base_url=base_url or _DEFAULT_BASE_URL,
            expires_at=getattr(entry, "expires_at", None),
        )


__all__ = ["AnthropicOAuthAdapter"]
