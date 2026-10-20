"""openzoo provider profile.

openzoo (https://openzoo.fun, npm ``openzoo``) is a local proxy that pays for
LLM inference per call over x402 — on-chain micropayments from a burner
wallet the proxy holds. There is no account and no API key. The user runs
``npx openzoo``; the proxy listens on ``http://localhost:8402/v1`` as an
OpenAI-compatible endpoint and forwards each paid call to the public gateway
(``https://x402-tokens.fly.dev/v1``), which answers HTTP 402 to anything
unpaid. Only the local proxy can settle those 402s, so this profile's
``base_url`` is the LOCAL proxy, never the gateway.

The proxy ignores the bearer value (payment replaces the key), but Hermes'
API-key resolver refuses to run a provider with an empty credential, so
``OPENZOO_API_KEY`` must be set to any non-empty value — ``sk-openzoo`` is
the documented placeholder. The value only ever travels to the loopback
proxy (or, for a public tunnel, as the tunnel's own printed bearer over the
tunnel's TLS); it is never forwarded to the upstream gateway.

Model discovery: ``GET /v1/models`` is free and returns OpenRouter-shaped
rows. Rows carrying a ``kind`` field (``image`` / ``video``) are media
models and are skipped so the chat picker only lists chat-completions
targets. Prices in that catalog are a ceiling (OpenRouter-direct basis);
the gateway charges at most that. Receipts land in ``~/.openzoo/proxy.log``.
"""

from __future__ import annotations

import json
import logging
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

DEFAULT_OPENZOO_BASE_URL = "http://localhost:8402/v1"

# Any non-empty bearer satisfies the proxy; documented placeholder.
OPENZOO_PLACEHOLDER_API_KEY = "sk-openzoo"


class OpenZooProfile(ProviderProfile):
    """openzoo — OpenAI-compatible local proxy that pays per call over x402."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the chat catalog from the local proxy, dropping media rows.

        The catalog endpoint is free (no 402), so no bearer is sent. Rows
        with a ``kind`` field are image/video generation models served on
        other endpoints and must not appear in the chat-completions picker.
        """
        effective_base = (base_url or "").strip() or self.base_url
        if not effective_base:
            return None

        req = urllib.request.Request(effective_base.rstrip("/") + "/models")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())

        from hermes_cli.urllib_security import open_credentialed_url

        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("fetch_models(openzoo): %s", exc)
            return None

        items = data if isinstance(data, list) else data.get("data", [])
        return [
            m["id"]
            for m in items
            if isinstance(m, dict) and m.get("id") and not m.get("kind")
        ]


openzoo = OpenZooProfile(
    name="openzoo",
    aliases=("open-zoo", "zoo"),
    display_name="openzoo",
    description="openzoo — pay-per-call inference over x402 via the local proxy (no API key)",
    signup_url="https://openzoo.fun",
    env_vars=("OPENZOO_API_KEY", "OPENZOO_BASE_URL"),
    base_url=DEFAULT_OPENZOO_BASE_URL,
    auth_type="api_key",
    # The gateway's own router; the cheapest sensible default for side tasks.
    default_aux_model="openzoo/auto",
    # Offline floor for the picker; the live /v1/models catalog is
    # authoritative and is merged in when the proxy is reachable. Every id
    # below is verified against openzoo@0.50.84's published catalog — the
    # proxy uses bare ids, not vendor-prefixed OpenRouter slugs.
    fallback_models=(
        "openzoo/auto",
        "claude-sonnet-5",
        "claude-fable-5",
        "grok-4",
        "deepseek-reasoner",
        "gemini-2.5-pro",
    ),
)

register_provider(openzoo)
