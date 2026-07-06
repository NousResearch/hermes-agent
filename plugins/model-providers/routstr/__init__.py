"""Routstr provider profile.

Routstr is a decentralized OpenAI-compatible AI inference router. Users can
point Hermes at the official Routstr node (api.routstr.com), the Non-KYC AI
node (api.nonkycai.com), or any private Routstr-compatible endpoint. The API
key is passed as a Bearer token and the /models endpoint returns an
OpenRouter-shaped catalog with per-token USD pricing.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


def _format_price_per_mtok(per_token_str: str) -> str:
    """Convert a per-token price string to a $/Mtok label."""
    try:
        val = float(per_token_str)
    except (TypeError, ValueError):
        return "?"
    if val == 0:
        return "free"
    per_m = val * 1_000_000
    return f"${per_m:.2f}"


class RoutstrProfile(ProviderProfile):
    """Routstr — decentralized OpenAI-compatible inference router."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the live model list from a Routstr node's /models endpoint."""
        effective_base = (base_url or self.base_url or "").rstrip("/")
        if not effective_base:
            return None
        url = effective_base + "/models"
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", f"hermes-cli/{_HERMES_VERSION}")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            items = data if isinstance(data, list) else data.get("data", [])
            return [m["id"] for m in items if isinstance(m, dict) and "id" in m]
        except Exception as exc:
            logger.debug("fetch_models(routstr, %s): %s", url, exc)
            return None

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Routstr-specific extra_body fields.

        Routstr supports provider preference routing via extra_body.provider
        when an upstream aggregator is used. For now we keep this minimal and
        only forward session_id if present.
        """
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        return body


routstr = RoutstrProfile(
    name="routstr",
    aliases=("routstr-ai",),
    display_name="Routstr",
    description="Routstr — decentralized OpenAI-compatible AI inference router",
    signup_url="https://chat.routstr.com/?tab=apikeys",
    env_vars=("ROUTSTR_API_KEY", "ROUTSTR_BASE_URL"),
    base_url="https://api.routstr.com/v1",
    auth_type="api_key",
    hostname="api.routstr.com",
    supports_health_check=True,
    default_aux_model="gemini-3.1-flash-lite-preview",
    fallback_models=(
        "claude-sonnet-5",
        "gpt-5.5",
        "deepseek-v4-pro",
        "gemini-3.5-flash",
    ),
)

register_provider(routstr)
