"""Tinfoil.sh provider profile — HPKE end-to-end encrypted AI inference.

Tinfoil.sh provides secure, end-to-end encrypted connections to AI models
using HPKE (RFC 9180) secure enclaves. This provider plugin integrates the
Tinfoil.sh SDK into the Hermes Agent workflow.

The SDK transparently handles:
  - HPKE protocol handshake for end-to-end encryption
  - Secure enclave routing for inference
  - Connection-time verification for audit compliance
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from providers import register_provider
from providers.base import ProviderProfile
from hermes_cli.models import _HERMES_USER_AGENT
from agent.model_metadata import _resolve_requests_verify

logger = logging.getLogger(__name__)

# Base inference endpoint for Tinfoil.sh — all model traffic routes
# through this gateway, which handles HPKE encryption and secure enclave
# routing transparently.
_TINFOIL_BASE_URL = "https://inference.tinfoil.sh/v1"

# Well-known Tinfoil.sh model endpoints.
# Each model name maps to a specific inference endpoint behind the
# HPKE-encrypted gateway. Users can override these in config.yaml
# under the ``tinfoil.endpoints`` section.
# This is also the fallback list used when the live API is unreachable.
_TINFOIL_FALLBACK_ENDPOINTS: dict[str, str] = {
    "kimi-k2-6": "kimi-k2-6",
    "glm-5-1": "glm-5-1",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "gemma4-31b": "gemma4-31b",
    "qwen3-vl-30b": "qwen3-vl-30b",
    "llama3-3-70b": "llama3-3-70b",
    "gpt-oss-120b": "gpt-oss-120b",
}

# Process-level cache of the live model list. Populated on the first
# successful ``fetch_models`` call and reused thereafter so model discovery
# doesn't hit the network on every resolution. Left as ``None`` on failure
# so a transient error doesn't poison the cache — the caller falls back to
# the static ``fallback_models`` list and discovery is retried next call.
_TINFOIL_MODEL_CACHE: list[str] | None = None


def _tinfoil_headers() -> dict[str, str]:
    """Build Tinfoil-specific request headers.

    Returns an ``X-Tinfoil-Mode`` header to signal the HPKE encryption
    mode to the gateway. When the ``X-Tinfoil-Sdk-Version`` header is
    present, the gateway can verify the client SDK version for
    compatibility.
    """
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    sdk_version = os.environ.get("TINFOIL_SDK_VERSION", "")
    if sdk_version:
        headers["X-Tinfoil-Sdk-Version"] = sdk_version
    return headers


class TinfoilProfile(ProviderProfile):
    """Tinfoil.sh provider — HPKE end-to-end encrypted AI inference.

    Routes requests through the Tinfoil.sh gateway which handles:
      - HPKE (RFC 9180) protocol handshake
      - Secure enclave routing
      - Connection-time verification
      - End-to-end encryption of all inference payloads

    Model-specific endpoints are configured via the ``tinfoil.endpoints``
    section in ``config.yaml``, or fall back to the built-in mappings.
    Automatic model discovery hits the OpenAI-compatible ``/v1/models``
    endpoint and filters for chat-capable models.
    """

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        global _TINFOIL_MODEL_CACHE
        if _TINFOIL_MODEL_CACHE is not None:
            return _TINFOIL_MODEL_CACHE

        url = f"{self.base_url.rstrip('/')}/models"
        headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": _HERMES_USER_AGENT,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        for k, v in self.default_headers.items():
            headers.setdefault(k, v)

        verify = _resolve_requests_verify()
        try:
            with httpx.Client(verify=verify, timeout=timeout) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            items = data if isinstance(data, list) else data.get("data", [])
            chat_models = [
                m for m in items
                if isinstance(m, dict) and m.get("id") and m.get("type") == "chat"
            ]
            model_ids = [m["id"] for m in chat_models]
            # Only cache a non-empty discovery result. Caching an empty list
            # would mask the static fallback_models for the rest of the
            # process if the gateway briefly returned no chat models.
            if model_ids:
                _TINFOIL_MODEL_CACHE = model_ids
            return model_ids
        except Exception as exc:
            logger.debug("Tinfoil fetch_models: %s", exc)
            return None

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Attach Tinfoil-specific fields to the request body.

        Wraps the model selection so the gateway knows which secure
        enclave to route to.  For models discovered via the live API
        but not in the fallback map, the model name is used as the
        endpoint path directly.
        """
        body: dict[str, Any] = {}
        model = (context.get("model") or "").strip().lower()

        # Strip provider prefix if present (e.g. "tinfoil/kimi-k2-6" -> "kimi-k2-6")
        if model.startswith("tinfoil/"):
            model = model[len("tinfoil/") :]

        # Map model name to Tinfoil endpoint path if configured
        endpoints: dict[str, str] = context.get("tinfoil_endpoints") or {}
        endpoint_path = endpoints.get(model)
        if endpoint_path is not None:
            body["tinfoil_endpoint"] = endpoint_path
        elif model in _TINFOIL_FALLBACK_ENDPOINTS:
            body["tinfoil_endpoint"] = _TINFOIL_FALLBACK_ENDPOINTS[model]
        elif model:
            # Use the model name itself as the endpoint — needed for
            # models discovered via the live /v1/models API that aren't
            # in the static fallback map yet.
            body["tinfoil_endpoint"] = model

        return body

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Instruct the transport to include Tinfoil-specific headers.

        The ``default_headers`` from the profile are forwarded to the
        OpenAI-compatible SDK constructor so every request carries the
        appropriate Tinfoil metadata.
        """
        extra_headers = _tinfoil_headers()
        top_kwargs: dict[str, Any] = {}
        if extra_headers:
            top_kwargs["extra_headers"] = extra_headers
        return {}, top_kwargs


tinfoil = TinfoilProfile(
    name="tinfoil",
    aliases=("tinfoil-sh", "tinfoil.sh"),
    api_mode="tinfoil_ehbp",
    env_vars=("TINFOIL_API_KEY",),
    display_name="Tinfoil.sh",
    description="Tinfoil.sh — HPKE end-to-end encrypted AI inference",
    signup_url="https://tinfoil.sh",
    base_url=_TINFOIL_BASE_URL,
    fallback_models=(
        "kimi-k2-6",
        "glm-5-1",
        "deepseek-v4-pro",
        "gemma4-31b",
        "qwen3-vl-30b",
        "llama3-3-70b",
        "gpt-oss-120b",
    ),
    default_headers=_tinfoil_headers(),
    auth_type="api_key",
    default_aux_model="gemma4-31b",
)

register_provider(tinfoil)
