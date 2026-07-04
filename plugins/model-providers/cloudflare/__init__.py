"""Cloudflare Workers AI provider profile."""

from __future__ import annotations

import os
from typing import Any

from agent import cloudflare_workers_ai as cloudflare_catalog
from providers import register_provider
from providers.base import ProviderProfile


def _needs_runtime_resolution(api_key: str | None, base_url: str | None) -> bool:
    normalized = str(base_url or "").strip()
    return (not api_key) or (not normalized) or ("ACCOUNT_ID" in normalized.upper())


def _resolve_credentials(api_key: str | None, base_url: str | None, provider_id: str) -> tuple[str, str]:
    """Resolve Cloudflare Workers AI credentials, substituting CLOUDFLARE_ACCOUNT_ID."""
    if _needs_runtime_resolution(api_key, base_url):
        try:
            from hermes_cli.cloudflare import resolve_cloudflare_runtime_credentials
            creds = resolve_cloudflare_runtime_credentials(provider_id)
            api_key = api_key or str(creds.get("api_key") or "").strip()
            resolved_base_url = str(creds.get("base_url") or "").strip()
            if not base_url or "ACCOUNT_ID" in str(base_url).upper():
                base_url = resolved_base_url
        except Exception:
            pass
    return str(api_key or "").strip(), str(base_url or "").strip().rstrip("/")


class CloudflareProfile(ProviderProfile):
    """Cloudflare Workers AI — native catalog + OpenAI-compatible inference."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        session_id: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        extra_headers: dict[str, str] = {}

        if supports_reasoning:
            if reasoning_config is not None:
                extra_body["reasoning"] = dict(reasoning_config)
                effort = str(reasoning_config.get("effort") or "").strip().lower()
                if effort == "none" or reasoning_config.get("enabled") is False:
                    extra_body["think"] = False
            else:
                extra_body["reasoning"] = {"enabled": True, "effort": "medium"}

        # x-session-affinity routes same-session requests to the same replica
        # so Workers AI's prefix cache stays warm across an agent loop.
        # https://developers.cloudflare.com/workers-ai/features/prompt-caching/
        if session_id:
            extra_headers["x-session-affinity"] = str(session_id)

        return extra_body, ({"extra_headers": extra_headers} if extra_headers else {})

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        api_key, base_url = _resolve_credentials(api_key, base_url, self.name)
        if not api_key or not base_url:
            return None
        entries = cloudflare_catalog.fetch_cloudflare_model_catalog(api_key, base_url, timeout=timeout)
        if not entries:
            return None
        payload = {"result": entries}
        return cloudflare_catalog.cloudflare_model_names(payload)

    def get_model_capabilities(
        self,
        *,
        model: str,
        provider_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
        **context: Any,
    ):
        api_key, base_url = _resolve_credentials(api_key, base_url, provider_id or self.name)
        if not api_key or not base_url:
            return None
        return cloudflare_catalog.get_cloudflare_model_capabilities(api_key, base_url, model, timeout=timeout)


cloudflare = CloudflareProfile(
    name="cloudflare",
    aliases=(
        "cloudflare-workers-ai",
        "workers-ai",
        "workersai",
        "cf",
        "cf-ai",
        "custom:cloudflare",
        "custom:cloudflare-workers-ai",
        "custom:workers-ai",
        "custom:workersai",
    ),
    display_name="Cloudflare Workers AI",
    description="Cloudflare Workers AI (native catalog + OpenAI-compatible inference)",
    signup_url="https://developers.cloudflare.com/workers-ai/",
    env_vars=("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_BASE_URL", "CLOUDFLARE_ACCOUNT_ID"),
    base_url="https://api.cloudflare.com/client/v4/accounts/ACCOUNT_ID/ai/v1",
    auth_type="api_key",
)

register_provider(cloudflare)
