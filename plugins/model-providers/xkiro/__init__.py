"""xKiro model-provider profiles.

xKiro exposes one multi-model catalog through OpenAI Chat Completions and
Anthropic Messages. Wire model IDs stay in their full ``vendor/model`` form.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlparse

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

_XKIRO_BASE = "https://api.xkiro.com/v1"
_XKIRO_MODELS_URL = f"{_XKIRO_BASE}/models"
_XKIRO_ENV = ("XKIRO_API_KEY", "XKIRO_BASE_URL")
_XKIRO_PRICING_TTL_SECONDS = 300.0


def _models_url(base_url: str | None) -> str:
    caller_base = (base_url or "").strip().rstrip("/")
    if not caller_base:
        return _XKIRO_MODELS_URL
    # xKiro's Anthropic SDK guide correctly recommends the origin without
    # ``/v1`` because that SDK appends it. Catalog URLs still require ``/v1``.
    parsed = urlparse(caller_base)
    if parsed.hostname == "api.xkiro.com" and not parsed.path.rstrip("/"):
        caller_base = f"{caller_base}/v1"
    return f"{caller_base}/models"


def _fetch_xkiro_catalog(
    *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0,
) -> list[dict[str, Any]] | None:
    """Fetch the public chat catalog; ``None`` means unavailable or malformed."""
    from hermes_cli.urllib_security import open_credentialed_url

    request = urllib.request.Request(_models_url(base_url))
    request.add_header("Accept", "application/json")
    request.add_header("User-Agent", _profile_user_agent())
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    try:
        with open_credentialed_url(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("fetch_models(xkiro): %s", exc)
        return None

    items = payload if isinstance(payload, list) else payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None
    return [
        item for item in items
        if isinstance(item, dict) and item.get("modality") in (None, "chat")
    ]


def _per_token_usd(value: Any) -> str | None:
    """Convert a non-negative USD-per-million value to a precise per-token string."""
    if value is None or value == "":
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not amount.is_finite() or amount < 0:
        return None
    return format(amount / Decimal(1_000_000), "f")


def _catalog_pricing(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Normalize xKiro's documented USD/MTok pricing into Hermes picker units."""
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        model_id = item.get("id")
        pricing = item.get("pricing")
        if not isinstance(model_id, str) or not model_id.strip() or not isinstance(pricing, dict):
            continue
        model_id = model_id.strip()
        if pricing.get("currency") != "USD" or pricing.get("unit") != "per_1m_tokens":
            continue

        prompt = _per_token_usd(pricing.get("input"))
        completion = _per_token_usd(pricing.get("output"))
        if prompt is None and completion is None:
            continue
        entry: dict[str, Any] = {"prompt": prompt or "", "completion": completion or ""}
        for source, target in (("cache_read", "input_cache_read"), ("cache_write", "input_cache_write")):
            normalized = _per_token_usd(pricing.get(source))
            if normalized is not None:
                entry[target] = normalized
        result[model_id] = entry
    return result


class XKiroProfile(ProviderProfile):
    """xKiro's shared catalog plus OpenAI-compatible inference profile."""

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0,
    ) -> list[str] | None:
        items = _fetch_xkiro_catalog(api_key=api_key, base_url=base_url, timeout=timeout)
        if items is None:
            return None
        return list(dict.fromkeys(
            item["id"].strip()
            for item in items
            if isinstance(item.get("id"), str) and item["id"].strip()
        ))

    def fetch_model_pricing(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0,
    ) -> dict[str, dict[str, Any]] | None:
        items = _fetch_xkiro_catalog(api_key=api_key, base_url=base_url, timeout=timeout)
        return None if items is None else _catalog_pricing(items)


xkiro = XKiroProfile(
    name="xkiro",
    aliases=("xkiro-ai",),
    api_mode="chat_completions",
    env_vars=_XKIRO_ENV,
    display_name="xKiro",
    description="xKiro — multi-model API gateway",
    signup_url="https://xkiro.com/",
    base_url=_XKIRO_BASE,
    models_url=_XKIRO_MODELS_URL,
    prefer_live_model_catalog=True,
    public_model_catalog=True,
    default_aux_model="qwen/qwen3.5-flash:free",
    pricing_cache_ttl_seconds=_XKIRO_PRICING_TTL_SECONDS,
)

xkiro_anthropic = XKiroProfile(
    name="xkiro-anthropic",
    aliases=("xkiro-claude",),
    api_mode="anthropic_messages",
    env_vars=_XKIRO_ENV,
    display_name="xKiro (Anthropic)",
    description="xKiro — multi-model API via Anthropic Messages",
    signup_url="https://xkiro.com/",
    base_url=_XKIRO_BASE,
    models_url=_XKIRO_MODELS_URL,
    prefer_live_model_catalog=True,
    public_model_catalog=True,
    default_aux_model="qwen/qwen3.5-flash:free",
    pricing_cache_ttl_seconds=_XKIRO_PRICING_TTL_SECONDS,
    preserve_anthropic_model_id=True,
)

register_provider(xkiro)
register_provider(xkiro_anthropic)
