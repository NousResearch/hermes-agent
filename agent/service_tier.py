"""Priority eligibility and provider-reported tiers, independent of prompt state."""

from typing import Any
from urllib.parse import urlsplit


_XAI_API_PROVIDERS = frozenset({"xai", "x-ai", "x.ai", "grok"})


def supports_xai_priority(model: str | None) -> bool:
    """Known API models only; a new Grok version does not imply Priority support."""
    from agent.model_metadata import is_grok_46_family

    name = str(model or "").strip().lower().replace("_", "-").rsplit("/", 1)[-1]
    return is_grok_46_family(name) or name == "grok-4.7"


def xai_priority_route(model: str | None, provider: str | None, base_url: str | None) -> bool:
    """Only the direct API-key route; OAuth, regional hosts and proxies are separate products."""
    if not supports_xai_priority(model):
        return False
    if provider and str(provider).strip().lower() not in _XAI_API_PROVIDERS:
        return False
    try:
        url = urlsplit(str(base_url or ""))
        return (
            url.scheme == "https" and url.hostname == "api.x.ai" and url.port in (None, 443)
            and url.path.rstrip("/") == "/v1" and not (url.username or url.password or url.query or url.fragment)
        )
    except ValueError:
        return False


def filter_xai_service_tier(kwargs: dict, *, model: str, provider: str | None,
                            base_url: str | None, is_xai: bool) -> None:
    """Sanitize both SDK representations; extra_body wins on the actual wire."""
    if not (is_xai or supports_xai_priority(model)):
        return
    extra = kwargs.get("extra_body")
    tier = extra.get("service_tier", kwargs.get("service_tier")) if isinstance(extra, dict) else kwargs.get("service_tier")
    kwargs.pop("service_tier", None)
    if isinstance(extra, dict) and "service_tier" in extra:
        kwargs["extra_body"] = {key: value for key, value in extra.items() if key != "service_tier"}
    if tier == "priority" and xai_priority_route(model, provider, base_url):
        kwargs["service_tier"] = tier


def served_service_tier(response: Any) -> str | None:
    """The response is authoritative; a requested tier is never evidence of service."""
    value = response.get("service_tier") if isinstance(response, dict) else getattr(response, "service_tier", None)
    return value.strip() if isinstance(value, str) and value.strip() else None
