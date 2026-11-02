"""Provider-agnostic billing/credit recovery links for billing-classified failures.

Detection lives in :mod:`agent.error_classifier` (``FailoverReason.billing``); the
:class:`BillingBlock` rides the turn result and the gateway ``message.complete`` event
so every surface renders one structured signal instead of re-parsing error text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from utils import base_url_host_matches


@dataclass
class BillingBlock:
    """Structured billing-wall descriptor shared across every surface. ``is_nous`` is the routing bit:
    Nous has an in-app billing surface (Settings → Billing, ``/topup``) preferred over ``billing_url``;
    third-party providers have none, so ``billing_url`` is the deep link needed."""

    provider: str
    provider_label: str
    model: str
    billing_url: Optional[str]
    is_nous: bool
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _Provider:
    label: str
    url: str
    slugs: tuple[str, ...]
    hosts: tuple[str, ...] = ()


# Single source of truth: slug(s) + base_url host(s) → curated "add credits" page.
# Hosts back the OpenAI-compatible fallback where the slug is a generic bucket but
# base_url reveals the real upstream. Unknown providers get a readable label, no URL.
_PROVIDERS: tuple[_Provider, ...] = (
    _Provider("OpenAI", "https://platform.openai.com/settings/organization/billing", ("openai",), ("api.openai.com",)),
    _Provider("Anthropic", "https://console.anthropic.com/settings/billing", ("anthropic",), ("api.anthropic.com",)),
    _Provider("OpenRouter", "https://openrouter.ai/settings/credits", ("openrouter",), ("openrouter.ai",)),
    _Provider("xAI", "https://console.x.ai/team/default/billing", ("xai", "xai-oauth"), ("api.x.ai",)),
    _Provider("DeepSeek", "https://platform.deepseek.com/top_up", ("deepseek",), ("api.deepseek.com",)),
    _Provider("Groq", "https://console.groq.com/settings/billing", ("groq",), ("api.groq.com",)),
    _Provider("Mistral", "https://console.mistral.ai/billing", ("mistral",), ("api.mistral.ai",)),
    _Provider("Together AI", "https://api.together.ai/settings/billing", ("together",), ("api.together.ai", "api.together.xyz")),
    _Provider("Fireworks AI", "https://fireworks.ai/account/billing", ("fireworks",), ("fireworks.ai",)),
    _Provider("Perplexity", "https://www.perplexity.ai/settings/api", ("perplexity",), ("perplexity.ai",)),
    _Provider("Google AI", "https://aistudio.google.com/app/billing", ("google", "gemini"), ("generativelanguage.googleapis.com",)),
    _Provider("Cohere", "https://dashboard.cohere.com/billing", ("cohere",)),
    _Provider("Moonshot AI", "https://platform.moonshot.ai/console/pay", ("moonshot",)),
    _Provider("NVIDIA", "https://build.nvidia.com/settings/billing", ("nvidia",)),
)

_BY_SLUG: dict[str, _Provider] = {slug: p for p in _PROVIDERS for slug in p.slugs}


def is_nous_inference_route(provider: str, base_url: str) -> bool:
    """True when the failing route is the Nous-managed inference gateway."""
    return (provider or "").strip().lower() == "nous" or base_url_host_matches(str(base_url or ""), "inference-api.nousresearch.com")


def _nous_billing_url() -> Optional[str]:
    """Best-effort Nous portal billing URL (text-surface fallback; Nous prefers the in-app flow)."""
    try:
        from hermes_cli.nous_account import nous_portal_billing_url
        return nous_portal_billing_url(None)
    except Exception:
        return "https://portal.nousresearch.com/billing"


def _resolve_provider_link(slug: str, base_url: str) -> tuple[str, Optional[str]]:
    """Resolve ``(label, url)``: exact slug → base_url host → readable-label fallback."""
    base = str(base_url or "")
    hit = _BY_SLUG.get(slug) or next(
        (p for p in _PROVIDERS if any(base_url_host_matches(base, host) for host in p.hosts)), None
    )
    if hit:
        return hit.label, hit.url
    return slug.replace("_", " ").replace("-", " ").strip().title() or "your provider", None


def build_billing_block(*, provider: str, base_url: str, model: str, message: str = "") -> BillingBlock:
    """Billing descriptor for a billing-classified failure; ``message`` (agent-loop guidance) passes through unchanged."""
    slug = (provider or "").strip().lower()
    model = (model or "").strip()
    if is_nous_inference_route(slug, base_url):
        return BillingBlock(slug or "nous", "Nous Portal", model, _nous_billing_url(), True, message or "")
    label, url = _resolve_provider_link(slug, base_url)
    return BillingBlock(slug, label, model, url, False, message or "")
