from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Literal, Optional

from agent.model_metadata import fetch_endpoint_model_metadata, fetch_model_metadata
from utils import base_url_host_matches

DEFAULT_PRICING = {"input": 0.0, "output": 0.0}

_ZERO = Decimal("0")
_ONE_MILLION = Decimal("1000000")
_NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"

CostStatus = Literal["actual", "estimated", "included", "unknown"]
CostSource = Literal[
    "provider_cost_api",
    "provider_generation_api",
    "provider_models_api",
    "official_docs_snapshot",
    "user_override",
    "custom_contract",
    "none",
]


@dataclass(frozen=True)
class CanonicalUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    request_count: int = 1
    raw_usage: Optional[dict[str, Any]] = None

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


@dataclass(frozen=True)
class BillingRoute:
    provider: str
    model: str
    base_url: str = ""
    billing_mode: str = "unknown"


@dataclass(frozen=True)
class PricingEntry:
    input_cost_per_million: Optional[Decimal] = None
    output_cost_per_million: Optional[Decimal] = None
    cache_read_cost_per_million: Optional[Decimal] = None
    cache_write_cost_per_million: Optional[Decimal] = None
    request_cost: Optional[Decimal] = None
    # Optional request-context tier. Some providers (Sakana Fugu Ultra) price
    # the entire request at a higher tier once prompt context crosses a threshold.
    context_tier_threshold_tokens: Optional[int] = None
    context_tier_input_cost_per_million: Optional[Decimal] = None
    context_tier_output_cost_per_million: Optional[Decimal] = None
    context_tier_cache_read_cost_per_million: Optional[Decimal] = None
    context_tier_cache_write_cost_per_million: Optional[Decimal] = None
    source: CostSource = "none"
    source_url: Optional[str] = None
    pricing_version: Optional[str] = None
    fetched_at: Optional[datetime] = None


@dataclass(frozen=True)
class CostResult:
    amount_usd: Optional[Decimal]
    status: CostStatus
    source: CostSource
    label: str
    fetched_at: Optional[datetime] = None
    pricing_version: Optional[str] = None
    notes: tuple[str, ...] = ()


_UTC_NOW = lambda: datetime.now(timezone.utc)


# Official docs snapshot entries. Models whose published pricing and cache
# semantics are stable enough to encode exactly.
_SAKANA_FUGU_ULTRA_PRICING = PricingEntry(
    input_cost_per_million=Decimal("5.00"),
    output_cost_per_million=Decimal("30.00"),
    cache_read_cost_per_million=Decimal("0.50"),
    context_tier_threshold_tokens=272_000,
    context_tier_input_cost_per_million=Decimal("10.00"),
    context_tier_output_cost_per_million=Decimal("45.00"),
    context_tier_cache_read_cost_per_million=Decimal("1.00"),
    source="official_docs_snapshot",
    source_url="https://sakana.ai/fugu/",
    pricing_version="sakana-fugu-ultra-20260615",
)

_OFFICIAL_DOCS_PRICING: Dict[tuple[str, str], PricingEntry] = {
    # ── Anthropic Claude 4.8 ─────────────────────────────────────────────
    # Same $5/$25 base pricing as 4.6/4.7.  Fast-mode variant is a separate
    # model ID with 2x premium (vs the 6x premium on older Opus generations).
    # Source: https://openrouter.ai/anthropic/claude-opus-4.8
    (
        "anthropic",
        "claude-opus-4-8",
    ): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    (
        "anthropic",
        "claude-opus-4-8-fast",
    ): PricingEntry(
        input_cost_per_million=Decimal("10.00"),
        output_cost_per_million=Decimal("50.00"),
        cache_read_cost_per_million=Decimal("1.00"),
        cache_write_cost_per_million=Decimal("12.50"),
        source="official_docs_snapshot",
        source_url="https://openrouter.ai/anthropic/claude-opus-4.8-fast",
        pricing_version="anthropic-pricing-2026-05",
    ),
    # ── Anthropic Claude 4.7 ─────────────────────────────────────────────
    # Opus 4.5/4.6/4.7 share $5/$25 pricing (new tokenizer, up to 35% more
    # tokens for the same text).
    # Source: https://platform.claude.com/docs/en/about-claude/pricing
    (
        "anthropic",
        "claude-opus-4-7",
    ): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    (
        "anthropic",
        "claude-opus-4-7-20250507",
    ): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    # ── Anthropic Claude 4.6 ─────────────────────────────────────────────
    (
        "anthropic",
        "claude-opus-4-6",
    ): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    (
        "anthropic",
        "claude-opus-4-6-20250414",
    ): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    (
        "anthropic",
        "claude-sonnet-4-6",
    ): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    (
        "anthropic",
        "claude-sonnet-4-6-20250414",
    ): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    # ── Anthropic Claude 4.5 ─────────────────────────────────────────────
    (
        "anthropic",
        "claude-opus-4-5",
    ): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    (
        "anthropic",
        "claude-sonnet-4-5",
    ): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    (
        "anthropic",
        "claude-haiku-4-5",
    ): PricingEntry(
        input_cost_per_million=Decimal("1.00"),
        output_cost_per_million=Decimal("5.00"),
        cache_read_cost_per_million=Decimal("0.10"),
        cache_write_cost_per_million=Decimal("1.25"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    # ── Anthropic Claude 4 / 4.1 ─────────────────────────────────────────
    (
        "anthropic",
        "claude-opus-4-20250514",
    ): PricingEntry(
        input_cost_per_million=Decimal("15.00"),
        output_cost_per_million=Decimal("75.00"),
        cache_read_cost_per_million=Decimal("1.50"),
        cache_write_cost_per_million=Decimal("18.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    (
        "anthropic",
        "claude-sonnet-4-20250514",
    ): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    # OpenAI
    (
        "openai",
        "gpt-4o",
    ): PricingEntry(
        input_cost_per_million=Decimal("2.50"),
        output_cost_per_million=Decimal("10.00"),
        cache_read_cost_per_million=Decimal("1.25"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    (
        "openai",
        "gpt-4o-mini",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.15"),
        output_cost_per_million=Decimal("0.60"),
        cache_read_cost_per_million=Decimal("0.075"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    (
        "openai",
        "gpt-4.1",
    ): PricingEntry(
        input_cost_per_million=Decimal("2.00"),
        output_cost_per_million=Decimal("8.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    (
        "openai",
        "gpt-4.1-mini",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.40"),
        output_cost_per_million=Decimal("1.60"),
        cache_read_cost_per_million=Decimal("0.10"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    (
        "openai",
        "gpt-4.1-nano",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.10"),
        output_cost_per_million=Decimal("0.40"),
        cache_read_cost_per_million=Decimal("0.025"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    (
        "openai",
        "o3",
    ): PricingEntry(
        input_cost_per_million=Decimal("10.00"),
        output_cost_per_million=Decimal("40.00"),
        cache_read_cost_per_million=Decimal("2.50"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    (
        "openai",
        "o3-mini",
    ): PricingEntry(
        input_cost_per_million=Decimal("1.10"),
        output_cost_per_million=Decimal("4.40"),
        cache_read_cost_per_million=Decimal("0.55"),
        source="official_docs_snapshot",
        source_url="https://openai.com/api/pricing/",
        pricing_version="openai-pricing-2026-03-16",
    ),
    # ── Anthropic older models (pre-4.5 generation) ────────────────────────
    (
        "anthropic",
        "claude-3-5-sonnet-20241022",
    ): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    (
        "anthropic",
        "claude-3-5-haiku-20241022",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.80"),
        output_cost_per_million=Decimal("4.00"),
        cache_read_cost_per_million=Decimal("0.08"),
        cache_write_cost_per_million=Decimal("1.00"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    (
        "anthropic",
        "claude-3-opus-20240229",
    ): PricingEntry(
        input_cost_per_million=Decimal("15.00"),
        output_cost_per_million=Decimal("75.00"),
        cache_read_cost_per_million=Decimal("1.50"),
        cache_write_cost_per_million=Decimal("18.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    (
        "anthropic",
        "claude-3-haiku-20240307",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.25"),
        output_cost_per_million=Decimal("1.25"),
        cache_read_cost_per_million=Decimal("0.03"),
        cache_write_cost_per_million=Decimal("0.30"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-05",
    ),
    # DeepSeek
    (
        "deepseek",
        "deepseek-chat",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.14"),
        output_cost_per_million=Decimal("0.28"),
        source="official_docs_snapshot",
        source_url="https://api-docs.deepseek.com/quick_start/pricing",
        pricing_version="deepseek-pricing-2026-03-16",
    ),
    (
        "deepseek",
        "deepseek-reasoner",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.55"),
        output_cost_per_million=Decimal("2.19"),
        source="official_docs_snapshot",
        source_url="https://api-docs.deepseek.com/quick_start/pricing",
        pricing_version="deepseek-pricing-2026-03-16",
    ),
    (
        "deepseek",
        "deepseek-v4-pro",
    ): PricingEntry(
        input_cost_per_million=Decimal("1.74"),
        output_cost_per_million=Decimal("3.48"),
        cache_read_cost_per_million=Decimal("0.0145"),
        source="official_docs_snapshot",
        source_url="https://api-docs.deepseek.com/quick_start/pricing",
        pricing_version="deepseek-pricing-2026-05-12",
    ),
    # Sakana Fugu Ultra — product page pricing for fugu-ultra-20260615.
    ("sakana-fugu", "fugu-ultra"): _SAKANA_FUGU_ULTRA_PRICING,
    ("sakana-fugu", "fugu-ultra-20260615"): _SAKANA_FUGU_ULTRA_PRICING,
    # Google Gemini
    (
        "google",
        "gemini-2.5-pro",
    ): PricingEntry(
        input_cost_per_million=Decimal("1.25"),
        output_cost_per_million=Decimal("10.00"),
        source="official_docs_snapshot",
        source_url="https://ai.google.dev/pricing",
        pricing_version="google-pricing-2026-03-16",
    ),
    (
        "google",
        "gemini-2.5-flash",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.15"),
        output_cost_per_million=Decimal("0.60"),
        source="official_docs_snapshot",
        source_url="https://ai.google.dev/pricing",
        pricing_version="google-pricing-2026-03-16",
    ),
    (
        "google",
        "gemini-2.0-flash",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.10"),
        output_cost_per_million=Decimal("0.40"),
        source="official_docs_snapshot",
        source_url="https://ai.google.dev/pricing",
        pricing_version="google-pricing-2026-03-16",
    ),
    # AWS Bedrock — pricing per the Bedrock pricing page.
    # Bedrock charges the same per-token rates as the model provider but
    # through AWS billing.  These are the on-demand prices (no commitment).
    # Source: https://aws.amazon.com/bedrock/pricing/
    (
        "bedrock",
        "anthropic.claude-opus-4-6",
    ): PricingEntry(
        input_cost_per_million=Decimal("15.00"),
        output_cost_per_million=Decimal("75.00"),
        cache_read_cost_per_million=Decimal("1.50"),
        cache_write_cost_per_million=Decimal("18.75"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/bedrock/pricing/",
        pricing_version="bedrock-pricing-2026-04",
    ),
    (
        "bedrock",
        "anthropic.claude-sonnet-4-6",
    ): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/bedrock/pricing/",
        pricing_version="bedrock-pricing-2026-04",
    ),
    (
        "bedrock",
        "anthropic.claude-sonnet-4-5",
    ): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/bedrock/pricing/",
        pricing_version="bedrock-pricing-2026-04",
    ),
    (
        "bedrock",
        "anthropic.claude-haiku-4-5",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.80"),
        output_cost_per_million=Decimal("4.00"),
        cache_read_cost_per_million=Decimal("0.08"),
        cache_write_cost_per_million=Decimal("1.00"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/bedrock/pricing/",
        pricing_version="bedrock-pricing-2026-04",
    ),
    (
        "bedrock",
        "amazon.nova-pro",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.80"),
        output_cost_per_million=Decimal("3.20"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/bedrock/pricing/",
        pricing_version="bedrock-pricing-2026-04",
    ),
    (
        "bedrock",
        "amazon.nova-lite",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.06"),
        output_cost_per_million=Decimal("0.24"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/bedrock/pricing/",
        pricing_version="bedrock-pricing-2026-04",
    ),
    (
        "bedrock",
        "amazon.nova-micro",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.035"),
        output_cost_per_million=Decimal("0.14"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/bedrock/pricing/",
        pricing_version="bedrock-pricing-2026-04",
    ),
    # MiniMax
    (
        "minimax",
        "minimax-m2.7",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.30"),
        output_cost_per_million=Decimal("1.20"),
        source="official_docs_snapshot",
        pricing_version="minimax-pricing-2026-04",
    ),
    (
        "minimax-cn",
        "minimax-m2.7",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.30"),
        output_cost_per_million=Decimal("1.20"),
        source="official_docs_snapshot",
        pricing_version="minimax-pricing-2026-04",
    ),
}


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _usage_to_dict(value: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> Any:
    """Best-effort JSON-like copy of provider usage details.

    Providers can add billable nested usage fields before Hermes knows how to
    aggregate them. Preserve the raw shape so observability/cost tooling can
    inspect those fields instead of losing them at normalization time.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if _depth > 16:
        return "<max-depth>"
    if _seen is None:
        _seen = set()
    obj_id = id(value)
    if obj_id in _seen:
        return "<cycle>"
    _seen.add(obj_id)
    if isinstance(value, dict):
        return {
            str(k): _usage_to_dict(v, _seen=_seen, _depth=_depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_usage_to_dict(v, _seen=_seen, _depth=_depth + 1) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _usage_to_dict(value.model_dump(), _seen=_seen, _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {
                str(k): _usage_to_dict(v, _seen=_seen, _depth=_depth + 1)
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass
    return str(value)


def resolve_billing_route(
    model_name: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BillingRoute:
    provider_name = (provider or "").strip().lower()
    base = (base_url or "").strip().lower()
    model = (model_name or "").strip()
    if not provider_name and "/" in model:
        inferred_provider, bare_model = model.split("/", 1)
        if inferred_provider in {"anthropic", "openai", "google"}:
            provider_name = inferred_provider
            model = bare_model

    if provider_name == "openai-codex":
        return BillingRoute(provider="openai-codex", model=model, base_url=base_url or "", billing_mode="subscription_included")
    if provider_name == "openrouter" or base_url_host_matches(base_url or "", "openrouter.ai"):
        return BillingRoute(provider="openrouter", model=model, base_url=base_url or "", billing_mode="official_models_api")
    if provider_name == "nous" or base_url_host_matches(base_url or "", "inference-api.nousresearch.com"):
        return BillingRoute(provider="nous", model=model, base_url=base_url or _NOUS_DEFAULT_BASE_URL, billing_mode="official_models_api")
    if provider_name == "anthropic":
        return BillingRoute(provider="anthropic", model=model.split("/")[-1], base_url=base_url or "", billing_mode="official_docs_snapshot")
    if provider_name == "openai":
        return BillingRoute(provider="openai", model=model.split("/")[-1], base_url=base_url or "", billing_mode="official_docs_snapshot")
    if provider_name in {"sakana-fugu", "sakana", "fugu", "sakana-ai", "sakanaai"} or base_url_host_matches(base_url or "", "api.sakana.ai"):
        return BillingRoute(provider="sakana-fugu", model=model.split("/")[-1], base_url=base_url or "", billing_mode="official_docs_snapshot")
    if provider_name in {"minimax", "minimax-cn"}:
        return BillingRoute(provider=provider_name, model=model.split("/")[-1], base_url=base_url or "", billing_mode="official_docs_snapshot")
    if provider_name in {"custom", "local"} or (base and "localhost" in base):
        return BillingRoute(provider=provider_name or "custom", model=model, base_url=base_url or "", billing_mode="unknown")
    return BillingRoute(provider=provider_name or "unknown", model=model.split("/")[-1] if model else "", base_url=base_url or "", billing_mode="unknown")


def _normalize_bedrock_model_name(model: str) -> str:
    """Normalize a Bedrock model id to its bare foundation-model form.

    Bedrock cross-region inference profiles prefix the foundation model id
    with a region scope (``us.`` / ``global.`` / ``eu.`` / ``ap.`` / ``jp.``),
    e.g. ``us.anthropic.claude-opus-4-7``.  The pricing table is keyed on the
    bare ``anthropic.claude-*`` id, so the prefix must be stripped before the
    lookup or every cross-region session prices as unknown.  Mirrors the
    prefix list in ``bedrock_adapter.is_anthropic_bedrock_model``.  Also
    normalizes dot-notation version numbers (``4.7`` → ``4-7``).
    """
    name = model.lower().strip()
    for prefix in ("us.", "global.", "eu.", "ap.", "jp."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = re.sub(r"(\d+)\.(\d+)", r"\1-\2", name)
    return name


def _normalize_anthropic_model_name(model: str) -> str:
    """Normalize Anthropic model name variants to canonical form.

    Handles:
      - Dot notation: claude-opus-4.7 → claude-opus-4-7
      - Short aliases: claude-opus-4.7 → claude-opus-4-7
      - Strips anthropic/ prefix if present
    """
    name = model.lower().strip()
    if name.startswith("anthropic/"):
        name = name[len("anthropic/"):]
    # Normalize dots to dashes in version numbers (e.g. 4.7 → 4-7, 4.6 → 4-6)
    # But preserve the rest of the name structure
    name = re.sub(r"(\d+)\.(\d+)", r"\1-\2", name)
    return name


def _lookup_official_docs_pricing(route: BillingRoute) -> Optional[PricingEntry]:
    model = route.model.lower()
    # Direct lookup first
    entry = _OFFICIAL_DOCS_PRICING.get((route.provider, model))
    if entry:
        return entry
    # Try normalized name for Anthropic (handles dot-notation like opus-4.7)
    if route.provider == "anthropic":
        normalized = _normalize_anthropic_model_name(model)
        if normalized != model:
            entry = _OFFICIAL_DOCS_PRICING.get((route.provider, normalized))
            if entry:
                return entry
    # Bedrock cross-region inference profiles carry a region prefix
    # (us./global./eu./...) that the bare pricing keys don't have.
    if route.provider == "bedrock":
        normalized = _normalize_bedrock_model_name(model)
        if normalized != model:
            entry = _OFFICIAL_DOCS_PRICING.get((route.provider, normalized))
            if entry:
                return entry
    return None


def _openrouter_pricing_entry(route: BillingRoute) -> Optional[PricingEntry]:
    return _pricing_entry_from_metadata(
        fetch_model_metadata(),
        route.model,
        source_url="https://openrouter.ai/docs/api/api-reference/models/get-models",
        pricing_version="openrouter-models-api",
    )


def _pricing_entry_from_metadata(
    metadata: Dict[str, Dict[str, Any]],
    model_id: str,
    *,
    source_url: str,
    pricing_version: str,
) -> Optional[PricingEntry]:
    if model_id not in metadata:
        return None
    pricing = metadata[model_id].get("pricing") or {}
    prompt = _to_decimal(pricing.get("prompt"))
    completion = _to_decimal(pricing.get("completion"))
    request = _to_decimal(pricing.get("request"))
    cache_read = _to_decimal(
        pricing.get("cache_read")
        or pricing.get("cached_prompt")
        or pricing.get("input_cache_read")
    )
    cache_write = _to_decimal(
        pricing.get("cache_write")
        or pricing.get("cache_creation")
        or pricing.get("input_cache_write")
    )
    if prompt is None and completion is None and request is None:
        return None

    def _per_token_to_per_million(value: Optional[Decimal]) -> Optional[Decimal]:
        if value is None:
            return None
        return value * _ONE_MILLION

    return PricingEntry(
        input_cost_per_million=_per_token_to_per_million(prompt),
        output_cost_per_million=_per_token_to_per_million(completion),
        cache_read_cost_per_million=_per_token_to_per_million(cache_read),
        cache_write_cost_per_million=_per_token_to_per_million(cache_write),
        request_cost=request,
        source="provider_models_api",
        source_url=source_url,
        pricing_version=pricing_version,
        fetched_at=_UTC_NOW(),
    )


def get_pricing_entry(
    model_name: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[PricingEntry]:
    route = resolve_billing_route(model_name, provider=provider, base_url=base_url)
    if route.billing_mode == "subscription_included":
        return PricingEntry(
            input_cost_per_million=_ZERO,
            output_cost_per_million=_ZERO,
            cache_read_cost_per_million=_ZERO,
            cache_write_cost_per_million=_ZERO,
            source="none",
            pricing_version="included-route",
        )
    if route.provider == "sakana-fugu":
        entry = _lookup_official_docs_pricing(route)
        if entry:
            return entry
    if route.provider == "openrouter":
        return _openrouter_pricing_entry(route)
    if route.base_url:
        entry = _pricing_entry_from_metadata(
            fetch_endpoint_model_metadata(route.base_url, api_key=api_key or ""),
            route.model,
            source_url=f"{route.base_url.rstrip('/')}/models",
            pricing_version="openai-compatible-models-api",
        )
        if entry:
            return entry
    return _lookup_official_docs_pricing(route)


def normalize_usage(
    response_usage: Any,
    *,
    provider: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> CanonicalUsage:
    """Normalize raw API response usage into canonical token buckets.

    Handles three API shapes:
    - Anthropic: input_tokens/output_tokens/cache_read_input_tokens/cache_creation_input_tokens
    - Codex Responses: input_tokens includes cache tokens; input_tokens_details.cached_tokens separates them
    - OpenAI Chat Completions: prompt_tokens includes cache tokens; prompt_tokens_details.cached_tokens separates them

    In both Codex and OpenAI modes, input_tokens is derived by subtracting cache
    tokens from the total — the API contract is that input/prompt totals include
    cached tokens and the details object breaks them out.
    """
    if not response_usage:
        return CanonicalUsage()

    provider_name = (provider or "").strip().lower()
    mode = (api_mode or "").strip().lower()
    raw_usage = _usage_to_dict(response_usage)

    if mode == "anthropic_messages" or provider_name == "anthropic":
        input_tokens = _to_int(getattr(response_usage, "input_tokens", 0))
        output_tokens = _to_int(getattr(response_usage, "output_tokens", 0))
        cache_read_tokens = _to_int(getattr(response_usage, "cache_read_input_tokens", 0))
        cache_write_tokens = _to_int(getattr(response_usage, "cache_creation_input_tokens", 0))
    elif mode == "codex_responses":
        input_total = _to_int(getattr(response_usage, "input_tokens", 0))
        output_tokens = _to_int(getattr(response_usage, "output_tokens", 0))
        details = getattr(response_usage, "input_tokens_details", None)
        base_cache_read_tokens = _to_int(
            getattr(details, "cached_tokens", 0) if details else 0
        )
        # Sakana Fugu Ultra reports orchestration usage in separate nested
        # fields (Vercel AI SDK integration notes these are billable in
        # addition to top-level input/output tokens). Treat those fields as
        # additive for Sakana only; raw usage remains preserved for any provider
        # so future route-specific accounting can opt in deliberately.
        is_sakana_fugu_usage = provider_name in {
            "sakana-fugu", "sakana", "fugu", "sakana-ai", "sakanaai"
        }
        orchestration_input_tokens = _to_int(
            getattr(details, "orchestration_input_tokens", 0)
            if details and is_sakana_fugu_usage
            else 0
        )
        orchestration_cache_read_tokens = _to_int(
            getattr(details, "orchestration_input_cached_tokens", 0)
            if details and is_sakana_fugu_usage
            else 0
        )
        cache_read_tokens = base_cache_read_tokens + orchestration_cache_read_tokens
        cache_write_tokens = _to_int(
            getattr(details, "cache_creation_tokens", 0) if details else 0
        )
        input_tokens = max(0, input_total - base_cache_read_tokens - cache_write_tokens)
        input_tokens += orchestration_input_tokens
    else:
        prompt_total = _to_int(getattr(response_usage, "prompt_tokens", 0))
        output_tokens = _to_int(getattr(response_usage, "completion_tokens", 0))
        details = getattr(response_usage, "prompt_tokens_details", None)
        # Primary: OpenAI-style prompt_tokens_details. Fallback: Anthropic-style
        # top-level fields that some OpenAI-compatible proxies (OpenRouter, Cline)
        # expose when routing Claude models — without this
        # fallback, cache writes are undercounted as 0 and cache reads can be
        # missed when the proxy only surfaces them at the top level.
        # Port of cline/cline#10266.
        cache_read_tokens = _to_int(getattr(details, "cached_tokens", 0) if details else 0)
        if not cache_read_tokens:
            cache_read_tokens = _to_int(getattr(response_usage, "cache_read_input_tokens", 0))
        cache_write_tokens = _to_int(
            getattr(details, "cache_write_tokens", 0) if details else 0
        )
        if not cache_write_tokens:
            cache_write_tokens = _to_int(
                getattr(response_usage, "cache_creation_input_tokens", 0)
            )
        input_tokens = max(0, prompt_total - cache_read_tokens - cache_write_tokens)

    reasoning_tokens = 0
    output_details = getattr(response_usage, "output_tokens_details", None)
    if output_details:
        reasoning_tokens = _to_int(getattr(output_details, "reasoning_tokens", 0))
        if provider_name in {"sakana-fugu", "sakana", "fugu", "sakana-ai", "sakanaai"}:
            output_tokens += _to_int(
                getattr(output_details, "orchestration_output_tokens", 0)
            )

    return CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=reasoning_tokens,
        raw_usage=raw_usage if isinstance(raw_usage, dict) else None,
    )


def estimate_usage_cost(
    model_name: str,
    usage: CanonicalUsage,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> CostResult:
    route = resolve_billing_route(model_name, provider=provider, base_url=base_url)
    if route.billing_mode == "subscription_included":
        return CostResult(
            amount_usd=_ZERO,
            status="included",
            source="none",
            label="included",
            pricing_version="included-route",
        )

    entry = get_pricing_entry(model_name, provider=provider, base_url=base_url, api_key=api_key)
    if not entry:
        return CostResult(amount_usd=None, status="unknown", source="none", label="n/a")

    notes: list[str] = []
    amount = _ZERO

    input_rate = entry.input_cost_per_million
    output_rate = entry.output_cost_per_million
    cache_read_rate = entry.cache_read_cost_per_million
    cache_write_rate = entry.cache_write_cost_per_million
    if (
        entry.context_tier_threshold_tokens is not None
        and usage.prompt_tokens > entry.context_tier_threshold_tokens
    ):
        input_rate = (
            entry.context_tier_input_cost_per_million
            if entry.context_tier_input_cost_per_million is not None
            else input_rate
        )
        output_rate = (
            entry.context_tier_output_cost_per_million
            if entry.context_tier_output_cost_per_million is not None
            else output_rate
        )
        cache_read_rate = (
            entry.context_tier_cache_read_cost_per_million
            if entry.context_tier_cache_read_cost_per_million is not None
            else cache_read_rate
        )
        cache_write_rate = (
            entry.context_tier_cache_write_cost_per_million
            if entry.context_tier_cache_write_cost_per_million is not None
            else cache_write_rate
        )
        notes.append(
            f"long-context pricing tier applies above {entry.context_tier_threshold_tokens:,} prompt tokens"
        )

    if usage.input_tokens and input_rate is None:
        return CostResult(amount_usd=None, status="unknown", source=entry.source, label="n/a")
    if usage.output_tokens and output_rate is None:
        return CostResult(amount_usd=None, status="unknown", source=entry.source, label="n/a")
    if usage.cache_read_tokens:
        if cache_read_rate is None:
            return CostResult(
                amount_usd=None,
                status="unknown",
                source=entry.source,
                label="n/a",
                notes=("cache-read pricing unavailable for route",),
            )
    if usage.cache_write_tokens:
        if cache_write_rate is None:
            return CostResult(
                amount_usd=None,
                status="unknown",
                source=entry.source,
                label="n/a",
                notes=("cache-write pricing unavailable for route",),
            )

    if input_rate is not None:
        amount += Decimal(usage.input_tokens) * input_rate / _ONE_MILLION
    if output_rate is not None:
        amount += Decimal(usage.output_tokens) * output_rate / _ONE_MILLION
    if cache_read_rate is not None:
        amount += Decimal(usage.cache_read_tokens) * cache_read_rate / _ONE_MILLION
    if cache_write_rate is not None:
        amount += Decimal(usage.cache_write_tokens) * cache_write_rate / _ONE_MILLION
    if entry.request_cost is not None and usage.request_count:
        amount += Decimal(usage.request_count) * entry.request_cost

    status: CostStatus = "estimated"
    label = f"~${amount:.2f}"
    if entry.source == "none" and amount == _ZERO:
        status = "included"
        label = "included"

    if route.provider == "openrouter":
        notes.append("OpenRouter cost is estimated from the models API until reconciled.")

    return CostResult(
        amount_usd=amount,
        status=status,
        source=entry.source,
        label=label,
        fetched_at=entry.fetched_at,
        pricing_version=entry.pricing_version,
        notes=tuple(notes),
    )


def has_known_pricing(
    model_name: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bool:
    """Check whether we have pricing data for this model+route.

    Uses direct lookup instead of routing through the full estimation
    pipeline — avoids creating dummy usage objects just to check status.
    """
    route = resolve_billing_route(model_name, provider=provider, base_url=base_url)
    if route.billing_mode == "subscription_included":
        return True
    entry = get_pricing_entry(model_name, provider=provider, base_url=base_url, api_key=api_key)
    return entry is not None



def format_duration_compact(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 24:
        remaining_min = int(minutes % 60)
        return f"{int(hours)}h {remaining_min}m" if remaining_min else f"{int(hours)}h"
    days = hours / 24
    return f"{days:.1f}d"


def format_token_count_compact(value: int) -> str:
    abs_value = abs(int(value))
    if abs_value < 1_000:
        return str(int(value))

    sign = "-" if value < 0 else ""
    units = ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))
    for threshold, suffix in units:
        if abs_value >= threshold:
            scaled = abs_value / threshold
            if scaled < 10:
                text = f"{scaled:.2f}"
            elif scaled < 100:
                text = f"{scaled:.1f}"
            else:
                text = f"{scaled:.0f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"

    return f"{value:,}"
