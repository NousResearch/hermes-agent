from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Literal, Optional

from agent.model_metadata import fetch_endpoint_model_metadata, fetch_model_metadata
from utils import base_url_host_matches

DEFAULT_PRICING = {"input": 0.0, "output": 0.0}

_ZERO = Decimal("0")
_ONE = Decimal("1")
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

    def __add__(self, other: "CanonicalUsage") -> "CanonicalUsage":
        """Sum two usage buckets (e.g. MoA advisor fan-out + aggregator).

        ``raw_usage`` is dropped on the sum — it describes a single API
        response and cannot be meaningfully merged. ``request_count`` adds so
        callers can see how many underlying API calls a combined figure covers.
        """
        if not isinstance(other, CanonicalUsage):
            return NotImplemented
        return CanonicalUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            request_count=self.request_count + other.request_count,
            raw_usage=None,
        )


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
    source: CostSource = "none"
    source_url: Optional[str] = None
    pricing_version: Optional[str] = None
    fetched_at: Optional[datetime] = None


@dataclass(frozen=True)
class CostBreakdown:
    """Per-bucket cost decomposition (Decimal USD) for a single priced row.

    Populated in ``_estimate_usage_cost_base`` at the amount-accumulation site
    so EVERY priced path (normal row, codex real-price, bedrock-normalized
    anthropic retry, family fallback) carries per-bucket data. Consumed by the
    langfuse plugin to emit per-type ``cost_details`` instead of recomputing
    from ``get_pricing_entry`` (which returns the $0 included entry for codex).
    """

    input_usd: Decimal = _ZERO
    output_usd: Decimal = _ZERO
    cache_read_usd: Decimal = _ZERO
    cache_write_usd: Decimal = _ZERO
    request_usd: Decimal = _ZERO

    @property
    def total_usd(self) -> Decimal:
        return (
            self.input_usd
            + self.output_usd
            + self.cache_read_usd
            + self.cache_write_usd
            + self.request_usd
        )

    def scaled(self, factor: Decimal) -> "CostBreakdown":
        """Return a copy with every bucket multiplied by ``factor`` (used by the
        bedrock cross-region uplift knob so the breakdown stays consistent with
        the scaled total)."""
        if factor == _ONE:
            return self
        return CostBreakdown(
            input_usd=self.input_usd * factor,
            output_usd=self.output_usd * factor,
            cache_read_usd=self.cache_read_usd * factor,
            cache_write_usd=self.cache_write_usd * factor,
            request_usd=self.request_usd * factor,
        )

    def as_langfuse_cost_details(self, total_fallback: Optional[Decimal] = None) -> dict[str, float]:
        """Langfuse ``cost_details`` keyed to match the usage_details keys.

        Emits per-bucket costs when present; otherwise falls back to a single
        ``total`` key so a corrected total is never dropped to $0 buckets.
        """
        details: dict[str, float] = {}
        if self.input_usd:
            details["input"] = float(self.input_usd)
        if self.output_usd:
            details["output"] = float(self.output_usd)
        if self.cache_read_usd:
            details["cache_read_input_tokens"] = float(self.cache_read_usd)
        if self.cache_write_usd:
            details["cache_creation_input_tokens"] = float(self.cache_write_usd)
        if self.request_usd:
            details["request"] = float(self.request_usd)
        if not details:
            fallback = total_fallback if total_fallback is not None else self.total_usd
            if fallback:
                details["total"] = float(fallback)
        return details


@dataclass(frozen=True)
class CostResult:
    amount_usd: Optional[Decimal]
    status: CostStatus
    source: CostSource
    label: str
    fetched_at: Optional[datetime] = None
    pricing_version: Optional[str] = None
    notes: tuple[str, ...] = ()
    # Per-bucket decomposition for langfuse/per-surface consumers (Phase 5).
    breakdown: Optional[CostBreakdown] = None
    # The pre-correction base amount (set when corrections repriced the row).
    base_amount_usd: Optional[Decimal] = None
    # Human-readable description of any corrections applied to this row.
    adjustments: tuple[str, ...] = ()


_UTC_NOW = lambda: datetime.now(timezone.utc)


# Official docs snapshot entries. Models whose published pricing and cache
# semantics are stable enough to encode exactly.
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
    # Opus 4.5/4.7/4.8 dropped to $5/$25 (1/3 of the 4.6 price); 4.6 stays
    # $15/$75. Cache rates follow Anthropic's 0.1x read / 1.25x write multipliers.
    # Source: https://aws.amazon.com/blogs/machine-learning/claude-opus-4-5-now-in-amazon-bedrock
    (
        "bedrock",
        "anthropic.claude-opus-4-8",
    ): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/blogs/machine-learning/claude-opus-4-5-now-in-amazon-bedrock",
        pricing_version="bedrock-pricing-2026-04",
    ),
    (
        "bedrock",
        "anthropic.claude-opus-4-7",
    ): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/bedrock/pricing/",
        pricing_version="bedrock-pricing-2026-04",
    ),
    (
        "bedrock",
        "anthropic.claude-opus-4-5",
    ): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("25.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://aws.amazon.com/blogs/machine-learning/claude-opus-4-5-now-in-amazon-bedrock",
        pricing_version="bedrock-pricing-2026-04",
    ),
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
    if provider_name == "bedrock" or base_url_host_matches(base_url or "", "amazonaws.com"):
        # AWS Bedrock resells third-party models (Anthropic Claude, Amazon Nova)
        # at the provider's published per-token rates through AWS billing. The
        # model id carries a cross-Region inference-profile prefix
        # (us./eu./apac./global./us-gov.) that is NOT part of the pricing key —
        # strip it and look up the exact (bedrock, <model>) official-docs entry.
        # base_url is deliberately cleared so get_pricing_entry does the pure
        # dict lookup and never attempts an endpoint /models metadata fetch
        # (this route is hit on every throttled cost render; no network).
        # NOTE: geographic profiles (us./eu./apac.) cost ~10% more than the
        # global profile; we estimate at the published base rate and the cost
        # label is "~$" (estimate), so a geo run reads ~10% low by design.
        return BillingRoute(provider="bedrock", model=_normalize_bedrock_model_name(model), base_url="", billing_mode="official_docs_snapshot")
    if provider_name == "anthropic":
        return BillingRoute(provider="anthropic", model=model.split("/")[-1], base_url=base_url or "", billing_mode="official_docs_snapshot")
    if provider_name == "openai":
        return BillingRoute(provider="openai", model=model.split("/")[-1], base_url=base_url or "", billing_mode="official_docs_snapshot")
    if provider_name in {"minimax", "minimax-cn"}:
        return BillingRoute(provider=provider_name, model=model.split("/")[-1], base_url=base_url or "", billing_mode="official_docs_snapshot")
    # Vertex AI hosts the same Gemini models as Google AI Studio; price them
    # off the gemini official-docs snapshot. Strip the "google/" vendor prefix
    # the OpenAI-compat endpoint requires so the pricing key matches.
    if provider_name == "vertex" or base_url_host_matches(base_url or "", "aiplatform.googleapis.com"):
        return BillingRoute(provider="gemini", model=model.split("/")[-1], base_url=base_url or "", billing_mode="official_docs_snapshot")
    if provider_name in {"custom", "local"} or (base and "localhost" in base):
        return BillingRoute(provider=provider_name or "custom", model=model, base_url=base_url or "", billing_mode="unknown")
    return BillingRoute(provider=provider_name or "unknown", model=model.split("/")[-1] if model else "", base_url=base_url or "", billing_mode="unknown")


def _normalize_bedrock_model_name(model: str) -> str:
    """Normalize a Bedrock model id to the exact official pricing key.

    Bedrock inference ids may carry a cross-Region/global prefix (``us.``,
    ``eu.``, ``apac.``, ``global.``, etc.) that is not part of the pricing key.
    Keep the vendor/model id exact (``anthropic.*`` / ``amazon.*``), strip only
    known routing prefixes, and normalize dot-version variants (``4.7`` →
    ``4-7``). Unknown ids still fail safe to unknown instead of fuzzy pricing.
    """
    name = (model or "").lower().strip()
    for prefix in ("us-gov.", "global.", "apac.", "us.", "eu.", "ap.", "jp."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return re.sub(r"(\d+)\.(\d+)", r"\1-\2", name)


# ════════════════════════════════════════════════════════════════════════════
# Config-gated cost corrections (company-OAuth codex repricing + provider-
# mislabel family fallback + optional bedrock cross-region uplift).
#
# These are lifted VERBATIM from the offline spend-attribution core
# (~/.hermes/scripts/spend_core.py) so that every Hermes surface — the workflow
# bubble, /usage, the delegate footer, langfuse, insights — and spend_core can
# share ONE pricing path instead of forking it. They are INERT until wired into
# estimate_usage_cost behind an explicit config gate (default OFF = upstream).
# See agent/usage_pricing corrections wiring + hermes_cli.config corrections
# loader.
# ════════════════════════════════════════════════════════════════════════════

# Alfredo's Codex runs on the COMPANY OpenAI OAuth — billed at real API prices,
# NOT a personal flat-rate ChatGPT plan. The canonical pricing routes any
# openai-codex model to subscription_included ($0), which is correct for a
# normal user but WRONG for this setup. These tables reprice at official OpenAI
# API rates (Standard + Priority tiers), USD per 1M tokens: (input, cached_input,
# cache_write, output). Cache-write pricing is None when OpenAI does not publish
# a separate rate. Source: https://developers.openai.com/api/docs/pricing
CODEX_PRICING_STANDARD: Dict[str, tuple[float, float, Optional[float], float]] = {
    "gpt-5.6": (5.00, 0.50, 6.25, 30.00),
    "gpt-5.6-sol": (5.00, 0.50, 6.25, 30.00),
    "gpt-5.6-terra": (2.50, 0.25, 3.125, 15.00),
    "gpt-5.6-luna": (1.00, 0.10, 1.25, 6.00),
    "gpt-5.5": (5.00, 0.50, None, 30.00),
    "gpt-5.4": (2.50, 0.25, None, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, None, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, None, 1.25),
    "gpt-5": (5.00, 0.50, None, 30.00),  # alias safety -> 5.5-class flagship
}
CODEX_PRICING_PRIORITY: Dict[str, tuple[float, float, Optional[float], float]] = {
    "gpt-5.6": (10.00, 1.00, 12.50, 60.00),
    "gpt-5.6-sol": (10.00, 1.00, 12.50, 60.00),
    "gpt-5.6-terra": (5.00, 0.50, 6.25, 30.00),
    "gpt-5.6-luna": (2.00, 0.20, 2.50, 12.00),
    "gpt-5.5": (12.50, 1.25, None, 75.00),
    "gpt-5.4": (5.00, 0.50, None, 30.00),
    "gpt-5.4-mini": (1.50, 0.15, None, 9.00),
    "gpt-5.4-nano": (0.50, 0.05, None, 3.125),  # not officially listed; 2.5x standard
    "gpt-5": (12.50, 1.25, None, 75.00),
}


def codex_cost(
    model: str,
    usage: CanonicalUsage,
    *,
    tier: str = "priority",
) -> tuple[Optional[float], Optional[str]]:
    """Real OpenAI API cost for a company-OAuth Codex session. Returns
    (usd, matched_key), or (None, None) if the model isn't in our table.

    Uses the SAME additive model as the canonical pricer: input_tokens and
    cache_read_tokens are separate buckets (input excludes the cached part).

    NOTE [B3]: the arithmetic is kept as VERBATIM float math
    (inp*c_in/1e6 + ...) — NOT a Decimal rewrite — because the frozen golden
    oracle was captured from this exact float arithmetic, so a Decimal port
    would shift last-digit rounding and break exact parity.
    """
    table = (
        CODEX_PRICING_PRIORITY
        if (tier or "priority").strip().lower() == "priority"
        else CODEX_PRICING_STANDARD
    )
    key = (model or "").strip().lower()
    rates = table.get(key)
    if rates is None:
        return None, None
    c_in, c_cached, c_write, c_out = rates
    inp = usage.input_tokens or 0
    out = usage.output_tokens or 0
    cr = usage.cache_read_tokens or 0
    cw = usage.cache_write_tokens or 0
    usd = (
        inp * c_in / 1e6
        + cr * c_cached / 1e6
        + (cw * c_write / 1e6 if c_write is not None else 0)
        + out * c_out / 1e6
    )
    return usd, key


# Bedrock cross-region / provider / version / date decorations to strip before
# routing the bare model name through the `anthropic` official-docs snapshot.
# Verbatim from spend_core.normalize_bedrock so the family-fallback path matches.
_BEDROCK_REGION_PREFIX = re.compile(r"^(us|eu|apac|us-gov)\.")
_BEDROCK_PROVIDER_PREFIX = re.compile(r"^(anthropic|amazon|meta|mistral|cohere)\.")
_BEDROCK_VERSION_SUFFIX = re.compile(r"-v\d+:\d+$")
_BEDROCK_VERSION_SUFFIX_BARE = re.compile(r"-v\d+$")
_BEDROCK_DATE_SUFFIX = re.compile(r"-\d{8}$")


def _strip_bedrock_decorations(model: str) -> str:
    """Strip region/provider/version/date decorations from a Bedrock model id
    so the bare name (e.g. ``claude-opus-4-8``) can route through the
    ``anthropic`` official-docs pricing. Verbatim port of
    spend_core.normalize_bedrock (used by the C2 family fallback)."""
    m = model or ""
    m = _BEDROCK_REGION_PREFIX.sub("", m)
    m = _BEDROCK_PROVIDER_PREFIX.sub("", m)
    m = _BEDROCK_VERSION_SUFFIX.sub("", m)
    m = _BEDROCK_VERSION_SUFFIX_BARE.sub("", m)
    m = _BEDROCK_DATE_SUFFIX.sub("", m)
    return m


@dataclass(frozen=True)
class CorrectionsConfig:
    """Explicit, immutable cost-corrections gate.

    All-off by default = upstream byte-identical behavior. Passed explicitly by
    spend_core (so a cron never depends on ambient config.yaml and can never
    double-apply the dashboard's bedrock uplift), or loaded ambiently by
    ``load_corrections_config`` for live surfaces.
    """

    enabled: bool = False
    codex_tier: str = "priority"
    # Decision 1 ships the mechanism gated, DEFAULT 1 (= option A, no uplift in
    # the shared engine). Flipping to option B is a one-line config change.
    bedrock_cross_region_factor: Decimal = _ONE


# Sentinel passed on EVERY internal estimate_usage_cost retry so the retry
# never re-enters apply_corrections with ambient config ON (re-entrancy /
# bedrock double-factor guard) [B4/R3].
_NO_CORRECTIONS = CorrectionsConfig(enabled=False)


# ── corrections config loader (mtime-cached, fail-safe, default all-off) ─────
# (path, mtime_ns, size) -> CorrectionsConfig
_CORRECTIONS_CACHE: dict[str, tuple[int, int, "CorrectionsConfig"]] = {}


def _corrections_config_path() -> str:
    """Path of the config.yaml whose ``cost_corrections`` block we read.

    Indirected through a function so tests can monkeypatch it to a temp file.
    """
    try:
        from hermes_cli.config import get_config_path

        return str(get_config_path())
    except Exception:
        return os.path.expanduser("~/.hermes/config.yaml")


def _coerce_factor(value: Any) -> Decimal:
    try:
        d = Decimal(str(value))
        return d if d > 0 else _ONE
    except Exception:
        return _ONE


def _corrections_from_block(block: Any) -> CorrectionsConfig:
    """Build a CorrectionsConfig from a parsed ``cost_corrections`` mapping.

    Fail-safe: a non-mapping or missing block yields all-off defaults. The
    ``SPEND_CODEX_TIER`` env alias wins over the block's ``codex_tier``.
    """
    enabled = False
    tier = "priority"
    factor = _ONE
    if isinstance(block, dict):
        enabled = bool(block.get("enabled", False))
        raw_tier = str(block.get("codex_tier", "priority") or "priority").strip().lower()
        tier = raw_tier if raw_tier in ("standard", "priority") else "priority"
        if "bedrock_cross_region_factor" in block:
            factor = _coerce_factor(block.get("bedrock_cross_region_factor"))
    env_tier = (os.environ.get("SPEND_CODEX_TIER") or "").strip().lower()
    if env_tier in ("standard", "priority"):
        tier = env_tier
    return CorrectionsConfig(
        enabled=enabled, codex_tier=tier, bedrock_cross_region_factor=factor
    )


def load_corrections_config(*, force_reload: bool = False) -> CorrectionsConfig:
    """Load the ambient ``cost_corrections`` config block.

    mtime/size-cached, fail-safe, never raises, default all-off. Live surfaces
    (bubble, /usage, delegate footer, langfuse, insights) inherit corrections
    through this; ``spend_core`` passes an EXPLICIT CorrectionsConfig instead so
    it never depends on ambient config and can never double-apply the dashboard
    uplift.
    """
    path = _corrections_config_path()
    try:
        st = os.stat(path)
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        return _corrections_from_block(None)

    if not force_reload:
        cached = _CORRECTIONS_CACHE.get(path)
        if cached is not None and cached[0] == sig[0] and cached[1] == sig[1]:
            return cached[2]

    try:
        import yaml

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        block = data.get("cost_corrections") if isinstance(data, dict) else None
    except Exception:
        block = None

    cfg = _corrections_from_block(block)
    _CORRECTIONS_CACHE[path] = (sig[0], sig[1], cfg)
    return cfg


def _route_is_bedrock(
    model: str, provider: Optional[str], base_url: Optional[str]
) -> bool:
    return (
        resolve_billing_route(model, provider=provider, base_url=base_url).provider
        == "bedrock"
    )


def _scale_result(result: CostResult, factor: Decimal, note: str) -> CostResult:
    """Multiply a priced CostResult (amount + breakdown) by the bedrock
    cross-region factor. A no-op when factor == 1 (Decision-1 default)."""
    if factor == _ONE or result.amount_usd is None:
        return result
    scaled_breakdown = (
        result.breakdown.scaled(factor) if result.breakdown is not None else None
    )
    return replace(
        result,
        amount_usd=result.amount_usd * factor,
        base_amount_usd=result.amount_usd,
        breakdown=scaled_breakdown,
        adjustments=result.adjustments + (note,),
        label=f"~${result.amount_usd * factor:.2f}",
    )


def _codex_repriced_result(
    base: CostResult,
    model: str,
    usage: CanonicalUsage,
    cfg: CorrectionsConfig,
) -> Optional[CostResult]:
    """Reprice a company-OAuth codex row at real OpenAI API rates. Returns a
    new CostResult, or None if the model isn't in the codex table (so the
    caller FALLS THROUGH instead of returning $0)."""
    usd, key = codex_cost(model, usage, tier=cfg.codex_tier)
    if usd is None or key is None:
        return None
    # Per-bucket breakdown from the SAME verbatim float arithmetic.
    table = (
        CODEX_PRICING_PRIORITY
        if (cfg.codex_tier or "priority").strip().lower() == "priority"
        else CODEX_PRICING_STANDARD
    )
    c_in, c_cached, c_write, c_out = table[key]
    inp = usage.input_tokens or 0
    out = usage.output_tokens or 0
    cr = usage.cache_read_tokens or 0
    cw = usage.cache_write_tokens or 0
    breakdown = CostBreakdown(
        input_usd=Decimal(str(inp * c_in / 1e6)),
        output_usd=Decimal(str(out * c_out / 1e6)),
        cache_read_usd=Decimal(str(cr * c_cached / 1e6)),
        cache_write_usd=Decimal(
            str(cw * c_write / 1e6 if c_write is not None else 0)
        ),
    )
    amount = Decimal(str(usd))
    return CostResult(
        amount_usd=amount,
        status="estimated",
        source="official_docs_snapshot",
        label=f"~${amount:.2f}",
        pricing_version=f"openai-api-2026-07-14 (company OAuth real API price, tier={cfg.codex_tier})",
        breakdown=breakdown,
        base_amount_usd=base.amount_usd,
        adjustments=("codex-real-api-price",),
    )


def apply_corrections(
    base: CostResult,
    model: str,
    usage: CanonicalUsage,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    cfg: CorrectionsConfig,
) -> CostResult:
    """Config-gated repricing. Pure. Ports ``spend_core.session_cost``
    branch-for-branch in the EXACT current order [B4]:

      1. company-OAuth codex ``included`` → real OpenAI API price; if the model
         is NOT in the codex table, codex_cost returns None and we FALL THROUGH
         (do NOT return $0 — spend_core relies on this fall-through);
      2. otherwise a normally-priced row (estimated/actual/included) → keep,
         applying the optional bedrock cross-region uplift on bedrock rows;
      3. unknown → bedrock-normalized anthropic retry;
      4. C2 model-family fallback: gpt-*/o1/o3/o4 → codex_cost; claude/anthropic
         → bedrock-normalized anthropic retry (price by what the MODEL is).

    Every internal ``estimate_usage_cost`` retry passes
    ``corrections=_NO_CORRECTIONS`` so it never re-enters apply_corrections with
    ambient config ON (re-entrancy / bedrock double-factor guard) [B4].
    """
    prov = (provider or "").lower()
    ml = (model or "").lower()
    factor = cfg.bedrock_cross_region_factor
    # The bedrock cross-region uplift keys on the BILLING ROUTE being bedrock —
    # the same signal the dashboard uses (billing_provider == "bedrock"). A
    # mislabeled row (e.g. us.anthropic.* stamped openai-codex) is NOT billed
    # through AWS, so it must never receive the uplift. Under Decision-1 (A)
    # factor == 1 so this is a no-op everywhere; the gate matters only if B is
    # later enabled.
    is_bedrock_route = _route_is_bedrock(model, provider, base_url)

    # ── Branch 1: company-OAuth codex included → real API price ──────────────
    codex_included = base.status == "included" and prov == "openai-codex"
    if codex_included:
        repriced = _codex_repriced_result(base, model, usage, cfg)
        if repriced is not None:
            return repriced
        # Codex provider but NOT a gpt-* in the table (e.g. a mislabeled
        # anthropic id stamped openai-codex). Do NOT accept the $0 'included'
        # result — FALL THROUGH to the model-family fallback below.
    # ── Branch 2: normal priced row → keep (+ optional bedrock uplift) ───────
    elif base.status in ("estimated", "actual", "included"):
        if base.status in ("estimated", "actual") and is_bedrock_route:
            return _scale_result(base, factor, "bedrock-cross-region-uplift")
        return base

    # ── Branch 3: unknown → bedrock-normalized anthropic retry ───────────────
    if prov == "bedrock" or ml.startswith(("us.", "eu.", "apac.")):
        nm = _strip_bedrock_decorations(model)
        if nm and nm != model:
            res2 = estimate_usage_cost(
                nm, usage, provider="anthropic", corrections=_NO_CORRECTIONS
            )
            if res2.status in ("estimated", "actual"):
                out2 = replace(res2, adjustments=("bedrock-family-fallback",))
                if is_bedrock_route:
                    return _scale_result(out2, factor, "bedrock-cross-region-uplift")
                return out2

    # ── Branch 4: C2 model-family fallback ───────────────────────────────────
    if ml.startswith(("gpt-", "o1", "o3", "o4")):
        repriced = _codex_repriced_result(base, model, usage, cfg)
        if repriced is not None:
            return replace(
                repriced, adjustments=("codex-family-fallback",)
            )
    if "claude" in ml or "anthropic" in ml or ml.startswith(("us.", "eu.", "apac.")):
        nm = _strip_bedrock_decorations(model)
        cand = nm if nm else model
        res3 = estimate_usage_cost(
            cand, usage, provider="anthropic", corrections=_NO_CORRECTIONS
        )
        if res3.status in ("estimated", "actual"):
            out3 = replace(res3, adjustments=("claude-family-fallback",))
            # Uplift only when the original row was genuinely billed through
            # bedrock (matches the dashboard's billing_provider keying).
            if is_bedrock_route:
                return _scale_result(out3, factor, "bedrock-cross-region-uplift")
            return out3

    # Nothing matched — preserve the base result (spend_core returns res.status
    # with $0 here; we keep the richer base unchanged, which is equivalent).
    return base


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

    if mode == "anthropic_messages" or provider_name == "anthropic":
        input_tokens = _to_int(getattr(response_usage, "input_tokens", 0))
        output_tokens = _to_int(getattr(response_usage, "output_tokens", 0))
        cache_read_tokens = _to_int(getattr(response_usage, "cache_read_input_tokens", 0))
        cache_write_tokens = _to_int(getattr(response_usage, "cache_creation_input_tokens", 0))
    elif mode == "codex_responses":
        input_total = _to_int(getattr(response_usage, "input_tokens", 0))
        output_tokens = _to_int(getattr(response_usage, "output_tokens", 0))
        details = getattr(response_usage, "input_tokens_details", None)
        cache_read_tokens = _to_int(getattr(details, "cached_tokens", 0) if details else 0)
        cache_write_tokens = _to_int(
            getattr(details, "cache_write_tokens", 0) if details else 0
        )
        input_tokens = max(0, input_total - cache_read_tokens - cache_write_tokens)
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
    # Responses API shape: output_tokens_details.reasoning_tokens.
    # Chat Completions shape (OpenAI, OpenRouter, DeepSeek, etc.):
    # completion_tokens_details.reasoning_tokens. Reading only the former
    # left reasoning_tokens=0 for every chat_completions reasoning model —
    # hidden thinking was invisible in session accounting even though it
    # dominates output spend on models like deepseek-v4-flash (measured:
    # single calls burning 21K reasoning tokens to emit 500 visible tokens).
    output_details = getattr(response_usage, "output_tokens_details", None)
    if output_details:
        reasoning_tokens = _to_int(getattr(output_details, "reasoning_tokens", 0))
    if not reasoning_tokens:
        completion_details = getattr(response_usage, "completion_tokens_details", None)
        if completion_details:
            reasoning_tokens = _to_int(
                getattr(completion_details, "reasoning_tokens", 0)
            )

    return CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _estimate_usage_cost_base(
    model_name: str,
    usage: CanonicalUsage,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> CostResult:
    """The upstream pricing body (renamed). Pure, config-unaware: routes the
    model, looks up the entry, and accumulates the Decimal cost. [B2] populates
    ``breakdown`` at the amount-accumulation site so every priced path carries
    per-bucket data. The gated ``estimate_usage_cost`` wrapper calls this, then
    optionally ``apply_corrections``."""
    route = resolve_billing_route(model_name, provider=provider, base_url=base_url)
    if route.billing_mode == "subscription_included":
        return CostResult(
            amount_usd=_ZERO,
            status="included",
            source="none",
            label="included",
            pricing_version="included-route",
            breakdown=CostBreakdown(),
        )

    entry = get_pricing_entry(model_name, provider=provider, base_url=base_url, api_key=api_key)
    if not entry:
        return CostResult(amount_usd=None, status="unknown", source="none", label="n/a")

    notes: list[str] = []
    amount = _ZERO

    if usage.input_tokens and entry.input_cost_per_million is None:
        return CostResult(amount_usd=None, status="unknown", source=entry.source, label="n/a")
    if usage.output_tokens and entry.output_cost_per_million is None:
        return CostResult(amount_usd=None, status="unknown", source=entry.source, label="n/a")
    if usage.cache_read_tokens:
        if entry.cache_read_cost_per_million is None:
            return CostResult(
                amount_usd=None,
                status="unknown",
                source=entry.source,
                label="n/a",
                notes=("cache-read pricing unavailable for route",),
            )
    if usage.cache_write_tokens:
        if entry.cache_write_cost_per_million is None:
            return CostResult(
                amount_usd=None,
                status="unknown",
                source=entry.source,
                label="n/a",
                notes=("cache-write pricing unavailable for route",),
            )

    # [B2] Accumulate per-bucket so the breakdown is built alongside the total.
    input_usd = _ZERO
    output_usd = _ZERO
    cache_read_usd = _ZERO
    cache_write_usd = _ZERO
    request_usd = _ZERO
    if entry.input_cost_per_million is not None:
        input_usd = Decimal(usage.input_tokens) * entry.input_cost_per_million / _ONE_MILLION
    if entry.output_cost_per_million is not None:
        output_usd = Decimal(usage.output_tokens) * entry.output_cost_per_million / _ONE_MILLION
    if entry.cache_read_cost_per_million is not None:
        cache_read_usd = Decimal(usage.cache_read_tokens) * entry.cache_read_cost_per_million / _ONE_MILLION
    if entry.cache_write_cost_per_million is not None:
        cache_write_usd = Decimal(usage.cache_write_tokens) * entry.cache_write_cost_per_million / _ONE_MILLION
    if entry.request_cost is not None and usage.request_count:
        request_usd = Decimal(usage.request_count) * entry.request_cost
    amount = input_usd + output_usd + cache_read_usd + cache_write_usd + request_usd
    breakdown = CostBreakdown(
        input_usd=input_usd,
        output_usd=output_usd,
        cache_read_usd=cache_read_usd,
        cache_write_usd=cache_write_usd,
        request_usd=request_usd,
    )

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
        breakdown=breakdown,
    )


def estimate_usage_cost(
    model_name: str,
    usage: CanonicalUsage,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    corrections: Optional["CorrectionsConfig"] = None,
) -> CostResult:
    """Estimate the USD cost of a usage record.

    Behavior is byte-identical to upstream UNLESS config-gated cost corrections
    are active. Resolution of the gate:

    - ``corrections`` explicitly passed (e.g. ``_NO_CORRECTIONS`` from an
      internal retry, or spend_core's explicit config) → use it verbatim.
    - ``corrections is None`` → load the ambient ``cost_corrections`` config
      block (mtime-cached, fail-safe, default all-OFF).

    With the gate OFF the base result is returned unchanged (criterion 5:
    codex $0 included, bedrock base rate, upstream contract preserved). With
    the gate ON, ``apply_corrections`` reprices company-OAuth codex, applies
    the C2 model-family fallback for mislabeled providers, and applies the
    optional bedrock cross-region uplift.
    """
    base = _estimate_usage_cost_base(
        model_name, usage, provider=provider, base_url=base_url, api_key=api_key
    )
    cfg = corrections if corrections is not None else load_corrections_config()
    if not cfg.enabled:
        return base
    return apply_corrections(
        base, model_name, usage, provider=provider, base_url=base_url, cfg=cfg
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
