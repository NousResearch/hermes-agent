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


# Providers that front the Anthropic API and are priced at official Anthropic
# API rates (docs snapshot) for fleet cost *visibility*, labelled "estimated".
# Two flavors, same routing:
#   • Local subscription proxies / bridges / pools whose marginal cash cost is
#     $0 (flat Claude subscription / tailnet failover / local relay).
#   • Third-party Anthropic-compatible resellers billed in real cash but at
#     their own rate (e.g. `yunwu`, 云雾 API) — priced here at the SAME official
#     Anthropic snapshot the other Claude rows use, on purpose, so a Yunwu
#     claude-* turn reconciles against the rest of the fleet rather than getting
#     a bespoke rate table (the estimate is Anthropic list price, not Yunwu's
#     actual ⚡-credit cost — same "estimated" caveat as the $0 proxies).
# Provider names match the `provider:` keys used in ~/.hermes/config.yaml +
# plugins/model-providers/* across the fleet.
#
# This is the set of EXACT BASE names. The numbered failover family
# (claude-apx-N / claude-bpx-N, any integer N) is matched by PATTERN
# in is_notional_anthropic_provider() below — so a NEW failover lane (-apx-11,
# -bpx-12, …) can never again silently price as $0 the way the un-renamed lanes
# did. Use the predicate, not bare `in` membership, at every call site.
#
# 2026-07-08 rename: the api-proxy pool `claude-pool`/`claude-app` → `claude-apr`
# and its failover lanes `claude-api-proxy-fN` → `claude-apx-N`; the bridge pool
# → `claude-bpr` and `claude-bridge-fN` → `claude-bpx-N`. The old names are gone
# fleet-wide (no config/session/cron references) so they are dropped here — they
# were pricing correctly ONLY because the live traffic still carried them, which
# it no longer does. Leaving them would have kept the LIVE pools (`claude-apr`/
# `claude-bpr`) + their apx/bpx lanes silently UNPRICED (billing_mode=unknown).
NOTIONAL_ANTHROPIC_PROVIDERS = frozenset({
    "claude-api-proxy",
    "claude-bridge",
    "claude-apr",
    "claude-bpr",
    "yunwu",
})

# claude-apx-1, claude-bpx-2, … -<any integer>. Anchored + integer-only so it
# matches ONLY the disciplined failover naming and never an unrelated
# "claude-bridge-frobnicate". The pools themselves (claude-apr / claude-bpr) have
# no numbered family (failover happens inside the relay), so they stay exact base
# members above.
_NOTIONAL_ANTHROPIC_FN_RE = re.compile(r"^claude-(?:apx|bpx)-\d+$")


def is_notional_anthropic_provider(provider_name: Optional[str]) -> bool:
    """True if a provider key should be priced at notional Anthropic rates.

    Covers the exact base relays/proxies/pool AND the numbered -fN failover
    family by pattern, so adding a new -fN lane (or a config that references
    one) never needs a code change. Normalizes (strip/lower) defensively for
    direct callers; resolve_billing_route already normalizes before calling.
    """
    p = (provider_name or "").strip().lower()
    if not p:
        return False
    return p in NOTIONAL_ANTHROPIC_PROVIDERS or bool(
        _NOTIONAL_ANTHROPIC_FN_RE.match(p)
    )


# Local subscription BRIDGE that fronts the Google AI Ultra sub via the
# Antigravity CLI (`agy`). Marginal cash cost is $0 (flat $100/mo Ultra sub), but
# for fleet cost *visibility* we price its turns at the underlying vendors' official
# rates and label the result "estimated" — mirroring the notional-Anthropic relays.
# Unlike those single-vendor relays, this bridge is POLY-VENDOR: the one endpoint
# fronts Gemini 3.x AND Claude Opus/Sonnet 4.6 AND GPT-OSS 120B, so a turn is routed
# to the vendor INFERRED from its (alias-normalized) model id, not to a fixed vendor.
NOTIONAL_SUBSCRIPTION_BRIDGE_PROVIDERS = frozenset({
    "gemini-bridge",
})

# The gemini-bridge exposes short model ALIASES that are not pricing keys; map each
# to the canonical (priced) model id of the real model the Ultra sub fronts. Mirrors
# `_MODEL_ALIASES` in ~/.hermes/gemini-bridge/gemini_bridge.py (kept in sync there).
_GEMINI_BRIDGE_MODEL_ALIASES = {
    "gemini-flash": "gemini-3.5-flash",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "gemini-pro": "gemini-3.1-pro",
    "gemini-3.1-pro": "gemini-3.1-pro",
    "claude-opus": "claude-opus-4-6",
    "claude-sonnet": "claude-sonnet-4-6",
    "gpt-oss": "gpt-oss-120b",
}

# The exact set of canonical models the bridge actually fronts (the alias-map
# values). Only these route to a priced vendor; anything else — including a
# prefix-shaped typo/unsupported id like "gemini-2.0-flash" that _infer_vendor
# WOULD map to google — is deliberately routed "unknown" so a misconfigured or
# unsupported bridge model surfaces in diagnostics instead of masquerading as a
# valid priced route.
_GEMINI_BRIDGE_CANONICAL_MODELS = frozenset(_GEMINI_BRIDGE_MODEL_ALIASES.values())


def is_notional_subscription_bridge(provider_name: Optional[str]) -> bool:
    """True if a provider key is a notional (subscription-fronting) poly-vendor bridge.

    Currently just the gemini-bridge (Google AI Ultra sub via agy). Its model id is
    normalized through ``_GEMINI_BRIDGE_MODEL_ALIASES`` then priced under the vendor
    inferred from the id (google/anthropic/openai) at official-docs rates.
    """
    p = (provider_name or "").strip().lower()
    return bool(p) and p in NOTIONAL_SUBSCRIPTION_BRIDGE_PROVIDERS


def _normalize_gemini_bridge_model(model: str) -> str:
    """Map a gemini-bridge alias (gemini-flash, claude-opus, gpt-oss, …) to its
    canonical priced model id. Strips a leading ``gemini-bridge/`` provider prefix
    first, and passes through an already-canonical id unchanged."""
    name = (model or "").split("/")[-1].lower().strip()
    return _GEMINI_BRIDGE_MODEL_ALIASES.get(name, name)

# Notional pricing for the xAI Grok OAuth provider (xai-oauth). Marginal cash
# cost is $0 (covered by a flat SuperGrok / X Premium+ subscription), but for
# fleet cost *visibility* we price its turns at xAI's official API rates and
# label the result "estimated" — mirroring the notional-Anthropic relays above.
# xAI is SINGLE-vendor (the OAuth provider fronts only Grok models), so a turn
# routes to a fixed "xai" vendor (unlike the poly-vendor gemini-bridge). The
# metered direct-API provider ("xai", api.x.ai key) prices from the SAME snapshot
# entries below — it just bills real dollars instead of notional ones, so it is
# routed to the "xai" vendor there too (see resolve_billing_route()).
NOTIONAL_XAI_PROVIDERS = frozenset({
    "xai-oauth",
})


def is_notional_xai_provider(provider_name: Optional[str]) -> bool:
    """True if a provider key is the subscription-fronting xAI Grok OAuth relay.

    The xai-oauth provider fronts Grok models on a flat SuperGrok / X Premium+
    subscription (marginal cash $0); we price its turns at xAI's official API
    rates for cost visibility and label them "estimated". Single-vendor: every
    model behind it is an xAI Grok model, so it routes to the fixed "xai" vendor.
    """
    p = (provider_name or "").strip().lower()
    return bool(p) and p in NOTIONAL_XAI_PROVIDERS


# Notional pricing for ChatGPT-subscription Codex providers (openai-codex).
# Marginal cash cost is $0 (covered by a flat ChatGPT subscription), but for
# fleet cost *visibility* we price these at OpenRouter's live catalog rates for
# the underlying OpenAI model and label the result "estimated". Unlike the
# Anthropic proxies (which fall back to a static docs snapshot), OpenAI models
# are priced from the configured external pricing catalog (models.dev by
# default, OpenRouter as fallback) — the same dynamic source that powers
# `provider: openrouter` routes fleet-wide. See resolve_billing_route() and
# _external_pricing_entry().
NOTIONAL_OPENROUTER_PROVIDERS = frozenset({
    "openai-codex",
})


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
class CostResult:
    amount_usd: Optional[Decimal]
    status: CostStatus
    source: CostSource
    label: str
    fetched_at: Optional[datetime] = None
    pricing_version: Optional[str] = None
    notes: tuple[str, ...] = ()
    # Per-class cost breakdown (SPEC-C). The engine already computes each class
    # term below; these expose the parts so consumers (blackbox telemetry →
    # tokens.ace) can show WHY a turn cost what it did without a second pricing
    # system. They sum to ``amount_usd`` for a priced turn; all None when the
    # turn is unknown/unpriceable. ``cost_input_usd`` is the fresh-input class
    # (a.k.a. "uncached"). Cache-write-at-input-rate (OpenAI family, no separate
    # cache-write fee) is attributed to ``cost_cache_write_usd`` — it is the
    # real cost of those cache-write tokens.
    cost_input_usd: Optional[Decimal] = None
    cost_output_usd: Optional[Decimal] = None
    cost_cache_read_usd: Optional[Decimal] = None
    cost_cache_write_usd: Optional[Decimal] = None


_UTC_NOW = lambda: datetime.now(timezone.utc)


# Official docs snapshot entries. Models whose published pricing and cache
# semantics are stable enough to encode exactly.
_OFFICIAL_DOCS_PRICING: Dict[tuple[str, str], PricingEntry] = {
    # ── xAI Grok ─────────────────────────────────────────────────────────
    # Priced from OpenRouter's live catalog snapshot (per-1M in/out; cache
    # read = input_cache_read; xAI publishes no cache-write rate → None).
    # Keyed on the "xai" vendor: both the notional SuperGrok/Premium+ OAuth
    # relay (xai-oauth, is_notional_xai_provider → status "estimated", $0
    # real cash) AND the metered direct api.x.ai key (provider "xai") route
    # here. Source: https://openrouter.ai/x-ai
    (
        "xai",
        "grok-build-0.1",
    ): PricingEntry(
        input_cost_per_million=Decimal("1.00"),
        output_cost_per_million=Decimal("2.00"),
        cache_read_cost_per_million=Decimal("0.20"),
        source="official_docs_snapshot",
        source_url="https://openrouter.ai/x-ai/grok-build-0.1",
        pricing_version="xai-pricing-2026-07",
    ),
    (
        "xai",
        "grok-4.5",
    ): PricingEntry(
        input_cost_per_million=Decimal("2.00"),
        output_cost_per_million=Decimal("6.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        source="official_docs_snapshot",
        source_url="https://openrouter.ai/x-ai/grok-4.5",
        pricing_version="xai-pricing-2026-07",
    ),
    (
        "xai",
        "grok-4.3",
    ): PricingEntry(
        input_cost_per_million=Decimal("1.25"),
        output_cost_per_million=Decimal("2.50"),
        cache_read_cost_per_million=Decimal("0.20"),
        source="official_docs_snapshot",
        source_url="https://openrouter.ai/x-ai/grok-4.3",
        pricing_version="xai-pricing-2026-07",
    ),
    (
        "xai",
        "grok-4.20",
    ): PricingEntry(
        input_cost_per_million=Decimal("1.25"),
        output_cost_per_million=Decimal("2.50"),
        cache_read_cost_per_million=Decimal("0.20"),
        source="official_docs_snapshot",
        source_url="https://openrouter.ai/x-ai/grok-4.20",
        pricing_version="xai-pricing-2026-07",
    ),
    (
        "xai",
        "grok-4.20-multi-agent",
    ): PricingEntry(
        input_cost_per_million=Decimal("1.25"),
        output_cost_per_million=Decimal("2.50"),
        cache_read_cost_per_million=Decimal("0.20"),
        source="official_docs_snapshot",
        source_url="https://openrouter.ai/x-ai/grok-4.20-multi-agent",
        pricing_version="xai-pricing-2026-07",
    ),
    # ── OpenAI GPT-5.6 series (Sol/Terra/Luna) ───────────────────────────
    # Announced in limited preview 2026-06-26; GA 2026-07-09 at the same
    # rates (Sol $5/$30, Terra $2.50/$15, Luna $1/$6 per 1M in/out). Cache
    # writes are billed at 1.25x the uncached input rate; cache reads get the
    # standard 90% discount (0.10x input, confirmed: Sol $0.50/M cached).
    # Note: "Sol Fast mode" ($12.5/$75, up to 750 tok/s via Cerebras) is a
    # separate serving tier, not covered by these entries. The "-pro"
    # variants (high-effort modes, GA alongside base tiers) bill at the
    # SAME per-token rates and are aliased onto these entries below the
    # dict (they cost more per task by consuming more tokens, not by a
    # higher rate — verified against OpenRouter's live pricing 2026-07-09).
    # Source: https://openai.com/index/previewing-gpt-5-6-sol/
    (
        "openai",
        "gpt-5.6-sol",
    ): PricingEntry(
        input_cost_per_million=Decimal("5.00"),
        output_cost_per_million=Decimal("30.00"),
        cache_read_cost_per_million=Decimal("0.50"),
        cache_write_cost_per_million=Decimal("6.25"),
        source="official_docs_snapshot",
        source_url="https://openai.com/index/previewing-gpt-5-6-sol/",
        pricing_version="openai-gpt-5.6-2026-07",
    ),
    (
        "openai",
        "gpt-5.6-terra",
    ): PricingEntry(
        input_cost_per_million=Decimal("2.50"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.25"),
        cache_write_cost_per_million=Decimal("3.125"),
        source="official_docs_snapshot",
        source_url="https://openai.com/index/previewing-gpt-5-6-sol/",
        pricing_version="openai-gpt-5.6-2026-07",
    ),
    (
        "openai",
        "gpt-5.6-luna",
    ): PricingEntry(
        input_cost_per_million=Decimal("1.00"),
        output_cost_per_million=Decimal("6.00"),
        cache_read_cost_per_million=Decimal("0.10"),
        cache_write_cost_per_million=Decimal("1.25"),
        source="official_docs_snapshot",
        source_url="https://openai.com/index/previewing-gpt-5-6-sol/",
        pricing_version="openai-gpt-5.6-2026-07",
    ),
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
    # ── Anthropic Claude Fable 5 ─────────────────────────────────────────
    # Premium 1M-context model priced at the opus-4-8-fast tier ($10/$50,
    # cache read 0.1x input, cache write 1.25x input). Subscription relays
    # (claude-apr/-bpr/-api-proxy/-bridge) price NOTIONAL via
    # is_notional_anthropic_provider(); this entry supplies the rate.
    # Source: https://openrouter.ai/anthropic/claude-fable-5
    (
        "anthropic",
        "claude-fable-5",
    ): PricingEntry(
        input_cost_per_million=Decimal("10.00"),
        output_cost_per_million=Decimal("50.00"),
        cache_read_cost_per_million=Decimal("1.00"),
        cache_write_cost_per_million=Decimal("12.50"),
        source="official_docs_snapshot",
        source_url="https://openrouter.ai/anthropic/claude-fable-5",
        pricing_version="anthropic-pricing-2026-06",
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
    # Claude Sonnet 5 (released 2026-06-30). List price $3/$15; cache read $0.30 (0.1x input).
    # Intro pricing $2/$10 in/out runs through 2026-08-31 — the cost-book uses the
    # standing LIST rate (as the rest of this table does), so it does not under-count
    # once intro ends. Subscription relays (claude-apr/-bpr/-api-proxy/-bridge) price
    # NOTIONAL via is_notional_anthropic_provider(); this entry only prices the bare
    # "anthropic" provider (direct key / Bedrock / Vertex).
    (
        "anthropic",
        "claude-sonnet-5",
    ): PricingEntry(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        cache_read_cost_per_million=Decimal("0.30"),
        cache_write_cost_per_million=Decimal("3.75"),
        source="official_docs_snapshot",
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_version="anthropic-pricing-2026-06",
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
    # Gemini 3.x — fronted by the Google AI Ultra subscription (gemini-bridge).
    # Rates from the OpenRouter catalog (2026-07-05); the AI Studio pricing page
    # lists the same tiers. Priced here so notional gemini-bridge turns resolve.
    (
        "google",
        "gemini-3.5-flash",
    ): PricingEntry(
        input_cost_per_million=Decimal("1.50"),
        output_cost_per_million=Decimal("9.00"),
        source="official_docs_snapshot",
        source_url="https://ai.google.dev/pricing",
        pricing_version="google-pricing-2026-07",
    ),
    (
        "google",
        "gemini-3.1-pro",
    ): PricingEntry(
        input_cost_per_million=Decimal("2.00"),
        output_cost_per_million=Decimal("12.00"),
        source="official_docs_snapshot",
        source_url="https://ai.google.dev/pricing",
        pricing_version="google-pricing-2026-07",
    ),
    # GPT-OSS 120B — the open-weight model the Ultra sub also fronts via agy.
    # Rate from the OpenRouter catalog (2026-07-05).
    (
        "openai",
        "gpt-oss-120b",
    ): PricingEntry(
        input_cost_per_million=Decimal("0.03"),
        output_cost_per_million=Decimal("0.15"),
        source="official_docs_snapshot",
        source_url="https://openrouter.ai/openai/gpt-oss-120b",
        pricing_version="openai-pricing-2026-07",
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

# GPT-5.6 "-pro" high-effort variants bill at the same per-token rates as
# their base tiers (more tokens per task, not a higher rate). Alias them
# onto the base entries so the snapshot stays single-source.
for _base_56 in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
    _OFFICIAL_DOCS_PRICING[("openai", f"{_base_56}-pro")] = _OFFICIAL_DOCS_PRICING[
        ("openai", _base_56)
    ]
del _base_56


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

    # Notional pricing for local subscription proxies/bridges that front the
    # Anthropic API (Claude Code OAuth billing, tailnet failovers, etc.). The
    # marginal cash cost is $0 (covered by a flat subscription), but for fleet
    # cost *visibility* we price these at official Anthropic API rates and label
    # the result "estimated" so /cost cards, top, and session rollups carry
    # meaningful numbers. See NOTIONAL_ANTHROPIC_PROVIDERS.
    if is_notional_anthropic_provider(provider_name):
        return BillingRoute(
            provider="anthropic",
            model=model.split("/")[-1],
            base_url=base_url or "",
            billing_mode="official_docs_snapshot",
        )

    # Notional pricing for the xAI Grok OAuth provider (xai-oauth). Marginal cash
    # cost is $0 (flat SuperGrok / X Premium+ sub), but for cost *visibility* we
    # price its Grok turns at xAI's official API rates (status "estimated").
    # Single-vendor: everything behind it is an xAI Grok model, so route to the
    # fixed "xai" vendor. The metered direct-API provider ("xai", api.x.ai) hits
    # the SAME snapshot entries just below and bills real dollars.
    if is_notional_xai_provider(provider_name):
        return BillingRoute(
            provider="xai",
            model=model.split("/")[-1],
            base_url=base_url or "",
            billing_mode="official_docs_snapshot",
        )

    # Notional pricing for the poly-vendor Google AI Ultra bridge (gemini-bridge).
    # Marginal cash cost is $0 (flat Ultra sub), but for cost *visibility* we
    # normalize its alias (gemini-flash/claude-opus/gpt-oss/…) to the canonical
    # model id and route to the vendor INFERRED from that id (google/anthropic/
    # openai), priced at official-docs rates. A poly-vendor analogue of the
    # single-vendor notional-Anthropic relays above.
    if is_notional_subscription_bridge(provider_name):
        canonical = _normalize_gemini_bridge_model(model)
        # Only price models the bridge actually fronts. An id that isn't a known
        # fronted model — even a prefix-valid one like "gemini-2.0-flash" that
        # _infer_vendor_from_model would map to google — routes as unsupported so a
        # misconfigured/unsupported bridge model surfaces in diagnostics/rollups
        # instead of masquerading as a valid priced route.
        #
        # Uses the distinct "unsupported_notional" billing_mode (NOT bare
        # "unknown") so _lookup_official_docs_pricing can suppress its M1 vendor
        # fallback for THIS case only. A bare "unknown" would be re-priced by M1 —
        # which infers the vendor from the model id (gemini-2.0-flash → google) and
        # would resurrect exactly the $0.50-at-Google-rates route we mean to reject.
        # M1 must keep working for its real job (a vendor-named model on a
        # mismatched/unknown provider, e.g. a custom localhost endpoint), so we
        # can't blanket-guard M1 on "unknown"; the sentinel scopes the suppression.
        if canonical not in _GEMINI_BRIDGE_CANONICAL_MODELS:
            return BillingRoute(
                provider="unknown",
                model=canonical,
                base_url=base_url or "",
                billing_mode="unsupported_notional",
            )
        vendor = _infer_vendor_from_model(canonical)
        if not vendor:
            return BillingRoute(
                provider="unknown",
                model=canonical,
                base_url=base_url or "",
                billing_mode="unsupported_notional",
            )
        return BillingRoute(
            provider=vendor,
            model=canonical,
            base_url=base_url or "",
            billing_mode="official_docs_snapshot",
        )

    # Notional pricing for ChatGPT-subscription Codex routes. Marginal cash cost
    # is $0, but for cost *visibility* we resolve to the underlying OpenAI model
    # and price it from the live OpenRouter catalog (status "estimated"). The
    # OpenRouter catalog lists base + dated families (gpt-5.5, gpt-5.3-codex,
    # ...) but not every "-codex" variant (e.g. gpt-5.5-codex is absent while
    # gpt-5.5 is present), so _normalize_codex_model_name() strips a trailing
    # "-codex" as a fallback when the exact id is missing.
    if provider_name in NOTIONAL_OPENROUTER_PROVIDERS:
        return BillingRoute(
            provider="openrouter",
            model=model.split("/")[-1],
            base_url=base_url or "",
            billing_mode="official_models_api",
        )
    if provider_name == "openai-codex":
        return BillingRoute(provider="openai-codex", model=model, base_url=base_url or "", billing_mode="subscription_included")
    if provider_name == "openrouter" or base_url_host_matches(base_url or "", "openrouter.ai"):
        return BillingRoute(provider="openrouter", model=model, base_url=base_url or "", billing_mode="official_models_api")
    if provider_name == "nous" or base_url_host_matches(base_url or "", "inference-api.nousresearch.com"):
        return BillingRoute(provider="nous", model=model, base_url=base_url or _NOUS_DEFAULT_BASE_URL, billing_mode="official_models_api")
    if provider_name == "anthropic":
        return BillingRoute(provider="anthropic", model=model.split("/")[-1], base_url=base_url or "", billing_mode="official_docs_snapshot")
    # "openai-api" is the picker/registry slug for direct api.openai.com; it
    # bills identically to bare "openai", so normalize it here — otherwise the
    # ("openai", <model>) _OFFICIAL_DOCS_PRICING keys are unreachable from the
    # openai-api provider path.
    if provider_name in {"openai", "openai-api"}:
        return BillingRoute(provider="openai", model=model.split("/")[-1], base_url=base_url or "", billing_mode="official_docs_snapshot")
    if provider_name in {"minimax", "minimax-cn"}:
        return BillingRoute(provider=provider_name, model=model.split("/")[-1], base_url=base_url or "", billing_mode="official_docs_snapshot")
    # Metered direct xAI API (api.x.ai, XAI_API_KEY). Bills real dollars; prices
    # from the same official-docs Grok snapshot as the notional xai-oauth relay.
    if provider_name in {"xai", "xai-api", "x-ai"} or base_url_host_matches(base_url or "", "api.x.ai"):
        return BillingRoute(provider="xai", model=model.split("/")[-1], base_url=base_url or "", billing_mode="official_docs_snapshot")
    # Vertex AI hosts the same Gemini models as Google AI Studio; price them
    # off the gemini official-docs snapshot. Strip the "google/" vendor prefix
    # the OpenAI-compat endpoint requires so the pricing key matches.
    if provider_name == "vertex" or base_url_host_matches(base_url or "", "aiplatform.googleapis.com"):
        return BillingRoute(provider="gemini", model=model.split("/")[-1], base_url=base_url or "", billing_mode="official_docs_snapshot")
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


# Trailing 8-digit release-date suffix on a NEW-scheme Anthropic model id, e.g.
# claude-haiku-4-5-20251001 → the dated alias of claude-haiku-4-5. Anchored to
# the end and require a preceding "-N" version segment so we only strip a date
# that was APPENDED to an already-versioned name — never the canonical OLD-scheme
# ids whose date IS part of the name (claude-3-5-haiku-20241022, where stripping
# would land on the non-existent claude-3-5-haiku). Used ONLY as a last-resort
# snapshot fallback, after direct + dot-normalized lookups, so a real dated entry
# always wins.
_ANTHROPIC_DATED_SUFFIX_RE = re.compile(r"^(claude-.*-\d+-\d+)-\d{8}$")


def _strip_anthropic_release_date(name: str) -> Optional[str]:
    """Strip a trailing -YYYYMMDD appended to a versioned new-scheme name.

    claude-haiku-4-5-20251001 → claude-haiku-4-5 ; claude-opus-4-8-20260115 →
    claude-opus-4-8. Returns None when there is no such suffix to strip (incl.
    the old-scheme claude-3-5-haiku-20241022, which lacks the -N-N version tail
    before the date and so is left intact for its own direct entry).
    """
    m = _ANTHROPIC_DATED_SUFFIX_RE.match(name)
    return m.group(1) if m else None


def _infer_vendor_from_model(model: str) -> Optional[str]:
    """Infer the pricing vendor from an unambiguous model-id prefix (M1, SPEC §5B).

    Assumes these prefixes are vendor-EXCLUSIVE (true 2026-06-30): a future prefix
    collision (a non-Anthropic ``claude-*``, etc.) must revisit this map — see
    SPEC INV-2/RR-1. Used ONLY as a last-resort pricing fallback in
    ``_lookup_official_docs_pricing`` — a real ``(provider, model)`` entry always
    wins first, so this never overrides a specific price. Returns None when the id
    names no known vendor.
    """
    name = (model or "").lower().strip()
    if name.startswith("claude-"):
        return "anthropic"
    if name.startswith("gpt-") or re.fullmatch(r"o[1-9][a-z0-9.\-]*", name):
        return "openai"
    if name.startswith("gemini-"):
        return "google"
    if name.startswith("grok-"):
        return "xai"
    return None


def _lookup_official_docs_pricing(route: BillingRoute) -> Optional[PricingEntry]:
    """Resolve a route to a static official-docs pricing entry, most-specific first.

    Lookup precedence (first hit wins, so a more-specific entry never loses to a
    fallback): exact ``(provider, model)`` → for Anthropic, the dot-normalized
    name (e.g. ``opus-4.7`` → ``opus-4-7``) → then the date-stripped base of a
    versioned new-scheme id (``claude-haiku-4-5-20251001`` → ``claude-haiku-4-5``).
    Returns None when no entry matches at any tier.
    """
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
        # Last-resort: strip a trailing -YYYYMMDD release date appended to a
        # versioned new-scheme id and retry on the base (claude-haiku-4-5-
        # 20251001 → claude-haiku-4-5). Runs AFTER direct + dot-normalized so a
        # real dated entry (e.g. claude-3-5-haiku-20241022) always wins on its
        # own key. Fixes the dated-Haiku unpriced gap (audit 2026-06-17).
        base = _strip_anthropic_release_date(normalized)
        if base and base != normalized:
            entry = _OFFICIAL_DOCS_PRICING.get((route.provider, base))
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
    # LAST-RESORT vendor fallback (M1, SPEC §5B): the (provider, model) is unpriced
    # at every specific tier above, but the model id unambiguously names a vendor.
    # Retry the WHOLE tiered lookup under that vendor as provider. ONE hop only:
    # guarded by ``vendor != route.provider`` so the inner call (provider == vendor)
    # cannot re-infer-and-recurse (RC-5 termination). Fixes Class A (openai/claude-*)
    # and every future proxy/bridge lane with a vendor-named model.
    #
    # EXCEPTION: a notional subscription bridge that already rejected this model as
    # unsupported (billing_mode "unsupported_notional") must NOT be re-priced here.
    # M1 would infer the vendor from the id (e.g. gemini-2.0-flash → google) and
    # resurrect the exact $0.50-at-Google-rates route resolve_billing_route
    # deliberately refused — masking a misconfigured/unsupported bridge model as a
    # valid priced route. The bridge's supported set is known and small, so an
    # unsupported-but-vendor-named id stays unpriced by design.
    if route.billing_mode == "unsupported_notional":
        return None
    vendor = _infer_vendor_from_model(route.model)
    if vendor and vendor != route.provider:
        return _lookup_official_docs_pricing(
            BillingRoute(
                provider=vendor,
                model=route.model,
                base_url=route.base_url,
                billing_mode=route.billing_mode,
            )
        )
    return None


def _normalize_codex_model_name(model: str) -> Optional[str]:
    """Map a Codex model id to its underlying OpenAI catalog id when they differ.

    OpenRouter lists base/dated OpenAI families (gpt-5.5, gpt-5.3-codex, ...) but
    not every "-codex" variant. When an exact id is missing we strip a trailing
    "-codex" segment so e.g. gpt-5.5-codex falls back to gpt-5.5. Returns None if
    no normalization applies (i.e. the name has no "-codex" suffix to strip).
    """
    name = model.lower().strip()
    if name.endswith("-codex"):
        return name[: -len("-codex")]
    return None


# --- External (dynamic-catalog) pricing sources ------------------------------
#
# For models NOT in the curated _OFFICIAL_DOCS_PRICING snapshot (e.g. the
# gpt-5.x family behind the openai-codex notional relay), price falls back to a
# live external catalog. Historically this was OpenRouter's models API only.
# It is now a CONFIGURABLE source (config.yaml `pricing.external_source`),
# defaulting to models.dev — a git-backed registry that already powers the
# fleet's context-window resolution and carries per-million cost fields. The
# curated snapshot ALWAYS takes precedence over any external source (see
# get_pricing_entry): the snapshot encodes deliberate corrections a live
# catalog can't know — most importantly the list-price-not-intro-promo rule
# (e.g. Sonnet 5's standing $3/$15, where models.dev/OpenRouter track the
# temporary $2/$10 intro rate and would under-bill once the promo lapses).

_VALID_PRICING_SOURCES = ("models_dev", "openrouter")
_DEFAULT_PRICING_SOURCE = "models_dev"


def _pricing_source_order() -> tuple:
    """Resolve the external pricing-source preference order from config.yaml.

    Reads `pricing.external_source` (default 'models_dev'). Returns a tuple with
    the configured primary first and the other valid source(s) after it as
    automatic fallback, so a model missing from the primary catalog still
    resolves from the secondary. Any unrecognized/missing value degrades to the
    default order (never raises — pricing must not break a turn).
    """
    primary = _DEFAULT_PRICING_SOURCE
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        pricing_cfg = cfg.get("pricing")
        if isinstance(pricing_cfg, dict):
            candidate = pricing_cfg.get("external_source")
            if isinstance(candidate, str) and candidate.strip().lower() in _VALID_PRICING_SOURCES:
                primary = candidate.strip().lower()
    except Exception:
        primary = _DEFAULT_PRICING_SOURCE
    # Primary first, then the remaining valid sources as ordered fallback.
    return (primary, *[s for s in _VALID_PRICING_SOURCES if s != primary])


def _models_dev_pricing_entry(route: BillingRoute) -> Optional[PricingEntry]:
    """Build a PricingEntry from the models.dev registry cost block.

    models.dev stores cost already in per-million USD, so (unlike the OpenRouter
    path) no per-token→per-million conversion is applied. Resolution mirrors the
    context-window path: try the model's mapped provider first, then a
    cross-provider scan (for aggregator ids that belong to no single mapped
    provider). Returns None if models.dev has no usable cost for the id.
    """
    from agent.models_dev import (
        lookup_models_dev_pricing,
        lookup_models_dev_pricing_any_provider,
    )

    cost = lookup_models_dev_pricing(route.provider, route.model)
    if cost is None:
        cost = lookup_models_dev_pricing_any_provider(route.model)
    if cost is None:
        # "-codex" variant absent → price the stripped base model id.
        fallback = _normalize_codex_model_name(route.model)
        if fallback:
            cost = lookup_models_dev_pricing_any_provider(fallback)
    if cost is None:
        return None

    inp = _to_decimal(cost.get("input"))
    out = _to_decimal(cost.get("output"))
    if inp is None and out is None:
        return None
    cache_read = _to_decimal(cost.get("cache_read"))
    cache_write = _to_decimal(cost.get("cache_write"))
    return PricingEntry(
        input_cost_per_million=inp,
        output_cost_per_million=out,
        cache_read_cost_per_million=cache_read,
        cache_write_cost_per_million=cache_write,
        request_cost=None,
        source="provider_models_api",
        source_url="https://models.dev/api.json",
        pricing_version="models-dev-api",
        fetched_at=_UTC_NOW(),
    )


def _openrouter_pricing_entry(route: BillingRoute) -> Optional[PricingEntry]:
    metadata = fetch_model_metadata()
    entry = _pricing_entry_from_metadata(
        metadata,
        route.model,
        source_url="https://openrouter.ai/docs/api/api-reference/models/get-models",
        pricing_version="openrouter-models-api",
    )
    if entry is not None:
        return entry
    # Fallback: a "-codex" variant absent from the catalog → price the base model.
    fallback = _normalize_codex_model_name(route.model)
    if fallback:
        return _pricing_entry_from_metadata(
            metadata,
            fallback,
            source_url="https://openrouter.ai/docs/api/api-reference/models/get-models",
            pricing_version="openrouter-models-api",
        )
    return None


# Dispatch table: pricing-source name → entry builder.
_PRICING_SOURCE_BUILDERS = {
    "models_dev": _models_dev_pricing_entry,
    "openrouter": _openrouter_pricing_entry,
}


def _external_pricing_entry(route: BillingRoute) -> Optional[PricingEntry]:
    """Price a route from the configured external catalog(s).

    Tries the config-selected primary source (default models.dev), then the
    other valid source as fallback, so a model absent from the primary catalog
    still resolves. This is ONLY consulted after the curated snapshot has
    missed (see get_pricing_entry), so the snapshot's deliberate corrections
    always win. Never raises.
    """
    for source in _pricing_source_order():
        builder = _PRICING_SOURCE_BUILDERS.get(source)
        if builder is None:
            continue
        try:
            entry = builder(route)
        except Exception:
            entry = None
        if entry is not None:
            return entry
    return None


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
        # Notional relays (openai-codex, provider: openrouter routes) and any
        # model absent from the curated snapshot price from the configured
        # external catalog — models.dev by default, OpenRouter as fallback
        # (config.yaml `pricing.external_source`). See _external_pricing_entry.
        return _external_pricing_entry(route)
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
            getattr(details, "cache_creation_tokens", 0) if details else 0
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
            # No published cache-write rate. For OpenAI-family routes (and any
            # provider that doesn't charge a separate cache-write premium) the
            # live models API omits this field by design — cache-write tokens
            # are just input tokens that were also written to cache and are
            # billed at the regular input rate. So if we DO know the input
            # rate, price cache-write at the input rate rather than dropping
            # the entire turn as unpriced (which silently loses real spend).
            # Only bail when input pricing is ALSO missing (truly unpriceable).
            if entry.input_cost_per_million is None:
                return CostResult(
                    amount_usd=None,
                    status="unknown",
                    source=entry.source,
                    label="n/a",
                    notes=("cache-write pricing unavailable for route",),
                )
            notes.append(
                "cache-write priced at input rate (no separate cache-write rate published)"
            )

    cost_input = _ZERO
    cost_output = _ZERO
    cost_cache_read = _ZERO
    cost_cache_write = _ZERO
    if entry.input_cost_per_million is not None:
        cost_input = Decimal(usage.input_tokens) * entry.input_cost_per_million / _ONE_MILLION
        amount += cost_input
    if entry.output_cost_per_million is not None:
        cost_output = Decimal(usage.output_tokens) * entry.output_cost_per_million / _ONE_MILLION
        amount += cost_output
    if entry.cache_read_cost_per_million is not None:
        cost_cache_read = Decimal(usage.cache_read_tokens) * entry.cache_read_cost_per_million / _ONE_MILLION
        amount += cost_cache_read
    if entry.cache_write_cost_per_million is not None:
        cost_cache_write = Decimal(usage.cache_write_tokens) * entry.cache_write_cost_per_million / _ONE_MILLION
        amount += cost_cache_write
    elif usage.cache_write_tokens and entry.input_cost_per_million is not None:
        # Fallback: no published cache-write rate → bill at the input rate
        # (see the cache-write guard above). Correct for OpenAI-family routes.
        # Attributed to the cache-write class (it IS the cost of cache-write
        # tokens), not folded into input.
        cost_cache_write = Decimal(usage.cache_write_tokens) * entry.input_cost_per_million / _ONE_MILLION
        amount += cost_cache_write
    if entry.request_cost is not None and usage.request_count:
        # Per-request structural charge. Fold into the input/uncached class so
        # the four-class breakdown still sums exactly to amount_usd. (All fleet
        # routes have request_cost=None today; SPEC-C's backfill NULL-splits any
        # route that doesn't, since aggregate pricing can't reconstruct N calls.)
        req_amount = Decimal(usage.request_count) * entry.request_cost
        cost_input += req_amount
        amount += req_amount

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
        cost_input_usd=cost_input,
        cost_output_usd=cost_output,
        cost_cache_read_usd=cost_cache_read,
        cost_cache_write_usd=cost_cache_write,
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
