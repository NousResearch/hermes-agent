from __future__ import annotations

import logging
import re
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Literal, Optional

from agent.model_metadata import fetch_endpoint_model_metadata, fetch_model_metadata
from utils import base_url_host_matches, base_url_hostname

logger = logging.getLogger(__name__)

_ZERO = Decimal("0")
_ONE_MILLION = Decimal("1000000")
_NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"

# Below $0.01, render at 4 dp so cheap-model costs never display as $0.00.
# Sub-cent cost threshold: below $0.01, render at 4 decimal places so the display is non-zero (e.g. $0.0046
# instead of $0.00). See #79220.
_SUBCENT_THRESHOLD = Decimal("0.01")

# Attached to every CostResult with status="included" so consumers can
# distinguish "free because subscription" from "free because $0 pricing".
_INCLUDED_NOTE = "subscription-included; no provider invoice for usage"


def format_cost_label(amount: Decimal) -> str:
    """Cost display label: zero → "$0.00"; sub-cent → "~$0.0046" (4 dp, or
    "~$<0.0001" when it rounds to 0.0000 so the label never reads as zero);
    else "~$1.23". Shared by per-response labels and insights cost buckets.

    This fixes #79220 where sub-cent per-turn costs on cheap models (DeepSeek, etc.) rendered as "$0.00"
    despite amount_usd carrying full Decimal precision.
    """
    if amount == _ZERO:
        return "$0.00"
    if amount < _SUBCENT_THRESHOLD:
        label = f"~${amount:.4f}"
        # Compare the rendered label: a naive `< 0.00005` threshold misses
        # the exact boundary under ROUND_HALF_EVEN.
        # A positive amount that rounds to 0.0000 at 4 dp would render "~$0.0000" — a zero-looking label,
        # the exact #79220 dishonesty.
        return label if label != "~$0.0000" else "~$<0.0001"
    return f"~${amount:.2f}"

CostStatus = Literal["actual", "estimated", "included", "unknown"]
CostSource = Literal[
    "provider_cost_api", "provider_generation_api", "provider_models_api", "official_docs_snapshot",
    "user_override", "custom_contract", "none",
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
        """Sum two usage buckets. ``raw_usage`` (single-response detail) is
        dropped; ``request_count`` adds so callers see how many API calls a
        combined figure covers."""
        if not isinstance(other, CanonicalUsage):
            return NotImplemented
        return CanonicalUsage(**{
            f.name: getattr(self, f.name) + getattr(other, f.name)
            for f in fields(CanonicalUsage) if f.name != "raw_usage"
        })


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
    # Context-tiered pricing (e.g. Gemini Pro above 200k prompt tokens): when
    # ``usage.prompt_tokens`` exceeds ``tier_threshold_tokens`` the ``*_above``
    # rates replace the base rates for the WHOLE request (Google's semantics,
    # not marginal brackets). A None ``*_above`` falls back to its base rate.
    tier_threshold_tokens: Optional[int] = None
    input_cost_per_million_above: Optional[Decimal] = None
    output_cost_per_million_above: Optional[Decimal] = None
    cache_read_cost_per_million_above: Optional[Decimal] = None


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
_INCLUDED_ENTRY = PricingEntry(
    input_cost_per_million=_ZERO, output_cost_per_million=_ZERO, cache_read_cost_per_million=_ZERO,
    cache_write_cost_per_million=_ZERO, source="none", pricing_version="included-route",
)


def _snap(
    inp: str, out: str, cache_read: Optional[str] = None, cache_write: Optional[str] = None, *,
    version: str, url: Optional[str] = None, **tiers: Any,
) -> PricingEntry:
    """Build an official-docs snapshot entry from per-million USD rate strings."""
    return PricingEntry(
        input_cost_per_million=Decimal(inp), output_cost_per_million=Decimal(out),
        cache_read_cost_per_million=Decimal(cache_read) if cache_read is not None else None,
        cache_write_cost_per_million=Decimal(cache_write) if cache_write is not None else None,
        source="official_docs_snapshot", source_url=url, pricing_version=version, **tiers,
    )


# Official docs snapshot: models whose published pricing and cache semantics are
# stable enough to encode exactly. Each snapshot is (provider, source_url,
# pricing_version, {model-or-models: per-1M rates (input, output[, cache_read[,
# cache_write]])}); a tuple key shares one rate row across several model ids.
_BEDROCK_URL = "https://aws.amazon.com/bedrock/pricing/"
_ANTHROPIC_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
_GOOGLE_URL = "https://ai.google.dev/pricing"
_OPUS = ("5.00", "25.00", "0.50", "6.25")
_SONNET = ("3.00", "15.00", "0.30", "3.75")
_SNAPSHOTS: tuple[tuple[str, Optional[str], str, dict], ...] = (
    # OpenAI GPT-5.6 (Sol/Terra/Luna). Cache write = 1.25x input, cache read =
    # 0.10x input. "-pro" high-effort modes bill at the same per-token rates
    # (aliased below); "Sol Fast mode" is a separate tier, not covered.
    ("openai", "https://openai.com/index/previewing-gpt-5-6-sol/", "openai-gpt-5.6-2026-07", {
        "gpt-5.6-sol": ("5.00", "30.00", "0.50", "6.25"), "gpt-5.6-terra": ("2.50", "15.00", "0.25", "3.125"),
        "gpt-5.6-luna": ("1.00", "6.00", "0.10", "1.25"),
    }),
    # Claude 4.5/4.6/4.7/4.8 Opus share $5/$25 (new tokenizer, up to 35% more tokens).
    ("anthropic", _ANTHROPIC_URL, "anthropic-pricing-2026-05", {
        ("claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-7-20250507", "claude-opus-4-6",
         "claude-opus-4-6-20250414", "claude-opus-4-5"): _OPUS,
        ("claude-sonnet-4-6", "claude-sonnet-4-6-20250414", "claude-sonnet-4-5", "claude-sonnet-4-20250514",
         "claude-3-5-sonnet-20241022"): _SONNET,
        "claude-haiku-4-5": ("1.00", "5.00", "0.10", "1.25"),
        ("claude-opus-4-20250514", "claude-3-opus-20240229"): ("15.00", "75.00", "1.50", "18.75"),
        "claude-3-5-haiku-20241022": ("0.80", "4.00", "0.08", "1.00"),
        "claude-3-haiku-20240307": ("0.25", "1.25", "0.03", "0.30"),
    }),
    # Fast mode is a separate model id at a 2x premium.
    ("anthropic", "https://openrouter.ai/anthropic/claude-opus-4.8-fast", "anthropic-pricing-2026-05", {
        "claude-opus-4-8-fast": ("10.00", "50.00", "1.00", "12.50"),
    }),
    # Claude Sonnet 5: introductory $2/$10 through 2026-08-31, then $3/$15
    # (matching Sonnet 4.6). Update this entry when the intro window closes.
    ("anthropic", _ANTHROPIC_URL, "anthropic-pricing-2026-06-intro", {
        "claude-sonnet-5": ("2.00", "10.00", "0.20", "2.50"),
    }),
    ("openai", "https://openai.com/api/pricing/", "openai-pricing-2026-03-16", {
        "gpt-4o": ("2.50", "10.00", "1.25"), "gpt-4o-mini": ("0.15", "0.60", "0.075"),
        "gpt-4.1": ("2.00", "8.00", "0.50"), "gpt-4.1-mini": ("0.40", "1.60", "0.10"),
        "gpt-4.1-nano": ("0.10", "0.40", "0.025"), "o3": ("10.00", "40.00", "2.50"),
        "o3-mini": ("1.10", "4.40", "0.55"),
    }),
    # deepseek-chat / deepseek-reasoner are deprecated aliases of
    # deepseek-v4-flash's non-thinking / thinking modes — same rates.
    ("deepseek", "https://api-docs.deepseek.com/quick_start/pricing", "deepseek-pricing-2026-07", {
        ("deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash"): ("0.14", "0.28", "0.0028"),
        "deepseek-v4-pro": ("0.435", "0.87", "0.003625"),
    }),
    ("google", "https://ai.google.dev/gemini-api/docs/pricing", "google-pricing-2026-07-28", {
        "gemini-3.6-flash": ("1.50", "7.50", "0.15"), "gemini-3.5-flash-lite": ("0.30", "2.50", "0.03"),
    }),
    ("google", _GOOGLE_URL, "google-pricing-2026-07-07", {
        "gemini-3.5-flash": ("1.50", "9.00", "0.15"), "gemini-3.1-flash-lite": ("0.25", "1.50", "0.025"),
        "gemini-3-pro-preview": ("2.00", "12.00", "0.20"), "gemini-3-flash-preview": ("0.50", "3.00", "0.05"),
        "gemini-2.5-flash": ("0.15", "0.60", "0.015"), "gemini-2.0-flash": ("0.10", "0.40", "0.01"),
    }),
    # AWS Bedrock on-demand: same per-token rates as the model provider, billed
    # through AWS. Current-gen Claude rows are commercial-list snapshots (the AWS
    # Price List API had not published these SKUs machine-readably).
    ("bedrock", _BEDROCK_URL, "anthropic-list-2026-07", {
        ("anthropic.claude-opus-4-8", "anthropic.claude-opus-4-7", "anthropic.claude-opus-4-6"): _OPUS,
    }),
    ("bedrock", _BEDROCK_URL, "bedrock-pricing-2026-06", {"anthropic.claude-sonnet-5": _SONNET}),
    ("bedrock", _BEDROCK_URL, "bedrock-pricing-2026-04", {
        ("anthropic.claude-sonnet-4-6", "anthropic.claude-sonnet-4-5"): _SONNET,
        "anthropic.claude-haiku-4-5": ("0.80", "4.00", "0.08", "1.00"),
        "amazon.nova-pro": ("0.80", "3.20"), "amazon.nova-lite": ("0.06", "0.24"), "amazon.nova-micro": ("0.035", "0.14"),
    }),
    ("minimax", None, "minimax-pricing-2026-04", {"minimax-m2.7": ("0.30", "1.20")}),
    ("minimax-cn", None, "minimax-pricing-2026-04", {"minimax-m2.7": ("0.30", "1.20")}),
    # Fireworks AI serverless (Standard tier) publishes a per-model cached_input
    # rate (→ cache_read) but no separate cache_write rate. Fast/turbo tiers are
    # exposed as accounts/fireworks/routers/<name>, so rsplit("/", 1) yields
    # these distinct ids with their own (higher) rates.
    ("fireworks", "https://docs.fireworks.ai/serverless/pricing", "fireworks-pricing-2026-07", {
        "kimi-k2p6": ("0.95", "4.00", "0.16"), "kimi-k2p7-code": ("0.95", "4.00", "0.19"),
        "glm-5p2": ("1.40", "4.40", "0.14"), "deepseek-v4-pro": ("1.74", "3.48", "0.145"),
        "deepseek-v4-flash": ("0.14", "0.28", "0.028"), "qwen3p7-plus": ("0.40", "1.60", "0.08"),
        "minimax-m3": ("0.30", "1.20", "0.06"), "gpt-oss-120b": ("0.15", "0.60", "0.015"),
        "gpt-oss-20b": ("0.07", "0.30", "0.035"), "glm-5p1": ("1.40", "4.40", "0.26"),
        "minimax-m2p7": ("0.30", "1.20", "0.06"),
        ("kimi-k2p6-fast", "kimi-k2p6-turbo"): ("2.00", "8.00", "0.30"),
        "kimi-k2p7-code-fast": ("1.90", "8.00", "0.38"), "glm-5p2-fast": ("2.10", "6.60", "0.21"),
        "glm-5p1-fast": ("2.80", "8.80", "0.52"),
    }),
)

_OFFICIAL_DOCS_PRICING: Dict[tuple[str, str], PricingEntry] = {}
for _provider, _url, _version, _rows in _SNAPSHOTS:
    for _models, _rates in _rows.items():
        _entry = _snap(*_rates, version=_version, url=_url)
        for _model in ((_models,) if isinstance(_models, str) else _models):
            _OFFICIAL_DOCS_PRICING[(_provider, _model)] = _entry
del _SNAPSHOTS, _provider, _url, _version, _rows, _models, _rates, _entry, _model

# Context-tiered Gemini Pro: above 200k prompt tokens the *_above rates apply to
# the whole request (see PricingEntry).
_OFFICIAL_DOCS_PRICING[("google", "gemini-3.1-pro")] = _snap(
    "2.00", "12.00", "0.20", url=_GOOGLE_URL, version="google-pricing-2026-07-07",
    tier_threshold_tokens=200_000, input_cost_per_million_above=Decimal("4.00"),
    output_cost_per_million_above=Decimal("18.00"), cache_read_cost_per_million_above=Decimal("0.40"),
)
_OFFICIAL_DOCS_PRICING[("google", "gemini-2.5-pro")] = _snap(
    "1.25", "10.00", "0.125", url=_GOOGLE_URL, version="google-pricing-2026-07-07",
    tier_threshold_tokens=200_000, input_cost_per_million_above=Decimal("2.50"),
    output_cost_per_million_above=Decimal("15.00"),
)
del _BEDROCK_URL, _ANTHROPIC_URL, _GOOGLE_URL, _OPUS, _SONNET

# GPT-5.6 "-pro" high-effort variants bill at the base tier's per-token rates
# (more tokens per task, not a higher rate); the Hermes-side "-900k" Codex
# picker variants are the same model with the suffix stripped on the wire.
# The direct Gemini provider emits preview IDs for two models; key the snapshot
# by both the documented stable name and the emitted ID.
for _provider, _alias, _canonical in (
    *((("openai", f"{m}-{suffix}", m) for m in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna") for suffix in ("pro", "900k"))),
    ("google", "gemini-3.1-pro-preview", "gemini-3.1-pro"),
    ("google", "gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite"),
):
    _OFFICIAL_DOCS_PRICING[(_provider, _alias)] = _OFFICIAL_DOCS_PRICING[(_provider, _canonical)]
del _provider, _alias, _canonical


def _to_decimal(value: Any) -> Optional[Decimal]:
    try:
        return None if value is None else Decimal(str(value))
    except Exception:
        return None


def _usage_field(obj: Any, *path: str) -> int:
    """Non-negative int at ``obj.path[0].path[1]...``; 0 if any hop is falsy or
    non-numeric. Hops read dicts and attribute objects alike (the Responses API
    returns either); negative counters from providers are clamped so they cannot
    corrupt session accounting."""
    for hop in path:
        if not obj:
            return 0
        obj = obj.get(hop, 0) if isinstance(obj, dict) else getattr(obj, hop, 0)
    try:
        return max(0, int(obj or 0))
    except Exception:
        return 0


def _first_nonzero(obj: Any, *paths: tuple[str, ...]) -> int:
    """First non-zero ``_usage_field`` across candidate paths, else 0."""
    return next((v for v in (_usage_field(obj, *path) for path in paths) if v), 0)


# Picker slugs → snapshot provider key ("openai-api" is the slug for direct
# api.openai.com). Google and Fireworks are matched by name OR host below.
_SNAPSHOT_PROVIDER_ALIASES = {
    "anthropic": "anthropic", "openai": "openai", "openai-api": "openai", "minimax": "minimax", "minimax-cn": "minimax-cn",
}
# AI Studio and Vertex host the same Gemini models (the Vertex "google/" vendor
# prefix is stripped with the rest of the path).
_GOOGLE_PROVIDER_NAMES = {"google", "gemini", "vertex", "google-gemini", "google-ai-studio", "google-vertex", "vertex-ai"}


def resolve_billing_route(
    model_name: str, provider: Optional[str] = None, base_url: Optional[str] = None
) -> BillingRoute:
    provider_name = (provider or "").strip().lower()
    base = (base_url or "").strip().lower()
    model = (model_name or "").strip()
    if not provider_name and "/" in model:
        inferred_provider, bare_model = model.split("/", 1)
        if inferred_provider in {"anthropic", "openai", "google"}:
            provider_name = inferred_provider
            model = bare_model

    url = base_url or ""
    # Fireworks ids look like accounts/fireworks/models/<name>; keys use <name>.
    # Every other snapshot provider keys on the last path segment as well.
    bare = model.split("/")[-1]

    def host(name: str) -> bool:
        return base_url_host_matches(url, name)

    if provider_name == "openai-codex":
        return BillingRoute(provider="openai-codex", model=model, base_url=url, billing_mode="subscription_included")
    if provider_name == "openrouter" or host("openrouter.ai"):
        return BillingRoute(provider="openrouter", model=model, base_url=url, billing_mode="official_models_api")
    if provider_name == "nous" or host("inference-api.nousresearch.com"):
        return BillingRoute(provider="nous", model=model, base_url=base_url or _NOUS_DEFAULT_BASE_URL, billing_mode="official_models_api")
    snapshot_provider = _SNAPSHOT_PROVIDER_ALIASES.get(provider_name)
    if snapshot_provider is None:
        if (
            provider_name in _GOOGLE_PROVIDER_NAMES
            or host("aiplatform.googleapis.com") or host("generativelanguage.googleapis.com")
        ):
            snapshot_provider = "google"
        elif provider_name == "fireworks" or host("api.fireworks.ai"):
            snapshot_provider = "fireworks"
    if snapshot_provider:
        return BillingRoute(provider=snapshot_provider, model=bare, base_url=url, billing_mode="official_docs_snapshot")
    if provider_name in {"custom", "local"} or (base and base_url_hostname(base) in ("localhost", "127.0.0.1")):
        return BillingRoute(provider=provider_name or "custom", model=model, base_url=url, billing_mode="unknown")
    return BillingRoute(provider=provider_name or "unknown", model=bare if model else "", base_url=url, billing_mode="unknown")


_BEDROCK_REGION_PREFIXES = ("global.", "us.", "eu.", "apac.", "ap.", "au.", "jp.", "ca.", "sa.", "me.", "af.")
# Bedrock ids end in documented date/revision/profile components (``-20250514-v1:0``).
_BEDROCK_TRAILERS = (r":\d+$", r"-v\d+$", r"-\d{8}$")


def _strip_prefix(name: str, prefixes: tuple[str, ...]) -> str:
    """Drop the first matching prefix (at most one), else return ``name`` unchanged."""
    return next((name[len(p):] for p in prefixes if name.startswith(p)), name)


def _normalize_bedrock_model_name(model: str) -> str:
    """Bare foundation-model id: strip the cross-region inference-profile scope
    (``us.``/``global.``/...), map dotted versions (``4.7`` → ``4-7``), then
    strip the trailing date/revision/profile components."""
    name = re.sub(r"(\d+)\.(\d+)", r"\1-\2", _strip_prefix(model.lower().strip(), _BEDROCK_REGION_PREFIXES))
    for pattern in _BEDROCK_TRAILERS:
        name = re.sub(pattern, "", name)
    return name


def _normalize_anthropic_model_name(model: str) -> str:
    """Strip an ``anthropic/`` prefix and map dotted versions (4.7 → 4-7)."""
    return re.sub(r"(\d+)\.(\d+)", r"\1-\2", _strip_prefix(model.lower().strip(), ("anthropic/",)))


# Anthropic dot-notation (opus-4.7) and Bedrock region-prefixed ids need
# normalizing before a second lookup.
_MODEL_NORMALIZERS = {"anthropic": _normalize_anthropic_model_name, "bedrock": _normalize_bedrock_model_name}


def _lookup_official_docs_pricing(route: BillingRoute) -> Optional[PricingEntry]:
    model = route.model.lower()
    entry = _OFFICIAL_DOCS_PRICING.get((route.provider, model))
    if entry:
        return entry
    normalize = _MODEL_NORMALIZERS.get(route.provider)
    normalized = normalize(model) if normalize else model
    return _OFFICIAL_DOCS_PRICING.get((route.provider, normalized)) if normalized != model else None


def _openrouter_pricing_entry(route: BillingRoute) -> Optional[PricingEntry]:
    return _pricing_entry_from_metadata(
        fetch_model_metadata(), route.model,
        source_url="https://openrouter.ai/docs/api/api-reference/models/get-models",
        pricing_version="openrouter-models-api",
    )


def _pricing_entry_from_metadata(
    metadata: Dict[str, Dict[str, Any]], model_id: str, *, source_url: str, pricing_version: str
) -> Optional[PricingEntry]:
    if model_id not in metadata:
        return None
    pricing = metadata[model_id].get("pricing") or {}

    def per_million(key: str, *aliases: str) -> Optional[Decimal]:
        raw = pricing.get(key)
        for alias in aliases:  # alias chain is truthiness-based (``a or b or c``)
            raw = raw or pricing.get(alias)
        value = _to_decimal(raw)
        return None if value is None else value * _ONE_MILLION

    prompt = per_million("prompt")
    completion = per_million("completion")
    request = _to_decimal(pricing.get("request"))
    if prompt is None and completion is None and request is None:
        return None
    return PricingEntry(
        input_cost_per_million=prompt, output_cost_per_million=completion,
        cache_read_cost_per_million=per_million("cache_read", "cached_prompt", "input_cache_read"),
        cache_write_cost_per_million=per_million("cache_write", "cache_creation", "input_cache_write"),
        request_cost=request, source="provider_models_api", source_url=source_url,
        pricing_version=pricing_version, fetched_at=_UTC_NOW(),
    )


def get_pricing_entry(
    model_name: str, provider: Optional[str] = None, base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[PricingEntry]:
    route = resolve_billing_route(model_name, provider=provider, base_url=base_url)
    if route.billing_mode == "subscription_included":
        return _INCLUDED_ENTRY
    if route.provider == "openrouter":
        return _openrouter_pricing_entry(route)

    bundled_entry = _lookup_official_docs_pricing(route)
    if bundled_entry:
        return bundled_entry
    if route.base_url:
        entry = _pricing_entry_from_metadata(
            fetch_endpoint_model_metadata(route.base_url, api_key=api_key or ""), route.model,
            source_url=f"{route.base_url.rstrip('/')}/models",
            pricing_version="openai-compatible-models-api",
        )
        if entry:
            return entry
    return None


# Usage-field candidate paths per API shape: (input/prompt total, output, cache
# read, cache write); the first non-zero path wins.
_ANTHROPIC_USAGE_SHAPE = (
    (("input_tokens",),), (("output_tokens",),), (("cache_read_input_tokens",),), (("cache_creation_input_tokens",),)
)
# OpenAI's documented GPT-5.6+ field is `cache_write_tokens` (billed at 1.25x);
# `cache_creation_tokens` is a fallback for older endpoints.
_CODEX_USAGE_SHAPE = (
    (("input_tokens",),), (("output_tokens",),), (("input_tokens_details", "cached_tokens"),),
    (("input_tokens_details", "cache_write_tokens"), ("input_tokens_details", "cache_creation_tokens")),
)
# OpenAI-style names first, then Anthropic-style: local OpenAI-compatible
# servers (e.g. mlx_vlm.server) emit input_tokens/output_tokens and the OpenAI
# client preserves them as extra attributes. Cache reads: nested OpenAI shape,
# then Anthropic-style top-level fields exposed by proxies routing Claude
# (OpenRouter, Vercel AI Gateway, Cline), then DeepSeek's prompt_cache_hit_tokens,
# then Kimi/Moonshot's cached_tokens — without these, direct sessions show 0
# hits and bill hits at the full input rate.
_CHAT_USAGE_SHAPE = (
    (("prompt_tokens",), ("input_tokens",)),
    (("completion_tokens",), ("output_tokens",)),
    (("prompt_tokens_details", "cached_tokens"), ("cache_read_input_tokens",), ("prompt_cache_hit_tokens",), ("cached_tokens",)),
    (("prompt_tokens_details", "cache_write_tokens"), ("prompt_tokens_details", "cache_creation_input_tokens"),
     ("cache_creation_input_tokens",), ("cache_write_tokens",)),
)


def normalize_usage(
    response_usage: Any, *, provider: Optional[str] = None, api_mode: Optional[str] = None
) -> CanonicalUsage:
    """Normalize raw API response usage into canonical token buckets (Anthropic,
    Codex Responses, or OpenAI Chat Completions shape)."""
    if not response_usage:
        return CanonicalUsage()

    provider_name = (provider or "").strip().lower()
    mode = (api_mode or "").strip().lower()
    u = response_usage

    if mode == "anthropic_messages" or provider_name == "anthropic":
        shape = _ANTHROPIC_USAGE_SHAPE
    elif mode == "codex_responses":
        shape = _CODEX_USAGE_SHAPE
    else:
        shape = _CHAT_USAGE_SHAPE
    prompt_total, output_tokens, cache_read_tokens, cache_write_tokens = (
        _first_nonzero(u, *paths) for paths in shape
    )
    # Anthropic reports uncached input directly; Codex/Chat totals INCLUDE
    # cached tokens, so the cache buckets are subtracted back out.
    input_tokens = prompt_total if shape is _ANTHROPIC_USAGE_SHAPE else max(
        0, prompt_total - cache_read_tokens - cache_write_tokens
    )

    # Responses API: output_tokens_details.reasoning_tokens. Chat Completions
    # (OpenAI, OpenRouter, DeepSeek, ...): completion_tokens_details.reasoning_tokens.
    # Hidden thinking dominates output spend on reasoning models, so read both.
    reasoning_tokens = _first_nonzero(
        u, ("output_tokens_details", "reasoning_tokens"), ("completion_tokens_details", "reasoning_tokens")
    )

    # On MiniMax-M3's Anthropic wire, cache_read_input_tokens carries a constant
    # +128 floor and cache_creation is always 0, so cache_read is not a reliable
    # hit signal; the input_tokens drop between consecutive calls is.
    # Docs: https://platform.minimax.io/docs/api-reference/text-prompt-caching
    if provider_name in {"minimax", "minimax-cn"} and mode == "anthropic_messages":
        logger.debug(
            "cache_observability provider=%s mode=%s input_tokens=%s "
            "output_tokens=%s cache_read_tokens=%s cache_write_tokens=%s "
            "(note: on MiniMax-M3 cache_read carries a +128 constant "
            "floor and is not a reliable hit signal — track input_tokens "
            "drops across calls instead)",
            provider_name, mode, input_tokens, output_tokens,
            cache_read_tokens, cache_write_tokens,
        )

    return CanonicalUsage(
        input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens, reasoning_tokens=reasoning_tokens,
    )


def _unknown_cost(source: CostSource, *notes: str) -> CostResult:
    return CostResult(amount_usd=None, status="unknown", source=source, label="n/a", notes=notes)


def estimate_usage_cost(
    model_name: str, usage: CanonicalUsage, *, provider: Optional[str] = None,
    base_url: Optional[str] = None, api_key: Optional[str] = None,
) -> CostResult:
    route = resolve_billing_route(model_name, provider=provider, base_url=base_url)
    if route.billing_mode == "subscription_included":
        return CostResult(
            amount_usd=_ZERO, status="included", source="none", label="included",
            pricing_version="included-route", notes=(_INCLUDED_NOTE,),
        )

    entry = get_pricing_entry(model_name, provider=provider, base_url=base_url, api_key=api_key)
    if not entry:
        return _unknown_cost("none")

    # Whole-request context tier (e.g. Gemini Pro >200k prompts): above the
    # threshold the *_above rates apply to the entire request; None falls back.
    above = entry.tier_threshold_tokens is not None and usage.prompt_tokens > entry.tier_threshold_tokens
    amount = _ZERO
    for tokens, rate, rate_above, note in (
        (usage.input_tokens, entry.input_cost_per_million, entry.input_cost_per_million_above, ()),
        (usage.output_tokens, entry.output_cost_per_million, entry.output_cost_per_million_above, ()),
        (usage.cache_read_tokens, entry.cache_read_cost_per_million, entry.cache_read_cost_per_million_above,
         ("cache-read pricing unavailable for route",)),
        (usage.cache_write_tokens, entry.cache_write_cost_per_million, None,
         ("cache-write pricing unavailable for route",)),
    ):
        if above and rate_above is not None:
            rate = rate_above
        if rate is None:
            if tokens:
                return _unknown_cost(entry.source, *note)
            continue
        amount += Decimal(tokens) * rate / _ONE_MILLION
    if entry.request_cost is not None and usage.request_count:
        amount += Decimal(usage.request_count) * entry.request_cost

    notes: list[str] = []
    status: CostStatus = "estimated"
    label = format_cost_label(amount)
    if entry.source == "none" and amount == _ZERO:
        status = "included"
        label = "included"
        notes.append(_INCLUDED_NOTE)

    if route.provider == "openrouter":
        notes.append("OpenRouter cost is estimated from the models API until reconciled.")

    return CostResult(
        amount_usd=amount, status=status, source=entry.source, label=label,
        fetched_at=entry.fetched_at, pricing_version=entry.pricing_version, notes=tuple(notes),
    )


def has_known_pricing(
    model_name: str, provider: Optional[str] = None, base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bool:
    """True if pricing data exists for this model+route (direct lookup, no dummy usage)."""
    return get_pricing_entry(model_name, provider=provider, base_url=base_url, api_key=api_key) is not None


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
    return f"{hours / 24:.1f}d"


def format_token_count_compact(value: int) -> str:
    abs_value = abs(int(value))
    if abs_value < 1_000:
        return str(int(value))

    sign = "-" if value < 0 else ""
    threshold, suffix = next((t, sfx) for t, sfx in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")) if abs_value >= t)
    scaled = abs_value / threshold
    text = f"{scaled:.2f}" if scaled < 10 else f"{scaled:.1f}" if scaled < 100 else f"{scaled:.0f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{sign}{text}{suffix}"


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

DEFAULT_PRICING = {"input": 0.0, "output": 0.0}
# ---- END PLUGIN-COMPAT ----
