"""Model fallback heuristics and dynamic criteria resolution.

Allows specifying fallback criteria rather than hardcoded models, solving the problem
of ephemeral free models and dynamically selecting models based on heuristics such as:
- largest parameter count (e.g. 70b, 120b, 405b)
- greatest context window (e.g. 1M, 200k, 128k)
- smallest / most efficient model (e.g. 0.5b, 3b, 8b)
- latest model from a provider / vendor (newest release date / version)
- latest flash model (e.g. gemini-flash, deepseek-flash)

Blends multi-tier metadata:
1. Provider API live & cached metadata (OpenRouter /v1/models context_length, pricing, created, tools)
2. models.dev registry (context_window, release_date, cost, tool_call)
3. Model name heuristics (parameter extraction, context extraction, version/date parsing, tier words)
"""

from __future__ import annotations

import datetime
import fnmatch
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Precompiled regular expressions for high-throughput model name parsing
_RE_MOE_PARAM = re.compile(r"(?:^|[-_/ ])(\d+)x(\d+(?:\.\d+)?)[bB](?:[-_/ :]|$)", re.IGNORECASE)
_RE_MOE_ACTIVE_PARAM = re.compile(r"(?:^|[-_/ ])(\d+(?:\.\d+)?)[bB]-a\d+[bB](?:[-_/ :]|$)", re.IGNORECASE)
_RE_B_PARAM = re.compile(r"(?:^|[-_/ ])(\d+(?:\.\d+)?)[bB](?:[-_/ :]|$)", re.IGNORECASE)
_RE_CTX_MILLION = re.compile(r"(?:^|[-_/ ])(\d+(?:\.\d+)?)[mM](?:[-_/ :]|$)", re.IGNORECASE)
_RE_CTX_THOUSAND = re.compile(r"(?:^|[-_/ ])(\d+)[kK](?:[-_/ :]|$)", re.IGNORECASE)
_RE_VERSION_MULTI = re.compile(r"(?:^|[-_/v])(\d+(?:\.\d+)+)(?:[-_/:]|$)")
_RE_VERSION_SINGLE = re.compile(r"(?:^|[-_/v])(\d+)(?:[-_/:]|$)")
_RE_DATE_FULL = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")
_RE_DATE_MONTH_DAY = re.compile(r"(?:^|[-_])(0[1-9]|1[0-2])([0-3]\d)(?:[-_:]|$)")
_RE_SHORTHAND_SPLIT = re.compile(r"[:_,\s-]+")
_RE_QUALITATIVE_TIERS = re.compile(
    r"\b(nano|xs|micro|mini|haiku|small|lite|medium|plus|sonnet|pro|large|max|opus|ultra|super)\b|(-s-)",
    re.IGNORECASE,
)

# Known qualitative size tiers mapped to estimated parameter size in Billions
_QUALITATIVE_TIER_DICT: dict[str, float] = {
    "nano": 0.5,
    "xs": 1.0,
    "micro": 1.0,
    "mini": 3.0,
    "haiku": 7.0,
    "small": 7.0,
    "-s-": 7.0,
    "lite": 8.0,
    "medium": 14.0,
    "plus": 50.0,
    "sonnet": 70.0,
    "pro": 70.0,
    "large": 70.0,
    "max": 100.0,
    "opus": 100.0,
    "ultra": 200.0,
    "super": 300.0,
}
_QUALITATIVE_SIZE_TIERS: list[tuple[str, float]] = list(_QUALITATIVE_TIER_DICT.items())

_DESCRIBE_SORT_BY: dict[str, str] = {
    "largest_parameter_count": "largest parameter count",
    "largest": "largest parameter count",
    "greatest_context": "greatest context",
    "largest_context": "greatest context",
    "smallest": "smallest model",
    "latest": "latest model",
    "latest_flash": "latest flash model",
}


@dataclass
class FallbackCriteria:
    """Parsed criteria for dynamic fallback model resolution."""

    sort_by: str = "largest_parameter_count"
    provider: str = "nous"
    free_only: bool = True
    require_tools: bool = True
    flash_only: bool = False
    name_filter: str = ""
    vendor: str = ""
    min_context: int = 0
    max_context: int = 0
    reasoning_effort: Optional[str] = None
    max_candidates: int = 3
    raw_heuristic: str = ""

    @property
    def model_provider(self) -> str:
        """Alias for provider."""
        return self.provider

    @model_provider.setter
    def model_provider(self, val: str) -> None:
        self.provider = val

    def describe(self) -> str:
        """User-friendly summary of the criteria."""
        parts = []
        if self.provider and self.provider != "nous":
            parts.append(f"provider:{self.provider}")
        if self.flash_only:
            parts.append("flash")
        if self.vendor:
            parts.append(f"vendor:{self.vendor}")
        if self.name_filter:
            parts.append(f"filter:{self.name_filter}")
        if self.reasoning_effort and self.reasoning_effort != "default":
            parts.append(f"reasoning:{self.reasoning_effort}")

        sort_label = _DESCRIBE_SORT_BY.get(self.sort_by, self.sort_by)
        parts.append(sort_label)

        if self.free_only:
            parts.append("free")
        if self.min_context:
            parts.append(f"min_ctx:{self.min_context}")
        return " ".join(parts) or "default"

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict representation."""
        return asdict(self)


@dataclass
class ModelCandidateMetadata:
    """Enriched metadata for a model candidate evaluated against fallback criteria."""

    id: str
    name: str = ""
    provider: str = ""
    is_free: bool = False
    context_window: int = 0
    param_size_b: Optional[float] = None
    release_date: str = ""
    created_ts: float = 0.0
    date_snapshot: float = 0.0
    version_tuple: tuple[int, ...] = (0,)
    supports_tools: bool = False
    is_flash: bool = False
    vendor: str = ""
    source: str = "heuristic"

    def __post_init__(self):
        if not self.name:
            self.name = self.id


def is_free_model_slug(model_id: str) -> bool:
    """Detect if model identifier indicates a free tier model."""
    lower = model_id.lower()
    return (
        lower.endswith(":free")
        or "/free" in lower
        or "-free" in lower
        or ":free" in lower
        or lower.startswith("free-")
        or "free/" in lower
        or "stealth/union-alpha" in lower
    )


def is_flash_model(model_id: str, name: str = "") -> bool:
    """Detect if model is in the 'flash' / high-speed tier."""
    lower = f"{model_id} {name}".lower()
    return "flash" in lower or "haiku" in lower or "speed" in lower or "turbo" in lower


def extract_vendor(model_id: str) -> str:
    """Extract vendor/creator prefix from a model slug like 'google/gemini-2.5-flash'."""
    if "/" in model_id:
        return model_id.split("/", 1)[0].lower().strip()
    return ""


def extract_param_size_b(model_id: str, name: str = "") -> Optional[float]:
    """Extract parameter size in billions from model slug or display name."""
    text = f" {model_id} {name} "

    m = _RE_MOE_PARAM.search(text)
    if m:
        try:
            experts = int(m.group(1))
            size_per_expert = float(m.group(2))
            return round(experts * size_per_expert, 1)
        except (ValueError, TypeError):
            pass

    m = _RE_MOE_ACTIVE_PARAM.search(text)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass

    m = _RE_B_PARAM.search(text)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass

    m = _RE_QUALITATIVE_TIERS.search(text)
    if m:
        matched_term = (m.group(1) or m.group(2) or "").lower()
        if matched_term in _QUALITATIVE_TIER_DICT:
            return _QUALITATIVE_TIER_DICT[matched_term]

    return None


_K_POWER_OF_TWO: dict[int, int] = {
    4: 4096,
    8: 8192,
    16: 16384,
    32: 32768,
    64: 65536,
    128: 131072,
    256: 262144,
    512: 524288,
}


def extract_context_window_from_name(model_id: str, default: int = 0) -> int:
    """Extract context window in tokens from model slug."""
    text = f" {model_id} "
    m = _RE_CTX_MILLION.search(text)
    if m:
        try:
            val = float(m.group(1))
            if 0.1 <= val <= 32.0:
                return int(val * 1_000_000)
        except (ValueError, TypeError):
            pass

    m = _RE_CTX_THOUSAND.search(text)
    if m:
        try:
            val = int(m.group(1))
            if val in _K_POWER_OF_TWO:
                return _K_POWER_OF_TWO[val]
            if 4 <= val <= 2048:
                return val * 1000
        except (ValueError, TypeError):
            pass

    return default


def extract_version_tuple(model_id: str) -> tuple[int, ...]:
    """Extract version numbers (e.g. 3.3 or 2.5 or 1.5) from model ID."""
    m = _RE_VERSION_MULTI.search(model_id)
    if m:
        try:
            return tuple(int(p) for p in m.group(1).split("."))
        except (ValueError, TypeError):
            pass
    m = _RE_VERSION_SINGLE.search(model_id)
    if m:
        try:
            return (int(m.group(1)),)
        except (ValueError, TypeError):
            pass
    return (0,)


def extract_date_snapshot(model_id: str) -> float:
    """Extract date snapshot from model ID (e.g. 20241022 or 0813)."""
    m = _RE_DATE_FULL.search(model_id)
    if m:
        try:
            return float(m.group(1) + m.group(2) + m.group(3))
        except (ValueError, TypeError):
            pass
    m = _RE_DATE_MONTH_DAY.search(model_id)
    if m:
        try:
            return 20240000.0 + float(m.group(1) + m.group(2))
        except (ValueError, TypeError):
            pass
    return 0.0


# ─── Criteria Parsing ───────────────────────────────────────────────────────


def _apply_dict_fields(criteria: FallbackCriteria, raw: dict[str, Any], raw_dict: dict[str, Any]) -> None:
    """Apply dictionary fields to criteria object."""
    provider = (
        raw_dict.get("model_provider")
        or raw_dict.get("provider")
        or raw.get("model_provider")
        or raw.get("provider")
    )
    if provider:
        criteria.provider = str(provider).strip().lower()

    if "sort_by" in raw_dict:
        criteria.sort_by = str(raw_dict["sort_by"]).strip().lower()
    if "free" in raw_dict:
        criteria.free_only = bool(raw_dict["free"])
    elif "free_only" in raw_dict:
        criteria.free_only = bool(raw_dict["free_only"])
    if "require_tools" in raw_dict:
        criteria.require_tools = bool(raw_dict["require_tools"])
    if "flash" in raw_dict:
        criteria.flash_only = bool(raw_dict["flash"])
    elif "flash_only" in raw_dict:
        criteria.flash_only = bool(raw_dict["flash_only"])

    name_filter = raw_dict.get("filter") or raw_dict.get("name_filter") or raw_dict.get("pattern")
    if name_filter:
        criteria.name_filter = str(name_filter).strip().lower()

    vendor = raw_dict.get("vendor") or raw_dict.get("lab")
    if vendor:
        criteria.vendor = str(vendor).strip().lower()

    if "min_context" in raw_dict:
        try:
            criteria.min_context = int(raw_dict["min_context"])
        except (ValueError, TypeError):
            pass
    if "max_context" in raw_dict:
        try:
            criteria.max_context = int(raw_dict["max_context"])
        except (ValueError, TypeError):
            pass
    if "max_candidates" in raw_dict or "limit" in raw_dict:
        try:
            val = raw_dict.get("max_candidates") if "max_candidates" in raw_dict else raw_dict.get("limit")
            criteria.max_candidates = max(1, int(val))
        except (ValueError, TypeError):
            pass

    _apply_dict_reasoning_effort(criteria, raw, raw_dict)


def _apply_dict_reasoning_effort(criteria: FallbackCriteria, raw: dict[str, Any], raw_dict: dict[str, Any]) -> None:
    """Extract and assign reasoning effort setting from dictionary."""
    reasoning = None
    for key in ("reasoning_effort", "thinking", "thinking_effort", "reasoning"):
        if raw_dict.get(key) is not None:
            reasoning = raw_dict[key]
            break
        if isinstance(raw, dict) and raw.get(key) is not None:
            reasoning = raw[key]
            break
    if reasoning is not None:
        if isinstance(reasoning, bool):
            criteria.reasoning_effort = "none" if not reasoning else "default"
        elif isinstance(reasoning, str):
            criteria.reasoning_effort = reasoning.strip().lower()


def _apply_shorthand_tokens(criteria: FallbackCriteria, cleaned: str) -> None:
    """Parse shorthand tokens after key-value specifiers have been extracted."""
    parts = set(_RE_SHORTHAND_SPLIT.split(cleaned))

    if "free" in parts:
        criteria.free_only = True
    elif "paid" in parts or "all" in parts or "any" in parts:
        criteria.free_only = False

    if "flash" in parts:
        criteria.flash_only = True

    if "tools" in parts or "require_tools" in parts:
        criteria.require_tools = True
    elif "no_tools" in parts:
        criteria.require_tools = False

    if any(k in cleaned for k in ("greatest_context", "largest_context", "max_context", "longest_context")):
        criteria.sort_by = "greatest_context"
    elif any(k in cleaned for k in ("smallest", "min_params", "smallest_model")):
        criteria.sort_by = "smallest"
    elif any(k in cleaned for k in ("latest_flash", "newest_flash")):
        criteria.sort_by = "latest_flash"
        criteria.flash_only = True
    elif any(k in cleaned for k in ("latest", "newest", "recent")):
        criteria.sort_by = "latest"
    elif any(k in cleaned for k in ("largest_parameter_count", "largest_parameters", "largest_params", "largest_param", "largest", "max_params", "largest_model", "largest_size")):
        criteria.sort_by = "largest_parameter_count"
    elif criteria.flash_only:
        criteria.sort_by = "latest_flash"


def _apply_shorthand_string(criteria: FallbackCriteria, token: str) -> None:
    """Parse shorthand strings like 'largest_parameter_count_free', 'auto:free:greatest_context', 'latest_flash'."""
    cleaned = token.lower().strip()

    m = re.search(r"(?:model_provider|provider)[:=]([a-zA-Z0-9_\-]+)", cleaned)
    if m:
        criteria.provider = m.group(1).lower().strip()

    m = re.search(r"(?:vendor|lab)[:=]([a-zA-Z0-9_\-]+)", cleaned)
    if m:
        criteria.vendor = m.group(1).lower().strip()

    m = re.search(r"(?:filter|pattern)[:=]([a-zA-Z0-9_\-]+)", cleaned)
    if m:
        criteria.name_filter = m.group(1).lower().strip()

    m = re.search(r"(?:min_ctx|min_context)[:=](\d+)", cleaned)
    if m:
        criteria.min_context = int(m.group(1))

    m = re.search(r"(?:reasoning_effort|reasoning|thinking)[:=]([a-zA-Z0-9_\-]+)", cleaned)
    if m:
        criteria.reasoning_effort = m.group(1).lower().strip()

    if cleaned.startswith("nous:") or cleaned.startswith("nous_"):
        criteria.provider = "nous"
        cleaned = cleaned[5:].strip()
    elif cleaned.startswith("openrouter:") or cleaned.startswith("openrouter_"):
        criteria.provider = "openrouter"
        cleaned = cleaned[11:].strip()

    _apply_shorthand_tokens(criteria, cleaned)


def parse_fallback_criteria(raw: Any) -> FallbackCriteria:
    """Parse FallbackCriteria from dict, string, or entry.

    Supports:
    - Dict: {"criteria": {"sort_by": "largest_parameter_count", "free": True, ...}}
    - Top-level keys in dict: {"provider": "openrouter", "heuristic": "largest_parameter_count_free"}
    - Model string tokens: "auto:free:largest_parameter_count", "heuristic:latest_flash", "criteria:..."
    - Shorthand string: "largest_parameter_count_free", "greatest_context", "smallest", "latest_flash"
    """
    if isinstance(raw, FallbackCriteria):
        return raw

    criteria = FallbackCriteria()

    if isinstance(raw, dict):
        raw_dict = raw.get("criteria") if isinstance(raw.get("criteria"), dict) else raw
        raw_heuristic = str(
            raw.get("heuristic") or raw.get("criteria") or raw.get("sort_by") or ""
        ).strip()
        criteria.raw_heuristic = raw_heuristic

        if raw_heuristic and isinstance(raw_heuristic, str):
            _apply_shorthand_string(criteria, raw_heuristic)

        _apply_dict_fields(criteria, raw, raw_dict)

        model_str = str(raw.get("model") or "").strip()
        if model_str.startswith(("auto:", "heuristic:", "criteria:")):
            _apply_shorthand_string(criteria, model_str)

        return criteria

    if isinstance(raw, str):
        criteria.raw_heuristic = raw
        _apply_shorthand_string(criteria, raw)
        return criteria

    return criteria


# ─── Metadata Enrichment ───────────────────────────────────────────────────


def _enrich_from_raw_item(meta: ModelCandidateMetadata, model_id: str, raw_item: dict[str, Any]) -> None:
    """Extract metadata from raw OpenRouter API payload."""
    meta.name = str(raw_item.get("name") or model_id)
    if raw_item.get("context_length"):
        try:
            meta.context_window = int(raw_item["context_length"])
        except (ValueError, TypeError):
            pass
    if raw_item.get("created"):
        try:
            meta.created_ts = float(raw_item["created"])
        except (ValueError, TypeError):
            pass
    pricing = raw_item.get("pricing")
    if isinstance(pricing, dict):
        try:
            prompt_cost = float(pricing.get("prompt", 0))
            comp_cost = float(pricing.get("completion", 0))
            if prompt_cost == 0 and comp_cost == 0:
                meta.is_free = True
        except (ValueError, TypeError):
            pass
    params = raw_item.get("supported_parameters")
    if isinstance(params, list):
        meta.supports_tools = "tools" in params
    if is_flash_model(model_id, meta.name):
        meta.is_flash = True
    if not meta.param_size_b:
        meta.param_size_b = extract_param_size_b(model_id, meta.name)
    meta.source = "openrouter_api"


def _enrich_from_openrouter_cache(meta: ModelCandidateMetadata, model_id: str) -> None:
    """Query cached OpenRouter metadata catalog."""
    try:
        from agent.model_metadata import fetch_model_metadata
        catalog = fetch_model_metadata()
        entry = catalog.get(model_id) or catalog.get(model_id.split("/", 1)[-1])
        if isinstance(entry, dict):
            meta.name = str(entry.get("name") or model_id)
            if entry.get("context_length"):
                try:
                    meta.context_window = int(entry["context_length"])
                except (ValueError, TypeError):
                    pass
            pricing = entry.get("pricing")
            if isinstance(pricing, dict):
                try:
                    prompt_cost = float(pricing.get("prompt", 0) or 0)
                    comp_cost = float(pricing.get("completion", 0) or 0)
                    if prompt_cost == 0 and comp_cost == 0:
                        meta.is_free = True
                except (ValueError, TypeError):
                    pass
            params = entry.get("supported_parameters")
            if isinstance(params, list):
                meta.supports_tools = "tools" in params
            elif entry.get("tools") is not None:
                meta.supports_tools = bool(entry.get("tools"))
            meta.source = "openrouter_cache"
    except Exception as exc:
        logger.debug("OpenRouter cache lookup failed for %s: %s", model_id, exc, exc_info=True)


def _enrich_from_nous_pricing(meta: ModelCandidateMetadata, model_id: str) -> None:
    """Query Nous Portal pricing tables."""
    try:
        from hermes_cli.models import _is_model_free
        from hermes_cli import models_pricing as mp
        pricing = mp.get_pricing_for_provider("nous") or {}
        entry = pricing.get(model_id)
        if isinstance(entry, dict):
            if (
                entry.get("tools") is not False
                and not entry.get("generation")
                and not model_id.startswith(("voyageai/", "sentence-transformers/", "thenlper/", "baai/", "intfloat/"))
            ):
                meta.supports_tools = True
            meta.source = "nous_pricing"
        if _is_model_free(model_id, pricing):
            meta.is_free = True
    except Exception as exc:
        logger.debug("Nous pricing lookup failed for %s: %s", model_id, exc, exc_info=True)


def _enrich_from_models_dev(meta: ModelCandidateMetadata, model_id: str, provider: str) -> None:
    """Query models.dev static registry for context and release date."""
    try:
        from agent.models_dev import get_model_info
        m_info = get_model_info(provider, model_id)
        if m_info is not None:
            if m_info.name:
                meta.name = m_info.name
            if m_info.context_window and m_info.context_window > meta.context_window:
                meta.context_window = m_info.context_window
            if m_info.release_date:
                meta.release_date = m_info.release_date
                try:
                    parts = [int(p) for p in m_info.release_date.split("-") if p.isdigit()]
                    if len(parts) >= 3:
                        dt = datetime.datetime(parts[0], parts[1], parts[2], tzinfo=datetime.timezone.utc)
                        meta.created_ts = dt.timestamp()
                except (ValueError, TypeError):
                    pass
            if m_info.has_cost_data() and m_info.cost_input == 0 and m_info.cost_output == 0:
                meta.is_free = True
            if m_info.tool_call is not None:
                meta.supports_tools = m_info.tool_call
            if meta.source == "heuristic":
                meta.source = "models_dev"
    except Exception as exc:
        logger.debug("models.dev lookup failed for %s: %s", model_id, exc, exc_info=True)


def enrich_model_metadata(
    model_id: str,
    provider: str = "openrouter",
    raw_item: Optional[dict[str, Any]] = None,
    base_url: str = "",
) -> ModelCandidateMetadata:
    """Gather multi-tier metadata for a model candidate from API, registry, and heuristics."""
    meta = ModelCandidateMetadata(id=model_id, provider=provider)

    meta.vendor = extract_vendor(model_id) or provider
    meta.is_flash = is_flash_model(model_id)
    meta.param_size_b = extract_param_size_b(model_id)
    meta.context_window = extract_context_window_from_name(model_id, default=0)
    meta.version_tuple = extract_version_tuple(model_id)
    meta.date_snapshot = extract_date_snapshot(model_id)
    meta.is_free = is_free_model_slug(model_id)

    if isinstance(raw_item, dict):
        _enrich_from_raw_item(meta, model_id, raw_item)
        return meta

    if provider.lower() == "openrouter":
        _enrich_from_openrouter_cache(meta, model_id)
    elif provider.lower() in ("nous", "nousresearch", "nous-portal"):
        _enrich_from_nous_pricing(meta, model_id)

    _enrich_from_models_dev(meta, model_id, provider)

    if meta.context_window <= 0:
        try:
            from agent.model_metadata import (
                _config_override_context_length,
                _resolve_provider_aware_context_length,
                _resolve_endpoint_context_length,
                _longest_key_match,
                DEFAULT_CONTEXT_LENGTHS,
            )

            ctx = _config_override_context_length(model_id, base_url, provider, None)
            if ctx is None and base_url:
                ctx = _resolve_endpoint_context_length(model_id, base_url)
            if ctx is None:
                ctx = _resolve_provider_aware_context_length(model_id, base_url, "", provider, provider)
            if ctx is None:
                hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model_id.lower())
                if hit:
                    ctx = hit[1]
            if ctx and ctx > 0:
                meta.context_window = ctx
        except Exception as exc:
            logger.debug("Context length positive resolution failed for %s (%s): %s", model_id, provider, exc, exc_info=True)

    return meta


# ─── Candidate Gathering and Filtering ──────────────────────────────────────


_CANDIDATE_CACHE: dict[tuple[str, str, str], tuple[float, list[ModelCandidateMetadata]]] = {}
_CANDIDATE_CACHE_TTL = 300.0  # 5 minutes
_CANDIDATE_NEGATIVE_CACHE_TTL = 30.0  # 30 seconds for empty results


def clear_candidate_cache() -> None:
    """Clear the candidate models in-memory cache (for testing or manual refresh)."""
    _CANDIDATE_CACHE.clear()


def _gather_nous_candidates(provider_norm: str, force_refresh: bool = False) -> dict[str, ModelCandidateMetadata]:
    """Gather candidates for Nous Research Portal."""
    candidates_by_id: dict[str, ModelCandidateMetadata] = {}
    try:
        from hermes_cli.models import provider_model_ids, _is_model_free
        from hermes_cli import models_pricing as mp
        mids = list(provider_model_ids("nous", force_refresh=force_refresh) or [])
        pricing = mp.get_pricing_for_provider("nous") or {}
        for mid in pricing:
            if mid not in mids and _is_model_free(mid, pricing):
                mids.append(mid)
        for mid in mids:
            cand = enrich_model_metadata(mid, provider="nous")
            if _is_model_free(mid, pricing):
                cand.is_free = True
            candidates_by_id[mid] = cand
    except Exception as exc:
        logger.debug("Nous catalog probe failed in fallback heuristics: %s", exc, exc_info=True)
    return candidates_by_id


def _load_openrouter_live_catalog(provider_norm: str) -> dict[str, ModelCandidateMetadata]:
    candidates_by_id: dict[str, ModelCandidateMetadata] = {}
    try:
        from hermes_cli.models import (
            _OPENROUTER_CATALOG_URL,
            _fetch_live_catalog_index,
            _urlopen_model_catalog_request,
        )
        live = _fetch_live_catalog_index(_OPENROUTER_CATALOG_URL, 6.0, _urlopen_model_catalog_request)
        if live is not None:
            live_items, _ = live
            for item in live_items:
                if isinstance(item, dict):
                    mid = str(item.get("id") or "").strip()
                    if mid:
                        candidates_by_id[mid] = enrich_model_metadata(
                            mid, provider=provider_norm, raw_item=item
                        )
    except Exception as exc:
        logger.debug("Live OpenRouter catalog probe failed: %s", exc, exc_info=True)
    return candidates_by_id


def _load_openrouter_static_catalog(provider_norm: str) -> dict[str, ModelCandidateMetadata]:
    candidates_by_id: dict[str, ModelCandidateMetadata] = {}
    try:
        from hermes_cli.models_catalog_static import OPENROUTER_MODELS
        for mid, desc in OPENROUTER_MODELS:
            cand = enrich_model_metadata(mid, provider=provider_norm)
            if desc == "free" or mid.endswith(":free"):
                cand.is_free = True
            candidates_by_id[mid] = cand
    except Exception as exc:
        logger.debug("Static OpenRouter catalog probe failed: %s", exc, exc_info=True)
    return candidates_by_id


def _load_openrouter_disk_catalog(provider_norm: str) -> dict[str, ModelCandidateMetadata]:
    candidates_by_id: dict[str, ModelCandidateMetadata] = {}
    try:
        import json
        for p in (
            Path(__file__).resolve().parent.parent / "website" / "static" / "api" / "model-catalog.json",
            get_hermes_home() / "cache" / "model_catalog.json",
        ):
            if not p.is_file():
                continue
            with open(p, "r", encoding="utf-8-sig") as fh:
                data = json.load(fh)
            m_list = data.get("providers", {}).get("openrouter", {}).get("models", [])
            for item in m_list:
                mid = str(item.get("id") or "").strip()
                if mid and mid not in candidates_by_id:
                    cand = enrich_model_metadata(mid, provider=provider_norm)
                    if ":free" in mid or "free" in str(item.get("description") or "").lower():
                        cand.is_free = True
                    candidates_by_id[mid] = cand
            if candidates_by_id:
                break
    except Exception as exc:
        logger.debug("OpenRouter disk catalog load failed: %s", exc, exc_info=True)
    return candidates_by_id


def _gather_openrouter_candidates(provider_norm: str, force_refresh: bool = False) -> dict[str, ModelCandidateMetadata]:
    """Gather candidates for OpenRouter through live API, static catalog, or disk cache."""
    candidates_by_id = _load_openrouter_live_catalog(provider_norm)
    if not candidates_by_id:
        candidates_by_id = _load_openrouter_static_catalog(provider_norm)
    if not candidates_by_id:
        candidates_by_id = _load_openrouter_disk_catalog(provider_norm)

    return candidates_by_id


def _gather_generic_candidates(provider_norm: str, base_url: str = "", force_refresh: bool = False) -> dict[str, ModelCandidateMetadata]:
    """Gather candidate models for an arbitrary or custom provider."""
    candidates_by_id: dict[str, ModelCandidateMetadata] = {}
    try:
        from hermes_cli.models import provider_model_ids
        model_ids = provider_model_ids(provider_norm, force_refresh=force_refresh) or []
        for mid in model_ids:
            if mid and mid not in candidates_by_id:
                candidates_by_id[mid] = enrich_model_metadata(
                    mid, provider=provider_norm, base_url=base_url
                )
    except Exception as exc:
        logger.debug("Generic provider model probe failed: %s", exc, exc_info=True)

    if not candidates_by_id:
        try:
            from hermes_cli.models_catalog_static import _PROVIDER_MODELS
            static_list = _PROVIDER_MODELS.get(provider_norm, [])
            for mid in static_list:
                if mid and mid not in candidates_by_id:
                    candidates_by_id[mid] = enrich_model_metadata(
                        mid, provider=provider_norm, base_url=base_url
                    )
        except Exception as exc:
            logger.debug("Static provider model fallback failed: %s", exc, exc_info=True)

    return candidates_by_id


def get_candidate_models(
    provider: str = "nous",
    base_url: str | None = "",
    force_refresh: bool = False,
) -> list[ModelCandidateMetadata]:
    """Gather candidate models for a provider, enriched with metadata. Defaults to 'nous'."""
    from hermes_constants import hermes_home_key

    provider_norm = provider.lower().strip()
    home_key = hermes_home_key()
    base_url_norm = (base_url or "").strip()
    cache_key = (home_key, provider_norm, base_url_norm)
    now = time.monotonic()

    if not force_refresh and cache_key in _CANDIDATE_CACHE:
        cached_ts, cached_candidates = _CANDIDATE_CACHE[cache_key]
        ttl = _CANDIDATE_CACHE_TTL if cached_candidates else _CANDIDATE_NEGATIVE_CACHE_TTL
        if now - cached_ts < ttl:
            return cached_candidates

    candidates_by_id: dict[str, ModelCandidateMetadata] = {}

    if provider_norm in ("nous", "nousresearch", "nous-portal"):
        candidates_by_id.update(_gather_nous_candidates(provider_norm, force_refresh))
    elif provider_norm == "openrouter":
        candidates_by_id.update(_gather_openrouter_candidates(provider_norm, force_refresh))
    else:
        candidates_by_id.update(_gather_generic_candidates(provider_norm, base_url_norm, force_refresh))

    res = list(candidates_by_id.values())
    _CANDIDATE_CACHE[cache_key] = (now, res)
    return res


# ─── Candidate Ranking and Sorting ──────────────────────────────────────────


def _key_largest_params(c: ModelCandidateMetadata) -> tuple[Any, ...]:
    size_val = c.param_size_b if c.param_size_b is not None else 1.0
    return (
        size_val,
        c.context_window,
        c.created_ts,
        c.date_snapshot,
        c.version_tuple,
    )


def _key_greatest_context(c: ModelCandidateMetadata) -> tuple[Any, ...]:
    return (
        c.context_window,
        c.param_size_b if c.param_size_b is not None else 1.0,
        c.created_ts,
        c.date_snapshot,
        c.version_tuple,
    )


def _key_smallest(c: ModelCandidateMetadata) -> tuple[Any, ...]:
    return (
        -(c.param_size_b if c.param_size_b is not None and c.param_size_b > 0 else 999999.0),
        -c.context_window,
    )


def _key_latest(c: ModelCandidateMetadata) -> tuple[Any, ...]:
    has_date = bool(c.created_ts or c.date_snapshot)
    return (
        c.created_ts,
        c.date_snapshot,
        c.version_tuple if has_date else (0,),
        c.param_size_b if c.param_size_b is not None else 1.0,
        c.context_window,
    )


def _key_latest_flash(c: ModelCandidateMetadata) -> tuple[Any, ...]:
    has_date = bool(c.created_ts or c.date_snapshot)
    return (
        1 if c.is_flash else 0,
        c.created_ts,
        c.date_snapshot,
        c.version_tuple if has_date else (0,),
        c.param_size_b if c.param_size_b is not None else 1.0,
        c.context_window,
    )


_SORT_KEY_DISPATCH: dict[str, Callable[[ModelCandidateMetadata], tuple[Any, ...]]] = {
    "largest_parameter_count": _key_largest_params,
    "largest_parameters": _key_largest_params,
    "largest_params": _key_largest_params,
    "largest": _key_largest_params,
    "largest_size": _key_largest_params,
    "max_params": _key_largest_params,
    "greatest_context": _key_greatest_context,
    "largest_context": _key_greatest_context,
    "max_context": _key_greatest_context,
    "longest_context": _key_greatest_context,
    "smallest": _key_smallest,
    "smallest_parameters": _key_smallest,
    "smallest_size": _key_smallest,
    "min_params": _key_smallest,
    "latest": _key_latest,
    "newest": _key_latest,
    "most_recent": _key_latest,
    "latest_flash": _key_latest_flash,
}


def filter_and_rank_candidates(
    candidates: list[ModelCandidateMetadata],
    criteria: FallbackCriteria,
) -> list[ModelCandidateMetadata]:
    """Filter candidate models and sort them by the specified heuristic."""
    filtered: list[ModelCandidateMetadata] = []

    for cand in candidates:
        if criteria.free_only and not cand.is_free:
            continue
        if criteria.require_tools and not cand.supports_tools:
            continue
        if criteria.flash_only and not cand.is_flash:
            continue
        if criteria.vendor and criteria.vendor != cand.vendor:
            continue
        if criteria.name_filter:
            pattern = criteria.name_filter.lower()
            name_lower = f"{cand.id} {cand.name}".lower()
            if not fnmatch.fnmatch(name_lower, f"*{pattern}*"):
                continue
        if criteria.min_context > 0 and cand.context_window < criteria.min_context:
            continue
        if criteria.max_context > 0 and cand.context_window > criteria.max_context:
            continue
        filtered.append(cand)

    if not filtered:
        return []

    sort_by = criteria.sort_by.lower().strip()
    sort_fn = _SORT_KEY_DISPATCH.get(sort_by, _key_largest_params)
    filtered.sort(key=sort_fn, reverse=True)
    return filtered


# ─── Public Resolver ────────────────────────────────────────────────────────


def resolve_fallback_candidates(
    provider: str,
    criteria: FallbackCriteria | dict[str, Any] | str,
    base_url: str = "",
    max_candidates: Optional[int] = None,
) -> list[str]:
    """Resolve model IDs matching the criteria for a provider."""
    parsed_criteria = parse_fallback_criteria(criteria)
    entry = {"provider": provider, "base_url": base_url, "criteria": parsed_criteria.to_dict()}
    resolved = resolve_fallback_entry(entry, max_candidates=max_candidates)
    return [r["model"] for r in resolved if r.get("model")]


def resolve_fallback_entry(
    entry: dict[str, Any],
    max_candidates: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Resolve a raw fallback entry into one or more concrete fallback entries.

    If the entry already has a static model and no criteria/heuristic, returns [entry].
    If the entry specifies criteria or heuristic, resolves the top matching models
    for that provider and returns them as a list of entries in the fallback chain.
    """
    if not isinstance(entry, dict):
        return []

    provider = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or "").strip()
    has_criteria = bool(entry.get("criteria") or entry.get("heuristic")) or (
        model.startswith(("auto:", "heuristic:", "criteria:"))
    )

    if not has_criteria and model:
        return [entry]

    criteria = parse_fallback_criteria(entry)
    provider = str(
        entry.get("provider")
        or entry.get("model_provider")
        or criteria.provider
        or "nous"
    ).strip().lower()

    if not provider:
        provider = "nous"

    base_url = str(entry.get("base_url") or "").strip()
    limit = max_candidates or criteria.max_candidates or 3

    candidates = get_candidate_models(provider=provider, base_url=base_url)
    ranked = filter_and_rank_candidates(candidates, criteria)

    if not ranked:
        logger.warning(
            "No models found matching criteria %s for provider %s",
            criteria.describe(),
            provider,
        )
        return []

    entry_effort = None
    for key in ("reasoning_effort", "thinking_effort", "thinking", "reasoning"):
        if entry.get(key) is not None:
            entry_effort = entry[key]
            break
    if entry_effort is None and criteria.reasoning_effort is not None:
        entry_effort = criteria.reasoning_effort

    resolved: list[dict[str, Any]] = []
    for rank, cand in enumerate(ranked[:limit]):
        resolved_entry = {
            **entry,
            "provider": provider,
            "model": cand.id,
            "criteria_matched": criteria.describe(),
            "criteria_rank": rank + 1,
            "criteria_total_matched": len(ranked),
            "_is_heuristic": True,
            "_resolved_criteria": criteria.to_dict(),
        }
        resolved_entry.pop("criteria", None)
        resolved_entry.pop("heuristic", None)
        if entry_effort is not None:
            resolved_entry["reasoning_effort"] = entry_effort
        elif "reasoning_effort" in resolved_entry:
            del resolved_entry["reasoning_effort"]

        resolved.append(resolved_entry)

    logger.info(
        "Resolved heuristic fallback for %s (%s): %s",
        provider,
        criteria.describe(),
        " -> ".join(f"{r['model']} (#{r['criteria_rank']})" for r in resolved),
    )
    return resolved
