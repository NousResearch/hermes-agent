"""Provider-neutral model policy and deterministic routing for ``hermes jev``.

Jev assigns advisory tiers to catalog metadata. This module enforces the hard
floor, normalizes canonical model identity, and resolves model/provider routes
without coupling either decision to the execution agent choice.
"""

from __future__ import annotations

import math
import json
import os
import re
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home, mkdir_under_hermes_home


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
MODEL_TIERS = ("cheap_fast", "mid", "premium")
MODEL_CLASSIFICATIONS = ("below_floor",) + MODEL_TIERS
POLICY_FLOOR = "anthropic/claude-sonnet-4.6-equivalent"
NATIVE_CATALOG_SOURCES = {
    "openai-codex": "hermes_cli.models.provider_model_ids('openai-codex')",
    "anthropic": "hermes_cli.models.provider_model_ids('anthropic')",
}

_ANTHROPIC_ID = re.compile(
    r"^anthropic/claude-(?P<family>sonnet|opus)-(?P<version>\d+(?:\.\d+)?)"
    r"(?P<variant>[-:].*)?$"
)
_OPENAI_ID = re.compile(r"^openai/gpt-(?P<version>\d+(?:\.\d+)?)(?P<variant>[-:].*)?$")
_VERSION = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)*)(?=$|[-_.])")
_BATCH_SUFFIX = re.compile(r"(?:^|:)batch(?:$|:)")

# Used only for deterministic fixture/catalog-only inspection. Normal command
# paths ask Jev to classify each provisional candidate from its metadata.
_CHEAP_FAST_MAX_COST = 3.0
_MID_MAX_COST = 15.0
# Hermes' canonical Sonnet 4.6 metadata declares a 1M-token context window and
# the Anthropic adapter declares a 64k output limit. These are observable,
# versioned properties of the baseline named by POLICY_FLOOR, not estimates.
_EQUIVALENT_CONTEXT_MIN = 1_000_000
_EQUIVALENT_OUTPUT_MIN = 64_000

_CACHE_SCHEMA_VERSION = 1
_CACHE_POLICY_VERSION = 1
_CACHE_TTL_SECONDS = 60 * 60
_CACHE_MAX_MODELS = 4_000
_CACHE_MAX_BYTES = 8 * 1024 * 1024

_TIER_USE = {
    "cheap_fast": "routine, latency- or cost-sensitive bounded work",
    "mid": "normal implementation, analysis, and review work",
    "premium": "complex, ambiguous, broad-context, or high-cost-of-error work",
}


def fetch_openrouter_catalog(api_key: str) -> list[dict[str, Any]]:
    """Fetch and validate the authenticated OpenRouter model catalog."""
    import httpx

    response = httpx.get(
        OPENROUTER_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        timeout=30.0,
    )
    response.raise_for_status()
    document = response.json()
    data = document.get("data") if isinstance(document, dict) else None
    if not isinstance(data, list):
        from hermes_cli.jev import JevInputError

        raise JevInputError("OpenRouter models response is missing a data array")
    records = [_catalog_record(item) for item in data if isinstance(item, dict)]
    records = [record for record in records if record is not None]
    if len(records) > _CACHE_MAX_MODELS:
        from hermes_cli.jev import JevInputError

        raise JevInputError(
            f"OpenRouter models response exceeds the {_CACHE_MAX_MODELS}-model safety bound"
        )
    return records


def _catalog_record(item: dict[str, Any]) -> dict[str, Any] | None:
    """Copy only structured fields used by policy; discard publisher prose."""
    model_id = item.get("id")
    if not isinstance(model_id, str):
        return None
    architecture = item.get("architecture")
    pricing = item.get("pricing")
    top_provider = item.get("top_provider")
    return {
        "id": model_id,
        "context_length": item.get("context_length"),
        "supported_parameters": item.get("supported_parameters"),
        "architecture": {
            "input_modalities": architecture.get("input_modalities"),
            "output_modalities": architecture.get("output_modalities"),
        }
        if isinstance(architecture, dict)
        else None,
        "pricing": {
            "prompt": pricing.get("prompt"),
            "completion": pricing.get("completion"),
        }
        if isinstance(pricing, dict)
        else None,
        "top_provider": {
            "context_length": top_provider.get("context_length"),
            "max_completion_tokens": top_provider.get("max_completion_tokens"),
        }
        if isinstance(top_provider, dict)
        else None,
    }


def _sanitize_catalog(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = [_catalog_record(item) for item in catalog if isinstance(item, dict)]
    sanitized = [record for record in records if record is not None]
    if len(sanitized) > _CACHE_MAX_MODELS:
        from hermes_cli.jev import JevInputError

        raise JevInputError(
            f"OpenRouter models response exceeds the {_CACHE_MAX_MODELS}-model safety bound"
        )
    return sanitized


def _cache_path() -> Path:
    return get_hermes_home() / "cache" / "jev_models.json"


@contextmanager
def _cache_lock():
    """Yield whether a cache lock was acquired; cache I/O is strictly best-effort."""
    path = _cache_path().with_suffix(".lock")
    handle = None
    locked = False
    try:
        mkdir_under_hermes_home(path.parent)
        handle = path.open("a+", encoding="utf-8")
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if path.stat().st_size == 0:
                handle.write("0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        locked = True
    except OSError:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        yield False
        return
    try:
        yield True
    finally:
        try:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            if locked:
                try:
                    handle.close()
                except OSError:
                    pass


def _read_cache(now: float) -> dict[str, Any] | None:
    path = _cache_path()
    try:
        if not path.is_file() or path.stat().st_size > _CACHE_MAX_BYTES:
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    if document.get("schema_version") != _CACHE_SCHEMA_VERSION:
        return None
    if document.get("policy_version") != _CACHE_POLICY_VERSION:
        return None
    cached_at = document.get("cached_at")
    if not isinstance(cached_at, (int, float)) or isinstance(cached_at, bool):
        return None
    age = now - float(cached_at)
    if age < 0 or age >= _CACHE_TTL_SECONDS:
        return None
    catalog = document.get("catalog")
    if not isinstance(catalog, list) or len(catalog) > _CACHE_MAX_MODELS:
        return None
    if not all(isinstance(item, dict) for item in catalog):
        return None
    assignments = document.get("assignments")
    if assignments is not None and not isinstance(assignments, dict):
        return None
    if isinstance(assignments, dict) and not all(
        isinstance(model_id, str)
        and isinstance(assignment, dict)
        and assignment.get("tier") in MODEL_CLASSIFICATIONS
        for model_id, assignment in assignments.items()
    ):
        return None
    usage = document.get("jev_usage")
    if not isinstance(usage, list) or not all(isinstance(item, dict) for item in usage):
        return None
    return document


def _write_cache(
    catalog: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]] | None,
    usage: list[dict[str, Any]],
    now: float,
) -> None:
    from utils import atomic_json_write

    document = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "policy_version": _CACHE_POLICY_VERSION,
        "cached_at": now,
        "catalog": catalog,
        "assignments": assignments,
        "jev_usage": usage,
    }
    encoded = json.dumps(document, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _CACHE_MAX_BYTES:
        from hermes_cli.jev import JevInputError

        raise JevInputError("OpenRouter model cache exceeds its 8 MiB safety bound")
    path = _cache_path()
    mkdir_under_hermes_home(path.parent)
    atomic_json_write(path, document, mode=0o600)


def _live_catalog_resolution(
    api_key: str,
    classifier: Any,
    *,
    catalog_only: bool,
) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]] | None, list[dict[str, Any]]
]:
    """Fetch and optionally classify without depending on cache infrastructure."""
    catalog = _sanitize_catalog(fetch_openrouter_catalog(api_key))
    if catalog_only:
        return catalog, None, []
    assignments, usage = classifier(catalog, api_key)
    return catalog, assignments, usage


def _write_cache_best_effort(
    catalog: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]] | None,
    usage: list[dict[str, Any]],
    now: float,
) -> None:
    """Ignore cache-only failures after a usable live result has been obtained."""
    from hermes_cli.jev import JevInputError

    try:
        _write_cache(catalog, assignments, usage, now)
    except (OSError, JevInputError):
        return


def cached_catalog_resolution(
    api_key: str,
    classifier: Any,
    *,
    catalog_only: bool = False,
    now: float | None = None,
) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]] | None, list[dict[str, Any]]
]:
    """Return a profile-scoped TTL snapshot, filling it once across CLI processes."""
    resolved_now = time.time() if now is None else now
    with _cache_lock() as cache_available:
        if not cache_available:
            return _live_catalog_resolution(
                api_key, classifier, catalog_only=catalog_only
            )
        cached = _read_cache(resolved_now)
        if cached is None:
            catalog, assignments, usage = _live_catalog_resolution(
                api_key, classifier, catalog_only=catalog_only
            )
        else:
            catalog = cached["catalog"]
            assignments = cached["assignments"]
            usage = cached["jev_usage"]

        if catalog_only:
            if cached is None:
                _write_cache_best_effort(catalog, None, [], resolved_now)
            return catalog, None, []

        if assignments is None:
            assignments, usage = classifier(catalog, api_key)

        if cached is None or cached["assignments"] is None:
            _write_cache_best_effort(catalog, assignments, usage, resolved_now)
        return catalog, assignments, usage


def load_native_catalogs(*, force_refresh: bool = False) -> dict[str, list[str]]:
    """Return Hermes' best-known, provider-specific native model catalogs."""
    from hermes_cli.models import provider_model_ids

    return {
        provider: provider_model_ids(provider, force_refresh=force_refresh)
        for provider in NATIVE_CATALOG_SOURCES
    }


def _version(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _at_least(value: str, floor: str) -> bool:
    actual = _version(value)
    minimum = _version(floor)
    width = max(len(actual), len(minimum))
    return actual + (0,) * (width - len(actual)) >= minimum + (0,) * (
        width - len(minimum)
    )


def _canonical_identity(model_id: str) -> tuple[dict[str, Any] | None, str]:
    """Normalize an exact catalog id and apply explicit family/version floors."""
    if model_id.startswith("~"):
        return None, "alias_id"
    if _BATCH_SUFFIX.search(model_id):
        return None, "batch_only"
    canonical_id = model_id.split(":", 1)[0]
    if (
        "/" not in canonical_id
        or len(canonical_id) > 200
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", canonical_id) is None
    ):
        return None, "invalid_id"
    vendor, slug = canonical_id.split("/", 1)

    match = _ANTHROPIC_ID.fullmatch(canonical_id)
    if match:
        version = match["version"]
        if not _at_least(version, "4.6"):
            return None, "below_explicit_floor"
        return {
            "id": canonical_id,
            "vendor": "anthropic",
            "family": match["family"],
            "version": version,
            "variant": (match["variant"] or "").lstrip("-:"),
            "explicit_floor_family": True,
        }, "eligible"

    match = _OPENAI_ID.fullmatch(canonical_id)
    if match:
        version = match["version"]
        if not _at_least(version, "5.5"):
            return None, "below_explicit_floor"
        return {
            "id": canonical_id,
            "vendor": "openai",
            "family": "gpt",
            "version": version,
            "variant": (match["variant"] or "").lstrip("-:"),
            "explicit_floor_family": True,
        }, "eligible"

    version_match = _VERSION.search(slug)
    version = version_match.group(1) if version_match else None
    family = slug[: version_match.start()].rstrip("-_.") if version_match else slug
    variant = slug[version_match.end() :].lstrip("-_.") if version_match else ""
    return {
        "id": canonical_id,
        "vendor": vendor,
        "family": family or slug,
        "version": version,
        "variant": variant,
        "explicit_floor_family": False,
    }, "eligible"


def _nonnegative_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price >= 0 else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _strip_publisher(model_id: str) -> str:
    return model_id.split("/", 1)[-1].strip().lower()


def _normalize_anthropic_model_id(model_id: str) -> str:
    """Normalize numeric version separators without recognizing model families."""
    slug = _strip_publisher(model_id)
    return re.sub(r"(?<=\d)\.(?=\d)", "-", slug)


def _native_model_id(
    identity: dict[str, Any], native_catalogs: dict[str, list[str]]
) -> tuple[str, str] | None:
    """Return an exact native catalog match for a canonical OpenRouter model."""
    target = identity["id"].split("/", 1)[1]
    if identity["vendor"] == "openai":
        provider = "openai-codex"
        normalize = _strip_publisher
    elif identity["vendor"] == "anthropic":
        provider = "anthropic"
        normalize = _normalize_anthropic_model_id
    else:
        return None
    normalized_target = normalize(target)
    for model_id in native_catalogs.get(provider, []):
        if isinstance(model_id, str) and normalize(model_id) == normalized_target:
            return provider, model_id
    return None


def _provider_routes(
    identity: dict[str, Any],
    provider_model_id: str,
    native_catalogs: dict[str, list[str]],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    native = _native_model_id(identity, native_catalogs)
    if native is not None:
        provider, native_model_id = native
        routes.append({
            "provider": provider,
            "provider_model_id": native_model_id,
            "route_kind": "native",
            "compatible_agents": (
                ["codex"] if provider == "openai-codex" else ["claude"]
            ),
            "availability": "native_catalog_verified",
        })
    routes.append({
        "provider": "openrouter",
        "provider_model_id": provider_model_id,
        "route_kind": "provider_fallback" if routes else "provider_primary",
        "compatible_agents": ["codex", "claude"],
        "availability": "catalog_verified",
    })
    return routes


def _provisional_candidate(
    item: dict[str, Any], native_catalogs: dict[str, list[str]]
) -> tuple[dict[str, Any] | None, str]:
    model_id = item.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None, "invalid_id"
    identity, status = _canonical_identity(model_id)
    if identity is None:
        return None, status

    context_length = _positive_int(item.get("context_length"))
    top_provider = item.get("top_provider")
    if not isinstance(top_provider, dict):
        return None, "unavailable_provider"
    provider_context = _positive_int(top_provider.get("context_length"))
    max_completion = _positive_int(top_provider.get("max_completion_tokens"))
    if context_length is None or provider_context is None or max_completion is None:
        return None, "unavailable_provider"
    context_length = min(context_length, provider_context)
    if (
        context_length < _EQUIVALENT_CONTEXT_MIN
        or max_completion < _EQUIVALENT_OUTPUT_MIN
    ):
        return None, "below_observable_floor"

    pricing = item.get("pricing")
    if not isinstance(pricing, dict):
        return None, "invalid_pricing"
    prompt_price = _nonnegative_price(pricing.get("prompt"))
    completion_price = _nonnegative_price(pricing.get("completion"))
    if prompt_price is None or completion_price is None:
        return None, "invalid_pricing"

    parameters = item.get("supported_parameters")
    if not isinstance(parameters, list) or not all(
        isinstance(value, str) for value in parameters
    ):
        return None, "invalid_capabilities"
    supported = set(parameters)
    capabilities = {
        "tools": "tools" in supported and "tool_choice" in supported,
        "reasoning": "reasoning" in supported or "reasoning_effort" in supported,
        "structured_outputs": (
            "structured_outputs" in supported or "response_format" in supported
        ),
    }
    architecture = item.get("architecture")
    if not isinstance(architecture, dict):
        return None, "invalid_capabilities"
    inputs = architecture.get("input_modalities")
    outputs = architecture.get("output_modalities")
    if not isinstance(inputs, list) or "text" not in inputs:
        return None, "no_text_input"
    if not isinstance(outputs, list) or "text" not in outputs:
        return None, "no_text_output"

    effective_cost = (prompt_price * 0.25 + completion_price * 0.75) * 1_000_000
    return {
        "canonical_model": identity,
        "context_length": context_length,
        "max_completion_tokens": max_completion,
        "capabilities": capabilities,
        "pricing": {
            "prompt_per_token": prompt_price,
            "completion_per_token": completion_price,
            "weighted_per_million_tokens": round(effective_cost, 6),
        },
        "latency_observed": False,
        "provider_routes": _provider_routes(identity, model_id, native_catalogs),
    }, "eligible"


def provisional_catalog(
    catalog: list[dict[str, Any]],
    native_catalogs: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply validity/floor checks and deterministically merge canonical duplicates.

    Candidate-level context, capabilities, and pricing all come from one stable
    representative row: the exact unsuffixed canonical ID when present, otherwise
    the lexicographically smallest provider model ID. Route metadata is not blended
    into a synthetic model; every unique route is retained and ordered native first,
    then by provider and provider model ID.
    """
    native_catalogs = native_catalogs or {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    exclusions: Counter[str] = Counter()
    for item in catalog:
        candidate, status = _provisional_candidate(item, native_catalogs)
        if candidate is None:
            exclusions[status] += 1
            continue
        canonical_id = candidate["canonical_model"]["id"]
        grouped.setdefault(canonical_id, []).append(candidate)

    merged: list[dict[str, Any]] = []
    for canonical_id in sorted(grouped):
        candidates = grouped[canonical_id]

        def representative_key(candidate: dict[str, Any]) -> tuple[bool, str]:
            openrouter_id = next(
                route["provider_model_id"]
                for route in candidate["provider_routes"]
                if route["provider"] == "openrouter"
            )
            return openrouter_id != canonical_id, openrouter_id

        representative = min(candidates, key=representative_key)
        model = dict(representative)
        unique_routes = {
            (route["provider"], route["provider_model_id"]): route
            for candidate in candidates
            for route in candidate["provider_routes"]
        }
        model["provider_routes"] = sorted(
            unique_routes.values(),
            key=lambda route: (
                route["route_kind"] != "native",
                route["provider"],
                route["provider_model_id"],
            ),
        )
        merged.append(model)
        if len(candidates) > 1:
            exclusions["duplicate_route_merged"] += len(candidates) - 1
    return merged, dict(sorted(exclusions.items()))


def metadata_for_jev(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return the bounded, secret-free evidence Jev uses for model tiering."""
    return {
        "canonical_model": {"id": candidate["canonical_model"]["id"]},
        "context_length": candidate["context_length"],
        "max_completion_tokens": candidate["max_completion_tokens"],
        "capabilities": candidate["capabilities"],
        "pricing": candidate["pricing"],
        "latency_observed": candidate["latency_observed"],
    }


def _metadata_tier(candidate: dict[str, Any]) -> str:
    cost = candidate["pricing"]["weighted_per_million_tokens"]
    if cost <= _CHEAP_FAST_MAX_COST:
        return "cheap_fast"
    if cost <= _MID_MAX_COST:
        return "mid"
    return "premium"


def _classification(
    candidate: dict[str, Any], assignments: dict[str, dict[str, Any]] | None
) -> dict[str, Any]:
    canonical_id = candidate["canonical_model"]["id"]
    if assignments is None:
        return {
            "tier": _metadata_tier(candidate),
            "confidence": None,
            "source": "deterministic_catalog_only",
        }
    if canonical_id not in assignments:
        from hermes_cli.jev import JevInputError

        raise JevInputError(f"Jev did not classify candidate {canonical_id!r}")
    assignment = assignments[canonical_id]
    tier = assignment.get("tier")
    if tier not in MODEL_CLASSIFICATIONS:
        from hermes_cli.jev import JevInputError

        raise JevInputError(f"invalid Jev model tier for {canonical_id!r}")
    return {"tier": tier, "confidence": assignment.get("confidence"), "source": "jev"}


def classify_catalog(
    catalog: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]] | None = None,
    native_catalogs: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply Jev tier assignments after deterministic candidate checks."""
    provisional, exclusions = provisional_catalog(catalog, native_catalogs)
    exclusion_counts = Counter(exclusions)
    candidates: list[dict[str, Any]] = []
    for candidate in provisional:
        classification = _classification(candidate, assignments)
        if classification["tier"] == "below_floor":
            exclusion_counts["jev_below_floor"] += 1
            continue
        tier = classification["tier"]
        model = dict(candidate)
        model["tier"] = tier
        model["recommended_use"] = _TIER_USE[tier]
        model["tier_classification"] = classification
        model["tier_reasons"] = [
            f"{classification['source']} classified current metadata as {tier}",
            f"declares {candidate['context_length']}-token context and the reported capability flags",
            "catalog has no observed latency metric; selection does not invent one",
        ]
        candidates.append(model)
    candidates.sort(
        key=lambda row: (
            MODEL_TIERS.index(row["tier"]),
            row["canonical_model"]["id"],
        )
    )
    return candidates, dict(sorted(exclusion_counts.items()))


def model_requirements(
    *,
    min_context_length: int = 0,
    require_tools: bool = True,
    require_reasoning: bool = False,
    require_structured_outputs: bool = False,
) -> dict[str, Any]:
    if min_context_length < 0:
        from hermes_cli.jev import JevInputError

        raise JevInputError("minimum context length must not be negative")
    return {
        "min_context_length": min_context_length,
        "tools": require_tools,
        "reasoning": require_reasoning,
        "structured_outputs": require_structured_outputs,
    }


def _meets_requirements(
    candidate: dict[str, Any], requirements: dict[str, Any]
) -> bool:
    capabilities = candidate["capabilities"]
    return (
        candidate["context_length"] >= requirements["min_context_length"]
        and (not requirements["tools"] or capabilities["tools"])
        and (not requirements["reasoning"] or capabilities["reasoning"])
        and (
            not requirements["structured_outputs"] or capabilities["structured_outputs"]
        )
    )


def _candidate_order(candidate: dict[str, Any], requested_tier: str) -> tuple[Any, ...]:
    price = candidate["pricing"]["weighted_per_million_tokens"]
    context = candidate["context_length"]
    model_id = candidate["canonical_model"]["id"]
    if requested_tier == "premium":
        return (-context, price, model_id)
    return (price, -context, model_id)


def model_fallbacks(
    candidates: list[dict[str, Any]],
    requested_tier: str,
    requirements: dict[str, Any],
) -> list[dict[str, Any]]:
    if requested_tier not in MODEL_TIERS:
        from hermes_cli.jev import JevInputError

        raise JevInputError(f"invalid model tier {requested_tier!r}")
    minimum = MODEL_TIERS.index(requested_tier)
    chain: list[dict[str, Any]] = []
    for tier in MODEL_TIERS[minimum:]:
        tier_candidates = [
            candidate
            for candidate in candidates
            if candidate["tier"] == tier
            and _meets_requirements(candidate, requirements)
        ]
        chain.extend(
            sorted(
                tier_candidates, key=lambda row: _candidate_order(row, requested_tier)
            )
        )
    return chain


def _route_compatible(route: dict[str, Any], agent_choice: str) -> bool:
    compatible = set(route["compatible_agents"])
    if agent_choice == "both":
        return {"codex", "claude"} <= compatible
    if agent_choice in {"codex", "claude"}:
        return agent_choice in compatible
    return bool(compatible)


def _ordered_routes(
    candidate: dict[str, Any], agent_choice: str
) -> list[dict[str, Any]]:
    routes = [
        route
        for route in candidate["provider_routes"]
        if _route_compatible(route, agent_choice)
    ]
    return sorted(
        routes,
        key=lambda route: (
            route["route_kind"] != "native",
            route["provider"],
            route["provider_model_id"],
        ),
    )


def fallback_chain(
    candidates: list[dict[str, Any]],
    requested_tier: str,
    requirements: dict[str, Any],
    *,
    agent_choice: str = "either",
) -> list[dict[str, Any]]:
    """Flatten provider routes per model, then move same-tier to higher-tier."""
    chain: list[dict[str, Any]] = []
    for candidate in model_fallbacks(candidates, requested_tier, requirements):
        canonical_id = candidate["canonical_model"]["id"]
        for route in _ordered_routes(candidate, agent_choice):
            chain.append({
                "agent_compatibility": route["compatible_agents"],
                "provider": route["provider"],
                "provider_model_id": route["provider_model_id"],
                "route_kind": route["route_kind"],
                "canonical_model_id": canonical_id,
                "model_id": canonical_id,
                "tier": candidate["tier"],
            })
    return chain


def resolve_model(
    candidates: list[dict[str, Any]],
    requested_tier: str,
    requirements: dict[str, Any],
    *,
    agent_choice: str = "either",
) -> dict[str, Any]:
    """Resolve a Jev tier to one canonical model and ordered provider routes."""
    from hermes_cli.jev import JevInputError

    models = model_fallbacks(candidates, requested_tier, requirements)
    routes = fallback_chain(
        candidates, requested_tier, requirements, agent_choice=agent_choice
    )
    if not models or not routes:
        raise JevInputError(
            f"no authorized model route satisfies tier {requested_tier!r} or a higher tier"
        )
    selected_model = models[0]
    selected_route = routes[0]
    allowed_tiers = list(MODEL_TIERS[MODEL_TIERS.index(requested_tier) :])
    return {
        "provider": selected_route["provider"],
        "provider_model_id": selected_route["provider_model_id"],
        "canonical_model": selected_model["canonical_model"],
        "selected_model_id": selected_model["canonical_model"]["id"],
        "requested_tier": requested_tier,
        "tier": selected_model["tier"],
        "recommended_use": selected_model["recommended_use"],
        "agent_choice": agent_choice,
        "provider_routes": selected_model["provider_routes"],
        "fallback_chain": routes,
        "fallback_policy": {
            "route_before_model": True,
            "native_before_openrouter_when_compatible": True,
            "native_catalog_match_required": True,
            "allowed_tiers": allowed_tiers,
            "same_tier_then_higher": True,
            "lower_tier_forbidden": True,
            "runtime_must_not_silently_downgrade": True,
        },
        "requirements": requirements,
        "reasons": selected_model["tier_reasons"]
        + [
            "canonical model selection is independent from execution-agent selection",
            "provider routing is resolved after model selection and prefers a compatible, catalog-verified native route",
        ],
    }


def inspect_catalog(
    catalog: list[dict[str, Any]],
    *,
    assignments: dict[str, dict[str, Any]] | None = None,
    requested_tier: str | None = None,
    requirements: dict[str, Any] | None = None,
    native_catalogs: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build stable JSON for orchestration and human catalog inspection."""
    candidates, exclusions = classify_catalog(catalog, assignments, native_catalogs)
    requirements = requirements or model_requirements()
    tiers = {
        tier: [candidate for candidate in candidates if candidate["tier"] == tier]
        for tier in MODEL_TIERS
    }
    result: dict[str, Any] = {
        "catalog_provider": "openrouter",
        "agent_selection": "independent_not_performed",
        "policy": {
            "absolute_floor": POLICY_FLOOR,
            "catalog_source": "openrouter_authenticated_models_endpoint",
            "classification_source": (
                "jev_decisions"
                if assignments is not None
                else "deterministic_catalog_only"
            ),
            "provider_neutral_candidates": True,
            "catalog_verification_required": True,
            "native_catalog_sources": NATIVE_CATALOG_SOURCES,
            "batch_only_excluded": True,
            "provider_neutral_observable_minimum": {
                "context_length": _EQUIVALENT_CONTEXT_MIN,
                "max_completion_tokens": _EQUIVALENT_OUTPUT_MIN,
                "text_input_output": True,
                "optional_capabilities_filtered_by_requirements": [
                    "tools",
                    "reasoning",
                    "structured_outputs",
                ],
            },
            "tier_assignment": "jev" if assignments is not None else "catalog_only",
            "route_order": "exact catalog-matched compatible native provider, then OpenRouter, then next model",
        },
        "requirements": requirements,
        "candidate_count": len(candidates),
        "tiers": tiers,
        "excluded_counts": exclusions,
    }
    if requested_tier is not None:
        result["requested_tier"] = requested_tier
        result["fallback_chain"] = fallback_chain(
            candidates, requested_tier, requirements
        )
        result["fallback_policy"] = {
            "allowed_tiers": list(MODEL_TIERS[MODEL_TIERS.index(requested_tier) :]),
            "route_before_model": True,
            "lower_tier_forbidden": True,
        }
    return result
