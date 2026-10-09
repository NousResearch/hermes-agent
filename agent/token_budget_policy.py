"""Fail-closed provider+model token-budget policy resolution.

The ordinary model metadata resolver remains the source for legacy models.  This
module adds an opt-in, exact-match policy for routes where a model's documented
window differs from what the currently authenticated provider route proves.

A policy can only promote to an explicitly approved stage backed by fresh,
exact provider evidence.  Missing, stale, or account-mismatched evidence falls
back to ``safe_context_limit``; it never activates the declared target.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import math
from typing import Any, Mapping
from urllib.parse import urlparse


class TokenBudgetPolicyError(ValueError):
    """Raised when an enabled token-budget policy is unsafe or malformed."""


_FEATURE_PROVIDER = "openai-codex"
_REQUIRED_LEGACY_MODELS = frozenset(
    {"gpt-5.6-sol", "gpt-6-astra", "gpt-5.6-terra", "gpt-5.6-luna"}
)
_OPTIONAL_NEW_MODELS = frozenset({"gpt-6.1-sol", "gpt-6-luna"})
_FEATURE_MODELS = _REQUIRED_LEGACY_MODELS | _OPTIONAL_NEW_MODELS
# These models must obtain promotable route evidence from the active runtime
# catalogue. Static configuration may describe policy, but is not entitlement.
_RUNTIME_CATALOG_ONLY_MODELS = frozenset({"gpt-6-astra", *_OPTIONAL_NEW_MODELS})
_TRUSTED_EVIDENCE_SOURCES = frozenset({"codex_oauth_catalog"})
_CANONICAL_MODEL_BUDGETS: dict[str, dict[str, Any]] = {
    "gpt-5.6-sol": {
        "target_context": 1_000_000,
        "official_context": 1_050_000,
        "target_max_output": 128_000,
        "official_max_output": 128_000,
        "target_soft_budget": 700_000,
        "stage_threshold_ratio": 0.8,
        "stages": (272_000, 500_000, 750_000, 950_000, 1_000_000),
    },
    "gpt-6-astra": {
        # Target raised to the documented 872K/128K ceiling (Tales, 2026-09-14).
        # The resolver still caps the effective window at the fresh Codex OAuth
        # catalogue observation (272K today) or safe_context_limit without it,
        # so this only lets Astra grow when the account catalogue proves more.
        "target_context": 872_000,
        "official_context": 872_000,
        "target_max_output": 128_000,
        "official_max_output": 128_000,
        "target_soft_budget": 610_400,
        "stage_threshold_ratio": 0.7,
        "stages": (272_000, 500_000, 750_000, 872_000),
    },
    "gpt-5.6-terra": {
        "target_context": 750_000,
        "official_context": 1_050_000,
        "target_max_output": 96_000,
        "official_max_output": 128_000,
        "target_soft_budget": 525_000,
        "stage_threshold_ratio": 0.8,
        "stages": (272_000, 500_000, 750_000),
    },
    "gpt-5.6-luna": {
        "target_context": 500_000,
        "official_context": 1_050_000,
        "target_max_output": 64_000,
        "official_max_output": 128_000,
        "target_soft_budget": 350_000,
        "stage_threshold_ratio": 0.8,
        "stages": (272_000, 500_000),
    },
    # Public model documentation supplies only official ceilings.  The Codex
    # OAuth catalogue observed 272K for this account/route, so this migration
    # deliberately has one fixed operational stage and a 54,400-token reserve.
    "gpt-6.1-sol": {
        "target_context": 272_000,
        "official_context": 1_050_000,
        "target_max_output": 54_400,
        "official_max_output": 128_000,
        "target_soft_budget": 217_600,
        "stage_threshold_ratio": 0.8,
        "stages": (272_000,),
    },
    "gpt-6-luna": {
        "target_context": 272_000,
        "official_context": 1_050_000,
        "target_max_output": 54_400,
        "official_max_output": 128_000,
        "target_soft_budget": 217_600,
        "stage_threshold_ratio": 0.8,
        "stages": (272_000,),
    },
}
_CANONICAL_APPROVAL_STAGES = frozenset(
    stage
    for budget in _CANONICAL_MODEL_BUDGETS.values()
    for stage in budget["stages"]
)


def _safe_evidence_source(source: str) -> str:
    value = (source or "").strip()
    return value if value in _TRUSTED_EVIDENCE_SOURCES else "configured_evidence"


@dataclass(frozen=True)
class ProviderContextEvidence:
    provider: str
    model: str
    observed_context: int
    observed_at: datetime
    source: str
    account_key: str = ""
    route_key: str = ""
    observed_max_output: int | None = None


@dataclass(frozen=True)
class TokenBudgetResolution:
    provider: str
    model: str
    target_context: int
    official_context: int
    observed_context: int | None
    effective_context: int
    soft_budget: int
    official_max_output: int
    observed_max_output: int | None
    max_output: int
    compression_threshold: float
    stage: int
    approved_stage: int
    status: str
    evidence_source: str
    evidence_age_seconds: int | None
    evidence_fresh: bool
    account_fingerprint: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/session-safe representation for status surfaces."""
        data = asdict(self)
        return data


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TokenBudgetPolicyError(f"{field} must be a positive integer")
    return value


def _ratio(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TokenBudgetPolicyError(f"{field} must be a number between 0 and 1")
    ratio = float(value)
    if not math.isfinite(ratio) or not 0 < ratio < 1:
        raise TokenBudgetPolicyError(f"{field} must be a number between 0 and 1")
    return ratio


def validate_token_budget_policy_config(
    config: Mapping[str, Any] | None,
) -> None:
    """Validate every enabled route without making provider/network calls."""
    if config is None:
        return
    if not isinstance(config, Mapping):
        raise TokenBudgetPolicyError("config must be a mapping")

    raw_policy = config.get("token_budget_policy")
    if raw_policy is None:
        return
    if not isinstance(raw_policy, Mapping):
        raise TokenBudgetPolicyError("token_budget_policy must be a mapping")

    enabled = raw_policy.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TokenBudgetPolicyError("token_budget_policy.enabled must be a boolean")
    if not enabled:
        return

    _positive_int(
        raw_policy.get("evidence_ttl_seconds"),
        "token_budget_policy.evidence_ttl_seconds",
    )
    safe_context = _positive_int(
        raw_policy.get("safe_context_limit"),
        "token_budget_policy.safe_context_limit",
    )
    approved_stage = _positive_int(
        raw_policy.get("approved_stage"),
        "token_budget_policy.approved_stage",
    )
    if approved_stage not in _CANONICAL_APPROVAL_STAGES:
        raise TokenBudgetPolicyError(
            "token_budget_policy.approved_stage must be a canonical Feature 007 stage"
        )
    if safe_context not in _CANONICAL_APPROVAL_STAGES:
        raise TokenBudgetPolicyError(
            "token_budget_policy.safe_context_limit must be a canonical Feature 007 stage"
        )

    providers = raw_policy.get("providers")
    if not isinstance(providers, Mapping) or not providers:
        raise TokenBudgetPolicyError(
            "token_budget_policy.providers must be a non-empty mapping"
        )

    for provider, provider_config in providers.items():
        if not isinstance(provider, str) or not provider.strip():
            raise TokenBudgetPolicyError(
                "token_budget_policy.providers keys must be non-empty strings"
            )
        if provider != _FEATURE_PROVIDER:
            raise TokenBudgetPolicyError(
                "token_budget_policy provider key must be canonical "
                f"'{_FEATURE_PROVIDER}', got {provider!r}"
            )
        provider_key = provider
        if not isinstance(provider_config, Mapping):
            raise TokenBudgetPolicyError(
                f"token_budget_policy.providers.{provider} must be a mapping"
            )
        models = provider_config.get("models")
        if not isinstance(models, Mapping) or not models:
            raise TokenBudgetPolicyError(
                f"token_budget_policy.providers.{provider}.models must be a non-empty mapping"
            )
        model_keys = set(models)
        missing = sorted(_REQUIRED_LEGACY_MODELS - model_keys)
        extra = sorted(str(key) for key in model_keys - _FEATURE_MODELS)
        if missing or extra:
            raise TokenBudgetPolicyError(
                "enabled token-budget policy must configure exactly all required "
                f"legacy models and only optional allowlisted models; missing={missing}, "
                f"extra={extra}"
            )
        for model, model_config in models.items():
            if not isinstance(model, str) or not model.strip():
                raise TokenBudgetPolicyError(
                    f"token_budget_policy.providers.{provider}.models keys must be non-empty strings"
                )
            if model != model.strip():
                raise TokenBudgetPolicyError(
                    f"token_budget_policy model key must be canonical, got {model!r}"
                )
            model_key = model
            if model_key not in _FEATURE_MODELS:
                raise TokenBudgetPolicyError(
                    f"unsupported model in token-budget feature allowlist: {model}"
                )
            if not isinstance(model_config, Mapping):
                raise TokenBudgetPolicyError(
                    f"token_budget_policy.providers.{provider}.models.{model} must be a mapping"
                )
            canonical = _CANONICAL_MODEL_BUDGETS[model_key]
            for field, expected in canonical.items():
                actual = model_config.get(field)
                if field == "stages" and isinstance(actual, list):
                    actual = tuple(actual)
                if actual != expected:
                    raise TokenBudgetPolicyError(
                        f"{model_key}.{field} must equal the canonical Feature 007 "
                        f"value {expected!r}"
                    )
            model_approved_stage = model_config.get("approved_stage")
            if model_key in _OPTIONAL_NEW_MODELS and model_approved_stage is None:
                raise TokenBudgetPolicyError(
                    f"{model_key}.approved_stage is required for new Codex models"
                )
            if model_approved_stage is not None:
                model_approved_stage = _positive_int(
                    model_approved_stage, f"{model_key}.approved_stage"
                )
                if model_approved_stage not in _CANONICAL_APPROVAL_STAGES:
                    raise TokenBudgetPolicyError(
                        f"{model_key}.approved_stage must be a canonical Feature 007 stage"
                    )
                if (
                    model_key in _OPTIONAL_NEW_MODELS
                    and model_approved_stage != 272_000
                ):
                    raise TokenBudgetPolicyError(
                        f"{model_key}.approved_stage must remain 272000 in this migration"
                    )
            # The resolver is the canonical semantic validator for all model
            # fields. Missing evidence intentionally selects the safe cap.
            resolve_token_budget(
                config,
                provider=provider,
                model=model,
                evidence=None,
                _config_validated=True,
            )


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _config_evidence(
    raw: Any,
    *,
    provider: str,
    model: str,
) -> ProviderContextEvidence | None:
    if not isinstance(raw, Mapping):
        return None
    observed_at = _parse_timestamp(raw.get("observed_at"))
    observed_context = raw.get("observed_context")
    source = raw.get("source")
    if (
        observed_at is None
        or isinstance(observed_context, bool)
        or not isinstance(observed_context, int)
        or observed_context <= 0
        or not isinstance(source, str)
        or not source.strip()
    ):
        return None
    account_key = raw.get("account_key")
    observed_max_output = raw.get("observed_max_output")
    if isinstance(observed_max_output, bool) or (
        observed_max_output is not None
        and (not isinstance(observed_max_output, int) or observed_max_output <= 0)
    ):
        observed_max_output = None
    return ProviderContextEvidence(
        provider=provider,
        model=model,
        observed_context=observed_context,
        observed_at=observed_at,
        source=source.strip(),
        account_key=account_key.strip() if isinstance(account_key, str) else "",
        route_key=(
            raw.get("route_key", "").strip()
            if isinstance(raw.get("route_key"), str)
            else ""
        ),
        observed_max_output=observed_max_output,
    )


def _codex_route_key(base_url: str) -> str:
    """Return the canonical trusted Codex route identity, or ``""``."""
    try:
        parsed = urlparse(str(base_url or "").strip())
    except Exception:
        return ""
    host = (parsed.hostname or "").lower().rstrip(".")
    path = "/" + (parsed.path or "").strip("/")
    if host == "chatgpt.com" and path == "/backend-api/codex":
        return "chatgpt.com/backend-api/codex"
    return ""


def _account_fingerprint(account_key: str) -> str:
    value = (account_key or "").strip()
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _codex_catalog_observed_at(access_token: str, live: Mapping[str, int]) -> float | None:
    """Read a matching in-memory Codex catalogue timestamp compatibly.

    Feature 007 must not mutate ``model_metadata`` or call a second endpoint.
    The metadata cache has had two private tuple/key layouts, so accept either
    only when its catalog exactly matches the just-resolved authenticated one.
    """
    try:
        from agent import model_metadata

        cache = getattr(model_metadata, "_codex_oauth_context_cache", {})
        digest = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
    except Exception:
        return None
    if not isinstance(cache, Mapping):
        return None
    for key in (digest[:16], digest[:24]):
        cached = cache.get(key)
        if not isinstance(cached, tuple) or len(cached) != 2:
            continue
        first, second = cached
        if isinstance(first, Mapping) and isinstance(second, (int, float)):
            cached_catalog, cached_at = first, second
        elif isinstance(second, Mapping) and isinstance(first, (int, float)):
            cached_catalog, cached_at = second, first
        else:
            continue
        if (
            isinstance(cached_at, bool)
            or not isinstance(cached_at, (int, float))
            or dict(cached_catalog) != dict(live)
        ):
            continue
        return float(cached_at)
    return None


def detect_provider_context_evidence(
    *,
    provider: str,
    model: str,
    base_url: str = "",
    api_key: str = "",
    account_key: str = "",
    now: datetime | None = None,
) -> ProviderContextEvidence | None:
    """Return fresh exact-slug context evidence from a supported provider.

    The detector deliberately does not reuse family fallbacks or configured
    overrides: neither proves that this authenticated route accepts the exact
    model at the reported window.  Credentials are consumed only by the
    existing in-memory provider catalogue call and are never logged or stored.
    """
    provider_key = (provider or "").strip().lower()
    model_key = (model or "").strip()
    if provider_key != "openai-codex" or not model_key or not api_key:
        return None

    route_key = _codex_route_key(base_url)
    if not route_key:
        return None

    from agent.model_metadata import _fetch_codex_oauth_context_lengths_with_source

    try:
        live, fetched_from_http = _fetch_codex_oauth_context_lengths_with_source(api_key)
    except Exception:
        return None
    observed = live.get(model_key) if isinstance(live, Mapping) else None
    if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
        return None

    if fetched_from_http:
        # A source-aware fresh result is evidence from the authenticated
        # provider endpoint.  ``now`` is only its local observation time; it
        # is never used to relabel an unverified cache hit as fresh.
        observed_at = now or datetime.now(timezone.utc)
    else:
        # Same-process cache hits are promotable only when the exact catalogue
        # has a corresponding, verifiable cache timestamp.  A result without
        # that timestamp is deliberately no evidence, rather than fabricated
        # freshness at policy-resolution time.
        cached_at = _codex_catalog_observed_at(api_key, live)
        if cached_at is None:
            return None
        observed_at = datetime.fromtimestamp(cached_at, tz=timezone.utc)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    return ProviderContextEvidence(
        provider=provider_key,
        model=model_key,
        observed_context=observed,
        observed_at=observed_at.astimezone(timezone.utc),
        source="codex_oauth_catalog",
        account_key=(account_key or "").strip(),
        route_key=route_key,
    )


def resolve_runtime_token_budget(
    config: Mapping[str, Any] | None,
    *,
    provider: str,
    model: str,
    base_url: str = "",
    api_key: str = "",
    account_key: str = "",
    now: datetime | None = None,
) -> TokenBudgetResolution | None:
    """Detect exact live evidence when available, then resolve the policy."""
    current_account = (account_key or "").strip()
    if not current_account and api_key:
        current_account = "credential-fingerprint-" + _account_fingerprint(api_key)

    current_route = _codex_route_key(base_url)
    preflight = resolve_token_budget(
        config,
        provider=provider,
        model=model,
        evidence=None,
        account_key=current_account,
        route_key=current_route,
        now=now,
    )
    if preflight is None:
        return None

    evidence = detect_provider_context_evidence(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        account_key=current_account,
        now=now,
    )
    if evidence is None:
        return preflight
    return resolve_token_budget(
        config,
        provider=provider,
        model=model,
        evidence=evidence,
        account_key=current_account,
        route_key=current_route,
        now=now,
    )


def _runtime_account_identity(agent: Any) -> str:
    """Return a stable, non-secret identity for the active credential/account."""
    entry_id = str(getattr(agent, "_credential_pool_entry_id", "") or "").strip()
    if entry_id:
        return entry_id
    pool = getattr(agent, "_credential_pool", None)
    if pool is not None:
        try:
            entry_id = str(getattr(pool.current(), "id", "") or "").strip()
        except Exception:
            entry_id = ""
        if entry_id:
            return entry_id
    raw_key = getattr(agent, "api_key", "")
    if isinstance(raw_key, str) and raw_key:
        return "credential-fingerprint-" + _account_fingerprint(raw_key)
    return ""


def _runtime_route_identity(agent: Any) -> tuple[str, str, str, str, str]:
    """Return the exact live route+account identity used for reversible budgets."""
    return (
        str(getattr(agent, "provider", "") or "").strip().lower(),
        str(getattr(agent, "model", "") or "").strip(),
        str(getattr(agent, "base_url", "") or "").strip(),
        str(getattr(agent, "api_mode", "") or "").strip().lower(),
        _runtime_account_identity(agent),
    )


def _copy_runtime_value(value: Any) -> Any:
    """Copy mutable runtime state without requiring client objects to copy."""
    if isinstance(value, dict):
        return {key: _copy_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_runtime_value(item) for item in value]
    if isinstance(value, set):
        return {_copy_runtime_value(item) for item in value}
    if isinstance(value, tuple):
        return tuple(_copy_runtime_value(item) for item in value)
    return value


def _snapshot_compressor_state(compressor: Any) -> dict[str, Any] | None:
    """Capture every instance-owned compressor field for a route baseline."""
    if compressor is None:
        return None
    try:
        return _copy_runtime_value(dict(vars(compressor)))
    except TypeError:
        # Slotted plugin compressors have no general complete state contract.
        # Keep the route's scalar output baseline but never pretend we captured
        # fields that cannot be restored safely.
        return None


def _restore_compressor_state(compressor: Any, state: dict[str, Any] | None) -> None:
    """Restore a previously captured complete compressor baseline in place."""
    if compressor is None or state is None:
        return
    try:
        values = vars(compressor)
    except TypeError:
        return
    values.clear()
    values.update(_copy_runtime_value(state))


def _capture_route_baseline(
    agent: Any, route: tuple[str, str, str, str, str]
) -> None:
    """Record pre-policy runtime state once, never replacing an active baseline."""
    baselines = getattr(agent, "_token_budget_route_baselines", None)
    if not isinstance(baselines, dict):
        baselines = {}
        agent._token_budget_route_baselines = baselines
    if route in baselines:
        return
    compressor = getattr(agent, "context_compressor", None)
    # An in-place credential rotation keeps the same physical runtime and
    # compressor. Reuse that route's original pre-policy baseline; capturing
    # the current state here would record account A's policy-mutated window as
    # account B's provider default and could resurrect it after policy removal.
    physical_route = route[:-1]
    for existing_route, existing in tuple(baselines.items()):
        if (
            isinstance(existing_route, tuple)
            and existing_route[:-1] == physical_route
            and isinstance(existing, Mapping)
            and existing.get("compressor_object") is compressor
        ):
            baselines[route] = {
                "max_tokens": existing.get("max_tokens"),
                "compressor": _copy_runtime_value(existing.get("compressor")),
                "compressor_object": compressor,
            }
            return
    baselines[route] = {
        "max_tokens": getattr(agent, "max_tokens", None),
        "compressor": _snapshot_compressor_state(compressor),
        "compressor_object": compressor,
    }


def _restore_route_baseline(
    agent: Any, route: tuple[str, str, str, str, str]
) -> bool:
    """Restore and retire the exact route's pre-policy state, if available."""
    baselines = getattr(agent, "_token_budget_route_baselines", None)
    if not isinstance(baselines, dict):
        return False
    baseline = baselines.pop(route, None)
    if not isinstance(baseline, Mapping):
        return False
    agent.max_tokens = baseline.get("max_tokens")
    compressor = getattr(agent, "context_compressor", None)
    # A switch normally reuses the compressor instance. Do not copy fields
    # from a different plugin instance onto the current one.
    if compressor is baseline.get("compressor_object"):
        _restore_compressor_state(compressor, baseline.get("compressor"))
    return True


def apply_runtime_token_budget(
    agent: Any, config: Mapping[str, Any] | None
) -> TokenBudgetResolution | None:
    """Resolve policy for ``agent`` and apply output/status fields.

    Context-engine callers use the returned ``effective_context`` and
    ``compression_threshold`` while constructing or updating their compressor.
    Keeping that part explicit avoids coupling this module to one engine.
    """
    # Validate before touching last-known-good policy state or runtime fields.
    # This keeps lifecycle preflight and direct callers equally fail-closed.
    validate_token_budget_policy_config(config)
    route = _runtime_route_identity(agent)
    account_key = route[-1]
    raw_key = getattr(agent, "api_key", "")
    api_key = raw_key if isinstance(raw_key, str) else ""
    policy_cfg = (
        config.get("token_budget_policy") if isinstance(config, Mapping) else None
    )
    if isinstance(policy_cfg, Mapping):
        # Keep only the non-secret policy subtree as a last-known-good fallback
        # for in-place route transitions. Callers use this snapshot only when
        # re-reading config fails; successful reads still own hot reload/removal.
        agent._token_budget_policy_config = {
            "token_budget_policy": copy.deepcopy(dict(policy_cfg)),
        }
    else:
        agent._token_budget_policy_config = {}
    result = resolve_runtime_token_budget(
        config,
        provider=getattr(agent, "provider", ""),
        model=getattr(agent, "model", ""),
        base_url=getattr(agent, "base_url", ""),
        api_key=api_key,
        account_key=account_key,
    )

    configured_max = getattr(agent, "_configured_max_tokens", None)
    if not bool(getattr(agent, "_configured_max_tokens_captured", False)):
        # Legacy agents/tests predate the explicit capture marker. Preserve an
        # existing configured value; otherwise the current value is the only
        # available baseline. Capture even ``None`` so a later policy-derived
        # cap cannot be mistaken for a provider-default baseline.
        if configured_max is None:
            current_max = getattr(agent, "max_tokens", None)
            if (
                isinstance(current_max, int)
                and not isinstance(current_max, bool)
                and current_max > 0
            ):
                configured_max = current_max
        agent._configured_max_tokens = configured_max
        agent._configured_max_tokens_captured = True
    if result is None:
        agent._token_budget_status = None
        if not _restore_route_baseline(agent, route):
            # Legacy and non-policy routes use the configured output baseline.
            # A route baseline, when present, wins because it records a prior
            # policy application for this exact provider/model/endpoint.
            agent.max_tokens = configured_max
        return None

    # This happens after a switch/fallback has built the destination engine,
    # preserving the baseline produced by that transition. Subsequent policy
    # refreshes for the same route intentionally do not overwrite it.
    _capture_route_baseline(agent, route)
    if (
        isinstance(configured_max, int)
        and not isinstance(configured_max, bool)
        and configured_max > 0
    ):
        agent.max_tokens = min(configured_max, result.max_output)
    else:
        agent.max_tokens = result.max_output
    agent._token_budget_status = result.as_dict()
    agent._token_budget_status["runtime_max_output"] = agent.max_tokens
    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    api_mode = str(getattr(agent, "api_mode", "") or "").strip().lower()
    base_url = str(getattr(agent, "base_url", "") or "").strip().lower()
    codex_backend = bool(_codex_route_key(base_url))
    codex_provider_default = codex_backend or (
        api_mode == "codex_responses" and provider == _FEATURE_PROVIDER
    )
    agent._token_budget_status["output_cap_enforcement"] = (
        "provider_default" if codex_provider_default else "request_cap"
    )
    agent._token_budget_status["request_output_cap"] = (
        None if codex_provider_default else agent.max_tokens
    )
    return result


def resolve_token_budget(
    config: Mapping[str, Any] | None,
    *,
    provider: str,
    model: str,
    evidence: ProviderContextEvidence | None = None,
    account_key: str = "",
    route_key: str = "",
    now: datetime | None = None,
    _config_validated: bool = False,
) -> TokenBudgetResolution | None:
    """Resolve an exact provider+model policy into safe runtime budgets.

    Returns ``None`` when the feature is disabled or no exact entry exists.
    An enabled matching entry that is malformed raises
    :class:`TokenBudgetPolicyError` so startup fails visibly rather than
    silently applying an unsafe generic value.
    """
    if not isinstance(config, Mapping):
        return None
    policy = config.get("token_budget_policy")
    if not isinstance(policy, Mapping) or policy.get("enabled") is not True:
        return None
    if not _config_validated:
        validate_token_budget_policy_config(config)

    provider_key = (provider or "").strip().lower()
    model_key = (model or "").strip()
    providers = policy.get("providers")
    if not isinstance(providers, Mapping):
        raise TokenBudgetPolicyError("token_budget_policy.providers must be a mapping")
    provider_cfg = providers.get(provider_key)
    if not isinstance(provider_cfg, Mapping):
        return None
    models = provider_cfg.get("models")
    if not isinstance(models, Mapping):
        raise TokenBudgetPolicyError(
            f"token_budget_policy.providers.{provider_key}.models must be a mapping"
        )
    model_cfg = models.get(model_key)
    if not isinstance(model_cfg, Mapping):
        return None

    ttl = _positive_int(policy.get("evidence_ttl_seconds"), "evidence_ttl_seconds")
    safe_context = _positive_int(policy.get("safe_context_limit"), "safe_context_limit")
    global_approved_stage = _positive_int(
        policy.get("approved_stage"), "approved_stage"
    )
    target_context = _positive_int(model_cfg.get("target_context"), "target_context")
    official_context = _positive_int(
        model_cfg.get("official_context"), "official_context"
    )
    target_output = _positive_int(
        model_cfg.get("target_max_output"), "target_max_output"
    )
    official_output = _positive_int(
        model_cfg.get("official_max_output"), "official_max_output"
    )
    target_soft = _positive_int(
        model_cfg.get("target_soft_budget"), "target_soft_budget"
    )
    threshold_ratio = _ratio(
        model_cfg.get("stage_threshold_ratio"), "stage_threshold_ratio"
    )

    raw_stages = model_cfg.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise TokenBudgetPolicyError("stages must be a non-empty list")
    stages = sorted({_positive_int(value, "stages") for value in raw_stages})
    declared_cap = min(target_context, official_context)
    if safe_context > declared_cap:
        raise TokenBudgetPolicyError(
            "safe_context_limit cannot exceed target_context or official_context"
        )
    if any(stage > declared_cap for stage in stages):
        raise TokenBudgetPolicyError(
            "stages cannot exceed target_context or official_context"
        )
    if global_approved_stage not in _CANONICAL_APPROVAL_STAGES:
        raise TokenBudgetPolicyError(
            "approved_stage must be a canonical Feature 007 stage"
        )
    configured_model_stage = model_cfg.get("approved_stage")
    if model_key in _OPTIONAL_NEW_MODELS and configured_model_stage is None:
        raise TokenBudgetPolicyError(
            f"{model_key}.approved_stage is required for new Codex models"
        )
    if configured_model_stage is None:
        # Existing policies retain their global-only behavior unless they opt
        # into a narrower per-model ceiling.
        model_approved_stage = global_approved_stage
    else:
        model_approved_stage = _positive_int(
            configured_model_stage, f"{model_key}.approved_stage"
        )
        if model_approved_stage not in _CANONICAL_APPROVAL_STAGES:
            raise TokenBudgetPolicyError(
                f"{model_key}.approved_stage must be a canonical Feature 007 stage"
            )
    approved_stage = min(global_approved_stage, model_approved_stage)

    current_account = (account_key or "").strip()
    current_route = (route_key or "").strip()
    # Astra and the new Codex models are deliberately runtime-catalog-only:
    # static configured evidence cannot prove what the active OAuth account
    # and route accept, and must never promote them.
    selected_evidence = evidence
    if selected_evidence is None and model_key not in _RUNTIME_CATALOG_ONLY_MODELS:
        selected_evidence = _config_evidence(
            model_cfg.get("evidence"), provider=provider_key, model=model_key
        )
    evidence_source = ""
    evidence_age: int | None = None
    evidence_fresh = False
    evidence_identity_matches = False
    blocked_status = "blocked_missing_evidence"

    if selected_evidence is not None:
        exact_subject = (
            selected_evidence.provider.strip().lower() == provider_key
            and selected_evidence.model.strip() == model_key
        )
        evidence_route = selected_evidence.route_key.strip()
        route_matches = bool(
            current_route and evidence_route and current_route == evidence_route
        )
        evidence_account = selected_evidence.account_key.strip()
        account_matches = bool(
            current_account and evidence_account and current_account == evidence_account
        )
        evidence_identity_matches = exact_subject and route_matches and account_matches
        evidence_source = _safe_evidence_source(selected_evidence.source)
        source_is_trusted = selected_evidence.source.strip() in _TRUSTED_EVIDENCE_SOURCES
        reference_now = now or datetime.now(timezone.utc)
        if reference_now.tzinfo is None:
            reference_now = reference_now.replace(tzinfo=timezone.utc)
        observed_at = selected_evidence.observed_at
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        evidence_age = int(
            (
                reference_now.astimezone(timezone.utc)
                - observed_at.astimezone(timezone.utc)
            ).total_seconds()
        )
        if not exact_subject or not route_matches:
            blocked_status = "blocked_route_mismatch"
        elif not account_matches:
            blocked_status = "blocked_account_mismatch"
        elif model_key in _RUNTIME_CATALOG_ONLY_MODELS and not source_is_trusted:
            blocked_status = "blocked_untrusted_evidence"
        elif evidence_age < 0:
            blocked_status = "blocked_future_evidence"
        elif evidence_age >= ttl:
            blocked_status = "blocked_stale_evidence"
        else:
            evidence_fresh = True

    observed_context = selected_evidence.observed_context if selected_evidence else None
    observed_max_output = (
        selected_evidence.observed_max_output
        if selected_evidence is not None and evidence_identity_matches
        else None
    )
    if evidence_fresh and observed_context is not None:
        route_cap = observed_context
    elif evidence_identity_matches and observed_context is not None:
        # A stale *lower* observation from this exact identity is still a
        # conservative cap; a stale higher value must never raise fallback.
        route_cap = min(safe_context, observed_context)
    else:
        route_cap = safe_context

    route_candidate_cap = min(declared_cap, route_cap)
    candidate_cap = min(route_candidate_cap, approved_stage)
    approved = [stage for stage in stages if stage <= candidate_cap]
    effective_context = approved[-1] if approved else candidate_cap
    if effective_context <= 0:
        raise TokenBudgetPolicyError(
            "effective context resolved to a non-positive value"
        )

    soft_budget = min(target_soft, int(effective_context * threshold_ratio))
    if soft_budget <= 0 or soft_budget >= effective_context:
        raise TokenBudgetPolicyError(
            "resolved soft budget must leave a positive output reserve"
        )
    output_reserve = effective_context - soft_budget
    output_caps = [target_output, official_output, output_reserve]
    if observed_max_output is not None:
        output_caps.append(observed_max_output)
    max_output = min(output_caps)
    if max_output <= 0:
        raise TokenBudgetPolicyError("resolved max output must be positive")

    if evidence_fresh and approved_stage < route_candidate_cap:
        status = "capped_by_approved_stage"
    elif evidence_fresh:
        status = (
            "target_active"
            if effective_context >= target_context
            else "capped_by_provider_route"
        )
    else:
        status = blocked_status

    return TokenBudgetResolution(
        provider=provider_key,
        model=model_key,
        target_context=target_context,
        official_context=official_context,
        observed_context=observed_context,
        effective_context=effective_context,
        soft_budget=soft_budget,
        official_max_output=official_output,
        observed_max_output=observed_max_output,
        max_output=max_output,
        compression_threshold=soft_budget / effective_context,
        stage=effective_context,
        approved_stage=approved_stage,
        status=status,
        evidence_source=evidence_source,
        evidence_age_seconds=evidence_age,
        evidence_fresh=evidence_fresh,
        account_fingerprint=_account_fingerprint(
            selected_evidence.account_key if selected_evidence else current_account
        ),
    )
