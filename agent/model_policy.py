"""Central model orchestration policy resolver.

This module is intentionally pure: it chooses model/provider/fallback policy from
already-normalized runtime inputs, but it never reads credentials, environment
variables, config files, or remote metadata. Credential resolution remains owned
by the existing runtime/provider layers.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_SENSITIVE_SUBSTRINGS = (
    "api_key",
    "authorization",
    "token",
    "secret",
    "password",
    "bearer",
)

_FALLBACK_KEYS = ("fallback_models", "fallbacks", "fallback_chain")
_PROVIDER_OPTIONS_KEYS = ("providerOptions", "provider_options", "provider_options")


@dataclass(frozen=True)
class ResolvedModelPolicy:
    """Resolved model selection artifact safe to attach to runtime metadata."""

    primary_provider: str | None
    primary_model: str | None
    variant: str | None
    provider_options: dict[str, Any]
    fallback_chain: list[dict[str, str]]
    trace: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_provider": self.primary_provider,
            "primary_model": self.primary_model,
            "variant": self.variant,
            "provider_options": deepcopy(self.provider_options),
            "fallback_chain": deepcopy(self.fallback_chain),
            "trace": deepcopy(self.trace),
        }


def parse_model_variant(value: str | None) -> tuple[str | None, str | None]:
    """Split a single trailing inline ``model(variant)`` suffix.

    Slash-form models and colon variant tags such as ``:free`` are left intact.
    """

    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None
    if text.endswith(")") and "(" in text:
        model, variant = text.rsplit("(", 1)
        model = model.strip()
        variant = variant[:-1].strip()
        if model:
            return model, variant or None
    return text, None


def resolve_model_policy(
    *,
    named_agent: Mapping[str, Any] | None = None,
    archetype: str | None = None,
    specialist: str | None = None,
    route_category: str | None = None,
    runtime_mode: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    fallback_models: Iterable[Any] | None = None,
    fallbacks: Iterable[Any] | None = None,
    provider_options: Mapping[str, Any] | None = None,
    providerOptions: Mapping[str, Any] | None = None,
    route_policies: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | None = None,
    provider_chain: Iterable[Any] | None = None,
    request_size_tokens: int | None = None,
) -> ResolvedModelPolicy:
    """Resolve primary model, fallback chain, and a safe precedence trace.

    Precedence is explicit override -> named-agent override -> route-category
    default -> runtime ultrawork override -> user fallback -> provider chain ->
    system default. In practice ultrawork is scoped to named-agent policy and is
    activated only when ``runtime_mode == 'ultrawork'``.
    """

    named_agent_cfg = _as_mapping(named_agent)
    default_policy = _as_mapping(defaults)
    route_policy = _lookup_route_policy(route_policies, route_category)
    named_policy = _select_named_agent_policy(named_agent_cfg, runtime_mode)

    explicit_model, explicit_variant = parse_model_variant(model)
    explicit_provider = _clean_str(provider)
    explicit_provider_options = _first_mapping(provider_options, providerOptions)

    source_trace: list[dict[str, Any]] = []
    selection_source = "system_default"

    primary_provider = explicit_provider
    primary_model = explicit_model
    variant = explicit_variant

    if primary_provider or primary_model:
        selection_source = "explicit_override"
    else:
        for source_name, policy in (
            ("named_agent_ultrawork" if _is_ultrawork(runtime_mode) and _as_mapping(named_agent_cfg.get("ultrawork")) else "named_agent", named_policy),
            ("route_category", route_policy),
            ("system_default", default_policy),
        ):
            candidate_model, candidate_variant = parse_model_variant(policy.get("model"))
            candidate_provider = _clean_str(policy.get("provider"))
            if candidate_provider or candidate_model:
                primary_provider = candidate_provider
                primary_model = candidate_model
                variant = candidate_variant
                selection_source = source_name
                break

    if not primary_provider:
        primary_provider = (
            _clean_str(named_policy.get("provider"))
            or _clean_str(route_policy.get("provider"))
            or _clean_str(default_policy.get("provider"))
        )

    provider_options_source = "none"
    if explicit_provider_options:
        resolved_provider_options = _sanitize_mapping(explicit_provider_options)
        provider_options_source = "explicit_override"
    elif selection_source in {"named_agent", "named_agent_ultrawork"}:
        resolved_provider_options = _sanitize_mapping(_get_provider_options(named_policy))
        provider_options_source = selection_source
    elif selection_source == "route_category":
        resolved_provider_options = _sanitize_mapping(_get_provider_options(route_policy))
        provider_options_source = "route_category"
    elif selection_source == "system_default":
        resolved_provider_options = _sanitize_mapping(_get_provider_options(default_policy))
        provider_options_source = "system_default"
    else:
        resolved_provider_options = {}

    user_fallbacks = list(fallback_models if fallback_models is not None else (fallbacks or []))
    chain = _build_fallback_chain(
        primary_provider=primary_provider,
        primary_model=primary_model,
        active_provider=primary_provider,
        explicit_fallbacks=user_fallbacks,
        named_policy=named_policy,
        route_policy=route_policy,
        default_policy=default_policy,
        provider_chain=list(provider_chain or default_policy.get("provider_chain") or []),
    )

    source_trace.append(_trace("selection", selection_source, {"provider": primary_provider, "model": primary_model, "variant": variant}))
    if explicit_provider or explicit_model:
        source_trace.append(_trace("explicit_override", "user", {"provider": explicit_provider, "model": explicit_model, "variant": explicit_variant}))
    if _is_ultrawork(runtime_mode) and _as_mapping(named_agent_cfg.get("ultrawork")):
        source_trace.append(_trace("runtime_mode", "ultrawork", "named_agent_ultrawork_override_active"))
    if named_policy:
        source_trace.append(_trace("named_agent", _clean_str(named_agent_cfg.get("name")) or "configured", _policy_trace_value(named_policy)))
    if route_policy:
        source_trace.append(_trace("route_category", _clean_str(route_category), _policy_trace_value(route_policy)))
    if user_fallbacks:
        source_trace.append(_trace("user_fallback", "explicit", user_fallbacks))
    if provider_chain or default_policy.get("provider_chain"):
        source_trace.append(_trace("provider_chain", "configured", list(provider_chain or default_policy.get("provider_chain") or [])))
    if request_size_tokens is not None:
        source_trace.append(_trace("request_size_tokens", "input", int(max(request_size_tokens, 0))))
    source_trace.append(_trace("provider_options", provider_options_source, resolved_provider_options))
    source_trace.append(_trace("resolved", selection_source, {"provider": primary_provider, "model": primary_model, "variant": variant, "fallback_count": len(chain)}))

    return ResolvedModelPolicy(
        primary_provider=primary_provider,
        primary_model=primary_model,
        variant=variant,
        provider_options=resolved_provider_options,
        fallback_chain=chain,
        trace=source_trace,
    )


def _lookup_route_policy(route_policies: Mapping[str, Any] | None, route_category: str | None) -> dict[str, Any]:
    if not route_category or not isinstance(route_policies, Mapping):
        return {}
    wanted = str(route_category).strip()
    for key, value in route_policies.items():
        if str(key).strip() == wanted:
            return _as_mapping(value)
    return {}


def _select_named_agent_policy(named_agent: Mapping[str, Any], runtime_mode: str | None) -> dict[str, Any]:
    if not named_agent:
        return {}
    # Wave 2 owns the canonical named-agent schema.  Wave 3 only consumes a
    # mapping-shaped contract and fails closed on common disabled/unsupported
    # flags so a non-invocable named agent cannot accidentally win routing.
    if _policy_disabled(named_agent) or not _runtime_mode_supported(named_agent, runtime_mode):
        return {}
    base = deepcopy(_as_mapping(named_agent.get("model_policy")) or _as_mapping(named_agent))
    if _policy_disabled(base) or not _runtime_mode_supported(base, runtime_mode):
        return {}
    # Wave 2 is allowed to expose either fallback_models or fallbacks. Normalize
    # both here so Wave 3 can run as a prototype branch before Wave 2 merges.
    _normalize_policy_aliases_in_place(base)
    if not _is_ultrawork(runtime_mode):
        return base
    ultrawork = deepcopy(_as_mapping(base.get("ultrawork")))
    if not ultrawork:
        return base
    if _policy_disabled(ultrawork) or not _runtime_mode_supported(ultrawork, runtime_mode):
        return base
    _normalize_policy_aliases_in_place(ultrawork)
    merged = deepcopy(base)
    merged.pop("ultrawork", None)
    for key, value in ultrawork.items():
        merged[key] = deepcopy(value)
    _normalize_policy_aliases_in_place(merged)
    return merged


def _policy_disabled(policy: Mapping[str, Any]) -> bool:
    mode = _clean_str(policy.get("mode"))
    status = _clean_str(policy.get("status"))
    if mode == "disabled" or status == "disabled":
        return True
    for key in ("disabled", "disable"):
        if policy.get(key) is True:
            return True
    if policy.get("enabled") is False:
        return True
    return False


def _runtime_mode_supported(policy: Mapping[str, Any], runtime_mode: str | None) -> bool:
    supported = _first_present(policy, ("supported_runtime_modes", "runtime_modes", "allowed_runtime_modes"))
    if supported in (None, "", []):
        return True
    active = _clean_str(runtime_mode) or "default"
    if isinstance(supported, str):
        values = {part.strip() for part in supported.replace(",", " ").split() if part.strip()}
    elif isinstance(supported, Iterable) and not isinstance(supported, Mapping):
        values = {_clean_str(item) for item in supported}
        values.discard(None)
    else:
        return True
    return active in values


def _normalize_policy_aliases_in_place(policy: dict[str, Any]) -> None:
    if "fallbacks" not in policy and "fallback_models" in policy:
        policy["fallbacks"] = deepcopy(policy.get("fallback_models"))
    if "provider_options" not in policy and "providerOptions" in policy:
        policy["provider_options"] = deepcopy(policy.get("providerOptions"))


def _get_provider_options(policy: Mapping[str, Any]) -> dict[str, Any]:
    for key in _PROVIDER_OPTIONS_KEYS:
        value = policy.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _first_mapping(*values: Mapping[str, Any] | None) -> dict[str, Any]:
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _build_fallback_chain(
    *,
    primary_provider: str | None,
    primary_model: str | None,
    active_provider: str | None,
    explicit_fallbacks: Iterable[Any],
    named_policy: Mapping[str, Any],
    route_policy: Mapping[str, Any],
    default_policy: Mapping[str, Any],
    provider_chain: list[Any],
) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    seen: set[tuple[str | None, str | None]] = set()

    def add_entries(values: Any, provider_hint: str | None, source: str) -> None:
        for entry in _iter_fallback_entries(values, provider_hint):
            key = (_clean_str(entry["provider"]), _model_identity_key(entry["model"]))
            if key in seen:
                continue
            seen.add(key)
            if _same_target(entry, primary_provider, primary_model):
                continue
            entry = dict(entry)
            entry["source"] = source
            chain.append(entry)

    add_entries(explicit_fallbacks or [], active_provider, "user_fallback")
    add_entries(_first_present(named_policy, _FALLBACK_KEYS), _clean_str(named_policy.get("provider")) or active_provider, "named_agent")
    add_entries(_first_present(route_policy, _FALLBACK_KEYS), _clean_str(route_policy.get("provider")) or active_provider, "route_category")
    add_entries(_first_present(default_policy, _FALLBACK_KEYS), _clean_str(default_policy.get("provider")) or active_provider, "system_default")

    default_provider = _clean_str(default_policy.get("provider"))
    default_model, _ = parse_model_variant(default_policy.get("model"))
    if default_provider and default_model:
        add_entries([{"provider": default_provider, "model": default_model}], default_provider, "system_default")

    # Provider-chain handling is intentionally conservative: reuse the primary
    # model on alternate providers only when both model and provider are known.
    if primary_model:
        for raw_provider in provider_chain or []:
            chain_provider = _clean_str(raw_provider)
            if not chain_provider or chain_provider == primary_provider:
                continue
            add_entries([{"provider": chain_provider, "model": primary_model}], chain_provider, "provider_chain")

    return [{k: v for k, v in item.items() if k in {"provider", "model", "source"} and v} for item in chain]


def _first_present(policy: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in policy:
            return policy.get(key)
    return []


def _iter_fallback_entries(values: Any, provider_hint: str | None) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for raw in values or []:
        normalized = _normalize_fallback_entry(raw, provider_hint)
        if normalized is not None:
            entries.append(normalized)
    return entries


def _normalize_fallback_entry(value: Any, provider_hint: str | None) -> dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        provider = _clean_str(value.get("provider")) or provider_hint
        model = _clean_str(value.get("model"))
        if not provider or not model:
            return None
        model, _variant = parse_model_variant(model)
        if not model:
            return None
        return {"provider": provider, "model": model}

    text = _clean_str(value)
    if not text:
        return None

    provider = provider_hint
    model_text = text
    if ":" in text:
        maybe_provider, maybe_model = text.split(":", 1)
        # Provider shorthand is accepted only when the prefix is provider-like.
        # Ollama-style tags such as qwen:32b remain same-provider model IDs.
        if maybe_provider and maybe_model and not _looks_like_ollama_tag(maybe_model):
            provider = maybe_provider
            model_text = maybe_model
    model, _variant = parse_model_variant(model_text)
    if not provider or not model:
        return None
    return {"provider": provider, "model": model}


def _looks_like_ollama_tag(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in {"latest", "stable"} or lowered.endswith("b") or lowered.startswith("q")


def _same_target(entry: Mapping[str, str], provider: str | None, model: str | None) -> bool:
    return _clean_str(entry.get("provider")) == _clean_str(provider) and _model_identity_key(entry.get("model")) == _model_identity_key(model)


def _model_identity_key(model: str | None) -> str | None:
    cleaned = _clean_str(model)
    if cleaned is None:
        return None
    # Aggregator slugs often carry vendor/model while same-provider shorthand
    # carries only model. Treat those as the same fallback target for de-dupe.
    if "/" in cleaned:
        return cleaned.rsplit("/", 1)[-1]
    return cleaned


def _is_ultrawork(runtime_mode: str | None) -> bool:
    return _clean_str(runtime_mode) == "ultrawork"


def _policy_trace_value(policy: Mapping[str, Any]) -> dict[str, Any]:
    return _sanitize_mapping({
        "provider": policy.get("provider"),
        "model": policy.get("model"),
        "fallbacks": _first_present(policy, _FALLBACK_KEYS),
        "provider_options": _get_provider_options(policy),
    })


def _trace(stage: str, source: Any, value: Any) -> dict[str, Any]:
    return {"stage": stage, "source": source, "value": _sanitize_value(value)}


def _sanitize_mapping(value: Any) -> dict[str, Any]:
    sanitized = _sanitize_value(_as_mapping(value))
    return sanitized if isinstance(sanitized, dict) else {}


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                continue
            sanitized_item = _sanitize_value(item)
            if _contains_secret_text(sanitized_item):
                continue
            result[str(key)] = sanitized_item
        return result
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value if not _contains_secret_text(item)]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item) for item in value if not _contains_secret_text(item))
    if isinstance(value, str) and _contains_secret_text(value):
        return "[redacted]"
    return value


def _contains_secret_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(fragment in lowered for fragment in _SENSITIVE_SUBSTRINGS)


def _is_sensitive_key(value: Any) -> bool:
    return isinstance(value, str) and any(fragment in value.lower() for fragment in _SENSITIVE_SUBSTRINGS)


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["ResolvedModelPolicy", "parse_model_variant", "resolve_model_policy"]
