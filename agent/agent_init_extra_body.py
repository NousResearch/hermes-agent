"""Custom-provider ``extra_body`` scoping and ownership.

Moved out of ``agent/agent_init.py`` (file-line ratchet): the helpers that decide which
``extra_body`` values a custom/LM Studio provider entry contributes, and which values in a
live ``request_overrides["extra_body"]`` are config-owned (the ownership ledger) rather than
caller-owned. ``agent.agent_init`` re-exports the names tests and lazy imports expect.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional


def _normalized_custom_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def _custom_provider_model_matches(agent_model: str, entry: dict[str, Any]) -> bool:
    agent_model_norm = str(agent_model or "").strip().lower()
    # Multi-model entries (`providers.<name>.models` mapping / legacy `models:` list):
    # matching ANY catalog entry counts, else a provider whose `model` differs from the
    # session model drops its extra_body (e.g. OpenAI service_tier) → wrong billing tier.
    models = entry.get("models")
    catalog = [str(m).strip().lower() for m in models] if isinstance(models, (dict, list, tuple)) else []
    if catalog and agent_model_norm in catalog:
        return True
    provider_model = str(entry.get("model", "") or "").strip().lower()
    return (not provider_model and not catalog) or provider_model == agent_model_norm


def _custom_provider_extra_body_for_agent(
    *, provider: str, model: str, base_url: str, custom_providers: list[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    provider_norm = (provider or "").strip().lower()
    if provider_norm not in {"custom", "lmstudio"} and not provider_norm.startswith("custom:"):
        return None
    provider_key_filter = provider_norm.partition(":")[2].strip()
    target_url = _normalized_custom_base_url(base_url)
    if not target_url:
        return None

    fallback: Optional[dict[str, Any]] = None
    for entry in custom_providers or []:
        if not isinstance(entry, dict):
            continue
        entry_keys = {str(entry.get(key, "") or "").strip().lower() for key in ("provider_key", "name")}
        if provider_key_filter and provider_key_filter not in entry_keys:
            continue
        if _normalized_custom_base_url(entry.get("base_url")) != target_url:
            continue
        extra_body = entry.get("extra_body")
        if not isinstance(extra_body, dict) or not extra_body:
            continue
        explicit_local = str(entry.get("provider", "") or "").strip().lower() == "lmstudio"
        if provider_norm == "lmstudio" and not explicit_local:
            continue
        if explicit_local:
            # The registration's scope holds even through custom/named-custom aliases.
            pinned_model = entry.get("model")
            if not isinstance(pinned_model, str) or not pinned_model.strip() or pinned_model == "*" or pinned_model != model:
                continue
        if str(entry.get("model", "") or "").strip():
            if _custom_provider_model_matches(model, entry):
                return dict(extra_body)
        elif fallback is None:
            fallback = dict(extra_body)
    return fallback


def _config_defines_extra_body(node: Any) -> bool:
    """True when a non-empty ``extra_body`` mapping appears anywhere in a loaded config."""
    if isinstance(node, dict):
        return any((key == "extra_body" and isinstance(value, dict) and bool(value))
                   or _config_defines_extra_body(value) for key, value in node.items())
    return isinstance(node, list) and any(_config_defines_extra_body(item) for item in node)


def _record_custom_provider_extra_body(agent, extra_body: Any) -> None:
    """Ownership ledger: add what a config channel just placed in ``request_overrides["extra_body"]``.
    Deep copies, so mutating the live body can never rewrite its recorded provenance. An agent without
    a ledger has no provenance for the body it already holds either, so all of it is recorded."""
    ledger = getattr(agent, "_custom_provider_injected_extra_body", None)
    if not isinstance(ledger, dict):
        ledger = (getattr(agent, "request_overrides", None) or {}).get("extra_body")
        ledger = deepcopy(ledger) if isinstance(ledger, dict) else {}
    agent._custom_provider_injected_extra_body = {**ledger, **(deepcopy(extra_body) if isinstance(extra_body, dict) else {})}


def _record_constructor_extra_body(agent, config: Any) -> None:
    """Start the ledger. Every Hermes construction path (CLI, TUI, gateway, cron, ACP, delegation)
    passes runtime-resolved config as ``request_overrides``, indistinguishable from a caller's: while
    config defines any extra_body, or could not be read, that body is config-owned (fail closed).
    Only without any config extra_body is it the caller's, as upstream treats it."""
    agent._custom_provider_injected_extra_body = {}
    extra_body = (getattr(agent, "request_overrides", None) or {}).get("extra_body")
    if extra_body and (not isinstance(config, dict) or not config or _config_defines_extra_body(config)):
        _record_custom_provider_extra_body(agent, extra_body)


def _is_config_owned(value: Any, owned: Any) -> bool:
    """``value`` is, or partly is, config-placed ``owned``: equal, or a dict/list sharing any part of
    it, at any depth. A partly config-owned value has ambiguous provenance and counts as config-owned."""
    if value == owned:
        return True
    if isinstance(value, dict) and isinstance(owned, dict):
        return any(key in owned and _is_config_owned(item, owned[key]) for key, item in value.items())
    if isinstance(value, (list, tuple)) and isinstance(owned, (list, tuple)):
        return any(_is_config_owned(item, part) for item in value for part in owned)
    return False


def _caller_owned_extra_body(agent, *route_bodies: Any) -> dict[str, Any]:
    """The positively caller-owned part of ``agent``'s extra_body: values neither recorded in the
    ownership ledger nor set by one of ``route_bodies`` (the route being left; upstream's rule).
    Without a ledger nothing records where the body came from, so none of it is the caller's."""
    extra_body = (getattr(agent, "request_overrides", None) or {}).get("extra_body")
    ledger = getattr(agent, "_custom_provider_injected_extra_body", None)
    if not isinstance(extra_body, dict) or not isinstance(ledger, dict):
        return {}
    owned = [ledger, *(body for body in route_bodies if isinstance(body, dict))]
    return {key: deepcopy(value) for key, value in extra_body.items()
            if not any(key in body and _is_config_owned(value, body[key]) for body in owned)}


def _drop_config_owned_extra_body(agent, *route_bodies: Any) -> None:
    """Fail closed at a route boundary: only positively caller-owned extra_body crosses it."""
    overrides = dict(getattr(agent, "request_overrides", None) or {})
    if "extra_body" not in overrides:
        return
    kept = _caller_owned_extra_body(agent, *route_bodies)
    if kept:
        overrides["extra_body"] = kept
    else:
        overrides.pop("extra_body", None)
    agent.request_overrides = overrides


def _merge_custom_provider_extra_body(agent, custom_providers: list[dict[str, Any]]) -> None:
    extra_body = _custom_provider_extra_body_for_agent(
        provider=agent.provider, model=agent.model, base_url=agent.base_url,
        custom_providers=custom_providers,
    )
    if not extra_body:
        return
    overrides = dict(getattr(agent, "request_overrides", {}) or {})
    merged_extra_body = deepcopy(extra_body)
    existing_extra_body = overrides.get("extra_body")
    if isinstance(existing_extra_body, dict):
        merged_extra_body.update(existing_extra_body)
    overrides["extra_body"] = merged_extra_body
    agent.request_overrides = overrides
    # Existing values won the merge; the ledger records only what this merge placed.
    kept_keys = existing_extra_body.keys() if isinstance(existing_extra_body, dict) else ()
    _record_custom_provider_extra_body(agent, {key: value for key, value in extra_body.items() if key not in kept_keys})
