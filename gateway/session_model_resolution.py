"""Gateway session model selection, credential acquisition and canonical routing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse

from models.selection import (
    ExplicitAlias,
    ExplicitSelectionError,
    explicit_provider_hint,
    select_explicit_model,
)
from providers import match_configured_provider, normalize_provider
from providers.routing import InvocationRequest, resolve_invocation_route
from utils import base_url_origin

from application_model_aliases import alias_api_key, model_aliases_from_config, provider_reference_context
from application_model_facts import configured_model_matches, provider_facts, static_detection


@dataclass(frozen=True, slots=True)
class SessionModelResolution:
    model: str
    provider: str
    base_url: str
    api_mode: str
    runtime_kind: str
    provider_changed: bool
    api_key: str = ""


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _same_origin(left: str, right: str) -> bool:
    if not left or not right:
        return False
    try:
        return base_url_origin(left) == base_url_origin(right)
    except Exception:
        return False


def _resolve_runtime_credentials(**kwargs) -> dict[str, Any]:
    """Phase 6 seam: acquire credentials/runtime material without owning route semantics."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    return resolve_runtime_provider(**kwargs)


def resolve_session_model(
    *,
    config: Mapping[str, Any],
    raw_model: str,
    explicit_provider: str,
    current_provider: str,
    current_base_url: str,
    current_api_key: str = "",
) -> SessionModelResolution:
    """Resolve one explicit session mutation without a CLI model coordinator."""
    raw = _clean(raw_model)
    current = _clean(current_provider)
    requested = _clean(explicit_provider)
    aliases = model_aliases_from_config(config)
    direct = aliases.get(raw.lower())
    known, named_custom = provider_reference_context(config)
    qualified = (
        explicit_provider_hint(
            raw,
            known_provider_ids=known,
            named_custom_provider_ids=named_custom,
        )
        if not requested
        else ""
    )

    configured = configured_model_matches(raw, config)
    fact_providers = tuple(dict.fromkeys((
        current,
        requested,
        qualified,
        *(ref.provider for ref in configured),
        *(alias.ref.provider for alias in aliases.values()),
    )))
    facts = provider_facts(fact_providers, raw, config)
    try:
        selection = select_explicit_model(
            raw,
            current,
            explicit_provider=requested,
            provider_facts=facts,
            direct_aliases=tuple(
                ExplicitAlias(alias.name, alias.ref) for alias in aliases.values()
            ),
            configured_matches=configured,
            known_provider_ids=known,
            named_custom_provider_ids=named_custom,
            detection=static_detection(raw, current),
            hold_current_provider=(
                normalize_provider(current) == "custom"
                or current.startswith("custom:")
                or (urlparse(current_base_url).hostname or "") in {"localhost", "127.0.0.1"}
            ),
        )
    except (ExplicitSelectionError, ValueError) as exc:
        raise ValueError("model_resolution_failed") from exc
    if selection.selected is None:
        raise ValueError("model_resolution_failed")

    ref = selection.selected.ref
    target_provider = ref.provider
    route_request_provider = target_provider
    explicit_base_url = ""
    explicit_api_key = ""

    if direct is not None and selection.matched_alias:
        route_request_provider = (
            requested
            if requested
            else "custom"
            if direct.base_url
            else direct.ref.provider
        )
        explicit_base_url = direct.base_url
        explicit_api_key = alias_api_key(direct)

    configured_target = match_configured_provider(
        target_provider,
        providers=(
            config.get("providers")
            if isinstance(config.get("providers"), Mapping)
            else None
        ),
        custom_providers=(
            config.get("custom_providers")
            if isinstance(config.get("custom_providers"), list)
            else None
        ),
    )
    if configured_target is not None:
        route_request_provider = configured_target.provider_key or configured_target.identity
        explicit_base_url = explicit_base_url or configured_target.base_url

    same_provider = normalize_provider(target_provider) == normalize_provider(current)
    if same_provider and not requested and not explicit_base_url:
        runtime = {
            "provider": target_provider,
            "base_url": current_base_url,
            "api_key": current_api_key,
            "api_mode": "",
            "runtime_kind": "",
        }
    elif (
        direct is not None
        and explicit_base_url
        and not explicit_api_key
        and _same_origin(current_base_url, explicit_base_url)
    ):
        runtime = {
            "provider": target_provider,
            "base_url": explicit_base_url,
            "api_key": current_api_key,
            "api_mode": "",
            "runtime_kind": "",
        }
    else:
        runtime = _resolve_runtime_credentials(
            requested=route_request_provider,
            explicit_api_key=explicit_api_key or None,
            explicit_base_url=explicit_base_url or None,
            target_model=ref.model,
        )

    route = resolve_invocation_route(InvocationRequest(
        provider=_clean(runtime.get("provider")) or target_provider,
        model=ref.model,
        base_url=_clean(runtime.get("base_url")) or explicit_base_url or current_base_url,
        configured_api_mode=_clean(runtime.get("api_mode")),
        openai_runtime=(
            "codex_app_server" if runtime.get("runtime_kind") == "app_server" else ""
        ),
        requested_provider=route_request_provider or target_provider,
    ))
    return SessionModelResolution(
        model=route.model,
        provider=route.provider,
        base_url=route.base_url,
        api_mode=route.api_mode,
        runtime_kind=route.runtime_kind,
        provider_changed=route.provider != normalize_provider(current),
        api_key=_clean(runtime.get("api_key")),
    )


__all__ = ["SessionModelResolution", "resolve_session_model"]
