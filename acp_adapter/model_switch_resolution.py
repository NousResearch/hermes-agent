"""ACP-owned selection/credential transaction over canonical model and provider domains."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

from application_model_aliases import alias_api_key, model_aliases_from_config, provider_reference_context
from application_model_facts import configured_model_matches, provider_facts, static_detection
from application_model_switch_enrichment import validate_model_switch
from models.identity import parse_model_ref
from models.selection import ExplicitAlias, explicit_provider_hint, select_explicit_model
from providers import get_provider_profile, match_configured_provider, normalize_provider
from providers.routing import InvocationRequest, resolve_invocation_route


@dataclass(frozen=True, slots=True)
class AcpModelRoute:
    model: str
    provider: str
    base_url: str
    api_mode: str
    runtime: dict[str, Any]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _same_endpoint(left: str, right: str) -> bool:
    """Only an identical endpoint may reuse the live credential for an alias."""
    try:
        a, b = urlsplit(left), urlsplit(right)
        return bool(a.hostname and b.hostname and not any((
            a.username, a.password, b.username, b.password,
        )) and (
            a.scheme.lower(), a.hostname.lower(), a.port, a.path.rstrip("/"), a.query,
        ) == (
            b.scheme.lower(), b.hostname.lower(), b.port, b.path.rstrip("/"), b.query,
        ))
    except ValueError:
        return False


def _acquire(**kwargs) -> dict[str, Any]:
    """Phase 6 credential acquisition, never an alternate model/route authority."""
    from hermes_cli.runtime_provider import resolve_runtime_provider
    return resolve_runtime_provider(**kwargs)


def resolve_acp_model_switch(
    *, config: Mapping[str, Any], raw_model: str, current_provider: str,
    current_model: str, current_base_url: str = "", current_api_key: Any = "",
    current_runtime: Mapping[str, Any] | None = None,
    keep_endpoint: bool = False,
) -> AcpModelRoute:
    """Validate one ACP picker choice; do not mutate a session or global config."""
    raw, current = _clean(raw_model), _clean(current_provider) or "openrouter"
    if not raw:
        raise ValueError("modelId is required")
    aliases = model_aliases_from_config(config)
    known, named = provider_reference_context(config)
    hinted = explicit_provider_hint(
        raw, known_provider_ids=known, named_custom_provider_ids=named,
    )
    configured_matches = configured_model_matches(raw, config)
    facts = provider_facts(tuple(dict.fromkeys((
        current, hinted, *(ref.provider for ref in configured_matches),
        *(alias.ref.provider for alias in aliases.values()),
    ))), raw, config)
    selection = select_explicit_model(
        raw, current, provider_facts=facts,
        direct_aliases=tuple(ExplicitAlias(alias.name, alias.ref) for alias in aliases.values()),
        configured_matches=configured_matches, known_provider_ids=known,
        named_custom_provider_ids=named, detection=static_detection(raw, current),
        hold_current_provider=normalize_provider(current) == "custom"
                              or current.startswith("custom:"),
    )
    if selection.selected is None:
        raise ValueError(f"Could not resolve model '{raw}'")
    ref = selection.selected.ref
    provider = ref.provider
    alias_name = (
        parse_model_ref(
            raw, "", known_provider_ids=known, named_custom_provider_ids=named,
        ).model if hinted else raw
    ).lower()
    direct = aliases.get(alias_name)
    if direct is not None and not (
        selection.matched_alias or (hinted and ref.model == direct.ref.model)
    ):
        direct = None
    alias_base, alias_key = "", ""
    if direct is not None:
        alias_base = direct.base_url
        # Explicitly qualified providers must acquire their OWN credentials.
        alias_key = alias_api_key(direct) if not hinted else ""
        if alias_base and not hinted:
            provider = "custom"

    configured = match_configured_provider(
        provider, providers=config.get("providers"), custom_providers=config.get("custom_providers"),
    )
    request_provider = configured.provider_key or configured.identity if configured else provider
    effective_provider = configured.identity if configured else provider
    if configured is not None:
        alias_base = alias_base or configured.base_url
    if configured is None and get_provider_profile(request_provider) is None:
        raise ValueError(f"Unknown provider '{request_provider}'")
    if provider.startswith("custom:") and configured is None and not alias_base:
        raise ValueError(f"Unconfigured provider '{provider}'")
    same = normalize_provider(effective_provider) == normalize_provider(current)
    same_route = same and not alias_base and keep_endpoint
    if same_route:
        runtime = {**(current_runtime or {}), "provider": effective_provider,
                   "base_url": current_base_url, "api_key": current_api_key}
    elif alias_base and not alias_key and keep_endpoint and same and _same_endpoint(
        current_base_url, alias_base
    ):
        runtime = {**(current_runtime or {}), "provider": effective_provider,
                   "base_url": alias_base, "api_key": current_api_key}
    else:
        try:
            runtime = _acquire(
                requested=request_provider, target_model=ref.model,
                explicit_base_url=alias_base or None, explicit_api_key=alias_key or None,
            )
        except RuntimeError as exc:
            from agent.secret_scope import UnscopedSecretError
            if isinstance(exc, UnscopedSecretError):
                raise
            message = str(exc).lower()
            if any(term in message for term in ("credential", "api key", "not authenticated", "not logged")):
                raise ValueError(str(exc)) from exc
            raise
    route = resolve_invocation_route(InvocationRequest(
        provider=effective_provider, requested_provider=request_provider,
        configured_provider=effective_provider, model=ref.model,
        base_url=_clean(runtime.get("base_url")) or alias_base or (
            current_base_url if same_route else ""
        ),
        configured_api_mode=(
            _clean(configured.api_mode) if configured is not None
            else _clean(runtime.get("api_mode"))
        ),
        openai_runtime="codex_app_server" if runtime.get("runtime_kind") == "app_server" else "",
    ))
    validated = type("Selection", (), {
        "new_model": route.model, "target_provider": route.provider,
        "base_url": route.base_url, "api_mode": route.api_mode,
        "api_key": runtime.get("api_key") or "",
    })()
    validate_model_switch(validated, config)
    return AcpModelRoute(
        model=route.model, provider=route.provider, base_url=route.base_url,
        api_mode=route.api_mode,
        runtime={**runtime, "provider": route.provider, "base_url": route.base_url,
                 "api_mode": route.api_mode, "runtime_kind": route.runtime_kind},
    )


__all__ = ["AcpModelRoute", "resolve_acp_model_switch"]
