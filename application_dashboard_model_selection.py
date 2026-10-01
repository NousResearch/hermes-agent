"""Dashboard main-model selection: canonical selection and provider routing.

Only credential acquisition, config projection and model admission are application
operations. The model and route decisions belong to models/ and providers/.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from application_model_aliases import model_aliases_from_config, provider_reference_context
from application_model_facts import configured_model_matches, provider_facts, static_detection
from application_model_switch_enrichment import validate_model_switch
from models.selection import ExplicitAlias, select_explicit_model
from providers import get_provider_profile, match_configured_provider, normalize_provider
from providers.routing import InvocationRequest, resolve_invocation_route


@dataclass(frozen=True, slots=True)
class DashboardModelSelection:
    new_model: str
    target_provider: str
    base_url: str = ""
    api_mode: str = ""
    api_key: str = ""


def _acquire(**kwargs) -> dict[str, Any]:
    """Phase 6 only: acquire runtime secrets; never accept its model/route as authoritative."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    return resolve_runtime_provider(**kwargs)


def select_dashboard_main_model(
    *, config: Mapping[str, Any], provider: str, model: str,
    base_url: str = "", api_key: str = "",
) -> DashboardModelSelection:
    """Resolve a dashboard selection, without mutating profile config or process globals."""
    wanted = str(provider or "").strip()
    raw = str(model or "").strip()
    if not wanted or not raw:
        raise ValueError("provider and model required for main")
    current_cfg = config.get("model") if isinstance(config.get("model"), Mapping) else {}
    current = str(current_cfg.get("provider") or "").strip() or wanted
    known, named = provider_reference_context(config)
    aliases = model_aliases_from_config(config)
    matches = configured_model_matches(raw, config)
    chosen = select_explicit_model(
        raw, current, explicit_provider=wanted,
        provider_facts=provider_facts(tuple(dict.fromkeys((
            current, wanted, *(match.provider for match in matches),
            *(alias.ref.provider for alias in aliases.values()),
        ))), raw, config),
        direct_aliases=tuple(ExplicitAlias(alias.name, alias.ref) for alias in aliases.values()),
        configured_matches=matches, known_provider_ids=known,
        named_custom_provider_ids=named, detection=static_detection(raw, current),
    )
    if chosen.selected is None:
        raise ValueError(f"Could not resolve model '{raw}'")
    selected = chosen.selected.ref
    configured = match_configured_provider(
        wanted, providers=config.get("providers"), custom_providers=config.get("custom_providers"),
    )
    canonical = normalize_provider(wanted)
    bare_custom = canonical in {"custom", "local"} or wanted.lower() in {"custom", "local"}
    if configured is None and get_provider_profile(wanted) is None and not bare_custom:
        raise ValueError(f"Unknown provider '{wanted}'")
    if wanted.lower().startswith("custom:") and configured is None:
        raise ValueError(f"Unconfigured provider '{wanted}'")
    # providers: keys intentionally retain their configured bare spelling;
    # legacy named custom entries retain their durable custom:<name> identity.
    target = (
        configured.provider_key if configured is not None and configured.source == "providers"
        and not wanted.lower().startswith("custom:") else configured.identity
        if configured is not None else "custom" if bare_custom else selected.provider
    )
    direct = aliases.get(raw.lower())
    alias_url = direct.base_url if direct is not None and (
        chosen.matched_alias or selected.model == direct.ref.model
    ) else ""
    submitted_url = str(base_url or "").strip()
    declared_url = configured.base_url if configured is not None else ""
    explicit_url = submitted_url or alias_url or declared_url
    # The submitted custom endpoint is the route being selected, never a stale
    # CUSTOM_BASE_URL/model.base_url inherited from a previous selection.
    if bare_custom and (submitted_url or alias_url):
        # The user-selected endpoint must never inherit a key belonging to
        # another arbitrary custom host. Only the submitted inline key is used.
        runtime = {"provider": target, "base_url": explicit_url, "api_key": api_key}
    else:
        from hermes_cli.auth_constants import AuthError

        try:
            runtime = _acquire(
                requested=(configured.provider_key or configured.identity if configured else target),
                target_model=selected.model, explicit_base_url=explicit_url or None,
                explicit_api_key=(api_key or None) if bare_custom else None,
            )
        except AuthError as exc:
            # Bad user/provider authentication is a rejected assignment (400),
            # not an internal dashboard failure. Scoped-secret violations and
            # unexpected runtime errors retain their original failure status.
            raise ValueError(str(exc)) from exc
    route = resolve_invocation_route(InvocationRequest(
        provider=target, requested_provider=wanted, configured_provider=target,
        model=selected.model,
        base_url=(submitted_url or alias_url) if (submitted_url or alias_url) else (
            str(runtime.get("base_url") or "").strip() or explicit_url
        ),
        configured_api_mode=(
            configured.api_mode if configured is not None and configured.api_mode
            else str(runtime.get("api_mode") or "").strip()
        ),
        openai_runtime="codex_app_server" if runtime.get("runtime_kind") == "app_server" else "",
    ))
    result = DashboardModelSelection(
        new_model=route.model, target_provider=target,
        base_url=route.base_url, api_mode=route.api_mode,
        api_key=str(runtime.get("api_key") or ""),
    )
    validate_model_switch(result, config)
    return result


__all__ = ["DashboardModelSelection", "select_dashboard_main_model"]
