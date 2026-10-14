"""Native OAuth session override classification and next-turn credential resolution.

The turn resolver retains model and channel precedence; this phase owns only
refreshing the override's credential and preserving its existing failure behavior.
"""

from __future__ import annotations


_NATIVE_OAUTH_OVERRIDE_PROVIDERS = frozenset({"openai-codex", "xai-oauth"})


def _native_oauth_override_uses_owned_route(override: dict) -> bool:
    """Whether an override is a native OAuth route whose bearer may follow its live credential store.

    A provider label alone is insufficient: an explicit key on a proxy/custom endpoint remains a
    static session credential even when it is labelled ``openai-codex`` or ``xai-oauth``.
    """
    provider = str(override.get("provider") or "").strip().lower()
    if provider not in _NATIVE_OAUTH_OVERRIDE_PROVIDERS:
        return False
    from hermes_cli.providers import get_provider
    from hermes_cli.route_identity import normalize_route_base_url
    definition = get_provider(provider, allow_network=False)
    # User-configured providers may own proxy URLs; that is not evidence that the route is native.
    return bool(definition and definition.base_url and
                normalize_route_base_url(override.get("base_url")) ==
                normalize_route_base_url(definition.base_url))


def _adopt_live_native_oauth_credential(override: dict, runtime: dict) -> None:
    """Replace only the bearer/pool after proving native route continuity.

    Raises rather than replaying the snapshot or falling onto the global provider when continuity
    cannot be proven. The caller deliberately leaves model and all non-credential override fields intact.
    """
    from hermes_cli.route_identity import normalize_route_base_url

    provider = str(override.get("provider") or "").strip().lower()
    live_provider = str(runtime.get("provider") or "").strip().lower()
    requested = str(override.get("requested_provider") or provider).strip().lower()
    live_requested = str(runtime.get("requested_provider") or live_provider).strip().lower()
    old_base = normalize_route_base_url(override.get("base_url"))
    live_base = normalize_route_base_url(runtime.get("base_url"))
    if live_provider != provider or live_requested != requested or not old_base or live_base != old_base:
        raise RuntimeError("Session /model OAuth credential resolved to a different route identity")

    live_key = str(runtime.get("api_key") or "").strip()
    if not live_key:
        raise RuntimeError("Session /model OAuth credential resolution returned no access token")

    # /model pins a model/provider, not an account. The profile resolver owns pool selection,
    # including a deliberate account change; pass its exact bearer and pool to agent construction.
    override["api_key"] = live_key
    override["credential_pool"] = runtime.get("credential_pool")


def _resolve_session_override_runtime(
    override: dict | None, *, refresh_native_oauth: bool,
) -> tuple[dict | None, dict | None, dict | None]:
    """Resolve the override provider, retaining default fallback for non-native routes."""
    from gateway.run import _resolve_runtime_agent_kwargs_for_provider
    from gateway.run_turn import logger

    runtime_kwargs, unavailable_override = None, None
    if override and override.get("provider"):
        try:
            runtime_kwargs = _resolve_runtime_agent_kwargs_for_provider(
                override["provider"], target_model=override.get("model") or None)
        except Exception as exc:
            if refresh_native_oauth:
                raise RuntimeError(
                    f"Session /model OAuth credentials for {override['provider']} are unavailable"
                ) from exc
            # Layering the override on the default runtime sent its model to the default provider's
            # endpoint (openai-codex on the Nous URL). Run this turn on the whole default route and say
            # so; the persisted override is kept, so the next turn retries it.
            logger.warning("Session /model override provider %s unavailable: %s", override["provider"], exc)
            unavailable_override, override = override, None
        if runtime_kwargs is not None and refresh_native_oauth:
            _adopt_live_native_oauth_credential(override, runtime_kwargs)
    return runtime_kwargs, unavailable_override, override
