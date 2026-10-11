"""Provider readiness probe for the session authority's ``setup.runtime_check``.

Answers with the legacy ``tui_gateway/methods_config.py`` handler's payload so a client reads one
contract whichever backend it reached: ``ok``/``provider``/``model``/``source``, ``error`` on
failure (naming the configured pin, never the fallback chain's tail), ``free_tier_route`` on success.
"""


def check_runtime_readiness(requested=None, *, strict_profile_scope=False, resolve=None):
    """ok/error verdict for the runtime a session would be built with.

    Without ``requested`` the probe runs the SAME resolver as session creation (the configured
    model + provider pin, then the configured fallback chain) — a probe that ignores the chain
    shows onboarding for a backend whose sessions build fine (#111775). ``resolve`` replaces that
    resolver with the caller's own ``() -> (model, runtime)``. An explicit ``requested`` provider
    stays a strict single-provider check so onboarding can verify the provider just connected
    without another provider's fallback masking a failed connection. Both branches report the model.
    """
    from hermes_cli.auth import has_usable_secret
    from hermes_cli.config import load_config
    from hermes_cli.main import _has_any_provider_configured
    from hermes_cli.runtime_provider import resolve_runtime_provider, resolve_runtime_with_fallback

    config = load_config()
    cfg_model = config.get('model')
    if isinstance(cfg_model, dict):
        configured_model = str(cfg_model.get('default') or cfg_model.get('model') or '') or None
    else:
        configured_model = cfg_model if isinstance(cfg_model, str) and cfg_model else None
    if resolve is not None:
        model, runtime = resolve()
    elif requested:
        model = configured_model
        runtime = resolve_runtime_provider(requested=requested, target_model=configured_model)
    else:
        runtime, entry = resolve_runtime_with_fallback(config, target_model=configured_model)
        model = (entry or {}).get('model') or configured_model or runtime.get('model')
    configured = bool(_has_any_provider_configured(strict_profile_scope=strict_profile_scope))
    provider = runtime.get('provider') or 'provider'
    source = str(runtime.get('source') or '')
    # When the chain only resolves at its tail the runtime stops there; a failure must name the
    # provider the user pinned, not a tail they never chose (#124939). Auto mode names the route.
    pinned = requested or (str(cfg_model.get('provider') or '').strip() if isinstance(cfg_model, dict) else '')
    blamed = pinned or provider

    def fail(error, src):
        return {'ok': False, 'provider': blamed, 'model': model, 'source': src, 'error': error}

    if not configured and provider == 'bedrock' and source in {'iam-role', 'aws-sdk-default-chain'}:
        return fail('No Hermes provider is configured.', source)
    api_key = runtime.get('api_key')
    api_key_text = '' if callable(api_key) else str(api_key or '').strip()
    if not (callable(api_key) or api_key_text in {'aws-sdk', 'no-key-required'}
            or has_usable_secret(api_key_text) or bool(runtime.get('command'))):
        return fail(f'No usable credentials found for {blamed}.', runtime.get('source'))
    from hermes_cli.anon_auth import route_is_welcome_host
    # free_tier_route is keyed on the SELECTED route (the welcome host serves only nous/welcome), not
    # on profile state: a paid Nous key beside a free-tier identity must not read as free.
    return {'ok': True, 'provider': runtime.get('provider'), 'model': model, 'source': runtime.get('source'),
            'free_tier_route': provider == 'nous' and route_is_welcome_host(runtime.get('base_url'))}
