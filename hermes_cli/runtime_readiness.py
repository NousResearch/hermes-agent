"""Provider readiness probe shared by ``setup.runtime_check`` (TUI gateway) and the session authority."""


def check_runtime_readiness(requested=None, *, strict_profile_scope=False, resolve=None):
    """ok/error verdict for the runtime a session would be built with.

    ``resolve`` is an optional ``() -> (model, runtime)`` pair supplied by the caller that owns the
    session builder: without ``provider`` the TUI probe must run the SAME resolver as session creation
    (startup model + provider pin, then the configured fallback chain) — a probe that ignores the chain
    shows onboarding for a backend whose sessions build fine (#111775). An explicit ``requested``
    provider stays a strict single-provider check so onboarding can verify the provider just connected
    without another provider's fallback masking a failed connection. Both branches report the model.
    """
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.auth import has_usable_secret
    from hermes_cli.main import _has_any_provider_configured

    if resolve is not None:
        model, runtime = resolve()
    else:
        runtime = resolve_runtime_provider(requested=requested)
        model = runtime.get('model')
    configured = bool(_has_any_provider_configured(strict_profile_scope=strict_profile_scope))
    provider = runtime.get('provider') or 'provider'
    source = str(runtime.get('source') or '')
    result = {'ok': True, 'provider': runtime.get('provider'),
              'model': model, 'source': runtime.get('source')}
    if not configured and provider == 'bedrock' and source in {'iam-role', 'aws-sdk-default-chain'}:
        return {**result, 'ok': False, 'error': 'No Hermes provider is configured.'}
    api_key = runtime.get('api_key')
    api_key_text = '' if callable(api_key) else str(api_key or '').strip()
    if not (callable(api_key) or api_key_text in {'aws-sdk', 'no-key-required'}
            or has_usable_secret(api_key_text) or bool(runtime.get('command'))):
        return {**result, 'ok': False, 'error': f'No usable credentials found for {provider}.'}
    from hermes_cli.anon_auth import route_is_welcome_host
    # free_tier is keyed on the SELECTED route (the welcome host serves only nous/welcome), not
    # on profile state: a paid Nous key beside a free-tier identity must not read as free.
    result['free_tier'] = provider == 'nous' and route_is_welcome_host(runtime.get('base_url'))
    return result
