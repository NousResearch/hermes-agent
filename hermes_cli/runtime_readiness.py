"""Provider readiness shared by authenticated frontend transports."""


def check_runtime_readiness(requested=None, *, strict_profile_scope=False):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.auth import has_usable_secret
    from hermes_cli.main import _has_any_provider_configured

    runtime = resolve_runtime_provider(requested=requested)
    configured = bool(_has_any_provider_configured(strict_profile_scope=strict_profile_scope))
    provider = runtime.get('provider') or 'provider'
    source = str(runtime.get('source') or '')
    result = {'ok': True, 'provider': runtime.get('provider'),
              'model': runtime.get('model'), 'source': runtime.get('source')}
    if not configured and provider == 'bedrock' and source in {'iam-role', 'aws-sdk-default-chain'}:
        return {**result, 'ok': False, 'error': 'No Hermes provider is configured.'}
    api_key = runtime.get('api_key')
    api_key_text = '' if callable(api_key) else str(api_key or '').strip()
    if not (callable(api_key) or api_key_text in {'aws-sdk', 'no-key-required'}
            or has_usable_secret(api_key_text) or bool(runtime.get('command'))):
        return {**result, 'ok': False, 'error': f'No usable credentials found for {provider}.'}
    from hermes_cli.anon_auth import route_is_welcome_host
    result['free_tier'] = provider == 'nous' and route_is_welcome_host(runtime.get('base_url'))
    return result
