"""Resolve declared routing constraints before agent clients or metadata probes."""


def initial_provider_route(provider, base_url, api_mode):
    from providers import get_provider_profile
    from hermes_cli.providers import is_actual_route

    normalized = provider.strip().lower() if isinstance(provider, str) else ""
    profile = get_provider_profile(normalized) if normalized else None
    if profile is not None:
        if profile.fixed_base_url:
            base_url = profile.base_url
        if profile.fixed_api_mode:
            api_mode = profile.api_mode
    if is_actual_route(provider, base_url):
        from hermes_cli.auth import normalize_actual_base_url
        base_url = normalize_actual_base_url(base_url)
    return base_url, api_mode
