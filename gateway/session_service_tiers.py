"""Speed choices for one resolved model route and its owning account."""


def available_service_tiers(model, *, provider=None, base_url=None, access_token=None, include_ultrafast=True):
    from agent.codex_service_tiers import ultrafast_offered
    from hermes_cli.models import resolve_fast_mode_overrides
    if not model or resolve_fast_mode_overrides(model, provider=provider, base_url=base_url) is None:
        return []
    options = ['normal', 'fast', 'auto', 'cold']
    if include_ultrafast and ultrafast_offered(model, provider=provider, base_url=base_url,
                                               access_token=access_token):
        options.append('ultrafast')
    return options
