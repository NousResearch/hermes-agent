"""One resolved-route identity for launch-route gates.

Two launch routes are the same route when the runtime resolver would send them to the same
endpoint: compare what it resolves (canonical provider, configured named entry, effective
URL), never raw spellings. ``ollama``/``vllm`` resolve to ``custom``, ``deep-infra`` to
``deepinfra``, and a configured ``providers:``/``custom_providers`` name (or its
``custom:<name>`` spelling) to that entry.
"""


# Every ``model:`` key that belongs to one endpoint; a route change drops them all, so none of the
# replaced endpoint's values is paired with the new one:
# - route itself (provider, base_url): which endpoint the turn goes to;
# - wire protocol (api_mode): how that endpoint speaks; another host may speak another protocol;
# - window (context_length): the replaced model's context size, wrong for the new one;
# - credentials (api_key, api, api_key_from, key_env, api_key_env): a key authenticates only the
#   endpoint that issued it, and sending it anywhere else leaks it.
ROUTE_BOUND_KEYS = ('provider', 'base_url', 'api_mode', 'context_length',
                    'api_key', 'api', 'api_key_from', 'key_env', 'api_key_env')


def _named_entry(name: str, config: dict) -> dict | None:
    from hermes_cli.runtime_provider_custom import _get_named_custom_provider
    return _get_named_custom_provider(name, config=config, resolve_credentials=False)


def _canonical(name: str) -> str:
    from hermes_cli.models import normalize_provider
    canonical = normalize_provider(name)
    from hermes_cli.auth import AuthError, resolve_provider
    try:
        return resolve_provider(canonical)
    except AuthError:  # not a registry provider: its canonical spelling is its identity
        return canonical


def configured_entry_aliases(identity: str, config: dict) -> frozenset[str]:
    """Every spelling the configured entry behind a route identity accepts, empty when it names none.

    Raw provider resolution defers a bare name that is a canonical built-in to the built-in, so a
    ``providers:`` entry named ``deepseek`` is reachable only through its ``custom:`` spelling —
    the entry's own alias set is what tells its keys apart from the built-in's head-on name.
    """
    entry = _named_entry(str(identity or '').strip().lower(), config)
    if not entry:
        return frozenset()
    from hermes_cli.providers import custom_provider_aliases
    return custom_provider_aliases(str(entry.get('name') or ''), str(entry.get('provider_key') or ''))


def profile_route_identity(config: dict) -> str:
    """The identity the profile's own ``model`` block resolves to — ``custom`` for an endpoint with
    no provider named (or ``auto``), which already routes to ``custom`` at that URL."""
    held = dict(config.get('model') or {}) if isinstance(config.get('model'), dict) else {}
    if str(held.get('provider') or '').strip().lower() in ('', 'auto') and str(held.get('base_url') or '').strip():
        held['provider'] = 'custom'
    return resolved_route(held, config)[0]


def resolved_route(model_cfg: dict, config: dict) -> tuple[str, str]:
    """``(provider_identity, effective_base_url)`` the runtime resolves ``model_cfg`` to."""
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
    config = config if isinstance(config, dict) else {}
    name = str(model_cfg.get('provider') or '').strip().lower()
    if name in ('', 'auto'):
        return 'auto', ''
    base_url = str(model_cfg.get('base_url') or '').strip()
    entry = _named_entry(name, config)
    if entry:
        from hermes_cli.providers import custom_provider_slug
        identity = custom_provider_slug(str(entry.get('name') or ''), str(entry.get('provider_key') or ''))
        base_url = base_url or str(entry.get('base_url') or '').strip()
    else:
        identity = _canonical(name)
    return identity, base_url.rstrip('/')


def same_route(a_cfg: dict, b_cfg: dict, config: dict) -> bool:
    """True when both configs resolve to one endpoint; ``auto`` on either side keeps the route."""
    a_id, a_url = resolved_route(a_cfg, config)
    b_id, b_url = resolved_route(b_cfg, config)
    if 'auto' in (a_id, b_id):
        return True
    if a_url or b_url:
        from hermes_cli.route_identity import normalize_route_base_url
        return normalize_route_base_url(a_url) == normalize_route_base_url(b_url)
    return a_id == b_id
