"""Endpoint-scoped pool instances for pinned session routes."""

from hermes_cli.route_identity import normalize_route_base_url


def entry_endpoint(entry) -> str:
    if entry.provider == "openai-codex":
        from hermes_cli.auth_codex import _codex_pool_route_base_url
        base_url = _codex_pool_route_base_url(entry.base_url)
    else:
        base_url = entry.runtime_base_url
    return normalize_route_base_url(base_url)


def restrict_pool_to_endpoint(pool, base_url):
    """Narrow a freshly loaded pool before selection, refresh or rotation.

    Keep the loaded pool's persistence/ownership metadata. Its writes merge by row id, so omitting
    foreign rows from this instance never deletes or modifies those rows in the owning store.
    """
    if base_url:
        endpoint = normalize_route_base_url(base_url)
        pool._entries = [entry for entry in pool.entries() if entry_endpoint(entry) == endpoint]
    return pool
