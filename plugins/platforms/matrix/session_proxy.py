"""The proxy a Matrix ``ClientSession`` actually uses, for error redaction (kept out of the adapter facade)."""

from hermes_agent.gateway.platforms.base import env_proxy_for_redaction, gateway_trust_env


def _socks_connector_missing() -> bool:
    try:
        import aiohttp_socks
    except ImportError:
        return True
    return False


def matrix_redaction_proxy(proxy_url: str | None, homeserver: str | None) -> str | None:
    """Proxy to hand ``describe_proxy_error`` for an error from the session ``_create_matrix_session``
    built: *proxy_url* when the session dials it, else the environment the ``trust_env`` session
    reads — no proxy configured, or a SOCKS one ignored because ``aiohttp_socks`` is missing."""
    socks = bool(proxy_url) and proxy_url.split("://")[0].lower().startswith("socks")
    if proxy_url and not (socks and _socks_connector_missing()):
        return proxy_url
    return env_proxy_for_redaction(homeserver, trust_env=gateway_trust_env())
