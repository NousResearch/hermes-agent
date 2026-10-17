"""Profile-scoped platform proxy overrides and safe proxy logging."""

from urllib.parse import urlsplit


def resolve_platform_proxy_env(platform_env_var: str | None) -> str:
    from gateway.platforms._shared import get_scoped_secret

    value = ""
    if platform_env_var:
        # Preserve adapters' lowercase proxy variables without borrowing another
        # profile's values from the process environment under multiplexing.
        for key in dict.fromkeys((platform_env_var, platform_env_var.lower())):
            value = (get_scoped_secret(key, "") or "").strip()
            if value:
                break
    return value


def redact_proxy_url(url: str) -> str:
    """Return *url* with any userinfo password masked, safe for logging.

    Proxy URLs from env vars may embed credentials
    (``http://user:pass@host:port``) which must not leak into logs.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.password is None:
        return url
    host = parts.hostname or ""
    netloc = f"{parts.username or ''}:***@{host}"
    if parts.port is not None:
        netloc += f":{parts.port}"
    return parts._replace(netloc=netloc).geturl()
