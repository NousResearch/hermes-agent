"""Small helpers for composing URLs from configured network hosts."""


def format_url_host(host: str) -> str:
    """Return *host* in HTTP URL-authority form.

    IPv6 literals require brackets when followed by a port. Already-bracketed
    values are preserved so callers can safely pass either representation.
    """
    host = host.strip()
    if host.startswith("[") and host.endswith("]"):
        inner_host = host[1:-1]
        if ":" in inner_host:
            return f"[{inner_host.replace('%', '%25')}]"
        return host
    if ":" in host:
        return f"[{host.replace('%', '%25')}]"
    return host
