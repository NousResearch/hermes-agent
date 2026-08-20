"""Small helpers for composing URLs from configured network hosts."""


def format_url_host(host: str) -> str:
    """Return *host* in HTTP URL-authority form.

    IPv6 literals require brackets when followed by a port. Already-bracketed
    values are preserved so callers can safely pass either representation.
    """
    host = host.strip()
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        return f"[{host}]"
    return host
