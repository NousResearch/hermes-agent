"""Log-safe URL rendering for adapter logs (split from :mod:`gateway.platforms.base`)."""
from urllib.parse import urlsplit

_UNSAFE_URL_PLACEHOLDER = "<unparseable URL redacted>"


def _authority_for_log(raw: str, parsed) -> str | None:
    """``host[:port]`` rebuilt from the parsed fields (never the userinfo), or None when unclean.

    An ``@`` outside the parsed netloc means an unencoded ``/``, ``?`` or ``#`` in the password
    ended the authority early (``http://u:ab/cd@host`` parses as host ``u`` with path
    ``/cd@host``): the "host" is the username and the password sits in the path or query.
    """
    if raw.count("@") != parsed.netloc.count("@") or not parsed.hostname:
        return None
    # ``http://bob:SEKRET`` (userinfo, no host) parses as host ``bob`` with port ``SEKRET``:
    # anything after the colon that is not a valid port may be a password.
    try:
        port = parsed.port
    except ValueError:
        return None
    host = parsed.hostname
    host = f"[{host}]" if ":" in host else host
    return f"{host}:{port}" if port is not None else host


def safe_url_for_log(url: str, max_len: int = 80) -> str:
    """Return a URL string safe for logs (no query/fragment/userinfo)."""
    raw = str(url) if max_len > 0 and url is not None else ""
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        parsed = None
    # Without a parsed authority the userinfo can't be cut out reliably (``http://u:p@[::1``,
    # ``u:p@proxy:8080``, ``http:/u:p@proxy``): never echo text that may still hold credentials.
    if parsed is None or not (parsed.scheme and parsed.netloc) and "@" in raw:
        return _UNSAFE_URL_PLACEHOLDER[:max_len]
    safe = raw
    if parsed.scheme and parsed.netloc:
        authority = _authority_for_log(raw, parsed)
        if authority is None:
            return _UNSAFE_URL_PLACEHOLDER[:max_len]
        path = parsed.path or ""
        basename = path.rsplit("/", 1)[-1]
        tail = "" if path in ("", "/") else f"/.../{basename}" if basename else "/..."
        safe = f"{parsed.scheme}://{authority}{tail}"
    if len(safe) <= max_len:
        return safe
    return "." * max_len if max_len <= 3 else f"{safe[:max_len - 3]}..."
