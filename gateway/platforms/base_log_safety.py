"""Log-safe URL rendering for adapter logs (split from :mod:`gateway.platforms.base`)."""
import re
from urllib.parse import urlsplit

_UNSAFE_URL_PLACEHOLDER = "<unparseable URL redacted>"
_FILE_NAME = re.compile(r"[\w-]{1,40}\.[A-Za-z0-9]{1,5}")
# A host worth naming: a DNS name / IPv4 (``\w`` keeps IDN) or an IPv6 literal with a zone id.
_PLAUSIBLE_HOST = re.compile(r"[\w.-]+|[0-9a-f:.]+(%[\w.-]+)?")


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
    if not _PLAUSIBLE_HOST.fullmatch(host):
        return None
    host = f"[{host}]" if ":" in host else host
    return f"{host}:{port}" if port is not None else host


def _fit(safe: str, max_len: int) -> str:
    if len(safe) <= max_len:
        return safe
    return "." * max_len if max_len <= 3 else f"{safe[:max_len - 3]}..."


def safe_url_for_log(url: str, max_len: int = 80) -> str:
    """Return a URL string safe for logs (no query/fragment/userinfo).

    Only a parsed ``scheme://host[:port]`` and a media file name are kept. Without a parsed
    authority (``bob:<pw>``, ``//bob:<pw>``, ``http:bob:<pw>``, ``proxy:8080/<pw>``,
    ``u:p@proxy``, ``http://u:p@[::1``) no part of the text can be told apart from a
    credential, so a fixed placeholder is logged, never the input."""
    raw = str(url) if max_len > 0 and url is not None else ""
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return _UNSAFE_URL_PLACEHOLDER[:max_len]
    if parsed.scheme.lower() == "file" and not parsed.netloc and "@" not in raw:
        # A local media path (``file:///…/cache/image.png``) has no authority by definition.
        return _fit(raw.split("?", 1)[0].split("#", 1)[0], max_len)
    authority = _authority_for_log(raw, parsed) if parsed.scheme and parsed.netloc else None
    if authority is None:
        return _UNSAFE_URL_PLACEHOLDER[:max_len]
    path = parsed.path or ""
    basename = path.rsplit("/", 1)[-1]
    # Keep a media file name (``image.png``) for context, never a bare token segment
    # (``/services/T/B/<secret>``, ``/x/<secret>`` on a proxy URL).
    if not _FILE_NAME.fullmatch(basename):
        basename = ""
    tail = "" if path in ("", "/") else f"/.../{basename}" if basename else "/..."
    return _fit(f"{parsed.scheme}://{authority}{tail}", max_len)


def safe_proxy_url_for_log(url: str, max_len: int = 80) -> str:
    """``scheme://host[:port]`` of a proxy URL, or the placeholder: a proxy URL's path has no
    routing meaning, so none of it is logged (``/<token>.json`` passes the media file-name rule)."""
    safe = safe_url_for_log(url, max_len=len(str(url or "")) + 64)
    if safe == _UNSAFE_URL_PLACEHOLDER:
        return safe[:max_len]
    return _fit(safe.split("/...", 1)[0], max_len) if max_len > 0 else ""


_SCHEME_WORDS = frozenset({"http", "https", "socks", "socks4", "socks4a", "socks5", "socks5h"})


def redact_proxy_in_text(text, proxy_url: str | None) -> str:
    """*text* (an exception or its message) with *proxy_url* and each 4+ char piece of it that
    the log-safe rendering does not show replaced by ``***``.

    A malformed proxy URL comes back verbatim in transport errors: aiohttp's
    ``InvalidUrlClientError`` / ``NonHttpUrlClientError`` text is the whole request URL, httpx's
    ``Invalid port: '<pw>'`` a bare password fragment. Cutting out the URL alone misses the
    fragment, so every userinfo/path piece goes too."""
    text = "" if text is None else str(text)
    proxy = str(proxy_url or "")
    if not proxy or not text:
        return text
    shown = safe_proxy_url_for_log(proxy, max_len=len(proxy) + 64)
    pieces = {proxy, *(p for p in re.split(r"[:/@?#;\[\]\s]+", proxy)
                       if len(p) >= 4 and p.lower() not in _SCHEME_WORDS and p not in shown)}
    for piece in sorted(pieces, key=len, reverse=True):
        text = text.replace(piece, "***")
    return text
