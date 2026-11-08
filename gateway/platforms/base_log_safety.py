"""Log-safe URL rendering for adapter logs (split from :mod:`gateway.platforms.base`)."""
import os
import re
from urllib.parse import urlsplit
from urllib.request import getproxies_environment

_UNSAFE_URL_PLACEHOLDER = "<unparseable URL redacted>"
_FILE_NAME = re.compile(r"[\w-]{1,40}\.[A-Za-z0-9]{1,5}")
# A host worth naming: a DNS name / IPv4 (``\w`` keeps IDN) or an IPv6 literal with a zone id.
_PLAUSIBLE_HOST = re.compile(r"[\w.-]+|[0-9a-f:.]+(%[\w.-]+)?")
# A URL that is nothing but ``scheme://host[:port]`` (trailing slashes allowed): the only shape whose
# rendering can show all of it, modulo the case and port zero-padding that parsing normalises.
_BARE_AUTHORITY = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*)://(\[[^\]]*\]|[^:/@?#\[\]]*)(?::([0-9]{1,5}))?/*")


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


def describe_proxy_error(exc: BaseException, proxy_url: str | None) -> str | None:
    """``<ExceptionType> via <scheme://host:port>`` for an error raised through a proxy URL holding
    anything that rendering leaves out (userinfo, path, query, a malformed authority), else None:
    the caller keeps the full error text, which then cannot repeat a part of the URL not shown.

    Transport libraries echo a proxy URL into their error text percent-encoded, split or
    reformatted (yarl/httpx re-quote the password, httpx's ``Invalid port: '<pw>'`` is a bare
    fragment), so no substring redaction of that text is complete: none of it is shown."""
    proxy = str(proxy_url or "")
    if not proxy or _shown_in_full(proxy):
        return None
    return f"{type(exc).__name__} via {safe_proxy_url_for_log(proxy, max_len=len(proxy) + 64)}"


# Every ``<scheme>_proxy`` a transport here reads from the environment: requests/httpx/aiohttp
# (http, https, all; aiohttp also ws/wss), websockets (ws, wss, socks), grpc (grpc_proxy).
_ENV_PROXY_KEYS = frozenset(f"{scheme}_proxy" for scheme in ("https", "http", "all", "wss", "ws", "socks", "grpc"))


def env_proxy_for_redaction(*target_urls: str | None) -> str | None:
    """The proxy-environment value to pass to :func:`describe_proxy_error` for an error raised by a
    library that reads the proxy environment itself, or None when NO_PROXY bypasses every target
    (that request used no proxy, so its error text cannot echo one and is kept in full).

    The libraries disagree on which variable wins (urllib and requests prefer lowercase, httpx
    mounts every scheme, websockets reads ws/wss/socks), so no single pick is the one they used:
    any value, of any case or scheme, that rendering cannot show in full is returned first, and the
    first secret-free value only when every value is one. A secret-free ``ALL_PROXY`` beside a secret
    ``http_proxy`` therefore still redacts, and an ``ALL_PROXY``-only secret redacts an error from a
    library that ignores ``ALL_PROXY``: leak-safety over that edge."""
    values = [value.strip() for key, value in os.environ.items() if key.lower() in _ENV_PROXY_KEYS and value.strip()]
    # httpx builds a transport for every proxy variable (``http://`` prefixed when scheme-less) when
    # the client is created, so an unparseable one raises, echoing it, before NO_PROXY is consulted
    # unless NO_PROXY is ``*``.
    star_only = any(safe_proxy_url_for_log(v if "://" in v else f"http://{v}") == _UNSAFE_URL_PLACEHOLDER
                    for v in values)
    targets = [url for url in target_urls if url]
    if targets and all(_bypassed(url, star_only=star_only) for url in targets):
        return None
    return next((v for v in values if not _shown_in_full(v)), values[0] if values else None)


def _bypassed(url: str, *, star_only: bool) -> bool:
    """NO_PROXY (``no_proxy`` first, as urllib reads it) bypasses ``url`` in every library here:
    ``*`` alone, or (unless ``star_only``) an entry equal to the host or a ``.suffix`` entry the
    host is a subdomain of. That is the part of urllib's ``proxy_bypass_environment`` rule httplib2
    (bare entries match only exactly) and httpx (``.suffix`` skips the apex) share; ``host:port``,
    CIDR and other forms some libraries ignore are not counted, so a request they proxied keeps
    its redaction."""
    try:
        host = (urlsplit(str(url)).hostname or "").rstrip(".")
    except ValueError:
        return False
    no_proxy = getproxies_environment().get("no", "").strip().lower()
    if no_proxy == "*" or star_only:
        return bool(host) and no_proxy == "*"
    entries = [entry.strip() for entry in no_proxy.split(",")]
    return any(host and entry and (host.endswith(entry) if entry.startswith(".") else host == entry)
               for entry in entries)


def _shown_in_full(proxy: str) -> bool:
    """True when rendering prints all of ``proxy`` (lowercased scheme and host, integer port):
    ``HTTP://Proxy:03128`` holds nothing ``http://proxy:3128`` leaves out."""
    m = _BARE_AUTHORITY.fullmatch(proxy)
    canonical = m and f"{m[1].lower()}://{m[2].lower()}" + (f":{int(m[3])}" if m[3] else "")
    return bool(canonical) and safe_proxy_url_for_log(proxy, max_len=len(proxy) + 64) == canonical
