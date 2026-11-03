"""Log-safe URL rendering for adapter logs (split from :mod:`gateway.platforms.base`)."""
from urllib.parse import urlsplit


def safe_url_for_log(url: str, max_len: int = 80) -> str:
    """Return a URL string safe for logs (no query/fragment/userinfo)."""
    raw = str(url) if max_len > 0 and url is not None else ""
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except Exception:
        return raw[:max_len]
    safe = raw
    if parsed.scheme and parsed.netloc:
        # Strip potential embedded credentials (user:pass@host).
        path = parsed.path or ""
        basename = path.rsplit("/", 1)[-1]
        tail = "" if path in ("", "/") else f"/.../{basename}" if basename else "/..."
        safe = f"{parsed.scheme}://{parsed.netloc.rsplit('@', 1)[-1]}{tail}"
    if len(safe) <= max_len:
        return safe
    return "." * max_len if max_len <= 3 else f"{safe[:max_len - 3]}..."
