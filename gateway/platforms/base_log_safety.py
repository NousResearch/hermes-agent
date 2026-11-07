"""Log-safe URL rendering for adapter logs (split from :mod:`gateway.platforms.base`)."""
import re
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


_URL_IN_TEXT = re.compile(r"https?://[^\s'\"()<>]+")


def scrub_urls_in_text(text: str, max_len: int = 160) -> str:
    """Render a log message with every URL reduced to its log-safe form.

    Exception strings embed the request URL (an httpx ``HTTPStatusError`` keeps it verbatim) and
    inbound-media URLs carry signatures in their query string, so a message is not safe to log as-is.
    """
    return _URL_IN_TEXT.sub(lambda m: safe_url_for_log(m.group(0)), str(text))[:max_len]
