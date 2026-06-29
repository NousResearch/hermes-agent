import urllib.parse
from typing import Optional

def _base_url_hostname(url: Optional[str]) -> str:
    if not url: return ""
    try:
        return urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return ""

def is_openrouter_url(base_url: Optional[str], fallback_hostname: str = "") -> bool:
    """Return True when a base URL targets OpenRouter."""
    if base_url is not None:
        hostname = _base_url_hostname(base_url)
    else:
        hostname = fallback_hostname
    return hostname == "openrouter.ai"
