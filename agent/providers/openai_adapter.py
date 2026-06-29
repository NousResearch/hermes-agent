import urllib.parse
from typing import Any, Optional

def _base_url_hostname(url: Optional[str]) -> str:
    if not url: return ""
    try:
        return urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return ""

def is_direct_openai_url(base_url: Optional[str], fallback_hostname: str = "") -> bool:
    """Return True when a base URL targets OpenAI's native API."""
    if base_url is not None:
        hostname = _base_url_hostname(base_url)
    else:
        hostname = fallback_hostname
    return hostname == "api.openai.com"

def is_azure_openai_url(base_url: Optional[str], fallback_url: str = "") -> bool:
    """Return True when a base URL targets Azure OpenAI."""
    url = str(base_url).lower() if base_url is not None else fallback_url
    return "openai.azure.com" in url

def is_github_copilot_url(base_url: Optional[str], fallback_hostname: str = "") -> bool:
    """Return True when a base URL targets GitHub Copilot's API."""
    if base_url is not None:
        hostname = _base_url_hostname(base_url)
    else:
        hostname = fallback_hostname
    return hostname == "api.githubcopilot.com"

def max_tokens_param(
    model: str,
    value: int,
    base_url: Optional[str] = None,
    fallback_hostname: str = "",
    fallback_url: str = ""
) -> dict:
    """Determine whether to send max_tokens or max_completion_tokens."""
    # Inline import to avoid circular dependency
    from utils import model_forces_max_completion_tokens
    
    if (
        is_direct_openai_url(base_url, fallback_hostname)
        or is_azure_openai_url(base_url, fallback_url)
        or is_github_copilot_url(base_url, fallback_hostname)
        or model_forces_max_completion_tokens(model)
    ):
        return {"max_completion_tokens": value} # wait, this function in run_agent took 'value' as an arg
    return {"max_tokens": value}

from typing import Optional, Any

def requested_output_cap_from_api_kwargs(api_kwargs: Any) -> Optional[int]:
    """Extract the outgoing response token cap from a prepared request."""
    if not isinstance(api_kwargs, dict):
        return None
    for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        raw = api_kwargs.get(key)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None
