"""Profile-local Slack API endpoint configuration (not an HTTP CONNECT proxy)."""

from urllib.parse import urlsplit


def keyless_api_base_url(extra: dict) -> str | None:
    """An explicit edge endpoint opts into edge-held bot/app credentials.

    Never manufacture a placeholder for Slack itself, or load saved OAuth tokens
    into an edge endpoint: the edge owns exactly one workspace identity.
    """
    value = extra.get("keyless_api_base_url")
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("keyless_api_base_url must be an HTTP(S) API URL")
    value = value.strip()
    url = urlsplit(value)
    if (url.scheme not in {"http", "https"} or not url.hostname
            or url.username is not None or url.password is not None
            or url.query or url.fragment
            or url.hostname == "slack.com" or url.hostname.endswith(".slack.com")):
        raise ValueError("keyless_api_base_url must be an HTTP(S) edge API URL, not slack.com")
    return value.rstrip("/") + "/"


def has_keyless_credentials(config) -> bool:
    """Pure config probe for discovery and reconnect eligibility (no env fallback)."""
    try:
        return keyless_api_base_url(config.extra or {}) is not None
    except ValueError:
        return False
