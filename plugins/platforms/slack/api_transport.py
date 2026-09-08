"""Profile-local Slack API endpoint configuration (not an HTTP CONNECT proxy)."""

import re
from urllib.parse import unquote, urlsplit, urlunsplit


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


class KeylessFileError(ValueError):
    """Safe attachment diagnostic: messages never contain configured or incoming URLs."""


def keyless_file_url(extra: dict, source: str) -> str:
    """Map only Slack's private-file path onto an explicitly configured file edge.

    The edge owns a fixed files.slack.com backend and returns bytes directly;
    a Web API base URL does not imply this separate download capability.
    """
    value = extra.get("keyless_file_base_url")
    if not value:
        raise KeylessFileError("keyless Slack files require keyless_file_base_url on an edge that serves private files")
    try:
        base = keyless_api_base_url({"keyless_api_base_url": value})
        original, edge = urlsplit(source), urlsplit(base)
        if (original.scheme != "https" or original.hostname != "files.slack.com"
                or original.port not in (None, 443) or original.username is not None
                or original.password is not None or original.fragment
                or not re.fullmatch(r"/files-pri/T[A-Z0-9]+-F[A-Z0-9]+/.+", original.path)):
            raise ValueError
        for raw in (source, value):
            if any(ord(c) < 32 or ord(c) == 127 for c in raw):
                raise ValueError
        for path in (original.path, edge.path):
            decoded = unquote(path)
            if (re.search(r"%(?:2f|5c|25)|%(?![0-9a-f]{2})", path, re.I)
                    or "\\" in decoded or any(ord(c) < 32 or ord(c) == 127 for c in decoded)
                    or any(part in (".", "..") for part in decoded.split("/"))):
                raise ValueError
        return urlunsplit((edge.scheme, edge.netloc, edge.path.rstrip("/") + original.path,
                           original.query, ""))
    except (ValueError, TypeError):
        # Neither a file's query nor a malformed credential-bearing config belongs in logs.
        raise KeylessFileError("Invalid keyless Slack file URL or keyless_file_base_url") from None
