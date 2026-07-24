"""Validation helpers for Telegram Mini App URL buttons."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit


MAX_WEB_APP_BUTTON_LABEL_LENGTH = 64
MAX_WEB_APP_URL_LENGTH = 2048


def normalize_web_app_button(value: Any) -> dict[str, str] | None:
    """Validate and normalize an outbound Telegram Web App button payload.

    The public contract is ``{"label": str, "url": str}``.  Telegram only
    accepts HTTPS Web App URLs.  Credentials and control characters are
    rejected so an artifact publisher cannot accidentally turn a visible
    button into a credential-bearing link or malformed Bot API payload.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Telegram Web App button must be an object")
    if set(value) != {"label", "url"}:
        raise ValueError(
            "Telegram Web App button must contain exactly 'label' and 'url'"
        )

    label = value.get("label")
    url = value.get("url")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Telegram Web App button label must be a non-empty string")
    label = label.strip()
    if len(label) > MAX_WEB_APP_BUTTON_LABEL_LENGTH:
        raise ValueError(
            f"Telegram Web App button label must be at most "
            f"{MAX_WEB_APP_BUTTON_LABEL_LENGTH} characters"
        )
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in label):
        raise ValueError(
            "Telegram Web App button label must not contain control characters"
        )

    if not isinstance(url, str) or not url.strip():
        raise ValueError("Telegram Web App button URL must be a non-empty string")
    url = url.strip()
    if len(url) > MAX_WEB_APP_URL_LENGTH:
        raise ValueError(
            f"Telegram Web App button URL must be at most {MAX_WEB_APP_URL_LENGTH} characters"
        )
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in url):
        raise ValueError(
            "Telegram Web App button URL must not contain control characters"
        )

    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("Telegram Web App button URL must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Telegram Web App button URL must not contain credentials")

    return {"label": label, "url": url}
