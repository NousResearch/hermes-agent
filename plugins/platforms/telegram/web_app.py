"""Validation helpers for Telegram Mini App URL buttons."""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit


MAX_WEB_APP_BUTTON_LABEL_LENGTH = 64
MAX_WEB_APP_URL_LENGTH = 2048


def _parse_legacy_ipv4_address(hostname: str) -> ipaddress.IPv4Address | None:
    """Parse browser-compatible IPv4 spellings that ``ip_address`` rejects.

    WHATWG URL parsers accept one-to-four decimal, octal, or hexadecimal
    components (for example ``2130706433``, ``0177.0.0.1``, and
    ``0x7f000001``) and normalize each of them to ``127.0.0.1``. Telegram
    opens this URL in a client WebView, so validating only dotted decimal
    would let a loopback literal bypass the public-host contract.
    """
    parts = hostname.split(".")
    if not 1 <= len(parts) <= 4 or any(not part for part in parts):
        return None

    numbers: list[int] = []
    for part in parts:
        base = 10
        digits = part
        if part.lower().startswith("0x"):
            base = 16
            digits = part[2:]
        elif len(part) > 1 and part.startswith("0"):
            base = 8
            digits = part[1:]
        if not digits:
            digits = "0"
        try:
            numbers.append(int(digits, base))
        except ValueError:
            return None

    if any(number > 255 for number in numbers[:-1]):
        return None
    last_limit = 256 ** (5 - len(numbers))
    if numbers[-1] >= last_limit:
        return None

    value = numbers[-1]
    for index, number in enumerate(numbers[:-1]):
        value += number * (256 ** (3 - index))
    return ipaddress.IPv4Address(value)


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
    hostname = parsed.hostname.rstrip(".").lower()
    if "%" in hostname:
        raise ValueError("Telegram Web App button URL must use a public host")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Telegram Web App button URL must use a public host")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = _parse_legacy_ipv4_address(hostname)
    if address is not None and not address.is_global:
        raise ValueError("Telegram Web App button URL must use a public host")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("Telegram Web App button URL is malformed") from exc

    return {"label": label, "url": url}
