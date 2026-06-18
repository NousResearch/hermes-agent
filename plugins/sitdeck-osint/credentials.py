"""SitDeck credential helpers — secrets live in ~/.hermes/.env only."""

from __future__ import annotations

import os
import re
from typing import Any

SITDECK_EMAIL_ENV = "SITDECK_EMAIL"
SITDECK_PASSWORD_ENV = "SITDECK_PASSWORD"
DEFAULT_LOGIN_URL = "https://app.sitdeck.com/#login"
DEFAULT_APP_URL = "https://app.sitdeck.com/"
GLOBAL_PULSE_URL = "https://sitdeck.com/global-pulse"


def _normalize_email(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    if "@" in value:
        return value
    return f"{value}@gmail.com"


def get_credentials() -> dict[str, str]:
    """Read SitDeck login from environment (.env loaded by Hermes)."""
    email = _normalize_email(os.getenv(SITDECK_EMAIL_ENV, ""))
    password = (os.getenv(SITDECK_PASSWORD_ENV, "") or "").strip()
    return {"email": email, "password": password}


def credential_status() -> dict[str, Any]:
    creds = get_credentials()
    email = creds["email"]
    masked = ""
    if email:
        local, _, domain = email.partition("@")
        if local:
            masked = f"{local[:2]}***@{domain}" if domain else f"{local[:2]}***"
    return {
        "email_configured": bool(email),
        "password_configured": bool(creds["password"]),
        "email_masked": masked,
        "login_url": DEFAULT_LOGIN_URL,
        "app_url": DEFAULT_APP_URL,
        "env_vars": [SITDECK_EMAIL_ENV, SITDECK_PASSWORD_ENV],
    }


def redact_secrets(text: str, creds: dict[str, str] | None = None) -> str:
    """Strip email/password substrings from crawl output."""
    creds = creds or get_credentials()
    out = text
    for key in ("email", "password"):
        val = creds.get(key) or ""
        if len(val) >= 4:
            out = out.replace(val, "[REDACTED]")
    email = creds.get("email") or ""
    if email and "@" in email:
        local = email.split("@", 1)[0]
        if len(local) >= 3:
            out = re.sub(re.escape(local), "[REDACTED]", out, flags=re.IGNORECASE)
    return out
