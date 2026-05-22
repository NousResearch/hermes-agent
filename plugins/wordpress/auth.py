"""Environment-backed auth helpers for the WordPress plugin."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass

from .errors import WordPressConfigError


@dataclass(frozen=True)
class WordPressCredentials:
    base_url: str
    username: str
    app_password: str

    def authorization_header(self) -> str:
        token = f"{self.username}:{self.app_password}".encode("utf-8")
        return f"Basic {base64.b64encode(token).decode('ascii')}"


def get_credentials(
    *,
    base_url: str | None = None,
    username: str | None = None,
    app_password: str | None = None,
) -> WordPressCredentials:
    resolved_base_url = (base_url or os.getenv("WORDPRESS_BASE_URL", "")).strip()
    resolved_username = (username or os.getenv("WORDPRESS_USERNAME", "")).strip()
    resolved_app_password = (app_password or os.getenv("WORDPRESS_APP_PASSWORD", "")).strip()

    missing = [
        name
        for name, value in (
            ("WORDPRESS_BASE_URL", resolved_base_url),
            ("WORDPRESS_USERNAME", resolved_username),
            ("WORDPRESS_APP_PASSWORD", resolved_app_password),
        )
        if not value
    ]
    if missing:
        raise WordPressConfigError(
            "Missing WordPress configuration: " + ", ".join(missing)
        )

    return WordPressCredentials(
        base_url=resolved_base_url,
        username=resolved_username,
        app_password=resolved_app_password,
    )


def wordpress_requirements_met() -> bool:
    try:
        get_credentials(base_url="https://example.com")
        return True
    except WordPressConfigError:
        return False
