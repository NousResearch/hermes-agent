"""Resolve npm's registry before npm itself is available during bootstrap.

Only the registry setting is consumed; lock versions and hashes remain authoritative.
The URL templates stay canonical so a local mirror never gets written into the lock.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
from urllib.parse import urlsplit

DEFAULT_REGISTRY = "https://registry.npmjs.org/"


def registry_url() -> str:
    """Environment beats the user npmrc. No npm subprocess or installation required."""
    env = {key.lower(): value for key, value in os.environ.items()}
    value = env.get("npm_config_registry", "").strip()
    if not value:
        config = Path(env.get("npm_config_userconfig") or Path.home() / ".npmrc").expanduser()
        try:
            lines = config.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            lines = []
        for line in lines:
            key, sep, candidate = line.strip().partition("=")
            if sep and key.strip().lower() == "registry":
                value = candidate.strip().strip('"\'')
        value = re.sub(r"\$\{([^}]+)\}", lambda match: os.environ.get(match[1], match[0]), value)
    if not value:
        return DEFAULT_REGISTRY
    parsed = urlsplit(value)
    if (parsed.scheme not in {"https", "http"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or "${" in value):
        raise ValueError("npm registry must be an HTTP(S) URL without credentials, query or fragment")
    return value.rstrip("/") + "/"


def registry_download_url(url: str) -> str:
    """Route only public npm URLs; never rewrite unrelated artifact suppliers."""
    if not url.startswith(DEFAULT_REGISTRY):
        return url
    return registry_url() + url[len(DEFAULT_REGISTRY):]
