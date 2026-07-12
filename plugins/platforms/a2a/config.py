"""Non-secret A2A configuration under the active Hermes profile."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from hermes_cli.config import load_config, save_config

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_LOOPBACK_NAMES = {"localhost", "ip6-localhost", "ip6-loopback"}
_PRESERVE_PATHS = {
    ("platforms", "a2a"),
    ("platform_toolsets", "a2a"),
}


@dataclass(frozen=True)
class A2ASettings:
    enabled: bool
    extra: dict[str, Any] = field(repr=False)
    principals: dict[str, dict[str, str]]
    peers: dict[str, dict[str, str]]


def validate_name(value: str, *, label: str = "name") -> str:
    normalized = str(value or "").strip()
    if not _NAME_RE.fullmatch(normalized):
        raise ValueError(f"{label} must use 1-64 letters, digits, dot, dash, or underscore")
    return normalized


def _is_loopback(host: str) -> bool:
    lowered = host.strip("[]").lower()
    if lowered in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(lowered).is_loopback
    except ValueError:
        return False


def validate_peer_url(value: str) -> str:
    """Require HTTPS except for explicit loopback development endpoints."""
    raw = str(value or "").strip()
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("peer URL must be an absolute HTTP(S) URL")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("peer URL contains an invalid port") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("peer URL must not contain credentials")
    if parsed.fragment:
        raise ValueError("peer URL must not contain a fragment")
    if parsed.query:
        raise ValueError("peer URL must not contain a query string")
    if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
        raise ValueError("non-loopback A2A peers require HTTPS")
    path = parsed.path.rstrip("/") or ""
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


def validate_public_url(value: str, *, production: bool) -> str:
    """Validate the configured public JSON-RPC interface URL."""
    normalized = validate_peer_url(value)
    parsed = urlsplit(normalized)
    if production and parsed.scheme != "https":
        raise ValueError("production A2A public URL requires HTTPS")
    if parsed.scheme == "http" and not _is_loopback(parsed.hostname or ""):
        raise ValueError("development HTTP public URL must use loopback")
    return normalized


def configured_public_url(*, production: bool) -> str:
    settings = load_a2a_settings()
    value = settings.extra.get("public_url")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("platforms.a2a.extra.public_url must be configured")
    return validate_public_url(value, production=production)


def _mapping(value: Any, *, allowed_fields: frozenset[str]) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, str]] = {}
    for key, entry in value.items():
        if isinstance(key, str) and isinstance(entry, dict):
            result[key] = {
                str(k): str(v)
                for k, v in entry.items()
                if k in allowed_fields and isinstance(v, str)
            }
    return result


def load_a2a_settings() -> A2ASettings:
    root = load_config()
    platforms = root.get("platforms") if isinstance(root, dict) else {}
    platform = platforms.get("a2a") if isinstance(platforms, dict) else {}
    if not isinstance(platform, dict):
        platform = {}
    extra = platform.get("extra")
    if not isinstance(extra, dict):
        extra = {}
    return A2ASettings(
        enabled=bool(platform.get("enabled", False)),
        extra=dict(extra),
        principals=_mapping(
            extra.get("principals"),
            allowed_fields=frozenset({"credential_ref", "profile"}),
        ),
        peers=_mapping(
            extra.get("peers"),
            allowed_fields=frozenset({"credential_ref", "url", "generation"}),
        ),
    )


def update_a2a_config(mutator: Callable[[dict[str, Any]], None]) -> None:
    """Atomically persist a non-secret mutation without clobbering siblings."""
    root = load_config()
    if not isinstance(root, dict):
        root = {}
    mutator(root)
    save_config(root, preserve_keys=_PRESERVE_PATHS)


def a2a_extra(root: dict[str, Any]) -> dict[str, Any]:
    platforms = root.setdefault("platforms", {})
    if not isinstance(platforms, dict):
        raise ValueError("platforms config must be a mapping")
    platform = platforms.setdefault("a2a", {})
    if not isinstance(platform, dict):
        raise ValueError("platforms.a2a config must be a mapping")
    extra = platform.setdefault("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("platforms.a2a.extra config must be a mapping")
    return extra
