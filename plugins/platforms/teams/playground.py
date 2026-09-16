"""Opt-in seam for the local Microsoft 365 Agents Playground client.

The Playground is a UI/client: it receives the bot endpoint through ``-e`` and
posts Bot Framework Activities to that endpoint. Its connector callback is
provided dynamically in each incoming Activity's ``serviceUrl``.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse


class PlaygroundValidationError(ValueError):
    """The configured endpoint is not an explicitly permitted local endpoint."""


@dataclass(frozen=True)
class PlaygroundConfig:
    enabled: bool = False
    app_endpoint: str | None = None
    allow_private: bool = False
    ui_url: str = "http://127.0.0.1:56150"

    @classmethod
    def from_extra(cls, extra: dict | None) -> "PlaygroundConfig":
        extra = extra or {}
        raw = extra.get("playground_url") or extra.get("app_endpoint")
        allow = extra.get("playground_allow_private") in (True, "true", "1", 1)
        url = validate_playground_url(str(raw), allow_private=allow) if raw else None
        # An explicitly configured and validated endpoint opts this target in;
        # an empty extra keeps it disabled by default.  There is deliberately no
        # credential fallback from the production Teams fields.
        return cls(enabled=bool(url), app_endpoint=url, allow_private=allow)

    @property
    def client_id(self) -> None:
        return None

    @property
    def client_secret(self) -> None:
        return None


@dataclass(frozen=True)
class HandshakeResult:
    ok: bool
    message: str


def validate_playground_url(raw: str, *, allow_private: bool = False) -> str:
    """Accept only HTTP(S) local/private endpoints after an explicit opt-in.

    No credentials, query strings, fragments, public hosts, metadata addresses, or
    alternate schemes are accepted. This target must never become a generic SSRF
    proxy, and it never reuses production Teams credentials.
    """
    value = str(raw or "").strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise PlaygroundValidationError("Playground URL must be an HTTP(S) URL without credentials.")
    if parsed.query or parsed.fragment:
        raise PlaygroundValidationError("Playground endpoint cannot contain a query or fragment.")
    if parsed.path.rstrip("/") != "/api/messages":
        raise PlaygroundValidationError("Playground endpoint path must be /api/messages.")
    host = parsed.hostname.lower().rstrip(".")
    if host in {"169.254.169.254", "metadata.google.internal", "metadata.azure.internal"}:
        raise PlaygroundValidationError("Cloud metadata endpoints are not allowed.")
    local = host in {"localhost", "localhost.localdomain"} or host.endswith((".local", ".lan", ".internal"))
    try:
        address = ipaddress.ip_address(host)
        local = address.is_loopback or address.is_private or address.is_link_local
    except ValueError:
        pass
    if not local or not allow_private:
        raise PlaygroundValidationError("Playground URL must be a local/private endpoint with explicit opt-in.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise PlaygroundValidationError("Playground URL port is invalid.") from exc
    if port is not None and not 1 <= port <= 65535:
        raise PlaygroundValidationError("Playground URL port is invalid.")
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def generated_test_url(config: PlaygroundConfig) -> str | None:
    if not config.enabled or not config.app_endpoint:
        return None
    return config.ui_url


def generated_callback_url(config: PlaygroundConfig) -> str | None:
    if not config.enabled or not config.app_endpoint:
        return None
    return config.app_endpoint


def build_playground_command(config: PlaygroundConfig) -> str | None:
    """Return the safe local command used by the installed Playground CLI."""
    if not config.enabled or not config.app_endpoint:
        return None
    return f"agentsplayground -e {config.app_endpoint} -c emulator --disable-telemetry"


def build_activity_envelope(text: str, *, conversation_id: str, user_id: str) -> dict[str, object]:
    """Build the smallest generic Bot Framework message activity."""
    return {
        "type": "message", "text": text,
        "conversation": {"id": conversation_id}, "from": {"id": user_id},
        "channelId": "msteams",
    }


async def handshake(config: PlaygroundConfig) -> HandshakeResult:
    if not config.enabled or not config.app_endpoint:
        return HandshakeResult(False, "Teams Playground is disabled or not configured.")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0, follow_redirects=False) as client:
            origin = config.app_endpoint.rsplit("/api/messages", 1)[0]
            response = await client.get(f"{origin}/api/health")
            if response.status_code == 404:
                response = await client.get(f"{origin}/health")
            if 200 <= response.status_code < 300:
                return HandshakeResult(True, "Teams Playground health check passed.")
            return HandshakeResult(False, f"Playground health check returned HTTP {response.status_code}.")
    except (TimeoutError, httpx.TimeoutException):
        return HandshakeResult(False, "Playground handshake timed out.")
    except Exception:
        return HandshakeResult(False, "Playground handshake failed.")
