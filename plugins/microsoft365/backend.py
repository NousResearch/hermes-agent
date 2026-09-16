"""Microsoft 365 Plugin backend.

The SDK owns Graph protocol details. This module owns capability configuration,
permission derivation, side-effect-free preflight, and safe result shaping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

CAPABILITIES = ("outlook", "sharepoint", "calendar", "teams", "planner")
PERMISSIONS = {
    "outlook": {"Mail.ReadWrite"},
    "sharepoint": {"Sites.ReadWrite.All", "Files.ReadWrite.All"},
    "calendar": {"Calendars.ReadWrite"},
    "teams": {"Chat.ReadWrite", "ChannelMessage.Read.All", "ChannelMessage.Send"},
    "planner": {"Tasks.ReadWrite"},
}
SECRET_KEYS = frozenset({"client_secret", "access_token", "refresh_token"})


@dataclass(frozen=True)
class Microsoft365Settings:
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    user_id: str = "me"
    capabilities: Mapping[str, bool] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "Microsoft365Settings":
        raw = raw or {}
        flags = raw.get("capabilities") or {}
        if not isinstance(flags, Mapping):
            flags = {}
        return cls(
            tenant_id=str(raw.get("tenant_id") or ""),
            client_id=str(raw.get("client_id") or ""),
            client_secret=str(raw.get("client_secret") or ""),
            user_id=str(raw.get("user_id") or "me"),
            capabilities={name: bool(flags.get(name, False)) for name in CAPABILITIES},
        )

    def enabled(self, capability: str) -> bool:
        return bool(self.capabilities.get(capability, False))


def required_permissions(settings: Microsoft365Settings) -> set[str]:
    return set().union(*(PERMISSIONS[name] for name in CAPABILITIES if settings.enabled(name)))


def _redacted_settings(settings: Microsoft365Settings) -> dict[str, Any]:
    return {
        "tenant_id": settings.tenant_id,
        "client_id": settings.client_id,
        "user_id": settings.user_id,
        "client_secret": "[redacted]" if settings.client_secret else "",
        "capabilities": dict(settings.capabilities),
    }


def preflight(settings: Microsoft365Settings, *, sdk_available: bool | None = None) -> dict[str, Any]:
    """Validate locally only; never creates a client or contacts Microsoft."""
    if sdk_available is None:
        try:
            import msgraph  # noqa: F401
            import azure.identity  # noqa: F401
            sdk_available = True
        except ImportError:
            sdk_available = False
    missing = [key for key, value in (("tenant_id", settings.tenant_id), ("client_id", settings.client_id), ("client_secret", settings.client_secret)) if not value]
    if not any(settings.capabilities.values()):
        missing.append("capabilities")
    return {
        "ready": bool(sdk_available and not missing),
        "sdk_available": bool(sdk_available),
        "missing": missing,
        "enabled_capabilities": [name for name in CAPABILITIES if settings.enabled(name)],
        "required_permissions": sorted(required_permissions(settings)),
        "configuration": _redacted_settings(settings),
    }


def create_graph_client(settings: Microsoft365Settings):
    """Create the official SDK client; imports remain optional until used."""
    try:
        from azure.identity import ClientSecretCredential
        from msgraph import GraphServiceClient
    except ImportError as exc:
        raise RuntimeError("Install the Microsoft 365 Plugin extra: msgraph-sdk and azure-identity") from exc
    if not settings.tenant_id or not settings.client_id or not settings.client_secret:
        raise RuntimeError("Microsoft 365 Plugin credentials are not configured")
    credential = ClientSecretCredential(settings.tenant_id, settings.client_id, settings.client_secret)
    return GraphServiceClient(credentials=credential, scopes=["https://graph.microsoft.com/.default"])


def safe_result(value: Any, *, limit: int = 50) -> Any:
    """Normalize SDK model/dict output without exposing credential-shaped fields."""
    if hasattr(value, "__dict__"):
        value = {key: item for key, item in vars(value).items() if not key.startswith("_")}
    if isinstance(value, (bytes, bytearray)):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, Mapping):
        return {str(key): "[redacted]" if str(key).lower() in SECRET_KEYS else safe_result(item, limit=limit) for key, item in list(value.items())[:limit]}
    if isinstance(value, (list, tuple)):
        return [safe_result(item, limit=limit) for item in value[:limit]]
    return value
