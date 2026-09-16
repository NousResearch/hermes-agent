"""Microsoft 365 Plugin configuration and SDK boundary.

Graph requests are intentionally delegated to the official msgraph-sdk. The
operation-to-permission table is explicit so consent remains least-privilege.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

OPERATIONS = {
    "outlook": ("search", "read", "create_draft", "send"),
    "sharepoint": ("search", "read", "download_files", "upload_files"),
    "onedrive": ("search", "read", "download_files", "upload_files"),
    "calendar": ("search", "create_events"),
    "teams": ("list_teams", "list_channels", "search_messages", "send_messages"),
    "planner": ("search", "read", "create_tasks", "update_tasks"),
}
CAPABILITIES = tuple(OPERATIONS)

# Microsoft Graph delegated/application permission names. These are kept here,
# rather than inferred from service names, because read/write scopes differ by
# operation. The values are also the contract exercised by plugin tests/docs.
OPERATION_PERMISSIONS = {
    "outlook": {
        "search": {"Mail.Read"}, "read": {"Mail.Read"},
        "create_draft": {"Mail.ReadWrite"}, "send": {"Mail.Send"},
    },
    "sharepoint": {
        "search": {"Sites.Read.All"}, "read": {"Sites.Read.All"},
        "download_files": {"Files.Read.All"}, "upload_files": {"Files.ReadWrite.All"},
    },
    "onedrive": {
        "search": {"Files.Read.All"}, "read": {"Files.Read.All"},
        "download_files": {"Files.Read.All"}, "upload_files": {"Files.ReadWrite.All"},
    },
    "calendar": {
        "search": {"Calendars.Read"}, "create_events": {"Calendars.ReadWrite"},
    },
    "teams": {
        "list_teams": {"Team.ReadBasic.All"}, "list_channels": {"Channel.ReadBasic.All"},
        "search_messages": {"ChannelMessage.Read.All"}, "send_messages": {"ChannelMessage.Send"},
    },
    "planner": {
        "search": {"Tasks.Read"}, "read": {"Tasks.Read"},
        "create_tasks": {"Tasks.ReadWrite"}, "update_tasks": {"Tasks.ReadWrite"},
    },
}
SECRET_KEYS = frozenset({"client_secret", "access_token", "refresh_token"})


@dataclass(frozen=True)
class Microsoft365Settings:
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    user_id: str = "me"
    capabilities: Mapping[str, Mapping[str, bool]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "Microsoft365Settings":
        raw = raw or {}
        flags = raw.get("capabilities") or {}
        if not isinstance(flags, Mapping):
            flags = {}
        normalized = {}
        for service, operations in OPERATIONS.items():
            value = flags.get(service, False)
            if value is True:  # documented legacy form: enable every operation
                normalized[service] = {operation: True for operation in operations}
            elif isinstance(value, Mapping):
                normalized[service] = {operation: bool(value.get(operation, False)) for operation in operations}
            else:
                normalized[service] = {operation: False for operation in operations}
        return cls(
            tenant_id=str(raw.get("tenant_id") or ""), client_id=str(raw.get("client_id") or ""),
            client_secret=str(raw.get("client_secret") or ""), user_id=str(raw.get("user_id") or "me"),
            capabilities=normalized,
        )

    def operations(self, service: str) -> set[str]:
        return {operation for operation, enabled in self.capabilities.get(service, {}).items() if enabled}

    def enabled(self, service: str) -> bool:
        return bool(self.operations(service))


def required_permissions(settings: Microsoft365Settings) -> set[str]:
    return {permission for service in CAPABILITIES for operation in settings.operations(service)
            for permission in OPERATION_PERMISSIONS[service][operation]}


def _redacted_settings(settings: Microsoft365Settings) -> dict[str, Any]:
    return {"tenant_id": settings.tenant_id, "client_id": settings.client_id, "user_id": settings.user_id,
            "client_secret": "[redacted]" if settings.client_secret else "",
            "capabilities": {service: dict(flags) for service, flags in settings.capabilities.items()}}


def preflight(settings: Microsoft365Settings, *, sdk_available: bool | None = None) -> dict[str, Any]:
    """Validate locally only; never creates a client or contacts Microsoft."""
    if sdk_available is None:
        try:
            import msgraph  # noqa: F401
            import azure.identity  # noqa: F401
            sdk_available = True
        except ImportError:
            sdk_available = False
    missing = [key for key, value in (("tenant_id", settings.tenant_id), ("client_id", settings.client_id),
                                      ("client_secret", settings.client_secret)) if not value]
    if not any(settings.enabled(service) for service in CAPABILITIES):
        missing.append("capabilities")
    return {"ready": bool(sdk_available and not missing), "sdk_available": bool(sdk_available), "missing": missing,
            "enabled_capabilities": [service for service in CAPABILITIES if settings.enabled(service)],
            "enabled_operations": {service: sorted(settings.operations(service)) for service in CAPABILITIES if settings.enabled(service)},
            "required_permissions": sorted(required_permissions(settings)), "configuration": _redacted_settings(settings)}


def create_graph_client(settings: Microsoft365Settings):
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
    if hasattr(value, "__dict__"):
        value = {key: item for key, item in vars(value).items() if not key.startswith("_")}
    if isinstance(value, (bytes, bytearray)):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, Mapping):
        return {str(key): "[redacted]" if str(key).lower() in SECRET_KEYS else safe_result(item, limit=limit)
                for key, item in list(value.items())[:limit]}
    if isinstance(value, (list, tuple)):
        return [safe_result(item, limit=limit) for item in value[:limit]]
    return value
