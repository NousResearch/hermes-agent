"""Microsoft 365 Plugin configuration, permissions, and Graph SDK boundary."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping

OPERATIONS = {
    "outlook": ("search", "read", "create_draft", "send"),
    "sharepoint": ("search", "read", "download_files", "upload_files"),
    "onedrive": ("search", "read", "download_files", "upload_files"),
    "calendar": ("search", "create_events", "update_events"),
    "teams": ("list_teams", "list_channels", "search_messages", "send_messages"),
    "planner": ("list_task_lists", "search", "read", "create_tasks", "update_tasks"),
}
CAPABILITIES = tuple(OPERATIONS)
# Authentication is deliberately explicit: ClientSecretCredential obtains an
# app-only token for the Graph /.default scope. It does not implement delegated
# user-consent or authorization-code flow.
AUTHENTICATION_MODE = "client_credentials"
APPLICATION_PERMISSION_MODE = "application"
GRAPH_DEFAULT_SCOPE = "https://graph.microsoft.com/.default"
OPERATION_PERMISSIONS = {
    "outlook": {"search": {"Mail.Read"}, "read": {"Mail.Read"}, "create_draft": {"Mail.ReadWrite"}, "send": {"Mail.Send"}},
    "sharepoint": {"search": {"Sites.Read.All"}, "read": {"Sites.Read.All"}, "download_files": {"Files.Read.All"}, "upload_files": {"Files.ReadWrite.All"}},
    "onedrive": {"search": {"Files.Read.All"}, "read": {"Files.Read.All"}, "download_files": {"Files.Read.All"}, "upload_files": {"Files.ReadWrite.All"}},
    "calendar": {"search": {"Calendars.Read"}, "create_events": {"Calendars.ReadWrite"}, "update_events": {"Calendars.ReadWrite"}},
    "teams": {"list_teams": {"Team.ReadBasic.All"}, "list_channels": {"Channel.ReadBasic.All"}, "search_messages": {"Chat.Read.All", "ChannelMessage.Read.All"}, "send_messages": set()},
    "planner": {"list_task_lists": {"Tasks.Read.All"}, "search": {"Tasks.Read.All"}, "read": {"Tasks.Read.All"}, "create_tasks": {"Tasks.ReadWrite.All"}, "update_tasks": {"Tasks.ReadWrite.All"}},
}
# The Graph API exposes no application permission for sending channel/chat
# messages through this endpoint. Keep the requested operation in the tool
# surface, but never claim a delegated-only role makes app-only auth work.
UNSUPPORTED_APPLICATION_OPERATIONS = frozenset({"teams.send_messages"})
SECRET_KEYS = frozenset({"client_secret", "access_token", "refresh_token"})
WRITE_OPERATIONS = frozenset(op for ops in OPERATIONS.values() for op in ops if op in {"create_draft", "send", "upload_files", "create_events", "update_events", "send_messages", "create_tasks", "update_tasks"})

@dataclass(frozen=True)
class Microsoft365Settings:
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    user_id: str = "me"
    capabilities: Mapping[str, Mapping[str, bool]] = field(default_factory=dict)
    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "Microsoft365Settings":
        raw = raw or {}; flags = raw.get("capabilities") or {}; flags = flags if isinstance(flags, Mapping) else {}
        normalized = {}
        for service, operations in OPERATIONS.items():
            value = flags.get(service, False)
            normalized[service] = {op: True for op in operations} if value is True else {op: bool(value.get(op, False)) for op in operations} if isinstance(value, Mapping) else {op: False for op in operations}
        return cls(str(raw.get("tenant_id") or ""), str(raw.get("client_id") or ""), str(raw.get("client_secret") or ""), str(raw.get("user_id") or "me"), normalized)
    def operations(self, service: str) -> set[str]: return {op for op, enabled in self.capabilities.get(service, {}).items() if enabled}
    def enabled(self, service: str) -> bool: return bool(self.operations(service))

def required_permissions(settings: Microsoft365Settings) -> set[str]:
    return {p for service in CAPABILITIES for op in settings.operations(service) for p in OPERATION_PERMISSIONS[service][op]}

def unsupported_operations(settings: Microsoft365Settings) -> list[str]:
    return sorted(f"{service}.{op}" for service in CAPABILITIES for op in settings.operations(service) if f"{service}.{op}" in UNSUPPORTED_APPLICATION_OPERATIONS)

def _redacted_settings(settings):
    return {"tenant_id": settings.tenant_id, "client_id": settings.client_id, "user_id": settings.user_id, "client_secret": "[redacted]" if settings.client_secret else "", "capabilities": {s: dict(v) for s, v in settings.capabilities.items()}}

def preflight(settings, *, sdk_available=None):
    if sdk_available is None:
        try: import msgraph, azure.identity; sdk_available = True
        except ImportError: sdk_available = False
    missing = [k for k,v in (("tenant_id",settings.tenant_id),("client_id",settings.client_id),("client_secret",settings.client_secret)) if not v]
    if not any(settings.enabled(s) for s in CAPABILITIES): missing.append("capabilities")
    return {"ready": bool(sdk_available and not missing), "sdk_available": bool(sdk_available), "authentication_mode": AUTHENTICATION_MODE, "permission_mode": APPLICATION_PERMISSION_MODE, "graph_scope": GRAPH_DEFAULT_SCOPE, "missing": missing, "enabled_capabilities": [s for s in CAPABILITIES if settings.enabled(s)], "enabled_operations": {s: sorted(settings.operations(s)) for s in CAPABILITIES if settings.enabled(s)}, "required_permissions": sorted(required_permissions(settings)), "unsupported_operations": unsupported_operations(settings), "configuration": _redacted_settings(settings)}

def create_graph_client(settings):
    try:
        from azure.identity import ClientSecretCredential
        from msgraph import GraphServiceClient
    except ImportError as exc: raise RuntimeError("Install the Microsoft 365 Plugin extra: msgraph-sdk and azure-identity") from exc
    if not settings.tenant_id or not settings.client_id or not settings.client_secret: raise RuntimeError("Microsoft 365 Plugin credentials are not configured")
    return GraphServiceClient(credentials=ClientSecretCredential(settings.tenant_id, settings.client_id, settings.client_secret), scopes=[GRAPH_DEFAULT_SCOPE])

def safe_result(value, *, limit=50):
    if hasattr(value, "__dict__"): value = {k:v for k,v in vars(value).items() if not k.startswith("_")}
    if isinstance(value, (bytes, bytearray)): return {"type":"bytes", "size":len(value)}
    if isinstance(value, Mapping): return {str(k): "[redacted]" if str(k).lower() in SECRET_KEYS else safe_result(v, limit=limit) for k,v in list(value.items())[:limit]}
    if isinstance(value, (list, tuple)): return [safe_result(v, limit=limit) for v in value[:limit]]
    return value
