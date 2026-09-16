"""Configuration, support decisions, permissions, and safe result handling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

OPERATIONS = {
    "outlook": ("search", "read", "create_draft", "send"),
    "sharepoint": ("search", "read", "download_files", "upload_files"),
    "onedrive": ("search", "read", "download_files", "upload_files"),
    "calendar": ("search", "create_events", "update_events"),
    "teams": ("list_teams", "list_channels", "search_messages", "send_messages"),
    "todo": ("list_task_lists", "search", "read", "create_tasks", "update_tasks"),
    "planner": ("list_plans", "list_buckets", "list_tasks", "read", "create_tasks", "update_tasks"),
}
CAPABILITIES = tuple(OPERATIONS)
AUTHENTICATION_MODE = "client_credentials"
APPLICATION_PERMISSION_MODE = "application"
GRAPH_DEFAULT_SCOPE = "https://graph.microsoft.com/.default"
MAX_TRANSFER_BYTES = 10 * 1024 * 1024
MAX_PAGE_SIZE = 100

OPERATION_PERMISSIONS = {
    "outlook": {"search": {"Mail.Read"}, "read": {"Mail.Read"}, "create_draft": {"Mail.ReadWrite"}, "send": {"Mail.Send"}},
    "sharepoint": {"search": {"Sites.Read.All"}, "read": {"Sites.Read.All"}, "download_files": {"Files.Read.All"}, "upload_files": {"Files.ReadWrite.All"}},
    "onedrive": {"search": {"Files.Read.All"}, "read": {"Files.Read.All"}, "download_files": {"Files.Read.All"}, "upload_files": {"Files.ReadWrite.All"}},
    "calendar": {"search": {"Calendars.Read"}, "create_events": {"Calendars.ReadWrite"}, "update_events": {"Calendars.ReadWrite"}},
    "teams": {"list_teams": {"Team.ReadBasic.All"}, "list_channels": {"Channel.ReadBasic.All"}, "search_messages": {"Chat.Read.All", "ChannelMessage.Read.All"}, "send_messages": set()},
    "todo": {"list_task_lists": {"Tasks.Read.All"}, "search": {"Tasks.Read.All"}, "read": {"Tasks.Read.All"}, "create_tasks": {"Tasks.ReadWrite.All"}, "update_tasks": {"Tasks.ReadWrite.All"}},
    "planner": {"list_plans": {"Group.Read.All"}, "list_buckets": {"Group.Read.All"}, "list_tasks": {"Group.Read.All"}, "read": {"Tasks.Read.All"}, "create_tasks": {"Tasks.ReadWrite.All"}, "update_tasks": {"Tasks.ReadWrite.All"}},
}
UNSUPPORTED_APPLICATION_OPERATIONS = frozenset({"teams.send_messages"})
WRITE_OPERATIONS = frozenset({"create_draft", "send", "upload_files", "create_events", "update_events", "send_messages", "create_tasks", "update_tasks"})
SECRET_KEY_PARTS = ("secret", "token", "authorization", "cookie", "set-cookie", "api-key", "apikey", "password")

@dataclass(frozen=True)
class SupportStatus:
    code: str
    reason: str
    @property
    def supported(self) -> bool:
        return self.code == "supported"


def operation_support(auth_mode: str, capability: str, operation: str) -> SupportStatus:
    if capability not in OPERATIONS or operation not in OPERATIONS[capability]:
        return SupportStatus("unknown_operation", f"Unknown Microsoft 365 operation: {capability}.{operation}")
    if auth_mode in {AUTHENTICATION_MODE, APPLICATION_PERMISSION_MODE, "app_only", "application"} and f"{capability}.{operation}" in UNSUPPORTED_APPLICATION_OPERATIONS:
        return SupportStatus("unsupported_auth_mode", "This operation requires delegated authentication and is not supported by the configured application-only client")
    return SupportStatus("supported", "Supported by the configured authentication mode")

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
        flags = raw.get("capabilities") if isinstance(raw.get("capabilities"), Mapping) else {}
        normalized = {}
        for service, operations in OPERATIONS.items():
            value = flags.get(service, False)
            if value is True:
                normalized[service] = {op: True for op in operations}
            elif isinstance(value, Mapping):
                normalized[service] = {op: bool(value.get(op, False)) for op in operations}
            else:
                normalized[service] = {op: False for op in operations}
        return cls(str(raw.get("tenant_id") or ""), str(raw.get("client_id") or ""), str(raw.get("client_secret") or ""), "me" if "user_id" not in raw else str(raw.get("user_id") or ""), normalized)

    def operations(self, service: str) -> set[str]:
        return {op for op, enabled in self.capabilities.get(service, {}).items() if enabled}

    def enabled(self, service: str) -> bool:
        return bool(self.operations(service))


def permission_status(capability: str, operation: str) -> str:
    """Return an explicit local permission-report status, not a remote claim."""
    if capability not in OPERATION_PERMISSIONS or operation not in OPERATION_PERMISSIONS[capability]:
        return "not_verified"
    if not OPERATION_PERMISSIONS[capability][operation]:
        return "unsupported"
    return "admin_consent_required"


def required_permissions(settings: Microsoft365Settings) -> set[str]:
    return {permission for service in CAPABILITIES for op in settings.operations(service) for permission in OPERATION_PERMISSIONS[service][op]}


def unsupported_operations(settings: Microsoft365Settings) -> list[str]:
    return sorted(f"{service}.{op}" for service in CAPABILITIES for op in settings.operations(service) if not operation_support(AUTHENTICATION_MODE, service, op).supported)


def supported_operations(settings: Microsoft365Settings, capability: str) -> set[str]:
    return {op for op in settings.operations(capability) if operation_support(AUTHENTICATION_MODE, capability, op).supported}


def _requires_user_id(capability: str) -> bool:
    return capability not in {"sharepoint"}


def _redacted_settings(settings: Microsoft365Settings) -> dict[str, Any]:
    return {"tenant_id": settings.tenant_id, "client_id": settings.client_id, "user_id": settings.user_id, "client_secret": "[REDACTED]" if settings.client_secret else "", "capabilities": {service: dict(flags) for service, flags in settings.capabilities.items()}}


def preflight(settings: Microsoft365Settings, *, sdk_available: bool | None = None) -> dict[str, Any]:
    if sdk_available is None:
        try:
            import msgraph  # noqa: F401
            import azure.identity  # noqa: F401
            sdk_available = True
        except ImportError:
            sdk_available = False
    selected = [service for service in CAPABILITIES if settings.enabled(service)]
    missing = [key for key, value in (("tenant_id", settings.tenant_id), ("client_id", settings.client_id), ("client_secret", settings.client_secret)) if not value]
    if not selected:
        missing.append("capabilities")
    if any(_requires_user_id(service) and not settings.user_id.strip() for service in selected):
        missing.append("user_id")
    if any(_requires_user_id(service) and settings.user_id.strip().lower() == "me" for service in selected):
        missing.append("user_id")
    unsupported = unsupported_operations(settings)
    locally_ready = bool(sdk_available and not missing and not unsupported)
    operation_status = {f"{service}.{op}": {"authentication": operation_support(AUTHENTICATION_MODE, service, op).code, "permission": permission_status(service, op)} for service in selected for op in settings.operations(service)}
    return {"ready": locally_ready, "locally_ready": locally_ready, "sdk_available": bool(sdk_available), "authentication": "not_tested", "permissions": "not_tested", "admin_consent": "required" if selected else "not_applicable", "authentication_mode": AUTHENTICATION_MODE, "permission_mode": APPLICATION_PERMISSION_MODE, "graph_scope": GRAPH_DEFAULT_SCOPE, "missing": sorted(set(missing)), "enabled_capabilities": selected, "enabled_operations": {service: sorted(settings.operations(service)) for service in selected}, "operation_status": operation_status, "required_permissions": sorted(required_permissions(settings)), "unsupported_operations": unsupported, "configuration": _redacted_settings(settings)}


def create_graph_client(settings: Microsoft365Settings):
    try:
        from azure.identity import ClientSecretCredential
        from msgraph import GraphServiceClient
    except ImportError as exc:
        raise RuntimeError("Install the Microsoft 365 Plugin extra: msgraph-sdk and azure-identity") from exc
    if not settings.tenant_id or not settings.client_id or not settings.client_secret:
        raise RuntimeError("Microsoft 365 Plugin credentials are not configured")
    credential = ClientSecretCredential(settings.tenant_id, settings.client_id, settings.client_secret)
    return GraphServiceClient(credentials=credential, scopes=[GRAPH_DEFAULT_SCOPE])


def _secret_key(key: Any) -> bool:
    lowered = str(key).lower().replace("_", "-")
    return any(part in lowered for part in SECRET_KEY_PARTS)


def safe_result(value: Any, *, limit: int = 50, max_depth: int = 6, max_string: int = 4000, _depth: int = 0, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()
    if _depth > max_depth:
        return "[TRUNCATED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, str):
        return value if len(value) <= max_string else value[:max_string] + "...[TRUNCATED]"
    identity = id(value)
    if identity in _seen:
        return "[CYCLE]"
    _seen.add(identity)
    try:
        if isinstance(value, Mapping):
            result = {}
            for key, item in list(value.items())[:limit]:
                result[str(key)] = "[REDACTED]" if _secret_key(key) or str(key).lower() in {"headers", "request_information", "response"} else safe_result(item, limit=limit, max_depth=max_depth, max_string=max_string, _depth=_depth + 1, _seen=_seen)
            return result
        if isinstance(value, (list, tuple, set)):
            return [safe_result(item, limit=limit, max_depth=max_depth, max_string=max_string, _depth=_depth + 1, _seen=_seen) for item in list(value)[:limit]]
        if hasattr(value, "additional_data"):
            data = getattr(value, "additional_data", {}) or {}
            result = {"additional_data": safe_result(data, limit=limit, max_depth=max_depth, max_string=max_string, _depth=_depth + 1, _seen=_seen)}
            for name in ("id", "name", "subject", "title", "content", "size", "web_url", "created_date_time", "last_modified_date_time"):
                if hasattr(value, name):
                    result[name] = safe_result(getattr(value, name), limit=limit, max_depth=max_depth, max_string=max_string, _depth=_depth + 1, _seen=_seen)
            return result
        return safe_result(str(value), limit=limit, max_depth=max_depth, max_string=max_string, _depth=_depth + 1, _seen=_seen)
    finally:
        _seen.discard(identity)
