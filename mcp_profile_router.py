"""No-model MCP profile router primitives.

This module is intentionally small and side-effect free: it only parses
fully-qualified profile refs and exposes read-only local profile inventory for
the ChatGPT ↔ Hermes MCP router, plus policy-gated read-only workspace access.
It must not import or call the Hermes agent loop, provider clients, or AI coding
CLIs. Later filesystem/terminal tools should build on the explicit cost
metadata here rather than exposing arbitrary Hermes tools by default.
"""

from __future__ import annotations

import fnmatch
import json
import os
import posixpath
import re
from collections.abc import Iterable as IterableABC, Mapping as MappingABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from hermes_cli.profiles import (
    ProfileInfo,
    list_profiles as _list_local_profile_infos,
    normalize_profile_name,
    validate_profile_name,
)


COST_CLASS_NO_MODEL = "no_model"
COST_CLASS_EXTERNAL_API_NO_MODEL = "external_api_no_model"
COST_CLASS_MAY_CALL_AUX_MODEL = "may_call_aux_model"
COST_CLASS_SCHEDULES_FUTURE_MODEL = "schedules_future_model"
COST_CLASS_CALLS_HERMES_AGENT_MODEL = "calls_hermes_agent_model"

NO_MODEL_DEFAULT_COST_CLASSES = frozenset(
    {COST_CLASS_NO_MODEL, COST_CLASS_EXTERNAL_API_NO_MODEL}
)
DEFAULT_ALLOWED_HOSTS = frozenset({"local", "mac"})
LOCAL_HOST = "local"
PROFILE_ROUTER_CONFIG_KEY = "profile_router"
PROFILE_ROUTER_TOOL_GROUP = "profile_router"
DEFAULT_PROTECTED_BRANCHES = ("main", "master", "develop", "production")
DEFAULT_ALLOWED_COST_CLASSES = (COST_CLASS_NO_MODEL,)
SECRET_PATH_NAMES = frozenset(
    {
        ".ssh",
        "auth.json",
        "mcp_tokens",
        "mcp_tokens.json",
    }
)
SECRET_PATH_PREFIXES = (".env.",)
MAX_FILE_READ_LINES = 200
MAX_FILE_READ_CHARS = 60_000
MAX_FILE_SEARCH_RESULTS = 50
MAX_FILE_SEARCH_BYTES = 1_000_000
MAX_SEARCH_LINE_CHARS = 500
ALLOWED_SEARCH_OUTPUT_MODES = frozenset({"content", "files_only", "count"})


class ProfileRouterError(ValueError):
    """Raised when a profile-router input is invalid or unsupported."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ProfileRef:
    """A validated, fully qualified profile reference."""

    host: str
    profile: str

    @property
    def value(self) -> str:
        return f"{self.host}:{self.profile}"


@dataclass(frozen=True)
class HostRoutePolicy:
    """Host-scoped routing policy.

    Host policies are intentionally separate from profile policies so future
    filesystem tools can prove that profile roots are contained by a host-local
    allowlist before touching any path.
    """

    host: str
    enabled: bool = False
    allowed_roots: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfileRoutePolicy:
    """Deny-by-default policy for one fully-qualified profile ref."""

    ref: ProfileRef
    enabled: bool = False
    display_name: str = ""
    description: str = ""
    allowed_roots: tuple[str, ...] = ()
    allowed_tool_groups: tuple[str, ...] = ()
    messaging_allowed_recipients: tuple[str, ...] = ()
    allow_filesystem_read: bool = False
    allow_filesystem_write: bool = False
    allow_terminal: bool = False
    allow_messaging: bool = False
    allow_cron: bool = False
    allow_git_push: bool = False
    allow_deploy: bool = False
    allow_model_tools: bool = False
    protected_branches: tuple[str, ...] = DEFAULT_PROTECTED_BRANCHES
    allowed_cost_classes: tuple[str, ...] = DEFAULT_ALLOWED_COST_CLASSES


@dataclass(frozen=True)
class ProfileRouterPolicy:
    """Parsed profile-router policy from ``config.yaml``.

    An absent ``profile_router`` section exposes no profiles.  A profile is
    exposed only when the host is enabled, the profile is explicitly enabled,
    and the read-only ``profile_router`` tool group is allowed.
    """

    hosts: Mapping[str, HostRoutePolicy]
    profiles: Mapping[str, ProfileRoutePolicy]

    def is_profile_exposed(self, policy: ProfileRoutePolicy) -> bool:
        host_policy = self.hosts.get(policy.ref.host)
        return bool(
            host_policy
            and host_policy.enabled
            and policy.enabled
            and PROFILE_ROUTER_TOOL_GROUP in policy.allowed_tool_groups
        )

    def iter_profiles(
        self,
        *,
        host: str | None = None,
        active_only: bool = True,
    ) -> list[ProfileRoutePolicy]:
        selected: list[ProfileRoutePolicy] = []
        normalized_host = host.lower() if host else None
        for policy in self.profiles.values():
            if normalized_host and policy.ref.host != normalized_host:
                continue
            if active_only and not self.is_profile_exposed(policy):
                continue
            selected.append(policy)
        return selected

    def get_profile_policy(self, ref: ProfileRef) -> ProfileRoutePolicy:
        policy = self.profiles.get(ref.value)
        if policy is None:
            raise ProfileRouterError(
                "profile_not_enabled",
                f"Profile is not enabled in profile_router policy: {ref.value}",
            )
        if not self.is_profile_exposed(policy):
            raise ProfileRouterError(
                "profile_disabled",
                f"Profile is disabled by profile_router policy: {ref.value}",
            )
        return policy


def _policy_mapping(value: Any, field: str) -> MappingABC:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise ProfileRouterError("invalid_policy", f"{field} must be a mapping")
    return value


def _policy_bool(value: Any, field: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ProfileRouterError("invalid_policy", f"{field} must be a boolean")


def _policy_string_tuple(
    value: Any,
    field: str,
    *,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, IterableABC) and not isinstance(value, (bytes, MappingABC)):
        values = list(value)
    else:
        raise ProfileRouterError("invalid_policy", f"{field} must be a string list")

    normalized: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ProfileRouterError("invalid_policy", f"{field} entries must be strings")
        text = item.strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _policy_allowed_roots(value: Any, field: str) -> tuple[str, ...]:
    roots = []
    for root in _policy_string_tuple(value, field):
        if not root.startswith("/"):
            raise ProfileRouterError(
                "invalid_policy",
                f"{field} entries must be absolute host-local paths",
            )
        roots.append(posixpath.normpath(root))
    return tuple(dict.fromkeys(roots))


def _path_within_root(path: str, root: str) -> bool:
    normalized_path = posixpath.normpath(path)
    normalized_root = posixpath.normpath(root)
    return normalized_path == normalized_root or normalized_path.startswith(
        normalized_root.rstrip("/") + "/"
    )


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _normalize_absolute_host_path(path: str, field: str) -> str:
    if not isinstance(path, str):
        raise ProfileRouterError("invalid_path", f"{field} must be a string")
    text = path.strip()
    if not text:
        raise ProfileRouterError("invalid_path", f"{field} is required")
    if not text.startswith("/"):
        raise ProfileRouterError(
            "invalid_path", f"{field} must be an absolute host-local path"
        )
    return posixpath.normpath(text)


def _is_secret_path(path: str) -> bool:
    normalized = posixpath.normpath(path)
    parts = [part.lower() for part in normalized.split("/") if part]
    for part in parts:
        if part == ".env" or part in SECRET_PATH_NAMES:
            return True
        if any(part.startswith(prefix) for prefix in SECRET_PATH_PREFIXES):
            return True
    return False


def _ensure_not_secret_path(path: str) -> None:
    if _is_secret_path(path):
        raise ProfileRouterError(
            "secret_path_denied",
            "Path is blocked by the profile-router secret denylist",
        )


def _resolve_existing_local_path(path: str, field: str) -> Path:
    try:
        return Path(path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ProfileRouterError("path_not_found", f"{field} not found: {path}") from exc
    except OSError as exc:
        raise ProfileRouterError("invalid_path", f"{field} is not accessible: {path}") from exc


def _parse_host_policy(host_name: str, raw_policy: Any) -> HostRoutePolicy:
    host = str(host_name or "").strip().lower()
    if host not in DEFAULT_ALLOWED_HOSTS:
        raise ProfileRouterError("unsupported_host", f"Unsupported profile host: {host}")

    policy = _policy_mapping(raw_policy, f"hosts.{host}")
    return HostRoutePolicy(
        host=host,
        enabled=_policy_bool(policy.get("enabled"), f"hosts.{host}.enabled"),
        allowed_roots=_policy_allowed_roots(
            policy.get("allowed_roots"), f"hosts.{host}.allowed_roots"
        ),
    )


def _parse_profile_policy(
    profile_ref: str,
    raw_policy: Any,
    hosts: Mapping[str, HostRoutePolicy],
) -> ProfileRoutePolicy:
    ref = parse_profile_ref(profile_ref)
    policy = _policy_mapping(raw_policy, f"profiles.{ref.value}")
    enabled = _policy_bool(policy.get("enabled"), f"profiles.{ref.value}.enabled")

    raw_tool_groups = policy.get("allowed_tool_groups")
    allowed_tool_groups = _policy_string_tuple(
        raw_tool_groups,
        f"profiles.{ref.value}.allowed_tool_groups",
        default=(PROFILE_ROUTER_TOOL_GROUP,) if enabled and raw_tool_groups is None else (),
    )
    allowed_roots = _policy_allowed_roots(
        policy.get("allowed_roots"), f"profiles.{ref.value}.allowed_roots"
    )
    host_policy = hosts.get(ref.host)
    host_allowed_roots = host_policy.allowed_roots if host_policy is not None else ()
    if allowed_roots and not host_allowed_roots:
        raise ProfileRouterError(
            "root_outside_allowed_host_roots",
            f"Profile {ref.value} declares roots but host {ref.host} has no allowed_roots",
        )
    for root in allowed_roots:
        if not any(_path_within_root(root, host_root) for host_root in host_allowed_roots):
            raise ProfileRouterError(
                "root_outside_allowed_host_roots",
                f"Profile root {root} is outside host {ref.host} allowed_roots",
            )

    filesystem = _policy_mapping(policy.get("filesystem"), f"profiles.{ref.value}.filesystem")
    terminal = _policy_mapping(policy.get("terminal"), f"profiles.{ref.value}.terminal")
    messaging = _policy_mapping(policy.get("messaging"), f"profiles.{ref.value}.messaging")
    cron = _policy_mapping(policy.get("cron"), f"profiles.{ref.value}.cron")
    git = _policy_mapping(policy.get("git"), f"profiles.{ref.value}.git")
    deploy = _policy_mapping(policy.get("deploy"), f"profiles.{ref.value}.deploy")
    model_tools = _policy_mapping(
        policy.get("model_tools", policy.get("model_policy")),
        f"profiles.{ref.value}.model_tools",
    )
    allow_model_tools = _policy_bool(
        model_tools.get("allow_model_tools"),
        f"profiles.{ref.value}.model_tools.allow_model_tools",
    )
    allowed_cost_classes = _policy_string_tuple(
        model_tools.get("allowed_cost_classes"),
        f"profiles.{ref.value}.model_tools.allowed_cost_classes",
        default=DEFAULT_ALLOWED_COST_CLASSES,
    )
    if not allow_model_tools:
        unsafe_cost_classes = sorted(set(allowed_cost_classes) - NO_MODEL_DEFAULT_COST_CLASSES)
        if unsafe_cost_classes:
            raise ProfileRouterError(
                "model_cost_class_not_allowed",
                "Model-consuming cost classes require allow_model_tools=true: "
                + ", ".join(unsafe_cost_classes),
            )

    return ProfileRoutePolicy(
        ref=ref,
        enabled=enabled,
        display_name=str(policy.get("display_name") or "").strip(),
        description=str(policy.get("description") or "").strip(),
        allowed_roots=allowed_roots,
        allowed_tool_groups=allowed_tool_groups,
        messaging_allowed_recipients=_policy_string_tuple(
            messaging.get("allowed_recipients"),
            f"profiles.{ref.value}.messaging.allowed_recipients",
        ),
        allow_filesystem_read=_policy_bool(
            filesystem.get("read"), f"profiles.{ref.value}.filesystem.read"
        ),
        allow_filesystem_write=_policy_bool(
            filesystem.get("write"), f"profiles.{ref.value}.filesystem.write"
        ),
        allow_terminal=_policy_bool(
            terminal.get("enabled"), f"profiles.{ref.value}.terminal.enabled"
        ),
        allow_messaging=_policy_bool(
            messaging.get("enabled"), f"profiles.{ref.value}.messaging.enabled"
        ),
        allow_cron=_policy_bool(cron.get("enabled"), f"profiles.{ref.value}.cron.enabled"),
        allow_git_push=_policy_bool(
            git.get("allow_push"), f"profiles.{ref.value}.git.allow_push"
        ),
        allow_deploy=_policy_bool(
            deploy.get("enabled"), f"profiles.{ref.value}.deploy.enabled"
        ),
        allow_model_tools=allow_model_tools,
        protected_branches=_policy_string_tuple(
            git.get("protected_branches"),
            f"profiles.{ref.value}.git.protected_branches",
            default=DEFAULT_PROTECTED_BRANCHES,
        ),
        allowed_cost_classes=allowed_cost_classes,
    )


def load_profile_router_policy(config: Mapping[str, Any] | None = None) -> ProfileRouterPolicy:
    """Load explicit profile-router policy from config.

    Missing config is safe: it exposes no profiles and grants no filesystem,
    terminal, messaging, cron, deploy, or model-consuming capability.
    """

    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    if not isinstance(config, MappingABC):
        raise ProfileRouterError("invalid_policy", "config must be a mapping")

    raw_section = config.get(PROFILE_ROUTER_CONFIG_KEY) or {}
    section = _policy_mapping(raw_section, PROFILE_ROUTER_CONFIG_KEY)
    raw_hosts = _policy_mapping(section.get("hosts"), f"{PROFILE_ROUTER_CONFIG_KEY}.hosts")
    raw_profiles = _policy_mapping(
        section.get("profiles"), f"{PROFILE_ROUTER_CONFIG_KEY}.profiles"
    )

    hosts = {
        host: _parse_host_policy(host, host_policy)
        for host, host_policy in raw_hosts.items()
    }
    profiles = {
        parse_profile_ref(profile_ref).value: _parse_profile_policy(
            profile_ref, profile_policy, hosts
        )
        for profile_ref, profile_policy in raw_profiles.items()
    }
    return ProfileRouterPolicy(hosts=hosts, profiles=profiles)


@dataclass(frozen=True)
class RouterToolMetadata:
    """Security/cost metadata required before exposing a router tool."""

    name: str
    description: str
    cost_class: str
    llm_calls: int
    enabled_by_default: bool = True
    mutates_state: bool = False
    tool_group: str = "profile_router"


def parse_profile_ref(
    profile_ref: str,
    *,
    allowed_hosts: Iterable[str] = DEFAULT_ALLOWED_HOSTS,
) -> ProfileRef:
    """Parse and validate ``host:profile`` profile refs.

    The router boundary requires explicit host qualification so a public MCP
    client cannot accidentally target the wrong profile namespace.  Profile
    names are normalized with the same rules as ``hermes profile``.
    """

    if not isinstance(profile_ref, str):
        raise ProfileRouterError("invalid_profile_ref", "profile_ref must be a string")

    raw = profile_ref.strip()
    if raw.count(":") != 1:
        raise ProfileRouterError(
            "invalid_profile_ref",
            "profile_ref must use fully qualified form '<host>:<profile>'",
        )

    raw_host, raw_profile = raw.split(":", 1)
    host = raw_host.strip().lower()
    if not host:
        raise ProfileRouterError("invalid_profile_ref", "profile_ref host is required")

    allowed = {str(item).strip().lower() for item in allowed_hosts}
    if host not in allowed:
        raise ProfileRouterError("unsupported_host", f"Unsupported profile host: {host}")

    try:
        profile = normalize_profile_name(raw_profile)
        validate_profile_name(profile)
    except ValueError as exc:
        raise ProfileRouterError("invalid_profile_ref", str(exc)) from exc

    return ProfileRef(host=host, profile=profile)


ROUTER_TOOL_METADATA: Mapping[str, RouterToolMetadata] = {
    "profiles_list": RouterToolMetadata(
        name="profiles_list",
        description="List local Hermes profiles as fully-qualified refs.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
    "profile_get": RouterToolMetadata(
        name="profile_get",
        description="Get non-secret metadata for one local Hermes profile.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
    "profile_health": RouterToolMetadata(
        name="profile_health",
        description="Report read-only local profile health without executing tools.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
    "workspace_open": RouterToolMetadata(
        name="workspace_open",
        description="Open a policy-gated read-only local workspace with an opaque ID.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
    "workspace_get": RouterToolMetadata(
        name="workspace_get",
        description="Inspect an opened workspace by opaque ID without revealing its root.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
    "workspace_close": RouterToolMetadata(
        name="workspace_close",
        description="Close an opened workspace and remove its server-side registry entry.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
    "file_read": RouterToolMetadata(
        name="file_read",
        description="Read a paginated text file through an opened read-only workspace.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
    "file_search": RouterToolMetadata(
        name="file_search",
        description="Search text files through an opened read-only workspace.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
}


def get_router_tool_metadata() -> dict[str, dict]:
    """Return serializable metadata for all current profile-router tools."""

    return {name: asdict(meta) for name, meta in ROUTER_TOOL_METADATA.items()}


def assert_default_tools_are_no_model(
    metadata: Mapping[str, RouterToolMetadata] = ROUTER_TOOL_METADATA,
) -> None:
    """Fail closed if a default-exposed router tool may spend model tokens."""

    unsafe = [
        meta.name
        for meta in metadata.values()
        if meta.enabled_by_default
        and (meta.cost_class not in NO_MODEL_DEFAULT_COST_CLASSES or meta.llm_calls != 0)
    ]
    if unsafe:
        raise ProfileRouterError(
            "unsafe_default_tool_exposure",
            "Default profile-router tools must be no-model: " + ", ".join(sorted(unsafe)),
        )


@dataclass(frozen=True)
class WorkspaceMetadata:
    """Server-side metadata for one opened profile-router workspace.

    Workspace IDs are opaque to MCP clients.  The root is kept in server-side
    metadata and every future file/search/write/terminal path must resolve
    through this object before touching the filesystem.
    """

    workspace_id: str
    profile_ref: str
    host: str
    profile: str
    root: str
    mode: str = "checkout"
    read_only: bool = True
    cost_class: str = COST_CLASS_NO_MODEL
    llm_calls: int = 0

    def to_public_dict(self) -> dict:
        return asdict(self)


def _local_profile_exists(profile: str) -> bool:
    return any(info.name == profile for info in _list_local_profile_infos())


def _resolve_allowed_local_roots(allowed_roots: Iterable[str]) -> list[Path]:
    resolved: list[Path] = []
    for allowed_root in allowed_roots:
        _ensure_not_secret_path(allowed_root)
        resolved.append(_resolve_existing_local_path(allowed_root, "allowed root"))
    return resolved


def create_workspace_metadata(
    profile_ref: str,
    root: str,
    *,
    mode: str = "checkout",
    workspace_id: str | None = None,
    policy: ProfileRouterPolicy | None = None,
) -> WorkspaceMetadata:
    """Validate a local read workspace and return opaque no-model metadata.

    This helper intentionally does not expose an MCP tool yet.  It is the common
    safety gate future ``workspace_open``, ``file_read``, and ``file_search``
    handlers must use before resolving paths.
    """

    assert_default_tools_are_no_model()
    if mode not in {"checkout", "worktree", "read_only"}:
        raise ProfileRouterError("invalid_workspace_mode", f"Unsupported workspace mode: {mode}")

    ref = parse_profile_ref(profile_ref)
    if ref.host != LOCAL_HOST:
        raise ProfileRouterError(
            "unsupported_host",
            f"Only local workspaces are supported by this skeleton, got: {ref.host}",
        )
    if not _local_profile_exists(ref.profile):
        raise ProfileRouterError("profile_not_found", f"Profile not found: {ref.value}")

    router_policy = policy or load_profile_router_policy()
    route_policy = router_policy.get_profile_policy(ref)
    if not route_policy.allow_filesystem_read:
        raise ProfileRouterError(
            "filesystem_read_not_allowed",
            f"Filesystem read is disabled by profile_router policy: {ref.value}",
        )

    normalized_root = _normalize_absolute_host_path(root, "workspace root")
    _ensure_not_secret_path(normalized_root)
    if not any(_path_within_root(normalized_root, allowed_root) for allowed_root in route_policy.allowed_roots):
        raise ProfileRouterError(
            "workspace_root_not_allowed",
            f"Workspace root is outside allowed roots for {ref.value}",
        )

    resolved_root = _resolve_existing_local_path(normalized_root, "workspace root")
    allowed_roots = _resolve_allowed_local_roots(route_policy.allowed_roots)
    if not any(_path_is_relative_to(resolved_root, allowed_root) for allowed_root in allowed_roots):
        raise ProfileRouterError(
            "symlink_traversal_denied",
            f"Resolved workspace root escapes allowed roots for {ref.value}",
        )

    return WorkspaceMetadata(
        workspace_id=workspace_id or f"ws_{uuid4().hex}",
        profile_ref=ref.value,
        host=ref.host,
        profile=ref.profile,
        root=str(resolved_root),
        mode=mode,
        read_only=True,
    )


def resolve_workspace_path(
    workspace: WorkspaceMetadata,
    path: str,
    *,
    require_exists: bool = True,
) -> str:
    """Resolve a client path inside a workspace and reject escapes/secrets."""

    if not isinstance(workspace, WorkspaceMetadata):
        raise ProfileRouterError("invalid_workspace", "workspace metadata is required")
    if workspace.host != LOCAL_HOST:
        raise ProfileRouterError(
            "unsupported_host",
            f"Only local workspace paths are supported by this skeleton, got: {workspace.host}",
        )
    if not isinstance(path, str):
        raise ProfileRouterError("invalid_path", "path must be a string")

    raw_path = path.strip() or "."
    if raw_path.startswith("/"):
        raise ProfileRouterError("absolute_path_not_allowed", "path must be workspace-relative")

    normalized_candidate = posixpath.normpath(posixpath.join(workspace.root, raw_path))
    if not _path_within_root(normalized_candidate, workspace.root):
        raise ProfileRouterError("path_outside_workspace", "path escapes workspace root")
    _ensure_not_secret_path(normalized_candidate)

    resolved_root = _resolve_existing_local_path(workspace.root, "workspace root")
    if require_exists:
        resolved_candidate = _resolve_existing_local_path(normalized_candidate, "workspace path")
    else:
        resolved_parent = _resolve_existing_local_path(
            posixpath.dirname(normalized_candidate), "workspace path parent"
        )
        resolved_candidate = resolved_parent / posixpath.basename(normalized_candidate)

    if not _path_is_relative_to(resolved_candidate, resolved_root):
        raise ProfileRouterError("symlink_traversal_denied", "path escapes workspace root")
    _ensure_not_secret_path(str(resolved_candidate))
    return str(resolved_candidate)


class WorkspaceRegistry:
    """In-memory server-side registry for opaque workspace IDs.

    MCP clients only receive the opaque ``workspace_id``. Host-local roots stay
    in this registry and every read/search path is resolved through
    ``resolve_workspace_path`` before file access.
    """

    def __init__(self) -> None:
        self._workspaces: dict[str, WorkspaceMetadata] = {}

    def open(
        self,
        profile_ref: str,
        root: str,
        *,
        mode: str = "checkout",
    ) -> WorkspaceMetadata:
        workspace = create_workspace_metadata(profile_ref, root, mode=mode)
        self._workspaces[workspace.workspace_id] = workspace
        return workspace

    def get(self, workspace_id: str) -> WorkspaceMetadata:
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ProfileRouterError("invalid_workspace", "workspace_id is required")
        workspace = self._workspaces.get(workspace_id.strip())
        if workspace is None:
            raise ProfileRouterError(
                "workspace_not_found", f"Workspace is not open: {workspace_id}"
            )
        return workspace

    def close(self, workspace_id: str) -> WorkspaceMetadata:
        workspace = self.get(workspace_id)
        del self._workspaces[workspace.workspace_id]
        return workspace


DEFAULT_WORKSPACE_REGISTRY = WorkspaceRegistry()


def _public_workspace_dict(workspace: WorkspaceMetadata) -> dict:
    """Return workspace metadata safe for an external MCP client.

    The host-local root is intentionally omitted; clients must use the opaque
    ID and workspace-relative paths instead of learning or replaying server
    absolute paths.
    """

    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "host": workspace.host,
        "profile": workspace.profile,
        "mode": workspace.mode,
        "read_only": workspace.read_only,
        "cost_class": workspace.cost_class,
        "llm_calls": workspace.llm_calls,
    }


def _bounded_int(
    value: int | None,
    field: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileRouterError("invalid_pagination", f"{field} must be an integer")
    if value < minimum:
        raise ProfileRouterError(
            "invalid_pagination", f"{field} must be >= {minimum}"
        )
    return min(value, maximum)


def _ensure_text_file(path: Path) -> None:
    if not path.is_file():
        raise ProfileRouterError("not_a_file", f"Path is not a file: {path.name}")
    try:
        with path.open("rb") as handle:
            sample = handle.read(4096)
    except OSError as exc:
        raise ProfileRouterError("file_not_readable", f"File is not readable: {path.name}") from exc
    if b"\x00" in sample:
        raise ProfileRouterError("binary_file_not_supported", "Binary files are not readable")


def open_workspace(
    profile_ref: str,
    root: str,
    *,
    mode: str = "checkout",
    registry: WorkspaceRegistry | None = None,
) -> WorkspaceMetadata:
    """Open a read-only local workspace through the policy/secret gate."""

    assert_default_tools_are_no_model()
    return (registry or DEFAULT_WORKSPACE_REGISTRY).open(profile_ref, root, mode=mode)


def get_workspace(
    workspace_id: str,
    *,
    registry: WorkspaceRegistry | None = None,
) -> WorkspaceMetadata:
    """Return metadata for an opened workspace without invoking a model."""

    assert_default_tools_are_no_model()
    return (registry or DEFAULT_WORKSPACE_REGISTRY).get(workspace_id)


def close_workspace(
    workspace_id: str,
    *,
    registry: WorkspaceRegistry | None = None,
) -> WorkspaceMetadata:
    """Remove an opened workspace from the server-side registry."""

    assert_default_tools_are_no_model()
    return (registry or DEFAULT_WORKSPACE_REGISTRY).close(workspace_id)


def read_workspace_file(
    workspace_id: str,
    path: str,
    *,
    offset: int | None = 1,
    limit: int | None = MAX_FILE_READ_LINES,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Read a bounded text slice from an opened workspace."""

    assert_default_tools_are_no_model()
    line_offset = _bounded_int(
        offset,
        "offset",
        default=1,
        minimum=1,
        maximum=1_000_000,
    )
    line_limit = _bounded_int(
        limit,
        "limit",
        default=MAX_FILE_READ_LINES,
        minimum=1,
        maximum=MAX_FILE_READ_LINES,
    )
    workspace = (registry or DEFAULT_WORKSPACE_REGISTRY).get(workspace_id)
    resolved_path = Path(resolve_workspace_path(workspace, path))
    _ensure_text_file(resolved_path)

    selected_lines: list[str] = []
    chars = 0
    truncated = False
    try:
        with resolved_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number < line_offset:
                    continue
                if len(selected_lines) >= line_limit:
                    truncated = True
                    break
                remaining_chars = MAX_FILE_READ_CHARS - chars
                if remaining_chars <= 0:
                    truncated = True
                    break
                if len(line) > remaining_chars:
                    selected_lines.append(line[:remaining_chars])
                    truncated = True
                    break
                selected_lines.append(line)
                chars += len(line)
    except OSError as exc:
        raise ProfileRouterError("file_not_readable", f"File is not readable: {path}") from exc

    return {
        "workspace_id": workspace.workspace_id,
        "path": posixpath.normpath(path.strip() or "."),
        "offset": line_offset,
        "limit": line_limit,
        "content": "".join(selected_lines),
        "lines_returned": len(selected_lines),
        "truncated": truncated,
    }


def _workspace_relative_path(workspace: WorkspaceMetadata, path: Path) -> str:
    try:
        return path.relative_to(Path(workspace.root)).as_posix()
    except ValueError as exc:
        raise ProfileRouterError("path_outside_workspace", "path escapes workspace root") from exc


def _iter_search_candidates(
    workspace: WorkspaceMetadata,
    search_root: Path,
    *,
    file_glob: str | None,
):
    if search_root.is_file():
        rel_path = _workspace_relative_path(workspace, search_root)
        if not file_glob or fnmatch.fnmatch(rel_path, file_glob) or fnmatch.fnmatch(search_root.name, file_glob):
            yield rel_path, search_root
        return

    if not search_root.is_dir():
        raise ProfileRouterError("not_a_directory", "Search path must be a file or directory")

    for dirpath, dirnames, filenames in os.walk(search_root, followlinks=False):
        rel_dir = _workspace_relative_path(workspace, Path(dirpath))
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not _is_secret_path(posixpath.join(rel_dir, dirname))
        ]
        for filename in filenames:
            candidate = Path(dirpath) / filename
            rel_path = _workspace_relative_path(workspace, candidate)
            if file_glob and not (
                fnmatch.fnmatch(rel_path, file_glob)
                or fnmatch.fnmatch(filename, file_glob)
            ):
                continue
            yield rel_path, candidate


def search_workspace_files(
    workspace_id: str,
    pattern: str,
    *,
    path: str | None = None,
    file_glob: str | None = None,
    output_mode: str = "content",
    limit: int | None = MAX_FILE_SEARCH_RESULTS,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Search bounded text files inside an opened workspace."""

    assert_default_tools_are_no_model()
    if not isinstance(pattern, str) or not pattern:
        raise ProfileRouterError("invalid_search_pattern", "pattern must be a non-empty string")
    if file_glob is not None and not isinstance(file_glob, str):
        raise ProfileRouterError("invalid_file_glob", "file_glob must be a string")
    if output_mode not in ALLOWED_SEARCH_OUTPUT_MODES:
        raise ProfileRouterError(
            "invalid_output_mode",
            "output_mode must be one of: " + ", ".join(sorted(ALLOWED_SEARCH_OUTPUT_MODES)),
        )
    max_results = _bounded_int(
        limit,
        "limit",
        default=MAX_FILE_SEARCH_RESULTS,
        minimum=1,
        maximum=MAX_FILE_SEARCH_RESULTS,
    )
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ProfileRouterError("invalid_search_pattern", str(exc)) from exc

    workspace = (registry or DEFAULT_WORKSPACE_REGISTRY).get(workspace_id)
    search_path = path if path is not None else "."
    search_root = Path(resolve_workspace_path(workspace, search_path))

    matches: list[dict] = []
    files: list[str] = []
    counts: list[dict] = []
    skipped = {"secret": 0, "binary": 0, "large": 0, "unreadable": 0}
    truncated = False
    normalized_glob = file_glob.strip() if isinstance(file_glob, str) and file_glob.strip() else None

    for rel_path, candidate in _iter_search_candidates(
        workspace,
        search_root,
        file_glob=normalized_glob,
    ):
        try:
            resolved_candidate = Path(resolve_workspace_path(workspace, rel_path))
            _ensure_text_file(resolved_candidate)
        except ProfileRouterError as exc:
            if exc.code == "secret_path_denied":
                skipped["secret"] += 1
                continue
            if exc.code == "binary_file_not_supported":
                skipped["binary"] += 1
                continue
            skipped["unreadable"] += 1
            continue

        try:
            if resolved_candidate.stat().st_size > MAX_FILE_SEARCH_BYTES:
                skipped["large"] += 1
                continue
        except OSError:
            skipped["unreadable"] += 1
            continue

        file_match_count = 0
        try:
            with resolved_candidate.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not compiled.search(line):
                        continue
                    file_match_count += 1
                    if output_mode == "content":
                        matches.append(
                            {
                                "path": rel_path,
                                "line": line_number,
                                "content": line.rstrip("\n")[:MAX_SEARCH_LINE_CHARS],
                            }
                        )
                        if len(matches) >= max_results:
                            truncated = True
                            break
                if output_mode == "content" and truncated:
                    break
        except OSError:
            skipped["unreadable"] += 1
            continue

        if file_match_count:
            if output_mode == "files_only" and rel_path not in files:
                files.append(rel_path)
                if len(files) >= max_results:
                    truncated = True
                    break
            elif output_mode == "count":
                counts.append({"path": rel_path, "count": file_match_count})
                if len(counts) >= max_results:
                    truncated = True
                    break

    payload = {
        "workspace_id": workspace.workspace_id,
        "path": posixpath.normpath(search_path.strip() or "."),
        "pattern": pattern,
        "file_glob": file_glob,
        "output_mode": output_mode,
        "limit": max_results,
        "truncated": truncated,
        "skipped": skipped,
    }
    if output_mode == "content":
        payload["matches"] = matches
    elif output_mode == "files_only":
        payload["files"] = files
    else:
        payload["counts"] = counts
    return payload


def _safe_profile_summary(
    info: ProfileInfo,
    *,
    route_policy: ProfileRoutePolicy | None = None,
    router_policy: ProfileRouterPolicy | None = None,
) -> dict:
    """Convert ``ProfileInfo`` to public, non-secret router metadata."""

    summary = {
        "profile_ref": f"{LOCAL_HOST}:{info.name}",
        "host": LOCAL_HOST,
        "profile": info.name,
        "is_default": bool(info.is_default),
        "gateway_running": bool(info.gateway_running),
        "skill_count": int(info.skill_count or 0),
        "description": info.description or "",
    }
    if route_policy is not None and router_policy is not None:
        summary["display_name"] = route_policy.display_name or info.name
        if route_policy.description:
            summary["description"] = route_policy.description
        summary["policy"] = {
            "enabled": router_policy.is_profile_exposed(route_policy),
            "allowed_tool_groups": list(route_policy.allowed_tool_groups),
            "capabilities": {
                "filesystem_read": route_policy.allow_filesystem_read,
                "filesystem_write": route_policy.allow_filesystem_write,
                "terminal": route_policy.allow_terminal,
                "messaging": route_policy.allow_messaging,
                "cron": route_policy.allow_cron,
                "git_push": route_policy.allow_git_push,
                "deploy": route_policy.allow_deploy,
            },
            "messaging_recipients_configured": bool(
                route_policy.messaging_allowed_recipients
            ),
            "model_policy": {
                "allow_model_tools": route_policy.allow_model_tools,
                "allowed_cost_classes": list(route_policy.allowed_cost_classes),
            },
        }
    return summary


def list_local_profiles(
    *,
    active_only: bool = True,
    policy: ProfileRouterPolicy | None = None,
) -> list[dict]:
    """Return read-only summaries for policy-enabled local profiles."""

    assert_default_tools_are_no_model()
    router_policy = policy or load_profile_router_policy()
    infos = {info.name: info for info in _list_local_profile_infos()}
    summaries: list[dict] = []
    for route_policy in router_policy.iter_profiles(host=LOCAL_HOST, active_only=active_only):
        info = infos.get(route_policy.ref.profile)
        if info is None:
            continue
        summaries.append(
            _safe_profile_summary(
                info,
                route_policy=route_policy,
                router_policy=router_policy,
            )
        )
    return summaries


def get_local_profile(profile_ref: str) -> dict:
    """Return one local profile summary, or raise ``ProfileRouterError``."""

    assert_default_tools_are_no_model()
    ref = parse_profile_ref(profile_ref)
    if ref.host != LOCAL_HOST:
        raise ProfileRouterError(
            "unsupported_host",
            f"Only local profiles are supported by this skeleton, got: {ref.host}",
        )

    router_policy = load_profile_router_policy()
    router_policy.get_profile_policy(ref)
    for profile in list_local_profiles(active_only=False, policy=router_policy):
        if profile["profile"] == ref.profile:
            return profile

    raise ProfileRouterError("profile_not_found", f"Profile not found: {ref.value}")


def _tool_envelope(tool_name: str, payload: dict) -> str:
    meta = ROUTER_TOOL_METADATA[tool_name]
    data = {
        **payload,
        "cost_class": meta.cost_class,
        "llm_calls": meta.llm_calls,
    }
    return json.dumps(data, indent=2, sort_keys=True)


def _tool_error(tool_name: str, exc: ProfileRouterError) -> str:
    return _tool_envelope(
        tool_name,
        {
            "ok": False,
            "error": {"code": exc.code, "message": exc.message},
        },
    )


def profiles_list(active_only: bool = True) -> str:
    """MCP-ready wrapper: list local profiles without invoking any model."""

    try:
        return _tool_envelope(
            "profiles_list",
            {"ok": True, "profiles": list_local_profiles(active_only=active_only)},
        )
    except ProfileRouterError as exc:
        return _tool_error("profiles_list", exc)


def profile_get(profile_ref: str) -> str:
    """MCP-ready wrapper: get one local profile without invoking any model."""

    try:
        return _tool_envelope(
            "profile_get",
            {"ok": True, "profile": get_local_profile(profile_ref)},
        )
    except ProfileRouterError as exc:
        return _tool_error("profile_get", exc)


def profile_health(profile_ref: str) -> str:
    """MCP-ready wrapper: health summary for one local profile."""

    try:
        profile = get_local_profile(profile_ref)
        return _tool_envelope(
            "profile_health",
            {
                "ok": True,
                "profile_ref": profile["profile_ref"],
                "health": {
                    "status": "ok",
                    "host": profile["host"],
                    "profile": profile["profile"],
                    "gateway_running": profile["gateway_running"],
                },
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("profile_health", exc)


def workspace_open(profile_ref: str, root: str, mode: str = "checkout") -> str:
    """MCP-ready wrapper: open a read-only workspace without invoking a model."""

    try:
        workspace = open_workspace(profile_ref, root, mode=mode)
        return _tool_envelope(
            "workspace_open",
            {"ok": True, "workspace": _public_workspace_dict(workspace)},
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_open", exc)


def workspace_get(workspace_id: str) -> str:
    """MCP-ready wrapper: inspect an opened workspace by opaque ID."""

    try:
        workspace = get_workspace(workspace_id)
        return _tool_envelope(
            "workspace_get",
            {"ok": True, "workspace": _public_workspace_dict(workspace)},
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_get", exc)


def workspace_close(workspace_id: str) -> str:
    """MCP-ready wrapper: close an opened workspace registry entry."""

    try:
        workspace = close_workspace(workspace_id)
        return _tool_envelope(
            "workspace_close",
            {
                "ok": True,
                "closed": True,
                "workspace": _public_workspace_dict(workspace),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_close", exc)


def file_read(
    workspace_id: str,
    path: str,
    offset: int | None = 1,
    limit: int | None = MAX_FILE_READ_LINES,
) -> str:
    """MCP-ready wrapper: read a bounded text slice from a workspace."""

    try:
        return _tool_envelope(
            "file_read",
            {
                "ok": True,
                "file": read_workspace_file(
                    workspace_id,
                    path,
                    offset=offset,
                    limit=limit,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("file_read", exc)


def file_search(
    workspace_id: str,
    pattern: str,
    path: str | None = None,
    file_glob: str | None = None,
    output_mode: str = "content",
    limit: int | None = MAX_FILE_SEARCH_RESULTS,
) -> str:
    """MCP-ready wrapper: search bounded text files in a workspace."""

    try:
        return _tool_envelope(
            "file_search",
            {
                "ok": True,
                "search": search_workspace_files(
                    workspace_id,
                    pattern,
                    path=path,
                    file_glob=file_glob,
                    output_mode=output_mode,
                    limit=limit,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("file_search", exc)
