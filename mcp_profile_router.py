"""No-model MCP profile router primitives.

This module is intentionally small and side-effect free: it only parses
fully-qualified profile refs and exposes read-only local profile inventory for
the ChatGPT ↔ Hermes MCP router.  It must not import or call the Hermes agent
loop, provider clients, or AI coding CLIs.  Later filesystem/terminal tools
should build on the explicit cost metadata here rather than exposing arbitrary
Hermes tools by default.
"""

from __future__ import annotations

import json
import posixpath
from collections.abc import Iterable as IterableABC, Mapping as MappingABC
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

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
