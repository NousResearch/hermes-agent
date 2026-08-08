"""Contracts and validation for opt-in same-session cron jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

SESSION_TARGET_ISOLATED = "isolated"
SESSION_TARGET_CURRENT = "current"
SESSION_TARGETS = frozenset({SESSION_TARGET_ISOLATED, SESSION_TARGET_CURRENT})
CONTEXTUAL_BINDING_VERSION_LOGICAL_ROUTE = 2


@dataclass(frozen=True)
class ContextualExecutionPolicy:
    """Single fail-closed capability policy for unattended live-session turns."""

    internal: bool = True
    suppress_transport_output: bool = True
    hidden_user_entry: bool = True
    touch_human_activity: bool = False
    consume_interactive_state: bool = False
    async_delivery: bool = False
    mirror_delivery: bool = False
    allow_proxy: bool = False
    use_agent_cache: bool = False
    persist_agent_directly: bool = False
    # V1 is intentionally model-only.  Name-only capability filtering cannot
    # authenticate a registry handler, and even nominally safe mutation/search
    # tools can cross the admitted conversation boundary.  Keep this empty
    # until handlers can be identity-pinned and arguments sandboxed centrally.
    allowed_tool_names: frozenset[str] = frozenset()


CONTEXTUAL_EXECUTION_POLICY = ContextualExecutionPolicy()
CONTEXTUAL_ALLOWED_TOOLSETS: tuple[str, ...] = ()
CONTEXTUAL_DISABLED_TOOLSETS = (
    "browser",
    "clarify",
    "code_execution",
    "computer_use",
    "cronjob",
    "delegation",
    "discord",
    "discord_admin",
    "homeassistant",
    "image_gen",
    "kanban",
    "memory",
    "messaging",
    "project",
    "skills",
    "spotify",
    "terminal",
    "todo",
    "tts",
    "video_gen",
    "yuanbao",
)


def normalize_session_target(value: Any) -> str:
    target = str(value or SESSION_TARGET_ISOLATED).strip().lower()
    if target not in SESSION_TARGETS:
        raise ValueError("session_target must be 'isolated' or 'current'")
    return target


def capture_current_session_key() -> str:
    """Read the trusted task-local key; callers cannot supply an arbitrary key."""
    return capture_current_session_binding()["session_key"]


def capture_current_session_binding() -> Dict[str, Any]:
    """Capture one canonical live-gateway binding without process-env fallback."""
    from gateway.session_context import (
        NON_MESSAGING_SESSION_SURFACES,
        get_bound_session_context,
    )

    binding = get_bound_session_context()
    if binding is None:
        raise ValueError(
            "session_target='current' requires a gateway-bound messaging session; "
            "process environment values are not trusted"
        )
    platform = binding["platform"].strip().lower()
    if platform in NON_MESSAGING_SESSION_SURFACES:
        raise ValueError(
            "session_target='current' requires a gateway-bound messaging session"
        )
    route_instance_id = str(binding.get("route_instance_id") or "").strip()
    if not route_instance_id:
        raise ValueError(
            "session_target='current' requires a concrete authenticated logical route"
        )
    route_principal = binding.get("route_principal")
    if not isinstance(route_principal, dict):
        raise ValueError(
            "session_target='current' requires a trusted logical route principal"
        )
    return {
        "profile": binding.get("profile", ""),
        "session_key": binding["session_key"],
        "route_instance_id": route_instance_id,
        "platform": binding["platform"],
        "chat_type": binding.get("chat_type", ""),
        "chat_id": binding["chat_id"],
        "thread_id": binding.get("thread_id", ""),
        "user_id": binding["user_id"],
        "scope_id": str(route_principal.get("scope_id") or ""),
        "parent_chat_id": str(route_principal.get("parent_chat_id") or ""),
        "user_id_alt": str(route_principal.get("user_id_alt") or ""),
        "chat_id_alt": str(route_principal.get("chat_id_alt") or ""),
    }


def validate_contextual_origin(origin: Any) -> Dict[str, Any]:
    """Validate private immutable creator authority before an external effect."""
    if not isinstance(origin, dict):
        raise ValueError("Contextual creator authority is missing.")
    normalized = dict(origin)
    for field in ("platform", "chat_type", "chat_id", "user_id"):
        raw_value = normalized.get(field)
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError("Contextual creator authority is incomplete.")
        normalized[field] = raw_value.strip()
    for field in (
        "profile",
        "thread_id",
        "scope_id",
        "parent_chat_id",
        "user_id_alt",
        "chat_id_alt",
    ):
        raw_value = normalized.get(field)
        if raw_value is not None and not isinstance(raw_value, str):
            raise ValueError("Contextual creator authority is invalid.")
    normalized["profile"] = str(normalized.get("profile") or "")
    return normalized


def contextual_origin_from_binding(binding: Dict[str, Any]) -> Dict[str, Any]:
    """Derive execution/delivery origin from the exact persisted binding."""
    return validate_contextual_origin({
        "platform": str(binding.get("platform") or ""),
        "chat_type": str(binding.get("chat_type") or ""),
        "chat_id": str(binding.get("chat_id") or ""),
        "thread_id": str(binding.get("thread_id") or "") or None,
        "user_id": str(binding.get("user_id") or ""),
        "profile": str(binding.get("profile") or ""),
        "scope_id": str(binding.get("scope_id") or "") or None,
        "parent_chat_id": str(binding.get("parent_chat_id") or "") or None,
        "user_id_alt": str(binding.get("user_id_alt") or "") or None,
        "chat_id_alt": str(binding.get("chat_id_alt") or "") or None,
    })


def _has(job: Dict[str, Any], field: str) -> bool:
    value = job.get(field)
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return value not in (None, "", False)


def _validate_contextual_scheduler_provider() -> None:
    """V1 contextual turns require the gateway-owned in-process dispatcher."""
    try:
        from hermes_cli.config import cfg_get, load_config

        configured = str(
            cfg_get(load_config(), "cron", "provider", default="") or ""
        ).strip().lower()
    except Exception as exc:
        raise ValueError(
            "session_target='current' cannot verify the configured cron provider"
        ) from exc

    if configured not in {"", "builtin", "in-process", "inprocess"}:
        raise ValueError(
            "session_target='current' requires cron.provider='builtin'; "
            f"configured provider {configured!r} cannot preserve gateway-bound authority"
        )


def contextual_definition_route(job: Mapping[str, Any]) -> tuple[str, int]:
    """Return a legacy-v1 immutable physical route binding."""
    session_key = str(job.get("session_key") or "").strip()
    binding = job.get("context_binding")
    if not session_key or not isinstance(binding, Mapping):
        raise ValueError(
            "session_target='current' requires an immutable session binding"
        )
    if str(binding.get("session_key") or "").strip() != session_key:
        raise ValueError(
            "session_target='current' has a mismatched immutable session binding"
        )
    session_id = str(binding.get("session_id") or "").strip()
    routing_revision = binding.get("routing_revision")
    if (
        not session_id
        or isinstance(routing_revision, bool)
        or not isinstance(routing_revision, int)
        or routing_revision < 0
    ):
        raise ValueError(
            "session_target='current' requires an immutable session-id and "
            "routing-revision binding"
        )
    return session_id, routing_revision


def contextual_definition_route_instance(job: Mapping[str, Any]) -> str:
    """Return the immutable logical-route instance captured by a v2 definition."""
    session_key = str(job.get("session_key") or "").strip()
    binding = job.get("context_binding")
    if not session_key or not isinstance(binding, Mapping):
        raise ValueError(
            "session_target='current' requires an immutable logical route binding"
        )
    if str(binding.get("session_key") or "").strip() != session_key:
        raise ValueError(
            "session_target='current' has a mismatched immutable logical route binding"
        )
    route_instance_id = str(binding.get("route_instance_id") or "").strip()
    if not route_instance_id:
        raise ValueError(
            "session_target='current' requires an immutable route-instance binding"
        )
    for field in ("platform", "chat_id", "user_id"):
        if not str(binding.get(field) or "").strip():
            raise ValueError(
                "session_target='current' requires an immutable authenticated "
                f"logical route field: {field}"
            )
    return route_instance_id


def validate_contextual_job_shape(job: Dict[str, Any]) -> None:
    """Fail closed on settings whose runtime semantics diverge from live chat."""
    if normalize_session_target(job.get("session_target")) != SESSION_TARGET_CURRENT:
        return

    _validate_contextual_scheduler_provider()
    version = int(job.get("_contextual_binding_version") or 1)
    if version == CONTEXTUAL_BINDING_VERSION_LOGICAL_ROUTE:
        contextual_definition_route_instance(job)
    elif version == 1:
        contextual_definition_route(job)
    else:
        raise ValueError(
            f"unsupported contextual binding version: {version}"
        )

    conflicts: list[str] = []
    if bool(job.get("no_agent")):
        conflicts.append("no_agent")
    if _has(job, "script"):
        conflicts.append("script")
    if _has(job, "skills") or _has(job, "skill"):
        conflicts.append("skills")
    for field in (
        "workdir",
        "context_from",
        "enabled_toolsets",
        "model",
        "provider",
        "base_url",
        "monitor_script",
        "monitor_url",
    ):
        if _has(job, field):
            conflicts.append(field)
    if job.get("attach_to_session") is True:
        conflicts.append("attach_to_session")

    deliver = str(job.get("deliver") or "origin").strip().lower()
    if deliver not in {"origin", "local"}:
        conflicts.append("deliver")

    if conflicts:
        unique = ", ".join(dict.fromkeys(conflicts))
        raise ValueError(
            "session_target='current' is incompatible with: "
            f"{unique}. Contextual jobs must use the live session's model, "
            "tools, profile, source, and single-origin delivery."
        )


def contextual_live_tool_policy(enabled, disabled):
    """Return the exact unattended toolset allowlist and defensive denylist."""
    del enabled  # Current platform bundles are intentionally not inherited.
    denied = list(disabled or [])
    for name in CONTEXTUAL_DISABLED_TOOLSETS:
        if name not in denied:
            denied.append(name)
    return list(CONTEXTUAL_ALLOWED_TOOLSETS), denied


def contextual_allowed_tool_names() -> frozenset[str]:
    return CONTEXTUAL_EXECUTION_POLICY.allowed_tool_names


def filter_contextual_tool_schemas(schemas: Iterable[Any]) -> list[dict]:
    """Final-schema capability gate; unknown/plugin tools fail closed."""
    allowed = contextual_allowed_tool_names()
    result: list[dict] = []
    for schema in schemas or ():
        if not isinstance(schema, dict):
            continue
        function = schema.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name in allowed:
            result.append(schema)
    return result


def contextual_fields_for_write(
    target: Any,
    *,
    current_session_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Return canonical persisted fields, clearing stale keys for isolation."""
    normalized = normalize_session_target(target)
    if normalized == SESSION_TARGET_ISOLATED:
        return {}
    # Deliberately ignore any job/model-provided key.  Only the trusted
    # task-local capture (or an explicitly injected test value) is accepted.
    binding = capture_current_session_binding()
    key = str(current_session_key or binding["session_key"]).strip()
    if not key:
        raise ValueError("session_target='current' requires a non-empty live session key")
    if current_session_key is not None:
        binding = {**binding, "session_key": key}
    return {
        "session_target": normalized,
        "session_key": key,
        "context_binding": binding,
        "_contextual_binding_version": CONTEXTUAL_BINDING_VERSION_LOGICAL_ROUTE,
    }
