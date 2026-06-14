"""Generic runtime guard interface for gateway delivery paths.

The module is intentionally provider-neutral and disabled by default. It gives
future gateway insertion points a single policy surface without changing any
adapter behavior until a config explicitly enables a scoped provider.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


_STREAMING_POLICIES = frozenset({"allow", "disable", "guard_first_visible"})
_SURFACE_ACTIONS = frozenset({"allow", "guard", "disable", "block"})
_SENSITIVE_AUDIT_KEY_PARTS = (
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_MISSING = object()


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _normalize_text(value: Any, *, lower: bool = False) -> str:
    raw = _enum_value(value)
    if raw is None:
        return ""
    text = str(raw).strip()
    return text.lower() if lower else text


def _normalize_tuple(values: Any, *, lower: bool = False) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str) or not isinstance(values, (list, tuple, set, frozenset)):
        values = (values,)
    normalized = []
    for value in values:
        text = _normalize_text(value, lower=lower)
        if text:
            normalized.append(text)
    return tuple(normalized)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default
    return bool(value)


def _coerce_decision_allowed(value: Any) -> bool | None:
    if value is _MISSING:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _normalize_streaming_policy(value: Any) -> str:
    policy = _normalize_text(value, lower=True)
    return policy if policy in _STREAMING_POLICIES else "disable"


def _normalize_surface_action(value: Any, default: str) -> str:
    action = _normalize_text(value, lower=True)
    return action if action in _SURFACE_ACTIONS else default


def _mapping_from_config(config: Any) -> Mapping[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        if isinstance(config.get("runtime_guard"), Mapping):
            return config["runtime_guard"]
        return config
    runtime_guard = getattr(config, "runtime_guard", None)
    if isinstance(runtime_guard, Mapping):
        return runtime_guard
    return {}


def _sanitize_audit(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if any(part in lowered for part in _SENSITIVE_AUDIT_KEY_PARTS):
                continue
            sanitized[key_text] = _sanitize_audit(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_audit(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_audit(item) for item in value)
    return value


@dataclass(frozen=True)
class RuntimeGuardScope:
    """Scope selector for guarded gateway sessions.

    Empty dimensions are non-restrictive. Non-empty dimensions must all match.
    """

    platforms: tuple[str, ...] = ()
    chat_ids: tuple[str, ...] = ()
    thread_ids: tuple[str, ...] = ()
    parent_chat_ids: tuple[str, ...] = ()
    session_keys: tuple[str, ...] = ()
    guild_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "platforms", _normalize_tuple(self.platforms, lower=True))
        object.__setattr__(self, "chat_ids", _normalize_tuple(self.chat_ids))
        object.__setattr__(self, "thread_ids", _normalize_tuple(self.thread_ids))
        object.__setattr__(self, "parent_chat_ids", _normalize_tuple(self.parent_chat_ids))
        object.__setattr__(self, "session_keys", _normalize_tuple(self.session_keys))
        object.__setattr__(self, "guild_ids", _normalize_tuple(self.guild_ids))

    @classmethod
    def from_mapping(cls, data: Any) -> "RuntimeGuardScope":
        if isinstance(data, RuntimeGuardScope):
            return data
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            platforms=_normalize_tuple(data.get("platforms"), lower=True),
            chat_ids=_normalize_tuple(data.get("chat_ids")),
            thread_ids=_normalize_tuple(data.get("thread_ids")),
            parent_chat_ids=_normalize_tuple(data.get("parent_chat_ids")),
            session_keys=_normalize_tuple(data.get("session_keys")),
            guild_ids=_normalize_tuple(data.get("guild_ids")),
        )

    def matches(self, context: "GuardContext") -> bool:
        checks = (
            (self.platforms, context.normalized_platform),
            (self.chat_ids, _normalize_text(context.chat_id)),
            (self.thread_ids, _normalize_text(context.thread_id)),
            (self.parent_chat_ids, _normalize_text(context.parent_chat_id)),
            (self.session_keys, _normalize_text(context.session_key)),
            (self.guild_ids, _normalize_text(context.guild_id)),
        )
        return all(not allowed or value in allowed for allowed, value in checks)


@dataclass(frozen=True)
class RuntimeGuardStreamingConfig:
    policy: str = "disable"

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", _normalize_streaming_policy(self.policy))

    @classmethod
    def from_mapping(cls, data: Any) -> "RuntimeGuardStreamingConfig":
        if isinstance(data, RuntimeGuardStreamingConfig):
            return data
        if not isinstance(data, Mapping):
            return cls()
        return cls(policy=data.get("policy", "disable"))


@dataclass(frozen=True)
class RuntimeGuardDeliveryPolicy:
    assistant_final: str = "guard"
    assistant_stream: str = "disable"
    assistant_interim: str = "disable"
    delivery_router: str = "allow"
    tool_progress: str = "disable"
    command_ack: str = "allow"
    interaction_ack: str = "allow"
    send_message_tool: str = "block"
    send_message_reaction: str = "block"
    cron_delivery: str = "allow"
    kanban_notification: str = "allow"
    process_notification: str = "allow"

    def __post_init__(self) -> None:
        for field_info in dataclasses.fields(self):
            default = field_info.default
            value = getattr(self, field_info.name)
            object.__setattr__(
                self,
                field_info.name,
                _normalize_surface_action(value, str(default)),
            )

    @classmethod
    def from_mapping(cls, data: Any) -> "RuntimeGuardDeliveryPolicy":
        if isinstance(data, RuntimeGuardDeliveryPolicy):
            return data
        if not isinstance(data, Mapping):
            return cls()
        defaults = cls()
        values = {}
        for field_info in dataclasses.fields(cls):
            default = getattr(defaults, field_info.name)
            values[field_info.name] = _normalize_surface_action(
                data.get(field_info.name, default),
                default,
            )
        return cls(**values)

    def policy_for(self, surface: str) -> str:
        normalized = _normalize_text(surface, lower=True)
        if not normalized:
            return "allow"
        return getattr(self, normalized, "allow")


@dataclass(frozen=True)
class RuntimeGuardConfig:
    enabled: bool = False
    provider: str = "noop"
    dry_run: bool = True
    fail_closed: bool = True
    scope: RuntimeGuardScope = field(default_factory=RuntimeGuardScope)
    streaming: RuntimeGuardStreamingConfig = field(default_factory=RuntimeGuardStreamingConfig)
    delivery_surfaces: RuntimeGuardDeliveryPolicy = field(default_factory=RuntimeGuardDeliveryPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _coerce_bool(self.enabled, False))
        object.__setattr__(self, "provider", _normalize_text(self.provider, lower=True) or "noop")
        object.__setattr__(self, "dry_run", _coerce_bool(self.dry_run, True))
        object.__setattr__(self, "fail_closed", _coerce_bool(self.fail_closed, True))
        object.__setattr__(self, "scope", RuntimeGuardScope.from_mapping(self.scope))
        object.__setattr__(
            self,
            "streaming",
            RuntimeGuardStreamingConfig.from_mapping(self.streaming),
        )
        object.__setattr__(
            self,
            "delivery_surfaces",
            RuntimeGuardDeliveryPolicy.from_mapping(self.delivery_surfaces),
        )

    @classmethod
    def from_mapping(cls, data: Any) -> "RuntimeGuardConfig":
        if isinstance(data, RuntimeGuardConfig):
            return data
        config = _mapping_from_config(data)
        return cls(
            enabled=_coerce_bool(config.get("enabled"), False),
            provider=config.get("provider", "noop"),
            dry_run=_coerce_bool(config.get("dry_run"), True),
            fail_closed=_coerce_bool(config.get("fail_closed"), True),
            scope=RuntimeGuardScope.from_mapping(config.get("scope")),
            streaming=RuntimeGuardStreamingConfig.from_mapping(config.get("streaming")),
            delivery_surfaces=RuntimeGuardDeliveryPolicy.from_mapping(
                config.get("delivery_surfaces"),
            ),
        )


@dataclass(frozen=True)
class GuardContext:
    surface: str
    platform: Any = None
    chat_id: Any = None
    thread_id: Any = None
    parent_chat_id: Any = None
    guild_id: Any = None
    user_id: Any = None
    message_id: Any = None
    session_key: Any = None
    chat_type: str | None = None
    is_internal: bool = False
    command: str | None = None
    metadata: Mapping[str, Any] | None = None
    idempotency_key: str | None = None

    @property
    def normalized_surface(self) -> str:
        return _normalize_text(self.surface, lower=True)

    @property
    def normalized_platform(self) -> str:
        return _normalize_text(self.platform, lower=True)

    def with_surface(self, surface: str) -> "GuardContext":
        return dataclasses.replace(self, surface=surface)


@dataclass
class GuardDecision:
    allowed: bool
    reason: str = ""
    status: str = "allowed"
    dry_run: bool = False
    fail_closed: bool = False
    provider: str = ""
    surface: str = ""
    audit: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.audit = _sanitize_audit(self.audit or {})

    @classmethod
    def allow(
        cls,
        *,
        reason: str = "allowed",
        status: str = "allowed",
        provider: str = "",
        surface: str = "",
        dry_run: bool = False,
        fail_closed: bool = False,
        audit: Mapping[str, Any] | None = None,
    ) -> "GuardDecision":
        return cls(
            allowed=True,
            reason=reason,
            status=status,
            provider=provider,
            surface=surface,
            dry_run=dry_run,
            fail_closed=fail_closed,
            audit=dict(audit or {}),
        )

    @classmethod
    def block(
        cls,
        *,
        reason: str = "blocked",
        status: str = "blocked",
        provider: str = "",
        surface: str = "",
        fail_closed: bool = False,
        audit: Mapping[str, Any] | None = None,
    ) -> "GuardDecision":
        return cls(
            allowed=False,
            reason=reason,
            status=status,
            provider=provider,
            surface=surface,
            fail_closed=fail_closed,
            audit=dict(audit or {}),
        )


@runtime_checkable
class RuntimeLeaseGuard(Protocol):
    def check(self, context: GuardContext) -> GuardDecision:
        ...


class NoopRuntimeLeaseGuard:
    """Provider used when the runtime guard is disabled or unconfigured."""

    def check(self, context: GuardContext) -> GuardDecision:
        return GuardDecision.allow(
            reason="noop_runtime_guard",
            status="allowed",
            provider="noop",
            surface=context.normalized_surface,
        )


RuntimeGuardProviderFactory = Callable[[RuntimeGuardConfig], RuntimeLeaseGuard]
_runtime_guard_providers: dict[str, RuntimeLeaseGuard | RuntimeGuardProviderFactory | Any] = {
    "noop": NoopRuntimeLeaseGuard(),
}


def register_runtime_guard_provider(name: str, provider_factory_or_instance: Any) -> None:
    """Register a runtime guard provider by name.

    Providers are process-local and intentionally generic: a value may be an
    object with ``check(context)`` or a factory returning such an object.
    """

    normalized = _normalize_text(name, lower=True)
    if not normalized:
        raise ValueError("runtime guard provider name is required")
    if provider_factory_or_instance is None:
        raise ValueError("runtime guard provider is required")
    _runtime_guard_providers[normalized] = provider_factory_or_instance


class RuntimeGuardManager:
    def __init__(self, config: RuntimeGuardConfig | Mapping[str, Any] | Any | None = None):
        self.config = RuntimeGuardConfig.from_mapping(config)

    def is_scoped(self, context: GuardContext) -> bool:
        if not self.config.enabled:
            return False
        return self.config.scope.matches(context)

    def check(self, context: GuardContext) -> GuardDecision:
        if not self.config.enabled:
            return self._allow(context, reason="runtime_guard_disabled", status="disabled")
        if not self.is_scoped(context):
            return self._allow(context, reason="runtime_guard_out_of_scope", status="out_of_scope")

        try:
            provider = self._resolve_provider()
            raw_decision = provider.check(context)
            decision = self._coerce_decision(raw_decision, context)
        except Exception as exc:
            return self._provider_error_decision(context, exc)

        if decision.allowed:
            return decision
        if self.config.dry_run:
            return self._dry_run_allow(context, decision)
        return decision

    def should_disable_streaming(self, context: GuardContext) -> bool:
        if not self.config.enabled or self.config.dry_run or not self.is_scoped(context):
            return False
        return self.config.streaming.policy == "disable"

    def requires_first_visible_stream_guard(self, context: GuardContext) -> bool:
        if not self.config.enabled or not self.is_scoped(context):
            return False
        return self.config.streaming.policy == "guard_first_visible"

    def check_first_visible_stream(self, context: GuardContext) -> GuardDecision:
        stream_context = context.with_surface("assistant_stream")
        if not self.requires_first_visible_stream_guard(stream_context):
            return self._allow(
                stream_context,
                reason="stream_guard_not_required",
                status="stream_guard_not_required",
            )
        return self.check(stream_context)

    def surface_action(self, context_or_surface: GuardContext | str) -> str:
        surface = (
            context_or_surface.surface
            if isinstance(context_or_surface, GuardContext)
            else context_or_surface
        )
        return self.config.delivery_surfaces.policy_for(str(surface))

    def should_guard_surface(self, context: GuardContext) -> bool:
        return self.config.enabled and self.is_scoped(context) and self.surface_action(context) == "guard"

    def should_block_surface(self, context: GuardContext) -> bool:
        if not self.config.enabled or not self.is_scoped(context):
            return False
        return self.surface_action(context) == "block"

    def should_disable_surface(self, context: GuardContext) -> bool:
        if not self.config.enabled or not self.is_scoped(context):
            return False
        return self.surface_action(context) == "disable"

    def check_surface_policy(self, context: GuardContext) -> GuardDecision:
        if not self.config.enabled:
            return self._allow(context, reason="runtime_guard_disabled", status="disabled")
        if not self.is_scoped(context):
            return self._allow(context, reason="runtime_guard_out_of_scope", status="out_of_scope")

        action = self.surface_action(context)
        if action == "allow":
            return self._allow(context, reason="surface_policy_allow", status="surface_allowed")
        if action == "guard":
            return self.check(context)
        if action in {"block", "disable"}:
            decision = GuardDecision.block(
                reason=f"surface_policy_{action}",
                status="surface_blocked" if action == "block" else "surface_disabled",
                provider=self.config.provider,
                surface=context.normalized_surface,
                fail_closed=self.config.fail_closed,
            )
            if self.config.dry_run:
                return self._dry_run_allow(context, decision)
            return decision
        return self._allow(context, reason="surface_policy_allow", status="surface_allowed")

    def _resolve_provider(self) -> RuntimeLeaseGuard:
        entry = _runtime_guard_providers.get(self.config.provider)
        if entry is None:
            raise RuntimeError(f"runtime guard provider unavailable: {self.config.provider}")
        if not isinstance(entry, type) and hasattr(entry, "check"):
            return entry
        if callable(entry):
            provider = self._call_provider_factory(entry)
            if hasattr(provider, "check"):
                return provider
        raise RuntimeError(f"runtime guard provider invalid: {self.config.provider}")

    def _call_provider_factory(self, factory: Callable[..., Any]) -> Any:
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            return factory(self.config)

        required_positional = [
            param
            for param in signature.parameters.values()
            if param.default is inspect.Parameter.empty
            and param.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if required_positional:
            return factory(self.config)
        return factory()

    def _coerce_decision(self, raw_decision: Any, context: GuardContext) -> GuardDecision:
        if isinstance(raw_decision, GuardDecision):
            return GuardDecision(
                allowed=raw_decision.allowed,
                reason=raw_decision.reason,
                status=raw_decision.status,
                dry_run=raw_decision.dry_run,
                fail_closed=raw_decision.fail_closed,
                provider=raw_decision.provider or self.config.provider,
                surface=raw_decision.surface or context.normalized_surface,
                audit=raw_decision.audit,
            )
        if isinstance(raw_decision, Mapping):
            allowed = _coerce_decision_allowed(raw_decision.get("allowed", _MISSING))
            if allowed is None:
                raise RuntimeError(
                    f"provider_invalid_allowed: {type(raw_decision.get('allowed')).__name__}"
                )
            return GuardDecision(
                allowed=allowed,
                reason=str(raw_decision.get("reason", "")),
                status=str(raw_decision.get("status", "allowed")),
                dry_run=_coerce_bool(raw_decision.get("dry_run"), False),
                fail_closed=_coerce_bool(raw_decision.get("fail_closed"), False),
                provider=str(raw_decision.get("provider") or self.config.provider),
                surface=str(raw_decision.get("surface") or context.normalized_surface),
                audit=dict(raw_decision.get("audit") or {}),
            )
        if isinstance(raw_decision, bool):
            if raw_decision:
                return self._allow(context, reason="provider_allowed", status="allowed")
            return GuardDecision.block(
                reason="provider_denied",
                status="denied",
                provider=self.config.provider,
                surface=context.normalized_surface,
            )
        if raw_decision is None:
            return self._allow(context, reason="provider_no_decision", status="allowed")
        raise RuntimeError(f"runtime guard provider returned unsupported decision: {type(raw_decision)!r}")

    def _provider_error_decision(self, context: GuardContext, exc: Exception) -> GuardDecision:
        reason = f"runtime_guard_provider_error: {exc}"
        if self.config.fail_closed:
            return GuardDecision.block(
                reason=reason,
                status="provider_error",
                provider=self.config.provider,
                surface=context.normalized_surface,
                fail_closed=True,
            )
        return GuardDecision.allow(
            reason=reason,
            status="provider_error_allowed",
            provider=self.config.provider,
            surface=context.normalized_surface,
            fail_closed=False,
        )

    def _allow(self, context: GuardContext, *, reason: str, status: str) -> GuardDecision:
        return GuardDecision.allow(
            reason=reason,
            status=status,
            provider=self.config.provider,
            surface=context.normalized_surface,
            fail_closed=self.config.fail_closed,
        )

    def _dry_run_allow(self, context: GuardContext, blocked: GuardDecision) -> GuardDecision:
        return GuardDecision.allow(
            reason=f"dry_run_would_block: {blocked.reason}",
            status="dry_run_allowed",
            provider=blocked.provider or self.config.provider,
            surface=blocked.surface or context.normalized_surface,
            dry_run=True,
            fail_closed=blocked.fail_closed or self.config.fail_closed,
            audit={
                "would_block": True,
                "original_allowed": blocked.allowed,
                "original_status": blocked.status,
                "original_reason": blocked.reason,
                "original_audit": blocked.audit,
            },
        )


def get_runtime_guard_manager(config: RuntimeGuardConfig | Mapping[str, Any] | Any | None = None) -> RuntimeGuardManager:
    return RuntimeGuardManager(config)


def check_delivery_surface_policy(
    config: RuntimeGuardConfig | Mapping[str, Any] | Any | None,
    surface: str,
    *,
    platform: Any = None,
    chat_id: Any = None,
    thread_id: Any = None,
    parent_chat_id: Any = None,
    guild_id: Any = None,
    user_id: Any = None,
    message_id: Any = None,
    session_key: Any = None,
    chat_type: str | None = None,
    is_internal: bool = False,
    command: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> GuardDecision:
    """Evaluate a delivery surface without depending on live adapters.

    This is a pure policy/classification helper for local delivery paths such
    as the send_message tool, cron delivery, the delivery router, and kanban
    notifications. A missing or disabled runtime_guard config remains a no-op.
    """

    context = GuardContext(
        surface=surface,
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        parent_chat_id=parent_chat_id,
        guild_id=guild_id,
        user_id=user_id,
        message_id=message_id,
        session_key=session_key,
        chat_type=chat_type,
        is_internal=is_internal,
        command=command,
        metadata=metadata,
        idempotency_key=idempotency_key,
    )
    return get_runtime_guard_manager(config).check_surface_policy(context)
