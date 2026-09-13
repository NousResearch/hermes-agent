"""Provider-independent worker profile parsing, discovery, and route resolution.

``delegation.profiles`` maps a profile name to a provider/model pin (plus optional
reasoning_effort, max_iterations, fallback chain). This module is the single place profiles
are parsed, validated, and resolved into an immutable :class:`ProfileRoute` that both
``delegate_task`` and ``SubagentLifecycleService`` consume.

Selection precedence (see docs/plans/2026-09-04-delegation-model-profiles.md):
  per-task profile > top-level profile > ``delegation.default_profile`` > legacy
  ``delegation.provider/model`` (this module returns no profile) > parent inherit.

Design rules honored here:
- Credentials resolve via ``hermes_cli.runtime_provider.resolve_runtime_provider`` — the exact
  path the legacy ``delegation.provider`` branch uses. No duplicated credential logic.
- Explicit invalid profile, model, and effort requests fail before a batch launches.
- Discovery is config-only: it never resolves credentials or probes a provider.
- Profile policy can only narrow the parent and backend capability sets downstream.
- Imports of heavyweight collaborators are lazy (house style, avoids import cycles).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_PROFILE_KEYS = frozenset({
    "description", "instructions", "provider", "model", "reasoning_effort",
    "tool_policy", "workspace_context", "execution_limits", "enabled_routes",
    # Accepted legacy profile fields.
    "max_iterations", "fallback",
})
_FALLBACK_ENTRY_KEYS = frozenset({"provider", "model"})
_ROUTE_KEYS = frozenset({"provider", "model", "reasoning_effort"})
_TOOL_POLICY_KEYS = frozenset({"allowed_toolsets", "blocked_tools", "allowed_tools", "allowed_mcp_tools"})
_WORKSPACE_CONTEXT_KEYS = frozenset({"mode", "include_context_files", "include_memory"})
_EXECUTION_LIMIT_KEYS = frozenset({
    "max_iterations", "timeout_seconds", "max_followups", "max_tool_calls",
    "max_spawn_depth", "max_concurrent_children",
})
_ROUTING_MODES = frozenset({"profile_only", "dynamic"})
_VALID_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"})


@dataclass(frozen=True)
class FallbackTarget:
    """One provider/model pair in a profile's fallback chain."""
    provider: str
    model: str


@dataclass(frozen=True)
class EnabledRoute:
    """One operator-approved route available inside a worker profile."""
    provider: str
    model: str
    reasoning_effort: Optional[str] = None


@dataclass(frozen=True)
class ToolPolicy:
    """Requested tool narrowing. ``None`` means inherit the parent/toolset ceiling."""
    allowed_toolsets: Optional[Tuple[str, ...]] = None
    blocked_tools: Tuple[str, ...] = ()
    allowed_tools: Optional[Tuple[str, ...]] = None
    allowed_mcp_tools: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class WorkspaceContextPolicy:
    """Startup-only context policy; it never mutates an active conversation prompt."""
    mode: str = "inherit"
    include_context_files: Optional[bool] = None
    include_memory: Optional[bool] = None


@dataclass(frozen=True)
class ExecutionLimits:
    """Profile ceilings. Runtime-wide and parent ceilings are applied after these values."""
    max_iterations: Optional[int] = None
    timeout_seconds: Optional[float] = None
    max_followups: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_spawn_depth: Optional[int] = None
    max_concurrent_children: Optional[int] = None


@dataclass(frozen=True)
class ProfileSpec:
    """A parsed-and-validated ``delegation.profiles`` entry (no credentials yet)."""
    name: str
    provider: str
    model: str
    description: str = ""
    instructions: str = ""
    reasoning_effort: Optional[str] = None
    reasoning_config: Optional[dict] = None      # parse_reasoning_effort output, None = inherit
    max_iterations: Optional[int] = None
    fallback: Tuple[FallbackTarget, ...] = ()
    tool_policy: ToolPolicy = ToolPolicy()
    workspace_context: WorkspaceContextPolicy = WorkspaceContextPolicy()
    execution_limits: ExecutionLimits = ExecutionLimits()
    enabled_routes: Tuple[EnabledRoute, ...] = ()


@dataclass(frozen=True)
class ProfileRoute:
    """An immutable resolved route: profile spec + runtime-provider credentials."""
    requested_profile: str
    provider: Optional[str]
    model: str
    base_url: Optional[str]
    api_key: Optional[str]
    api_mode: Optional[str]
    reasoning_config: Optional[dict] = None
    max_iterations: Optional[int] = None
    fallback: Tuple[FallbackTarget, ...] = ()
    supports_tools: bool = True
    requested_provider: Optional[str] = None
    requested_model: Optional[str] = None
    requested_reasoning_effort: Optional[str] = None
    resolved_reasoning_effort: Optional[str] = None
    transmitted_model: Optional[str] = None
    provider_reported_model: Optional[str] = None
    tool_policy: ToolPolicy = ToolPolicy()
    workspace_context: WorkspaceContextPolicy = WorkspaceContextPolicy()
    execution_limits: ExecutionLimits = ExecutionLimits()


def _profiles_section(cfg: Optional[dict]) -> Any:
    return ((cfg or {}).get("profiles"))


def _text(value: Any, *, field: str, max_chars: int = 16_000) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}")
    if len(value) > max_chars:
        raise ValueError(f"{field} must be at most {max_chars} characters")
    return value.strip()


def _string_tuple(value: Any, *, field: str, optional: bool = False) -> Optional[Tuple[str, ...]]:
    if value is None and optional:
        return None
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return tuple(dict.fromkeys(item.strip() for item in value))


def _parse_effort(value: Any, *, field: str) -> Tuple[Optional[str], Optional[dict]]:
    if value is None or value == "":
        return None, None
    if value is False:
        normalized = "none"
    elif not isinstance(value, str):
        raise ValueError(f"{field} must be one of {sorted(_VALID_EFFORTS)}")
    else:
        normalized = value.strip().lower()
    if normalized not in _VALID_EFFORTS:
        raise ValueError(f"{field} '{value}' is not valid (allowed: {', '.join(sorted(_VALID_EFFORTS))})")
    from hermes_constants import parse_reasoning_effort
    return normalized, parse_reasoning_effort(normalized)


def _positive_int(value: Any, *, field: str, allow_zero: bool = False) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        qualifier = "a non-negative" if allow_zero else "a positive"
        raise ValueError(f"{field} must be {qualifier} integer")
    return value


def _positive_number(value: Any, *, field: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field} must be a positive number")
    return float(value)


def _parse_tool_policy(name: str, raw: Any) -> ToolPolicy:
    field = f"delegation.profiles.{name}.tool_policy"
    if raw in (None, ""):
        return ToolPolicy()
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be a mapping")
    unknown = set(raw) - _TOOL_POLICY_KEYS
    if unknown:
        raise ValueError(f"{field} has unknown keys: {sorted(unknown)}")
    return ToolPolicy(
        allowed_toolsets=_string_tuple(raw.get("allowed_toolsets"), field=f"{field}.allowed_toolsets", optional=True),
        blocked_tools=_string_tuple(raw.get("blocked_tools"), field=f"{field}.blocked_tools") or (),
        allowed_tools=_string_tuple(raw.get("allowed_tools"), field=f"{field}.allowed_tools", optional=True),
        allowed_mcp_tools=_string_tuple(raw.get("allowed_mcp_tools"), field=f"{field}.allowed_mcp_tools", optional=True),
    )


def _parse_workspace_context(name: str, raw: Any) -> WorkspaceContextPolicy:
    field = f"delegation.profiles.{name}.workspace_context"
    if raw in (None, ""):
        return WorkspaceContextPolicy()
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be a mapping")
    unknown = set(raw) - _WORKSPACE_CONTEXT_KEYS
    if unknown:
        raise ValueError(f"{field} has unknown keys: {sorted(unknown)}")
    mode = str(raw.get("mode") or "inherit").strip().lower()
    if mode not in {"inherit", "none"}:
        raise ValueError(f"{field}.mode must be 'inherit' or 'none'")
    values: Dict[str, Optional[bool]] = {}
    for key in ("include_context_files", "include_memory"):
        value = raw.get(key)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{field}.{key} must be true, false, or null")
        values[key] = value
    if mode == "none" and any(values.values()):
        raise ValueError(f"{field}.mode 'none' cannot enable context files or memory")
    return WorkspaceContextPolicy(mode=mode, **values)


def _parse_execution_limits(name: str, raw: Any, legacy_max_iterations: Any) -> ExecutionLimits:
    field = f"delegation.profiles.{name}.execution_limits"
    if raw in (None, ""):
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be a mapping")
    unknown = set(raw) - _EXECUTION_LIMIT_KEYS
    if unknown:
        raise ValueError(f"{field} has unknown keys: {sorted(unknown)}")
    if legacy_max_iterations is not None and raw.get("max_iterations") is not None:
        raise ValueError(
            f"delegation.profiles.{name} cannot set both max_iterations and execution_limits.max_iterations")
    max_iterations = raw.get("max_iterations", legacy_max_iterations)
    return ExecutionLimits(
        max_iterations=_positive_int(max_iterations, field=f"{field}.max_iterations"),
        timeout_seconds=_positive_number(raw.get("timeout_seconds"), field=f"{field}.timeout_seconds"),
        max_followups=_positive_int(raw.get("max_followups"), field=f"{field}.max_followups", allow_zero=True),
        max_tool_calls=_positive_int(raw.get("max_tool_calls"), field=f"{field}.max_tool_calls", allow_zero=True),
        max_spawn_depth=_positive_int(raw.get("max_spawn_depth"), field=f"{field}.max_spawn_depth", allow_zero=True),
        max_concurrent_children=_positive_int(
            raw.get("max_concurrent_children"), field=f"{field}.max_concurrent_children", allow_zero=True),
    )


def _parse_enabled_routes(name: str, raw: Any) -> Tuple[EnabledRoute, ...]:
    field = f"delegation.profiles.{name}.enabled_routes"
    if raw in (None, ""):
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list of route mappings")
    routes: List[EnabledRoute] = []
    for index, entry in enumerate(raw):
        item_field = f"{field}[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{item_field} must be a mapping")
        unknown = set(entry) - _ROUTE_KEYS
        if unknown:
            raise ValueError(f"{item_field} has unknown keys: {sorted(unknown)}")
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not model:
            raise ValueError(f"{item_field}.model must be a non-empty string")
        effort, _ = _parse_effort(entry.get("reasoning_effort"), field=f"{item_field}.reasoning_effort")
        _validate_known_effort(provider, model, effort, field=f"{item_field}.reasoning_effort")
        route = EnabledRoute(provider=provider, model=model, reasoning_effort=effort)
        if route not in routes:
            routes.append(route)
    return tuple(routes)


def _parse_enabled_models(raw: Any) -> Tuple[EnabledRoute, ...]:
    """Parse the optional global model ceiling without resolving provider state."""
    if raw in (None, ""):
        return ()
    if not isinstance(raw, list):
        raise ValueError("delegation.enabled_models must be a list")
    routes: List[EnabledRoute] = []
    for index, entry in enumerate(raw):
        field = f"delegation.enabled_models[{index}]"
        if isinstance(entry, str):
            value = entry.strip()
            if not value:
                raise ValueError(f"{field} must be non-empty")
            provider, slash, model = value.partition("/")
            route = EnabledRoute(provider=provider if slash else "", model=model if slash else provider)
        elif isinstance(entry, dict):
            unknown = set(entry) - {"provider", "model"}
            if unknown:
                raise ValueError(f"{field} has unknown keys: {sorted(unknown)}")
            provider = str(entry.get("provider") or "").strip()
            model = str(entry.get("model") or "").strip()
            if not model:
                raise ValueError(f"{field}.model must be a non-empty string")
            route = EnabledRoute(provider=provider, model=model)
        else:
            raise ValueError(f"{field} must be a provider/model string or mapping")
        if route not in routes:
            routes.append(route)
    return tuple(routes)


def _routing_mode(cfg: Optional[dict]) -> str:
    mode = str((cfg or {}).get("routing_mode") or "profile_only").strip().lower()
    if mode not in _ROUTING_MODES:
        raise ValueError(
            f"delegation.routing_mode '{mode}' is invalid (allowed: {', '.join(sorted(_ROUTING_MODES))})")
    return mode


def _validate_known_effort(provider: str, model: str, effort: Optional[str], *, field: str) -> None:
    if effort is None:
        return
    supported, source = _supported_efforts(provider, model)
    if supported is not None and effort not in supported:
        raise ValueError(
            f"{field} '{effort}' is unsupported for {provider or '(default)'}/{model} "
            f"(supported: {', '.join(supported)}; source: {source})")


def _parse_fallback(name: str, raw: Any) -> Tuple[FallbackTarget, ...]:
    if raw in (None, ""):
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"delegation.profiles.{name}.fallback must be a list of {{provider, model}} entries, "
            f"got {type(raw).__name__}")
    targets: List[FallbackTarget] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"delegation.profiles.{name}.fallback[{i}] must be a dict with provider + model, "
                f"got {type(entry).__name__}")
        unknown = set(entry) - _FALLBACK_ENTRY_KEYS
        if unknown:
            raise ValueError(
                f"delegation.profiles.{name}.fallback[{i}] has unknown keys: {sorted(unknown)} "
                f"(allowed: provider, model)")
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            raise ValueError(
                f"delegation.profiles.{name}.fallback[{i}] needs both 'provider' and 'model'")
        targets.append(FallbackTarget(provider=provider, model=model))
    return tuple(targets)


def _parse_profile(name: str, raw: Any) -> ProfileSpec:
    if not isinstance(raw, dict):
        raise ValueError(
            f"delegation.profiles.{name} must be a mapping (provider/model/...), "
            f"got {type(raw).__name__}")
    unknown = set(raw) - _PROFILE_KEYS
    if unknown:
        raise ValueError(
            f"delegation.profiles.{name} has unknown keys: {sorted(unknown)} "
            f"(allowed: {sorted(_PROFILE_KEYS)})")
    if not name.strip():
        raise ValueError("delegation.profiles contains an empty profile name")
    model = str(raw.get("model") or "").strip()
    if not model:
        raise ValueError(f"delegation.profiles.{name} is missing required 'model'")
    provider = str(raw.get("provider") or "").strip()

    effort, reasoning_config = _parse_effort(
        raw.get("reasoning_effort"), field=f"delegation.profiles.{name}.reasoning_effort")
    _validate_known_effort(
        provider, model, effort, field=f"delegation.profiles.{name}.reasoning_effort")
    execution_limits = _parse_execution_limits(name, raw.get("execution_limits"), raw.get("max_iterations"))

    return ProfileSpec(
        name=name, provider=provider, model=model, reasoning_config=reasoning_config,
        description=_text(raw.get("description"), field=f"delegation.profiles.{name}.description", max_chars=2_000),
        instructions=_text(raw.get("instructions"), field=f"delegation.profiles.{name}.instructions"),
        reasoning_effort=effort,
        max_iterations=execution_limits.max_iterations,
        fallback=_parse_fallback(name, raw.get("fallback")),
        tool_policy=_parse_tool_policy(name, raw.get("tool_policy")),
        workspace_context=_parse_workspace_context(name, raw.get("workspace_context")),
        execution_limits=execution_limits,
        enabled_routes=_parse_enabled_routes(name, raw.get("enabled_routes")),
    )


def parse_profiles(cfg: Optional[dict]) -> Dict[str, ProfileSpec]:
    """Parse+validate ``delegation.profiles`` from the delegation config section.

    Returns ``{}`` when no profiles are configured. Raises ``ValueError`` with an actionable
    message on the first malformed profile (use :func:`profile_config_errors` to collect all).
    """
    raw = _profiles_section(cfg)
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"delegation.profiles must be a mapping of name -> profile, got {type(raw).__name__}")
    return {str(name): _parse_profile(str(name), spec) for name, spec in raw.items()}


def profile_config_errors(cfg: Optional[dict]) -> List[str]:
    """All profile config problems as messages (for ``hermes config check``); never raises."""
    errors: List[str] = []
    raw = _profiles_section(cfg)
    specs: Dict[str, ProfileSpec] = {}
    if raw and not isinstance(raw, dict):
        errors.append(
            f"delegation.profiles must be a mapping of name -> profile, got {type(raw).__name__}")
    elif isinstance(raw, dict):
        for name, spec in raw.items():
            try:
                specs[str(name)] = _parse_profile(str(name), spec)
            except ValueError as exc:
                errors.append(str(exc))
    default_profile = str((cfg or {}).get("default_profile") or "").strip()
    if default_profile and default_profile not in specs:
        configured = ", ".join(sorted(specs)) or "(none)"
        errors.append(
            f"delegation.default_profile '{default_profile}' is not a configured profile "
            f"(configured: {configured})")
    mode = str((cfg or {}).get("routing_mode") or "profile_only").strip().lower()
    if mode not in _ROUTING_MODES:
        errors.append(
            f"delegation.routing_mode '{mode}' is invalid (allowed: {', '.join(sorted(_ROUTING_MODES))})")
    try:
        enabled_models = _parse_enabled_models((cfg or {}).get("enabled_models"))
    except ValueError as exc:
        errors.append(str(exc))
        enabled_models = ()
    if enabled_models:
        allowed = {(route.provider, route.model) for route in enabled_models}
        for spec in specs.values():
            candidates = (EnabledRoute(spec.provider, spec.model, spec.reasoning_effort), *spec.enabled_routes)
            for route in candidates:
                if (route.provider, route.model) not in allowed:
                    errors.append(
                        f"delegation.profiles.{spec.name} route {route.provider or '(default)'}/{route.model} "
                        "is not present in delegation.enabled_models")
    return errors


def select_profile_name(task_profile: Any, top_profile: Any, cfg: Optional[dict]) -> Optional[str]:
    """The profile name that applies, by precedence: per-task > top-level > default_profile.

    ``None`` = no profile applies (legacy ``delegation.provider/model`` or parent inherit take
    over downstream). Falsy values (None/False/''/0) at any level fall through to the next.
    """
    for candidate in (task_profile, top_profile, (cfg or {}).get("default_profile")):
        if candidate:
            name = str(candidate).strip()
            if name:
                return name
    return None


def _route_allowed(route: EnabledRoute, allowed: Tuple[EnabledRoute, ...]) -> bool:
    return not allowed or any(
        item.model == route.model and (not item.provider or item.provider == route.provider)
        for item in allowed
    )


def _select_route(
    spec: ProfileSpec,
    *,
    routing_mode: str,
    requested_provider: Optional[str],
    requested_model: Optional[str],
    requested_reasoning_effort: Optional[str],
    global_enabled_models: Tuple[EnabledRoute, ...],
) -> EnabledRoute:
    provider = str(requested_provider or "").strip()
    model = str(requested_model or "").strip()
    effort, _ = _parse_effort(
        requested_reasoning_effort,
        field=f"delegation task profile '{spec.name}' reasoning_effort",
    )
    has_override = bool(provider or model or effort is not None)
    primary = EnabledRoute(spec.provider, spec.model, spec.reasoning_effort)
    if not has_override:
        return primary
    if routing_mode != "dynamic":
        raise ValueError(
            f"delegation profile '{spec.name}' uses profile_only routing; per-task provider/model/effort overrides are disabled")
    target_provider = provider or primary.provider
    target_model = model or primary.model
    target_effort = effort if effort is not None else primary.reasoning_effort
    target = EnabledRoute(target_provider, target_model, target_effort)
    enabled = (primary, *(spec.enabled_routes or global_enabled_models))
    matching_route = next((
        item for item in enabled
        if item.provider == target.provider and item.model == target.model
        and (item.reasoning_effort is None or item.reasoning_effort == target.reasoning_effort)
    ), None)
    if matching_route is None:
        choices = ", ".join(
            f"{item.provider or '(default)'}/{item.model}"
            + (f"@{item.reasoning_effort}" if item.reasoning_effort else "")
            for item in enabled
        )
        raise ValueError(
            f"Requested route {target.provider or '(default)'}/{target.model}"
            f"{('@' + target.reasoning_effort) if target.reasoning_effort else ''} is not enabled for profile "
            f"'{spec.name}' (enabled: {choices})")
    _validate_known_effort(
        target.provider, target.model, target.reasoning_effort,
        field=f"delegation task profile '{spec.name}' reasoning_effort",
    )
    return target


def resolve_profile_route(
    name: str,
    cfg: Optional[dict],
    parent_agent: Any = None,
    *,
    requested_provider: Optional[str] = None,
    requested_model: Optional[str] = None,
    requested_reasoning_effort: Optional[str] = None,
) -> ProfileRoute:
    """Resolve profile *name* into an immutable :class:`ProfileRoute`.

    Raises ``ValueError`` (before any child construction) for an unknown name, listing the
    configured profile names. Credentials come from ``resolve_runtime_provider`` — the same
    ladder the legacy ``delegation.provider`` branch uses.
    """
    specs = parse_profiles(cfg)
    key = str(name or "").strip()
    if key not in specs:
        configured = ", ".join(sorted(specs)) or "(none configured)"
        raise ValueError(
            f"Unknown delegation profile '{key}'. Configured profiles: {configured}. "
            f"Add it under delegation.profiles in config.yaml or pick a configured name.")
    spec = specs[key]
    allowed_models = _parse_enabled_models((cfg or {}).get("enabled_models"))
    selected = _select_route(
        spec,
        routing_mode=_routing_mode(cfg),
        requested_provider=requested_provider,
        requested_model=requested_model,
        requested_reasoning_effort=requested_reasoning_effort,
        global_enabled_models=allowed_models,
    )
    if not _route_allowed(selected, allowed_models):
        raise ValueError(
            f"Requested route {selected.provider or '(default)'}/{selected.model} is outside "
            "the delegation.enabled_models ceiling")

    # Lazy imports (house style — delegate_tool_config.py does the same) to avoid import cycles.
    from hermes_cli.runtime_provider import resolve_runtime_provider
    runtime = resolve_runtime_provider(
        requested=selected.provider or None, target_model=selected.model) or {}

    supports_tools = True
    try:
        from agent.models_dev import get_model_capabilities
        caps = get_model_capabilities(selected.provider or "", selected.model, allow_network=False)
        if caps is not None:
            supports_tools = bool(getattr(caps, "supports_tools", True))
    except Exception as exc:  # capability metadata is advisory; never block resolution on it
        pass

    _, reasoning_config = _parse_effort(
        selected.reasoning_effort,
        field=f"delegation profile '{key}' resolved reasoning_effort",
    )

    return ProfileRoute(
        requested_profile=key,
        provider=runtime.get("provider") or selected.provider or None,
        model=selected.model,
        base_url=runtime.get("base_url"),
        api_key=runtime.get("api_key"),
        api_mode=runtime.get("api_mode"),
        reasoning_config=reasoning_config,
        max_iterations=spec.max_iterations,
        fallback=spec.fallback,
        supports_tools=supports_tools,
        requested_provider=str(requested_provider or "").strip() or None,
        requested_model=str(requested_model or "").strip() or None,
        requested_reasoning_effort=(str(requested_reasoning_effort).strip().lower()
                                    if requested_reasoning_effort is not None else None),
        resolved_reasoning_effort=selected.reasoning_effort,
        transmitted_model=None,
        provider_reported_model=None,
        tool_policy=spec.tool_policy,
        workspace_context=spec.workspace_context,
        execution_limits=spec.execution_limits,
    )


def _supported_efforts(provider: str, model: str) -> Tuple[Optional[List[str]], str]:
    """Return only capability tables Hermes already owns; unknown stays explicit."""
    key = provider.strip().lower()
    if key in {"openai", "openai-api", "openai-codex", "codex"}:
        from agent.reasoning_effort import codex_supported_efforts
        return list(codex_supported_efforts(model)), "agent.reasoning_effort"
    if key in {"kimi", "kimi-coding", "kimi-coding-cn", "moonshot"}:
        from agent.reasoning_effort import kimi_supported_efforts
        return list(kimi_supported_efforts(model)), "agent.reasoning_effort"
    if key in {"zai", "zhipu", "glm"}:
        from agent.reasoning_effort import GLM52_EFFORTS, GLM53_EFFORTS
        normalized = model.lower()
        if "glm-5.3" in normalized or "glm5.3" in normalized:
            return list(GLM53_EFFORTS), "agent.reasoning_effort"
        if "glm-5.2" in normalized or "glm5.2" in normalized:
            return list(GLM52_EFFORTS), "agent.reasoning_effort"
    return None, "unknown"


def _catalog_route(route: EnabledRoute) -> Dict[str, Any]:
    capabilities: Dict[str, Any] = {
        "supports_tools": None,
        "supports_vision": None,
        "supports_reasoning": None,
        "context_window": None,
        "max_output_tokens": None,
    }
    capability_source = "unknown"
    try:
        from agent.models_dev import get_model_capabilities
        caps = get_model_capabilities(route.provider, route.model, allow_network=False)
        if caps is not None:
            capabilities = {
                "supports_tools": bool(caps.supports_tools),
                "supports_vision": bool(caps.supports_vision),
                "supports_reasoning": bool(caps.supports_reasoning),
                "context_window": int(caps.context_window or 0) or None,
                "max_output_tokens": int(caps.max_output_tokens or 0) or None,
            }
            capability_source = "models.dev_cache"
    except Exception:
        pass
    supported_efforts, effort_source = _supported_efforts(route.provider, route.model)
    return {
        "provider": route.provider or None,
        "model": route.model,
        "reasoning_effort": route.reasoning_effort,
        "supported_efforts": supported_efforts,
        "capabilities": capabilities,
        "availability": {"status": "unknown", "reason": "not_checked"},
        "freshness": {"status": "unknown", "checked_at": None},
        "provenance": {
            "route": "delegation config",
            "capabilities": capability_source,
            "supported_efforts": effort_source,
        },
    }


def discover_workers(cfg: Optional[dict], parent_agent: Any = None) -> Dict[str, Any]:
    """Return a stable, credential-free worker catalog for CLI and ``delegate_task`` discovery.

    The catalog is intentionally honest about unknown availability and freshness. It reads cached
    capability metadata only and never mutates a tool schema or contacts a provider.
    """
    errors = profile_config_errors(cfg)
    profiles: List[Dict[str, Any]] = []
    raw = _profiles_section(cfg)
    if isinstance(raw, dict):
        for name in sorted(raw, key=str):
            try:
                spec = _parse_profile(str(name), raw[name])
            except ValueError:
                continue
            routes = (EnabledRoute(spec.provider, spec.model, spec.reasoning_effort), *spec.enabled_routes)
            route_catalog = [_catalog_route(route) for route in dict.fromkeys(routes)]
            primary = route_catalog[0]
            profiles.append({
                "name": spec.name,
                "description": spec.description,
                "instructions": spec.instructions or None,
                "provider": primary["provider"],
                "model": primary["model"],
                "reasoning_effort": primary["reasoning_effort"],
                "supported_efforts": primary["supported_efforts"],
                "capabilities": primary["capabilities"],
                "availability": primary["availability"],
                "freshness": primary["freshness"],
                "provenance": primary["provenance"],
                "tool_policy": {
                    "allowed_toolsets": list(spec.tool_policy.allowed_toolsets) if spec.tool_policy.allowed_toolsets is not None else None,
                    "blocked_tools": list(spec.tool_policy.blocked_tools),
                    "allowed_tools": list(spec.tool_policy.allowed_tools) if spec.tool_policy.allowed_tools is not None else None,
                    "allowed_mcp_tools": list(spec.tool_policy.allowed_mcp_tools) if spec.tool_policy.allowed_mcp_tools is not None else None,
                },
                "workspace_context": {
                    "mode": spec.workspace_context.mode,
                    "include_context_files": spec.workspace_context.include_context_files,
                    "include_memory": spec.workspace_context.include_memory,
                },
                "execution_limits": {
                    "max_iterations": spec.execution_limits.max_iterations,
                    "timeout_seconds": spec.execution_limits.timeout_seconds,
                    "max_followups": spec.execution_limits.max_followups,
                    "max_tool_calls": spec.execution_limits.max_tool_calls,
                    "max_spawn_depth": spec.execution_limits.max_spawn_depth,
                    "max_concurrent_children": spec.execution_limits.max_concurrent_children,
                },
                "enabled_routes": route_catalog,
            })
    enabled_models = _parse_enabled_models((cfg or {}).get("enabled_models")) if not any(
        error.startswith("delegation.enabled_models") for error in errors
    ) else ()
    return {
        "version": 1,
        "routing_mode": str((cfg or {}).get("routing_mode") or "profile_only").strip().lower(),
        "default_profile": str((cfg or {}).get("default_profile") or "").strip() or None,
        "enabled_models": [_catalog_route(route) for route in enabled_models],
        "profiles": profiles,
        "errors": errors,
    }
