"""Deterministic multi-model route selection for native delegation.

Hermes can preconfigure many ``delegation.routes`` entries — one per
provider:model pair — and let the orchestrator describe a task's difficulty
and required capabilities instead of pinning a provider by hand.  This module
owns the two halves of that contract:

* **The catalog.** ``load_route_catalog`` parses and validates the
  ``delegation.routing`` / ``delegation.routes`` config into typed
  ``DelegationRoute`` objects.  When the catalog is absent or disabled the
  legacy ``delegation.provider`` / ``delegation.model`` behavior is preserved
  byte-for-byte — the caller sees ``catalog.enabled is False`` and takes the
  old path.
* **The selector.** ``select_route`` is a pure function: given a catalog, a
  request, and an already-read usage view it returns a ``RouteDecision``.  It
  performs **no** network I/O, so a delegation never blocks on a quota probe.
  Cached usage is supplied by :mod:`agent.delegation_usage_cache`, which
  refreshes out of band.

Two backends are routable: ``native`` (``openai-codex`` and
``google-antigravity``) and ``claude-p``, a single, deliberately narrow
external-CLI backend that shells out to ``claude -p`` (Claude Code print
mode) so a route can consume the operator's Claude Pro/Max subscription.
Every other CLI-shelling backend (``agy -p``, ``codex exec``, OpenCode,
arbitrary commands) and Anthropic API-key provider substitution remain
deliberately unsupported — a config naming one fails loudly rather than
silently degrading to a different provider or backend.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    "BUILTIN_CAPABILITIES",
    "CLAUDE_P_BACKEND",
    "CLAUDE_P_TOOL_PROFILES",
    "NATIVE_ROUTABLE_PROVIDERS",
    "ROUTABLE_BACKENDS",
    "DelegationRoute",
    "ModelClass",
    "ProviderUsage",
    "RouteCatalog",
    "RouteConfigError",
    "RouteDecision",
    "RouteRequest",
    "TaskDifficulty",
    "UsageView",
    "load_route_catalog",
    "parse_difficulty",
    "parse_model_class",
    "select_route",
]


class RouteConfigError(ValueError):
    """Raised when ``delegation.routes`` is malformed.

    Loud by design: a typo'd provider or difficulty would otherwise silently
    drop a route out of the candidate set and send work to a different model
    than the operator configured.
    """


class TaskDifficulty(enum.Enum):
    """How hard the orchestrator judges a delegated task to be."""

    ROUTINE = "routine"
    STANDARD = "standard"
    COMPLEX = "complex"
    FRONTIER = "frontier"


@enum.unique
class ModelClass(enum.IntEnum):
    """Capability tier of a model, ordered weakest → strongest.

    Ordered (``IntEnum``) so ``minimum_model_class`` is a simple comparison
    and so models from different providers can be treated as equivalent when
    they share a class.
    """

    FAST = 0
    BALANCED = 1
    ADVANCED = 2
    FRONTIER = 3


#: Capability strings Hermes documents to the orchestrator.  Routes may
#: declare others (deployment-specific skills); required capabilities are
#: matched as a plain subset, so unknown strings simply never match unless a
#: route declares them too.
BUILTIN_CAPABILITIES: tuple[str, ...] = (
    "coding",
    "reasoning",
    "tool_use",
    "long_context",
    "vision",
    "review",
)

#: Providers this PR can route to natively.  Anything else is a config error.
NATIVE_ROUTABLE_PROVIDERS: frozenset[str] = frozenset(
    {"openai-codex", "google-antigravity"}
)

#: The single external-CLI backend this PR supports: Claude Code print mode
#: (``claude -p``), used only to consume a Claude Pro/Max subscription. Its
#: provider slug is fixed and does not belong to ``NATIVE_ROUTABLE_PROVIDERS``
#: — it is never dispatched through the native in-process agent path.
CLAUDE_P_BACKEND = "claude-p"
CLAUDE_P_PROVIDER = "claude-p"

#: Backends a route may declare. ``native`` dispatches through the in-process
#: AIAgent path; ``claude-p`` shells out to the ``claude`` CLI. No other
#: backend string is ever accepted.
ROUTABLE_BACKENDS: frozenset[str] = frozenset({"native", CLAUDE_P_BACKEND})

#: Fixed, least-privilege tool profiles a ``claude-p`` route may select via
#: its ``tool_profile`` field. Never an arbitrary CLI/tool string.
CLAUDE_P_TOOL_PROFILES: frozenset[str] = frozenset({"read_only", "review", "coding"})
DEFAULT_CLAUDE_P_TOOL_PROFILE = "read_only"

#: Bounded defaults/ceilings for a claude-p child. The ceilings are hard: a
#: route asking for more is clamped, never trusted, so a config typo cannot
#: hand an external CLI an unbounded turn/spend/wall-clock budget.
DEFAULT_CLAUDE_P_MAX_TURNS = 40
MAX_CLAUDE_P_MAX_TURNS = 200
DEFAULT_CLAUDE_P_MAX_BUDGET_USD = 5.0
MAX_CLAUDE_P_MAX_BUDGET_USD = 50.0
DEFAULT_CLAUDE_P_TIMEOUT_SECONDS = 900
MAX_CLAUDE_P_TIMEOUT_SECONDS = 3600
DEFAULT_CLAUDE_P_COOLDOWN_SECONDS = 60
MAX_CLAUDE_P_COOLDOWN_SECONDS = 3600

#: Policies for a route whose usage is unknown or expired.  ``fixed_priority``
#: keeps the route eligible at its configured priority — never treating the
#: unknown as "unlimited" (which would overrun a depleted account) nor as
#: "zero" (which would strand every route the moment a cache expires).
UNKNOWN_USAGE_POLICIES: frozenset[str] = frozenset({"fixed_priority", "skip"})

DEFAULT_USAGE_TTL_SECONDS = 300
DEFAULT_USAGE_STALE_SECONDS = 1800


@dataclass(frozen=True)
class DelegationRoute:
    """One validated provider:model destination in the catalog."""

    id: str
    provider: str
    model: str
    model_class: ModelClass
    task_difficulties: tuple[TaskDifficulty, ...]
    capabilities: frozenset[str] = frozenset()
    priority: int = 100
    reserve_remaining_percent: float = 0.0
    usage_window_prefixes: tuple[str, ...] = ()
    backend: str = "native"
    #: Only meaningful for ``backend == "claude-p"``: which fixed, least-
    #: privilege tool allowlist the child CLI is launched with. Defaults to
    #: read-only; ``coding`` must be opted into explicitly in route config.
    tool_profile: str = "read_only"
    #: Bounded execution limits for the claude-p child process.
    max_turns: int = 40
    max_budget_usd: float = 5.0
    timeout_seconds: int = 900
    cooldown_seconds: int = 60
    enabled: bool = True

    @property
    def is_claude_p(self) -> bool:
        return self.backend == CLAUDE_P_BACKEND

    @property
    def write_capable(self) -> bool:
        """True when this route's tool profile can modify the workdir."""
        return self.is_claude_p and self.tool_profile == "coding"


@dataclass(frozen=True)
class RouteCatalog:
    """Parsed ``delegation.routing`` block plus its validated routes."""

    enabled: bool = False
    routes: tuple[DelegationRoute, ...] = ()
    usage_ttl_seconds: int = DEFAULT_USAGE_TTL_SECONDS
    usage_stale_seconds: int = DEFAULT_USAGE_STALE_SECONDS
    unknown_usage: str = "fixed_priority"

    @property
    def active(self) -> bool:
        """True when the selector should run instead of the legacy path."""
        return bool(self.enabled and self.routes)


def _require_str(raw: Mapping[str, Any], key: str, *, where: str) -> str:
    value = str(raw.get(key) or "").strip()
    if not value:
        raise RouteConfigError(f"{where}: '{key}' is required and must be a non-empty string")
    return value


def _parse_model_class(value: Any, *, where: str) -> ModelClass:
    name = str(value or "").strip().lower()
    if not name:
        raise RouteConfigError(f"{where}: 'model_class' is required")
    try:
        return ModelClass[name.upper()]
    except KeyError:
        valid = ", ".join(m.name.lower() for m in ModelClass)
        raise RouteConfigError(
            f"{where}: unknown 'model_class' {value!r} (expected one of: {valid})"
        ) from None


def _parse_difficulty(value: Any, *, where: str, key: str = "task_difficulties") -> TaskDifficulty:
    name = str(value or "").strip().lower()
    try:
        return TaskDifficulty(name)
    except ValueError:
        valid = ", ".join(d.value for d in TaskDifficulty)
        raise RouteConfigError(
            f"{where}: unknown '{key}' entry {value!r} (expected one of: {valid})"
        ) from None


def _parse_difficulties(raw: Any, *, where: str) -> tuple[TaskDifficulty, ...]:
    if isinstance(raw, str) or not isinstance(raw, Iterable):
        raise RouteConfigError(f"{where}: 'task_difficulties' must be a list")
    parsed = tuple(_parse_difficulty(item, where=where) for item in raw)
    if not parsed:
        raise RouteConfigError(f"{where}: 'task_difficulties' must list at least one difficulty")
    # Deduplicate while preserving declaration order.
    seen: list[TaskDifficulty] = []
    for item in parsed:
        if item not in seen:
            seen.append(item)
    return tuple(seen)


def _parse_capabilities(raw: Any, *, where: str) -> frozenset[str]:
    if raw is None:
        return frozenset()
    if isinstance(raw, str) or not isinstance(raw, Iterable):
        raise RouteConfigError(f"{where}: 'capabilities' must be a list of strings")
    caps = {str(item or "").strip().lower() for item in raw}
    caps.discard("")
    return frozenset(caps)


def _parse_percent(raw: Any, key: str, *, where: str, default: float = 0.0) -> float:
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise RouteConfigError(f"{where}: '{key}' must be a number between 0 and 100") from None
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise RouteConfigError(
            f"{where}: '{key}'={raw!r} is out of range (expected 0-100)"
        )
    return value


def _parse_int(raw: Any, key: str, *, where: str, default: int, minimum: int = 0) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        raise RouteConfigError(f"{where}: '{key}' must be an integer") from None
    if value < minimum:
        raise RouteConfigError(f"{where}: '{key}'={raw!r} must be >= {minimum}")
    return value


def _parse_bounded_int(
    raw: Any, key: str, *, where: str, default: int, ceiling: int, minimum: int = 1
) -> int:
    """Parse an integer and clamp it into ``[minimum, ceiling]``.

    Clamping rather than rejecting keeps a slightly-too-large config working
    while guaranteeing the external CLI can never be handed an unbounded
    budget.
    """
    value = _parse_int(raw, key, where=where, default=default, minimum=minimum)
    return min(ceiling, value)


def _parse_bounded_float(
    raw: Any, key: str, *, where: str, default: float, ceiling: float, minimum: float
) -> float:
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise RouteConfigError(f"{where}: '{key}' must be a number") from None
    if not math.isfinite(value) or value < minimum:
        raise RouteConfigError(f"{where}: '{key}'={raw!r} must be >= {minimum}")
    return min(ceiling, value)


def _parse_route(raw: Any, index: int) -> DelegationRoute:
    where = f"delegation.routes[{index}]"
    if not isinstance(raw, Mapping):
        raise RouteConfigError(f"{where}: each route must be a mapping")

    route_id = _require_str(raw, "id", where=where)
    where = f"delegation.routes[{index}] (id={route_id!r})"

    backend = str(raw.get("backend") or "native").strip().lower()
    if backend not in ROUTABLE_BACKENDS:
        supported = ", ".join(sorted(ROUTABLE_BACKENDS))
        raise RouteConfigError(
            f"{where}: unsupported 'backend' {backend!r} (supported: {supported}). "
            f"Arbitrary CLI-shelling backends are not routable."
        )

    provider = _require_str(raw, "provider", where=where).lower()
    tool_profile = DEFAULT_CLAUDE_P_TOOL_PROFILE
    if backend == CLAUDE_P_BACKEND:
        # The claude-p backend has exactly one provider slug: it always runs
        # the local `claude` CLI against the operator's subscription. Naming a
        # native provider here would silently redirect subscription work onto
        # API billing, so it is a loud config error.
        if provider != CLAUDE_P_PROVIDER:
            raise RouteConfigError(
                f"{where}: backend 'claude-p' requires provider 'claude-p' "
                f"(got {provider!r}); it never substitutes an API-key provider."
            )
        tool_profile = str(
            raw.get("tool_profile") or DEFAULT_CLAUDE_P_TOOL_PROFILE
        ).strip().lower()
        if tool_profile not in CLAUDE_P_TOOL_PROFILES:
            valid = ", ".join(sorted(CLAUDE_P_TOOL_PROFILES))
            raise RouteConfigError(
                f"{where}: unknown 'tool_profile' {raw.get('tool_profile')!r} "
                f"(expected one of: {valid})"
            )
    elif provider not in NATIVE_ROUTABLE_PROVIDERS:
        supported = ", ".join(sorted(NATIVE_ROUTABLE_PROVIDERS))
        raise RouteConfigError(
            f"{where}: unsupported native 'provider' {provider!r} (supported: {supported})"
        )
    elif raw.get("tool_profile") is not None:
        raise RouteConfigError(
            f"{where}: 'tool_profile' only applies to backend 'claude-p'"
        )

    window_prefixes_raw = raw.get("usage_window_prefixes") or ()
    if isinstance(window_prefixes_raw, str) or not isinstance(window_prefixes_raw, Iterable):
        raise RouteConfigError(f"{where}: 'usage_window_prefixes' must be a list of strings")

    capabilities = _parse_capabilities(raw.get("capabilities"), where=where)
    if backend == CLAUDE_P_BACKEND and "coding" in capabilities and tool_profile != "coding":
        raise RouteConfigError(
            f"{where}: capability 'coding' requires tool_profile 'coding'; "
            f"profile {tool_profile!r} cannot edit files"
        )

    return DelegationRoute(
        id=route_id,
        provider=provider,
        model=_require_str(raw, "model", where=where),
        model_class=_parse_model_class(raw.get("model_class"), where=where),
        task_difficulties=_parse_difficulties(raw.get("task_difficulties"), where=where),
        capabilities=capabilities,
        priority=_parse_int(raw.get("priority"), "priority", where=where, default=100),
        reserve_remaining_percent=_parse_percent(
            raw.get("reserve_remaining_percent"), "reserve_remaining_percent", where=where
        ),
        usage_window_prefixes=tuple(
            str(p).strip() for p in window_prefixes_raw if str(p).strip()
        ),
        backend=backend,
        tool_profile=tool_profile,
        max_turns=_parse_bounded_int(
            raw.get("max_turns"), "max_turns", where=where,
            default=DEFAULT_CLAUDE_P_MAX_TURNS, ceiling=MAX_CLAUDE_P_MAX_TURNS,
        ),
        max_budget_usd=_parse_bounded_float(
            raw.get("max_budget_usd"), "max_budget_usd", where=where,
            default=DEFAULT_CLAUDE_P_MAX_BUDGET_USD,
            ceiling=MAX_CLAUDE_P_MAX_BUDGET_USD, minimum=0.01,
        ),
        timeout_seconds=_parse_bounded_int(
            raw.get("timeout_seconds"), "timeout_seconds", where=where,
            default=DEFAULT_CLAUDE_P_TIMEOUT_SECONDS,
            ceiling=MAX_CLAUDE_P_TIMEOUT_SECONDS,
        ),
        cooldown_seconds=_parse_bounded_int(
            raw.get("cooldown_seconds"), "cooldown_seconds", where=where,
            default=DEFAULT_CLAUDE_P_COOLDOWN_SECONDS,
            ceiling=MAX_CLAUDE_P_COOLDOWN_SECONDS,
        ),
        enabled=bool(raw.get("enabled", True)),
    )


def load_route_catalog(delegation_cfg: Optional[Mapping[str, Any]]) -> RouteCatalog:
    """Parse the ``delegation`` config block into a validated catalog.

    Returns a disabled, empty catalog when no ``routes`` are configured so
    callers fall through to the legacy fixed provider/model behavior.
    """
    cfg = delegation_cfg if isinstance(delegation_cfg, Mapping) else {}
    routing = cfg.get("routing")
    routing = routing if isinstance(routing, Mapping) else {}

    # A disabled catalog must be inert.  In particular, staged/future route
    # entries may name backends this build does not support; validating them
    # while routing is off would break the legacy provider/model path.
    if not bool(routing.get("enabled", False)):
        return RouteCatalog()

    routes_raw = cfg.get("routes")
    if routes_raw is None:
        return RouteCatalog()
    if isinstance(routes_raw, str) or not isinstance(routes_raw, Iterable):
        raise RouteConfigError("delegation.routes must be a list of route mappings")
    routes_raw = list(routes_raw)
    if not routes_raw:
        return RouteCatalog()

    routes: list[DelegationRoute] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(routes_raw):
        route = _parse_route(raw, index)
        if route.id in seen_ids:
            raise RouteConfigError(
                f"delegation.routes: duplicate route id {route.id!r}; ids must be unique"
            )
        seen_ids.add(route.id)
        routes.append(route)

    unknown_usage = str(routing.get("unknown_usage") or "fixed_priority").strip().lower()
    if unknown_usage not in UNKNOWN_USAGE_POLICIES:
        valid = ", ".join(sorted(UNKNOWN_USAGE_POLICIES))
        raise RouteConfigError(
            f"delegation.routing.unknown_usage={unknown_usage!r} is invalid (expected: {valid})"
        )

    usage_ttl_seconds = _parse_int(
        routing.get("usage_ttl_seconds"),
        "usage_ttl_seconds",
        where="delegation.routing",
        default=DEFAULT_USAGE_TTL_SECONDS,
    )
    usage_stale_seconds = _parse_int(
        routing.get("usage_stale_seconds"),
        "usage_stale_seconds",
        where="delegation.routing",
        default=DEFAULT_USAGE_STALE_SECONDS,
    )
    if usage_stale_seconds < usage_ttl_seconds:
        raise RouteConfigError(
            "delegation.routing: 'usage_stale_seconds' must be greater than or equal "
            "to 'usage_ttl_seconds'"
        )

    return RouteCatalog(
        enabled=bool(routes),
        routes=tuple(routes),
        usage_ttl_seconds=usage_ttl_seconds,
        usage_stale_seconds=usage_stale_seconds,
        unknown_usage=unknown_usage,
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderUsage:
    """Normalized, secret-free usage for one provider, as read from cache.

    ``remaining_percent`` is ``None`` when the provider reports windows we
    cannot turn into a percentage; ``freshness`` is one of ``fresh``,
    ``stale`` or ``unknown``.  This carries no token, account identity,
    project, header, or raw provider payload — see
    :mod:`agent.delegation_usage_cache` for the projection rules.
    """

    provider: str
    remaining_percent: Optional[float] = None
    freshness: str = "unknown"
    age_seconds: Optional[float] = None

    @property
    def usable(self) -> bool:
        """True when the reading may be compared against a route reserve."""
        return self.freshness in {"fresh", "stale"} and self.remaining_percent is not None


@dataclass(frozen=True)
class UsageView:
    """Immutable snapshot of cached usage handed to the pure selector."""

    entries: Mapping[str, ProviderUsage] = field(default_factory=dict)

    def for_route(self, route: "DelegationRoute") -> ProviderUsage:
        # Route-scoped readings take precedence because one provider account
        # may expose independent pools (for example Antigravity Gemini versus
        # Claude/GPT buckets).  Provider-keyed entries remain supported for
        # simple callers and backward-compatible tests.
        found = self.entries.get(route.id) or self.entries.get(route.provider)
        if found is None:
            return ProviderUsage(provider=route.provider)
        return found


@dataclass(frozen=True)
class RouteRequest:
    """What the orchestrator says about one delegated task."""

    difficulty: TaskDifficulty = TaskDifficulty.STANDARD
    difficulty_reason: Optional[str] = None
    required_capabilities: frozenset[str] = frozenset()
    minimum_model_class: Optional[ModelClass] = None
    route_id: Optional[str] = None


@dataclass(frozen=True)
class RouteDecision:
    """Structured, loggable outcome of :func:`select_route`."""

    selected: bool
    reason: str
    route_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    model_class: Optional[str] = None
    difficulty: Optional[str] = None
    difficulty_reason: Optional[str] = None
    usage_freshness: str = "unknown"
    usage_age_seconds: Optional[float] = None
    usage_remaining_percent: Optional[float] = None
    explicit_override: bool = False
    reserve_bypassed: bool = False
    considered_route_ids: tuple[str, ...] = ()
    #: Why each non-eligible route was dropped, in catalog order. Surfaced in
    #: result/progress metadata so an operator can see that a route was
    #: skipped for reserve rather than silently never considered.
    exclusions: tuple[str, ...] = ()


def parse_model_class(value: Any) -> ModelClass:
    """Public parser for a model-class string (raises ``RouteConfigError``)."""
    return _parse_model_class(value, where="minimum_model_class")


def parse_difficulty(value: Any) -> TaskDifficulty:
    """Public parser for a difficulty string (raises ``RouteConfigError``)."""
    return _parse_difficulty(value, where="difficulty", key="difficulty")


def _matches_request(
    route: DelegationRoute,
    request: RouteRequest,
    available_providers: frozenset[str],
    *,
    check_difficulty: bool = True,
    cooling_down_route_ids: frozenset[str] = frozenset(),
) -> Optional[str]:
    """Return a human-readable rejection reason, or None when the route fits."""
    if not route.enabled:
        return f"route {route.id!r} is disabled"
    if route.backend not in ROUTABLE_BACKENDS:
        return f"route {route.id!r} uses unsupported backend {route.backend!r}"
    if route.provider not in available_providers:
        return f"provider {route.provider!r} is not available (not authenticated or not installed)"
    if route.id in cooling_down_route_ids:
        # A recent pre-start rate-limit/overload/billing failure. Bounded and
        # decided BEFORE execution, so the next route is chosen cleanly; it is
        # never consulted once a task has started running tools.
        return f"route {route.id!r} is in a bounded cooldown after a recent startup failure"
    if check_difficulty and request.difficulty not in route.task_difficulties:
        return (
            f"route {route.id!r} does not serve difficulty {request.difficulty.value!r}"
        )
    if request.minimum_model_class is not None and route.model_class < request.minimum_model_class:
        return (
            f"route {route.id!r} model_class {route.model_class.name.lower()!r} is below "
            f"minimum {request.minimum_model_class.name.lower()!r}"
        )
    missing = set(request.required_capabilities) - set(route.capabilities)
    if missing:
        return (
            f"route {route.id!r} is missing required capabilities: "
            f"{', '.join(sorted(missing))}"
        )
    return None


def _usage_verdict(
    route: DelegationRoute, usage: ProviderUsage, unknown_usage: str
) -> Optional[str]:
    """Reject a route whose cached usage sits below its configured reserve.

    Unknown/expired usage never means "unlimited" (which would push work at a
    depleted account) nor "zero" (which would strand every route whenever the
    cache expires): under the default ``fixed_priority`` policy the route
    simply keeps competing at its configured priority.
    """
    if usage.usable:
        assert usage.remaining_percent is not None
        if usage.remaining_percent < route.reserve_remaining_percent:
            return (
                f"route {route.id!r} has {usage.remaining_percent:.0f}% remaining, "
                f"below its {route.reserve_remaining_percent:.0f}% reserve"
            )
        return None
    if unknown_usage == "skip":
        return f"route {route.id!r} has unknown usage and unknown_usage='skip'"
    return None


def _describe(route: DelegationRoute, usage: ProviderUsage, request: RouteRequest) -> str:
    if usage.usable and usage.remaining_percent is not None:
        usage_note = (
            f"{usage.remaining_percent:.0f}% remaining ({usage.freshness} usage)"
        )
    else:
        usage_note = "usage unknown (fixed priority)"
    return (
        f"{route.id} [{route.provider}:{route.model}, "
        f"{route.model_class.name.lower()} class] for {request.difficulty.value} task; "
        f"{usage_note}; priority {route.priority}"
    )


def select_route(
    catalog: RouteCatalog,
    request: RouteRequest,
    *,
    usage: UsageView,
    available_providers: frozenset[str],
    cooling_down_route_ids: frozenset[str] = frozenset(),
) -> RouteDecision:
    """Pick a route deterministically. Pure: never performs I/O.

    Order of resolution:

    1. An explicit ``request.route_id`` override, still validated for
       enabled/backend/provider-availability/capabilities.  It may bypass the
       reserve threshold, and the decision says so.
    2. Otherwise filter enabled native routes by difficulty, minimum model
       class, required capabilities and provider availability.
    3. Drop routes whose cached (fresh or allowed-stale) usage is below their
       reserve; apply ``unknown_usage`` for the rest.
    4. Break ties by ``priority`` ascending, then route id, so the same inputs
       always yield the same route.
    """
    base = {
        "difficulty": request.difficulty.value,
        "difficulty_reason": request.difficulty_reason,
    }

    if request.route_id:
        matched = next((r for r in catalog.routes if r.id == request.route_id), None)
        if matched is None:
            known = ", ".join(r.id for r in catalog.routes) or "none configured"
            return RouteDecision(
                selected=False,
                reason=f"explicit route {request.route_id!r} is not in the catalog (known: {known})",
                **base,
            )
        rejection = _matches_request(
            matched,
            request,
            available_providers,
            check_difficulty=False,
            cooling_down_route_ids=cooling_down_route_ids,
        )
        if rejection:
            return RouteDecision(
                selected=False,
                reason=f"explicit route rejected: {rejection}",
                considered_route_ids=(matched.id,),
                explicit_override=True,
                **base,
            )
        matched_usage = usage.for_route(matched)
        reserve_hit = _usage_verdict(matched, matched_usage, "fixed_priority")
        reason = f"explicit override: {_describe(matched, matched_usage, request)}"
        if reserve_hit:
            reason += f"; reserve bypassed by explicit override ({reserve_hit})"
        return RouteDecision(
            selected=True,
            reason=reason,
            route_id=matched.id,
            provider=matched.provider,
            model=matched.model,
            model_class=matched.model_class.name.lower(),
            usage_freshness=matched_usage.freshness,
            usage_age_seconds=matched_usage.age_seconds,
            usage_remaining_percent=matched_usage.remaining_percent,
            explicit_override=True,
            reserve_bypassed=bool(reserve_hit),
            considered_route_ids=(matched.id,),
            **base,
        )

    rejections: list[str] = []
    eligible: list[DelegationRoute] = []
    for route in catalog.routes:
        rejection = _matches_request(
            route, request, available_providers,
            cooling_down_route_ids=cooling_down_route_ids,
        )
        if rejection:
            rejections.append(rejection)
            continue
        usage_rejection = _usage_verdict(route, usage.for_route(route), catalog.unknown_usage)
        if usage_rejection:
            rejections.append(usage_rejection)
            continue
        eligible.append(route)

    # Usage-aware total ordering.  Once difficulty/model-class/capability and
    # reserve constraints are satisfied, prefer the route with the most known
    # remaining allowance.  Unknown usage is never treated as unlimited: it
    # competes only when no route has a usable fresh/stale reading.  Priority
    # and route id make equal readings deterministic.
    def _rank(route: DelegationRoute) -> tuple[Any, ...]:
        route_usage = usage.for_route(route)
        if route_usage.usable and route_usage.remaining_percent is not None:
            return (0, -route_usage.remaining_percent, route.priority, route.id)
        return (1, 0.0, route.priority, route.id)

    eligible.sort(key=_rank)
    considered = tuple(r.id for r in eligible)

    if not eligible:
        detail = "; ".join(rejections) if rejections else "no routes configured"
        return RouteDecision(
            selected=False,
            reason=f"no eligible route for {request.difficulty.value} task ({detail})",
            **base,
        )

    winner = eligible[0]
    winner_usage = usage.for_route(winner)
    reason = f"selected {_describe(winner, winner_usage, request)}"
    if rejections:
        reason += f"; skipped: {'; '.join(rejections)}"
    return RouteDecision(
        selected=True,
        reason=reason,
        route_id=winner.id,
        provider=winner.provider,
        model=winner.model,
        model_class=winner.model_class.name.lower(),
        usage_freshness=winner_usage.freshness,
        usage_age_seconds=winner_usage.age_seconds,
        usage_remaining_percent=winner_usage.remaining_percent,
        considered_route_ids=considered,
        exclusions=tuple(rejections),
        **base,
    )
