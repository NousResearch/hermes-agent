"""Controller-owned Kanban assignee routing guard.

Workers and prompts may propose assignees. The controller decides whether
that assignment is legal. This is not a workflow engine: it is a small
fail-closed allow/deny check used at create, assign, decompose, and
dispatch/claim.

Missing or ambiguous criticality is CRITICAL. A missing role on a
critical task is treated as implementation. engineer / engineer38 may
receive only explicit noncritical work. Critical work never remaps to a
fallback assignee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

CRITICALITY_CRITICAL = "critical"
CRITICALITY_NONCRITICAL = "noncritical"
KNOWN_CRITICALITIES = frozenset({CRITICALITY_CRITICAL, CRITICALITY_NONCRITICAL})

ROLE_IMPLEMENTATION = "implementation"
ROLE_ARCHITECTURE = "architecture"
ROLE_REVIEW = "review"
ROLE_NONCRITICAL = "noncritical"
KNOWN_ROLES = frozenset(
    {ROLE_IMPLEMENTATION, ROLE_ARCHITECTURE, ROLE_REVIEW, ROLE_NONCRITICAL}
)

SECOND_OPINION_PROFILES = frozenset({"reviewer-grok", "architect-grok"})

REASON_DENIED = "ROUTING_DENIED"
REASON_UNAVAILABLE = "ROUTING_UNAVAILABLE"
REASON_UNKNOWN = "ROUTING_UNKNOWN"
REASON_EMPTY = "ROUTING_EMPTY"
REASON_FALLBACK = "ROUTING_FALLBACK"
REASON_PREFLIGHT = "ROUTING_PREFLIGHT"

DEFAULT_ROUTING_CONFIG: dict[str, Any] = {
    "default_criticality": CRITICALITY_CRITICAL,
    "noncritical_only": ["engineer", "engineer38"],
    "critical": {
        ROLE_IMPLEMENTATION: ["engineer-grok"],
        ROLE_ARCHITECTURE: ["architect-sol"],
        ROLE_REVIEW: ["reviewer", "architect-sol"],
    },
    "unavailable": "block",
}


class RoutingGuardError(ValueError):
    """Fail-closed routing refusal with a structured reason code."""

    def __init__(self, code: str, message: str, *, details: Optional[dict] = None):
        self.code = code
        self.details = details or {}
        super().__init__(message)


@dataclass(frozen=True)
class RoutingDecision:
    assignee: Optional[str]
    criticality: str
    role: str
    second_opinion: bool = False


def default_routing_config() -> dict[str, Any]:
    return {
        "default_criticality": DEFAULT_ROUTING_CONFIG["default_criticality"],
        "noncritical_only": list(DEFAULT_ROUTING_CONFIG["noncritical_only"]),
        "critical": {
            role: list(names)
            for role, names in DEFAULT_ROUTING_CONFIG["critical"].items()
        },
        "unavailable": DEFAULT_ROUTING_CONFIG["unavailable"],
    }


def load_routing_config(cfg: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Return merged ``kanban.routing`` config, fail-closed to defaults."""
    routing = default_routing_config()
    if not isinstance(cfg, Mapping):
        return routing
    kanban_cfg = cfg.get("kanban") if "kanban" in cfg else cfg
    if not isinstance(kanban_cfg, Mapping):
        return routing
    raw = kanban_cfg.get("routing")
    if not isinstance(raw, Mapping):
        return routing

    default_crit = normalize_criticality(
        raw.get("default_criticality"),
        default=CRITICALITY_CRITICAL,
    )
    routing["default_criticality"] = default_crit

    noncritical_only = _string_list(raw.get("noncritical_only"))
    if noncritical_only:
        routing["noncritical_only"] = noncritical_only

    critical_raw = raw.get("critical")
    if isinstance(critical_raw, Mapping):
        for role in (ROLE_IMPLEMENTATION, ROLE_ARCHITECTURE, ROLE_REVIEW):
            names = _string_list(critical_raw.get(role))
            if names:
                routing["critical"][role] = names

    unavailable = str(raw.get("unavailable") or "block").strip().lower()
    routing["unavailable"] = "block" if unavailable != "allow" else "block"
    return routing


def load_routing_config_from_disk() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return load_routing_config(load_config() or {})
    except Exception:
        return default_routing_config()


def normalize_criticality(raw: Any, *, default: str = CRITICALITY_CRITICAL) -> str:
    if raw is None:
        return default if default in KNOWN_CRITICALITIES else CRITICALITY_CRITICAL
    text = str(raw).strip().lower()
    if not text:
        return default if default in KNOWN_CRITICALITIES else CRITICALITY_CRITICAL
    if text in KNOWN_CRITICALITIES:
        return text
    return CRITICALITY_CRITICAL


def normalize_role(raw: Any, *, criticality: str) -> str:
    if criticality == CRITICALITY_NONCRITICAL:
        if raw is None or not str(raw).strip():
            return ROLE_NONCRITICAL
        text = str(raw).strip().lower()
        return text if text in KNOWN_ROLES else ROLE_NONCRITICAL
    if raw is None or not str(raw).strip():
        return ROLE_IMPLEMENTATION
    text = str(raw).strip().lower()
    if text == ROLE_NONCRITICAL:
        return ROLE_IMPLEMENTATION
    if text in KNOWN_ROLES:
        return text
    return ROLE_IMPLEMENTATION


def normalize_second_opinion(raw: Any) -> bool:
    if raw is True or raw == 1:
        return True
    if isinstance(raw, str) and raw.strip().lower() in {"1", "true", "yes"}:
        return True
    return False


def infer_role_from_assignee(
    assignee: Any,
    *,
    routing: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Infer a role from a known policy name when the caller omitted one."""
    chosen = _canonical_name(assignee)
    if not chosen:
        return None
    cfg = routing or default_routing_config()
    critical = cfg.get("critical") or {}
    if isinstance(critical, Mapping):
        for role in (ROLE_IMPLEMENTATION, ROLE_ARCHITECTURE, ROLE_REVIEW):
            names = {
                str(n).strip() for n in (critical.get(role) or []) if str(n).strip()
            }
            if chosen in names:
                return role
    if chosen in SECOND_OPINION_PROFILES:
        return ROLE_REVIEW
    return None


def resolve_routing_fields(
    criticality: Any = None,
    role: Any = None,
    *,
    second_opinion: Any = False,
    routing: Optional[Mapping[str, Any]] = None,
    assignee: Any = None,
) -> tuple[str, str, bool]:
    cfg = routing or default_routing_config()
    crit = normalize_criticality(
        criticality,
        default=str(cfg.get("default_criticality") or CRITICALITY_CRITICAL),
    )
    raw_role = role
    if (raw_role is None or not str(raw_role).strip()) and assignee is not None:
        inferred = infer_role_from_assignee(assignee, routing=cfg)
        if inferred:
            raw_role = inferred
    resolved_role = normalize_role(raw_role, criticality=crit)
    return crit, resolved_role, normalize_second_opinion(second_opinion)


def policy_profile_names(routing: Optional[Mapping[str, Any]] = None) -> set[str]:
    cfg = routing or default_routing_config()
    names = {str(n).strip() for n in cfg.get("noncritical_only") or [] if str(n).strip()}
    critical = cfg.get("critical") or {}
    if isinstance(critical, Mapping):
        for group in critical.values():
            names.update(str(n).strip() for n in (group or []) if str(n).strip())
    names.update(SECOND_OPINION_PROFILES)
    return names


def allowed_critical_assignees(
    role: str,
    *,
    second_opinion: bool = False,
    routing: Optional[Mapping[str, Any]] = None,
) -> set[str]:
    cfg = routing or default_routing_config()
    critical = cfg.get("critical") or {}
    allowed = set()
    if isinstance(critical, Mapping):
        allowed.update(
            str(n).strip() for n in (critical.get(role) or []) if str(n).strip()
        )
    if second_opinion and role == ROLE_REVIEW:
        allowed.update(SECOND_OPINION_PROFILES)
    return allowed


def installed_profile_names() -> set[str]:
    """Controller-owned installed/authorized profile roster."""
    try:
        from hermes_cli.profiles import list_profiles
        return {str(p.name).strip() for p in list_profiles() if getattr(p, "name", None)}
    except Exception:
        return set()


def profile_is_available(name: Any) -> bool:
    """True when *name* is an installed/authorized Hermes profile."""
    chosen = _canonical_name(name)
    if not chosen:
        return False
    try:
        from hermes_cli.profiles import profile_exists
        if profile_exists(chosen):
            return True
    except Exception:
        pass
    return chosen in installed_profile_names()


def check_assignment(
    assignee: Any,
    *,
    criticality: Any = None,
    role: Any = None,
    second_opinion: Any = False,
    routing: Optional[Mapping[str, Any]] = None,
    valid_names: Optional[Iterable[str]] = None,
    apply_default: Optional[str] = None,
    require_available: bool = True,
) -> RoutingDecision:
    """Validate an assignee against the Sprint 3 routing policy.

    Empty critical assignee is a hard block — callers must not fill
    ``default_assignee`` / active profile / ``default``. Noncritical empty
    assignee may use ``apply_default`` only after that default itself
    passes this check. Arbitrary nonempty strings are not legal critical
    assignees; they must exist on the installed/authorized roster.
    """
    cfg = routing or default_routing_config()
    crit, resolved_role, second = resolve_routing_fields(
        criticality, role, second_opinion=second_opinion, routing=cfg, assignee=assignee,
    )
    chosen = _canonical_name(assignee)
    valid = {str(n).strip() for n in (valid_names or []) if str(n).strip()}
    details = {
        "assignee": chosen,
        "criticality": crit,
        "role": resolved_role,
        "second_opinion": second,
    }

    if not chosen:
        if crit == CRITICALITY_CRITICAL:
            raise RoutingGuardError(
                REASON_EMPTY,
                "critical work cannot be left unassigned or remapped to a fallback",
                details=details,
            )
        if apply_default:
            return check_assignment(
                apply_default,
                criticality=crit,
                role=resolved_role,
                second_opinion=second,
                routing=cfg,
                valid_names=valid_names,
                require_available=require_available,
            )
        return RoutingDecision(None, crit, resolved_role, second)

    if valid and chosen not in valid:
        if crit == CRITICALITY_CRITICAL:
            raise RoutingGuardError(
                REASON_UNAVAILABLE,
                f"required critical assignee {chosen!r} is missing or unknown",
                details=details,
            )
        if apply_default:
            return check_assignment(
                apply_default,
                criticality=crit,
                role=resolved_role,
                second_opinion=second,
                routing=cfg,
                valid_names=valid_names,
                require_available=require_available,
            )
        raise RoutingGuardError(
            REASON_UNKNOWN,
            f"assignee {chosen!r} is not an installed profile",
            details=details,
        )

    if require_available and not profile_is_available(chosen):
        if crit == CRITICALITY_CRITICAL:
            raise RoutingGuardError(
                REASON_UNAVAILABLE,
                f"required critical assignee {chosen!r} is missing or unknown",
                details=details,
            )
        # Noncritical dummy/test assignees remain legal. Critical work
        # never accepts an uninstalled name.

    noncritical_only = {
        str(n).strip() for n in (cfg.get("noncritical_only") or []) if str(n).strip()
    }
    if chosen in noncritical_only and crit != CRITICALITY_NONCRITICAL:
        raise RoutingGuardError(
            REASON_DENIED,
            f"{chosen} may receive only explicit noncritical work",
            details=details,
        )

    if crit == CRITICALITY_CRITICAL:
        allowed = allowed_critical_assignees(
            resolved_role, second_opinion=second, routing=cfg,
        )
        if chosen not in allowed:
            raise RoutingGuardError(
                REASON_DENIED,
                f"{chosen} is not an allowed {resolved_role} assignee for critical work",
                details=details,
            )
        if chosen in SECOND_OPINION_PROFILES and not second:
            raise RoutingGuardError(
                REASON_DENIED,
                f"{chosen} cannot silently become the sole critical certifier",
                details=details,
            )

    return RoutingDecision(chosen, crit, resolved_role, second)


def reject_critical_default_fallback(
    *,
    criticality: Any,
    default_assignee: Any,
    routing: Optional[Mapping[str, Any]] = None,
) -> None:
    """Dispatcher must not stamp a fallback onto unassigned critical work."""
    crit, role, _second = resolve_routing_fields(criticality, None, routing=routing)
    if crit != CRITICALITY_CRITICAL:
        return
    details = {
        "assignee": _canonical_name(default_assignee),
        "criticality": crit,
        "role": role,
        "source": "kanban.default_assignee",
    }
    raise RoutingGuardError(
        REASON_FALLBACK,
        "critical work cannot be remapped to default_assignee",
        details=details,
    )


def _canonical_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        from hermes_cli.profiles import normalize_profile_name
        return normalize_profile_name(text)
    except Exception:
        return text.lower()


def _string_list(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out
