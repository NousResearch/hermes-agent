"""Shared policy for explicit flagship-model overrides.

Flagship models remain available, but choosing one must be deliberate and
attributable. The guard is shared by Kanban and ``delegate_task`` so one
override surface cannot silently bypass the other.
"""

from __future__ import annotations

from typing import Optional


FIREPOWER_MODEL_SUBSTRINGS = ("gpt-6-astra", "claude-fable")


def canonical_model_pair(
    model: Optional[str], provider: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """Resolve a short alias to its real model id before any policy decision.

    A configured alias (``astra``) resolves to a concrete model id
    (``gpt-6-astra-900k``). Classifying the *raw* input would let the alias
    spelling walk straight past a substring guard, so every policy, audit, and
    announcement decision must run on the resolved pair.

    Best-effort: an unresolvable name is returned unchanged, which is also the
    behaviour for a plain model id that needs no resolution.
    """
    if not model:
        return model, provider
    raw = str(model).strip()
    if not raw:
        return model, provider
    # A "provider/model" route resolves on its model half.
    prefix, sep, tail = raw.rpartition("/")
    lookup = tail if sep else raw
    try:
        from hermes_cli.model_switch import resolve_alias

        resolved = resolve_alias(lookup, provider or "")
    except ImportError:  # pragma: no cover - upstream module must exist
        raise
    except Exception:
        # An ambiguous or uncatalogued alias is not a policy decision; fall
        # back to the literal input rather than failing the caller's write.
        return model, provider
    if not resolved:
        return model, provider
    resolved_provider, resolved_model, _alias = resolved
    return resolved_model, (provider or resolved_provider)


def is_firepower_model(model: Optional[str]) -> bool:
    """Return whether *model* resolves to a flagship/firepower-only family."""
    canonical_model, _ = canonical_model_pair(model)
    normalized = str(canonical_model or "").strip().lower()
    return bool(normalized) and any(
        banned in normalized for banned in FIREPOWER_MODEL_SUBSTRINGS
    )


def route_kind(route: Optional[str]) -> str:
    """Classify a route for dispatcher announcements."""
    return "firepower" if is_firepower_model(route) else "standard"


def firepower_guard_error(
    model: Optional[str],
    reason: Optional[str],
    *,
    reason_field: str = "--firepower",
) -> Optional[str]:
    """Return an actionable refusal message, or ``None`` when allowed."""
    if not is_firepower_model(model) or str(reason or "").strip():
        return None
    return (
        f"model {model!r} is firepower-only; provide {reason_field} "
        '"<reason>" to make this explicit and auditable'
    )


def format_firepower_audit(
    model: str,
    provider: Optional[str],
    reason: str,
) -> str:
    """Stable human-readable audit comment/log payload."""
    canonical_model, canonical_provider = canonical_model_pair(model, provider)
    route = (
        f"{canonical_provider}/{canonical_model}"
        if canonical_provider
        else canonical_model
    )
    return f"firepower override: route={route}; reason={reason.strip()}"
