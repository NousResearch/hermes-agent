"""Scope construction and fail-closed matching."""

from __future__ import annotations

from dataclasses import fields
from typing import Iterable

from .errors import IncompleteScope
from .model import ScopeRef, SurfaceRef

_SCOPE_FIELDS = tuple(item.name for item in fields(ScopeRef))


def scope_from_surface(
    surface: SurfaceRef,
    *,
    principal_id: str,
    project_id: str | None = None,
    continuity_id: str | None = None,
) -> ScopeRef:
    """Construct canonical policy scope from one authenticated surface."""
    return ScopeRef(
        profile=surface.profile,
        platform=surface.platform,
        scope_id=surface.scope_id,
        chat_id=surface.chat_id,
        thread_id=surface.thread_id,
        principal_id=principal_id,
        project_id=project_id,
        continuity_id=continuity_id,
    )


def require_scope_fields(scope: ScopeRef, required: Iterable[str]) -> None:
    """Reject missing scope discriminators instead of guessing them."""
    missing: list[str] = []
    for name in required:
        if name not in _SCOPE_FIELDS:
            raise ValueError(f"unknown scope field: {name}")
        value = getattr(scope, name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(name)
    if missing:
        raise IncompleteScope(f"missing required scope fields: {', '.join(sorted(missing))}")


def scope_matches(candidate: ScopeRef, selector: dict[str, str | None]) -> bool:
    """Match only declared selector dimensions; unknown dimensions fail closed."""
    for name, expected in selector.items():
        if name not in _SCOPE_FIELDS:
            return False
        if expected is None:
            continue
        if getattr(candidate, name) != expected:
            return False
    return True


def scope_is_within(candidate: ScopeRef, boundary: ScopeRef) -> bool:
    """Return whether candidate is no broader than boundary.

    Optional fields on the boundary act as wildcards only when absent there.
    Tenant/workspace, profile, platform, chat, and principal are always exact.
    """
    for name in ("profile", "platform", "scope_id", "chat_id", "principal_id"):
        if getattr(candidate, name) != getattr(boundary, name):
            return False
    for name in ("thread_id", "project_id", "continuity_id"):
        expected = getattr(boundary, name)
        if expected is not None and getattr(candidate, name) != expected:
            return False
    return True


def scope_redacted(scope: ScopeRef) -> dict[str, str | None]:
    """Expose stable digests for external identifiers in operator diagnostics."""
    import hashlib

    def digest(value: str | None) -> str | None:
        if value is None:
            return None
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    return {
        "profile": scope.profile,
        "platform": scope.platform,
        "scope_id_digest": digest(scope.scope_id),
        "chat_id_digest": digest(scope.chat_id),
        "thread_id_digest": digest(scope.thread_id),
        "principal_id": scope.principal_id,
        "project_id": scope.project_id,
        "continuity_id": scope.continuity_id,
    }
