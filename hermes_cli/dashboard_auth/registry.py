"""Module-level registry for DashboardAuthProvider instances. Plugins call ``register_provider``
via the plugin context hook at startup; the auth gate iterates ``list_providers()`` and uses
``get_provider`` to dispatch on the session's ``provider`` field."""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

from hermes_constants import hermes_home_key
from hermes_cli.dashboard_auth.base import DashboardAuthProvider, assert_protocol_compliance

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_providers: dict[str, DashboardAuthProvider] = {}
_scoped_providers: dict[str, dict[str, DashboardAuthProvider]] = {}
# Profile (scope_key) that currently owns each name in ``_providers`` — see
# ``register_global_provider``.
_provider_owners: dict[str, Optional[str]] = {}


def _merged(scope: Optional[str] = None) -> dict[str, DashboardAuthProvider]:
    providers = dict(_providers)
    providers.update(_scoped_providers.get(scope or hermes_home_key(), {}))
    return providers


def _target(scope: Optional[str], *, create: bool) -> dict[str, DashboardAuthProvider]:
    """Global map for ``scope is None``, else that scope's overlay."""
    if scope is None:
        return _providers
    return _scoped_providers.setdefault(scope, {}) if create else _scoped_providers.get(scope, {})


def _log_registered(kind: str, provider: DashboardAuthProvider) -> None:
    _log.info("dashboard-auth: registered %s%r (%s)", kind, provider.name, provider.display_name)


def register_provider(provider: DashboardAuthProvider, *, scope: Optional[str] = None) -> None:
    """Raises ``TypeError`` on protocol violation, ``ValueError`` on a duplicate name."""
    assert_protocol_compliance(type(provider))
    with _lock:
        target = _target(scope, create=True)
        effective = target if scope is None else _merged(scope)
        if provider.name in effective:
            raise ValueError(f"dashboard-auth provider already registered: {provider.name!r}")
        target[provider.name] = provider
    _log_registered("provider ", provider)


def get_provider(name: str, *, scope: Optional[str] = None) -> Optional[DashboardAuthProvider]:
    """Return the registered provider for ``name``, or None if unknown."""
    with _lock:
        return _merged(scope).get(name)


def snapshot_registration(
    name: str, *, scope: Optional[str] = None) -> Optional[DashboardAuthProvider]:
    with _lock:
        return _target(scope, create=False).get(name)


def restore_registration(
    name: str, current: DashboardAuthProvider, previous: Optional[DashboardAuthProvider],
    *, scope: Optional[str] = None) -> bool:
    """Restore a host-owned provider registration if it is still current."""
    with _lock:
        target = _target(scope, create=True)
        if target.get(name) is not current:
            return False
        if previous is None:
            target.pop(name, None)
        else:
            target[name] = previous
        if scope is not None and not target:
            _scoped_providers.pop(scope, None)
    return True


def list_providers(*, scope: Optional[str] = None) -> List[DashboardAuthProvider]:
    """All registered providers, in registration order."""
    with _lock:
        return list(_merged(scope).values())


def list_token_providers() -> List[DashboardAuthProvider]:
    """Providers with ``supports_token`` True, in registration order. The ``token_auth`` seam
    consults only these, so OAuth/password-only providers are never asked to ``verify_token``;
    empty => a token-authable route fails closed (401)."""
    return [p for p in list_providers() if getattr(p, "supports_token", False)]


def list_session_providers() -> List[DashboardAuthProvider]:
    """Providers with ``supports_session`` True (interactive cookie sessions); the login page,
    /auth/login and the gate's verify/refresh loops use only these."""
    return [p for p in list_providers() if getattr(p, "supports_session", True)]


def register_global_provider(provider: DashboardAuthProvider, *, scope: Optional[str] = None) -> bool:
    """Register a host-owned provider in the process-global slot (upsert). The registry is shared
    across every profile one dashboard process serves, so these outlive any per-home plugin
    manager: always targets ``_providers`` (never a per-home overlay), so a forced plugin
    re-discovery (e.g. after a password change) rotates the provider in place.

    ``scope`` identifies the profile whose plugin manager is registering. A same-name provider
    from a DIFFERENT scope than the one already holding the name is refused (warn, keep the live
    provider) rather than upserted — otherwise background plugin discovery for another profile
    (e.g. the dashboard UI following the machine's sticky active profile) can silently replace the
    provider actively validating an unrelated, already-authenticated Dashboard's sessions with one
    signed for a different profile (#106608). Returns whether the registration took effect, so the
    caller can tell an operator apart from a silent no-op.

    Pairs with ``unregister_global_provider`` for teardown of the exact object still current (#91701).
    """
    assert_protocol_compliance(type(provider))
    with _lock:
        owner = _provider_owners.get(provider.name)
        if (provider.name in _providers and owner is not None and scope is not None
                and owner != scope):
            _log.warning(
                "dashboard-auth: refusing to replace provider %r (owned by profile %r) with one "
                "registered for profile %r", provider.name, owner, scope)
            return False
        _providers[provider.name] = provider
        _provider_owners[provider.name] = scope
    _log_registered("global provider ", provider)
    return True


def unregister_global_provider(name: str, provider: DashboardAuthProvider) -> bool:
    """Remove a global registration if ``provider`` is still current (a stale handle whose
    provider was already replaced never clears the live one)."""
    with _lock:
        if _providers.get(name) is provider:
            _providers.pop(name, None)
            _provider_owners.pop(name, None)
            return True
    return False


def clear_providers() -> None:
    """Test-only: drop all registrations."""
    with _lock:
        _providers.clear()
        _scoped_providers.clear()
        _provider_owners.clear()
