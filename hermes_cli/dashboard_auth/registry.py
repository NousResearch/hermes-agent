"""Module-level registry for DashboardAuthProvider instances.

Plugins call ``register_provider`` via the plugin context hook at startup.
The auth gate middleware iterates ``list_providers()`` and uses
``get_provider`` to dispatch on the session's ``provider`` field.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

from hermes_constants import hermes_home_key
from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    assert_protocol_compliance,
)

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_providers: dict[str, DashboardAuthProvider] = {}
_scoped_providers: dict[str, dict[str, DashboardAuthProvider]] = {}


def _merged(scope: Optional[str] = None) -> dict[str, DashboardAuthProvider]:
    providers = dict(_providers)
    providers.update(_scoped_providers.get(scope or hermes_home_key(), {}))
    return providers


def register_provider(
    provider: DashboardAuthProvider,
    *,
    scope: Optional[str] = None,
) -> None:
    """Register a provider.

    Raises:
        TypeError: on protocol violation.
        ValueError: if a provider with the same name is already registered.
    """
    assert_protocol_compliance(type(provider))
    with _lock:
        target = _providers if scope is None else _scoped_providers.setdefault(scope, {})
        effective = target if scope is None else _merged(scope)
        if provider.name in effective:
            raise ValueError(
                f"dashboard-auth provider already registered: {provider.name!r}"
            )
        target[provider.name] = provider
    _log.info(
        "dashboard-auth: registered provider %r (%s)",
        provider.name, provider.display_name,
    )


def get_provider(
    name: str,
    *,
    scope: Optional[str] = None,
) -> Optional[DashboardAuthProvider]:
    """Return the registered provider for ``name``, or None if unknown."""
    with _lock:
        return _merged(scope).get(name)


def snapshot_registration(
    name: str,
    *,
    scope: Optional[str] = None,
) -> Optional[DashboardAuthProvider]:
    with _lock:
        target = _providers if scope is None else _scoped_providers.get(scope, {})
        return target.get(name)


def restore_registration(
    name: str,
    current: DashboardAuthProvider,
    previous: Optional[DashboardAuthProvider],
    *,
    scope: Optional[str] = None,
) -> bool:
    """Restore a host-owned provider registration if it is still current."""
    with _lock:
        target = _providers if scope is None else _scoped_providers.setdefault(scope, {})
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
    """Registered providers that support non-interactive token auth.

    The subset of ``list_providers()`` whose ``supports_token`` flag is True,
    in registration order. The ``token_auth`` middleware seam consults these
    (and only these) when a token-authable route is hit, so OAuth/password-only
    providers are never asked to ``verify_token``. Returns an empty list when
    no token provider is registered — a token-authable route then fails
    closed (401), never open.
    """
    return [p for p in list_providers() if getattr(p, "supports_token", False)]


def list_session_providers() -> List[DashboardAuthProvider]:
    """Registered providers with supports_session True (interactive cookie
    sessions). The login page, /auth/login, and the gate's verify/refresh loops
    consult only these. Mirror of list_token_providers.
    """
    return [p for p in list_providers() if getattr(p, "supports_session", True)]


def register_global_provider(provider: DashboardAuthProvider) -> None:
    """Register a host-owned provider in the process-global slot (upsert).

    The dashboard auth registry is process-global and shared across every
    profile the dashboard serves from one process, so its providers must
    outlive any single per-home plugin manager. Unlike ``register_provider``
    this always targets the global ``_providers`` map (never a per-home
    overlay) and *replaces* any same-name entry instead of raising, so a
    forced plugin re-discovery (e.g. after a password change) rotates the
    provider in place. Pairs with ``unregister_global_provider`` for teardown
    of the exact object still current (#91701).
    """
    assert_protocol_compliance(type(provider))
    with _lock:
        _providers[provider.name] = provider
    _log.info(
        "dashboard-auth: registered global provider %r (%s)",
        provider.name, provider.display_name,
    )


def unregister_global_provider(
    name: str,
    provider: DashboardAuthProvider,
) -> bool:
    """Remove a global provider registration if ``provider`` is still current.

    Identity-conditional so a stale handle (whose provider was already
    replaced by a later ``register_global_provider``) never clears the live
    registration.
    """
    with _lock:
        if _providers.get(name) is provider:
            _providers.pop(name, None)
            return True
    return False


def clear_providers() -> None:
    """Test-only: drop all registrations."""
    with _lock:
        _providers.clear()
        _scoped_providers.clear()
