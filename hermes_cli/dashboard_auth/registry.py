"""Module-level registry for DashboardAuthProvider instances. Plugins call ``register_provider``
via the plugin context hook at startup; the auth gate iterates ``list_providers()`` and uses
``get_provider`` to dispatch on the session's ``provider`` field.

Storage, the per-profile overlays and the plugin snapshot/restore pair are the shared
:class:`agent.provider_registry.ProviderRegistry`. Registration, lookup and listing keep their own
contract below: names are exact (no normalization), a duplicate raises, and lists keep
registration order."""
from __future__ import annotations

import logging
from typing import List, Optional

from agent.provider_registry import ProviderRegistry, exact_key
from hermes_constants import hermes_home_key, normalize_scope
from hermes_cli.dashboard_auth.base import DashboardAuthProvider, assert_protocol_compliance

_log = logging.getLogger(__name__)
_registry: ProviderRegistry[DashboardAuthProvider] = ProviderRegistry(
    label="Dashboard-auth", provider_cls=DashboardAuthProvider, logger=_log, normalize=exact_key,
)
# Bound by hand rather than via ``export()``: register/get/list below keep this registry's own
# contract, so only the storage and the snapshot/restore pair come from the shared engine.
_providers = _registry._providers
_scoped_providers = _registry._scoped_providers
_lock = _registry._lock
_target = _registry._target
snapshot_registration = _registry.snapshot_registration
restore_registration = _registry.restore_registration


def _merged(scope: Optional[str] = None) -> dict[str, DashboardAuthProvider]:
    """Lock-free merge for callers already holding ``_lock``."""
    providers = dict(_providers)
    providers.update(_scoped_providers.get(hermes_home_key(scope), {}))
    return providers


def _log_registered(kind: str, provider: DashboardAuthProvider) -> None:
    _log.info("dashboard-auth: registered %s%r (%s)", kind, provider.name, provider.display_name)


def register_provider(provider: DashboardAuthProvider, *, scope: Optional[str] = None) -> None:
    """Raises ``TypeError`` on protocol violation, ``ValueError`` on a duplicate name."""
    assert_protocol_compliance(type(provider))
    with _lock:
        scope = normalize_scope(scope)
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


def list_providers(*, scope: Optional[str] = None) -> list[DashboardAuthProvider]:
    """All registered providers, in registration order."""
    with _lock:
        return list(_merged(scope).values())


def list_token_providers() -> list[DashboardAuthProvider]:
    """Providers with ``supports_token`` True, in registration order. The ``token_auth`` seam
    consults only these, so OAuth/password-only providers are never asked to ``verify_token``;
    empty => a token-authable route fails closed (401)."""
    return [p for p in list_providers() if getattr(p, "supports_token", False)]


def list_session_providers() -> list[DashboardAuthProvider]:
    """Providers with ``supports_session`` True (interactive cookie sessions); the login page,
    /auth/login and the gate's verify/refresh loops use only these."""
    return [p for p in list_providers() if getattr(p, "supports_session", True)]


def list_interactive_providers() -> list[DashboardAuthProvider]:
    """Session providers that offer a sign-in flow (``interactive_login`` True). Every login
    surface — the login page, ``/api/auth/providers``, auto-SSO, native authorize selection and
    chooser, the ``native_pkce`` advertisement — uses this; the verify/refresh loops keep
    :func:`list_session_providers`, so a provisioned-credential provider still authenticates."""
    return [p for p in list_session_providers() if getattr(p, "interactive_login", True)]


def register_global_provider(provider: DashboardAuthProvider) -> None:
    """Register a host-owned provider in the process-global slot (upsert). The registry is shared
    across every profile one dashboard process serves, so these outlive any per-home plugin
    manager: always targets ``_providers`` (never a per-home overlay) and *replaces* a same-name
    entry instead of raising, so a forced plugin re-discovery (e.g. after a password change)
    rotates the provider in place. Pairs with ``unregister_global_provider``.

    Pairs with ``unregister_global_provider`` for teardown of the exact object still current (#91701).
    """
    assert_protocol_compliance(type(provider))
    with _lock:
        _providers[provider.name] = provider
    _log_registered("global provider ", provider)


def unregister_global_provider(name: str, provider: DashboardAuthProvider) -> bool:
    """Remove a global registration if ``provider`` is still current (a stale handle whose
    provider was already replaced never clears the live one)."""
    with _lock:
        if _providers.get(name) is provider:
            _providers.pop(name, None)
            return True
    return False


clear_providers = _registry.reset_for_tests  # Test-only: drop all registrations.
