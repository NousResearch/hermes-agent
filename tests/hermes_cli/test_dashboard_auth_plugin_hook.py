"""The plugin context exposes register_dashboard_auth_provider.

Mirrors the image-gen / memory-provider hooks (see plugins.py:531 for prior
art).
"""
from __future__ import annotations

import pytest

from hermes_cli.dashboard_auth import clear_providers, get_provider
from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider, LoginStart, Session,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.dashboard_auth import registry as _auth_registry
from hermes_constants import hermes_home_key


class _Stub(DashboardAuthProvider):
    name = "stub"
    display_name = "Stub IdP"

    def start_login(self, *, redirect_uri):
        return LoginStart(redirect_url="x", cookie_payload={})

    def complete_login(self, *, code, state, code_verifier, redirect_uri):
        return Session("u", "e", "n", "o", "stub", 0, "a", "r")

    def verify_session(self, *, access_token):
        return None

    def refresh_session(self, *, refresh_token):
        return Session("u", "e", "n", "o", "stub", 0, "a", "r")

    def revoke_session(self, *, refresh_token):
        return None


class _MinimalManager:
    """The fixture only needs whatever PluginContext touches at register-time.

    We don't import the real PluginManager because it pulls in the full
    plugin-discovery surface.  The hook we're testing only reads from
    ``ctx.manifest``, so the manager attributes don't matter — but we set
    the few that other PluginContext methods touch defensively.
    """

    _cli_ref = None
    _context_engine = None
    _tools: dict = {}


@pytest.fixture(autouse=True)
def _isolated_registry():
    clear_providers()
    yield
    clear_providers()


def _make_ctx(name: str = "dashboard-auth-stub") -> PluginContext:
    manifest = PluginManifest(name=name, version="0.0.1", description="stub")
    return PluginContext(manifest=manifest, manager=_MinimalManager())  # type: ignore[arg-type]


def test_plugin_ctx_exposes_register_dashboard_auth_provider():
    ctx = _make_ctx()
    assert hasattr(ctx, "register_dashboard_auth_provider")


def test_plugin_ctx_silently_ignores_non_provider(caplog):
    """Mirror image_gen behaviour: log warning, leave registry empty.

    We do NOT raise — a misbehaving plugin must not crash the host.
    """
    import logging
    ctx = _make_ctx("dashboard-auth-bad")
    with caplog.at_level(logging.WARNING):
        ctx.register_dashboard_auth_provider("not a provider")  # type: ignore[arg-type]
    assert get_provider("stub") is None
    assert any(
        "dashboard-auth-bad" in rec.message
        and "DashboardAuthProvider" in rec.message
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# #91701: a dashboard-auth provider is process-global host infrastructure. A
# per-home plugin manager is torn down routinely (profile-scoped dashboard
# activity, force re-discovery); that teardown must NOT empty the auth
# registry and lock the whole process out of sign-in.
# ---------------------------------------------------------------------------


class _Basic(DashboardAuthProvider):
    name = "basic"
    display_name = "Basic"

    def __init__(self, tag: str = "a") -> None:
        self.tag = tag

    def start_login(self, *, redirect_uri):
        return LoginStart(redirect_url="x", cookie_payload={})

    def complete_login(self, *, code, state, code_verifier, redirect_uri):
        return Session("u", "e", "n", "o", "basic", 0, "a", "r")

    def verify_session(self, *, access_token):
        return None

    def refresh_session(self, *, refresh_token):
        return None

    def revoke_session(self, *, refresh_token):
        return None


def _real_ctx() -> tuple[PluginManager, PluginContext]:
    manager = PluginManager(scope_key=hermes_home_key())
    manifest = PluginManifest(name="basic", version="0.0.1", kind="backend")
    return manager, PluginContext(manifest=manifest, manager=manager)


def test_auth_provider_registers_globally_not_in_home_overlay():
    """Registered in the process-global slot so every profile scope sees it."""
    manager, ctx = _real_ctx()
    ctx.register_dashboard_auth_provider(_Basic())
    assert "basic" in _auth_registry._providers
    assert "basic" not in _auth_registry._scoped_providers.get(
        manager.scope_key, {}
    )
    assert [p.name for p in _auth_registry.list_session_providers()] == ["basic"]


def test_auth_provider_survives_per_home_manager_unload():
    """Regression for #91701: routine per-home unload must not disable auth."""
    manager, ctx = _real_ctx()
    ctx.register_dashboard_auth_provider(_Basic())
    assert get_provider("basic") is not None

    # The exact teardown discover_and_load(force=True) / profile-scoped
    # activity drives; before the fix this emptied the registry permanently.
    manager.unload()

    assert get_provider("basic") is not None, (
        "auth provider was disposed by a per-home plugin-manager unload"
    )
    assert [p.name for p in _auth_registry.list_session_providers()] == ["basic"]


def test_auth_provider_kept_out_of_manager_teardown_order():
    """Persistent registration is not enrolled in reverse-order teardown."""
    manager, ctx = _real_ctx()
    ctx.register_dashboard_auth_provider(_Basic())
    assert manager._registration_order == []
    # Still attributed to the plugin for `hermes plugins list`.
    assert "basic" in manager._ownership_ledger


def test_auth_provider_re_register_rotates_in_place():
    """A forced re-discovery (e.g. password change) upserts the new provider."""
    manager, ctx = _real_ctx()
    old = _Basic("old")
    new = _Basic("new")
    stale = ctx.register_dashboard_auth_provider(old)
    ctx.register_dashboard_auth_provider(new)
    assert get_provider("basic") is new

    # The superseded handle is identity-conditional: disposing it is a no-op.
    stale.dispose()
    assert get_provider("basic") is new
