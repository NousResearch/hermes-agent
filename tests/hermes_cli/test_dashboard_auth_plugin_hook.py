"""The plugin context exposes register_dashboard_auth_provider.

Mirrors the image-gen / memory-provider hooks (see plugins.py:531 for prior
art).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.dashboard_auth import clear_providers, get_provider
from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider, LoginStart, Session,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.dashboard_auth import registry as _auth_registry
from hermes_constants import get_process_hermes_home, hermes_home_key


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
    _manager, ctx = _real_ctx()
    old = _Basic("old")
    new = _Basic("new")
    stale = ctx.register_dashboard_auth_provider(old)
    ctx.register_dashboard_auth_provider(new)
    assert get_provider("basic") is new

    # The superseded handle is identity-conditional: disposing it is a no-op.
    stale.dispose()
    assert get_provider("basic") is new


def test_profile_manager_cannot_replace_launch_home_auth_provider():
    """Profile plugin discovery must not rotate the dashboard's auth gate."""
    manager, ctx = _real_ctx()
    launch_provider = _Basic("launch-home")
    ctx.register_dashboard_auth_provider(launch_provider)

    profile_scope = hermes_home_key(Path(manager.scope_key) / "profiles" / "bot")
    profile_manager = PluginManager(scope_key=profile_scope)
    profile_ctx = PluginContext(
        PluginManifest(name="basic", version="0.0.1", kind="backend"),
        manager=profile_manager,
    )

    registration = profile_ctx.register_dashboard_auth_provider(_Basic("profile"))

    assert registration is None
    assert get_provider("basic") is launch_provider
    assert "basic" not in profile_manager._ownership_ledger


def test_profile_directory_can_be_the_process_launch_home(monkeypatch, tmp_path):
    """A process launched directly into a profile owns dashboard auth."""
    launch_home = tmp_path / "profiles" / "bot"
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    manager, ctx = _real_ctx()
    provider = _Basic("profile-launch-home")

    registration = ctx.register_dashboard_auth_provider(provider)

    assert manager.scope_key == hermes_home_key(get_process_hermes_home())
    assert registration is not None
    assert get_provider("basic") is provider


# ---------------------------------------------------------------------------
# #91701 follow-up: persistence must not outlive the plugin. A targeted
# unload (plugin disable/uninstall) and a re-discovery that drops the plugin
# must both release the process-global provider — only the ROUTINE
# unload-all path keeps it alive.
# ---------------------------------------------------------------------------


def test_targeted_unload_disposes_persistent_auth_provider():
    """Disabling the auth plugin removes its provider process-wide."""
    manager, ctx = _real_ctx()
    ctx.register_dashboard_auth_provider(_Basic())
    assert get_provider("basic") is not None

    # `hermes plugins disable basic` drives a targeted unload of that plugin.
    assert manager.unload("basic") is True

    assert get_provider("basic") is None, (
        "disabled auth plugin's provider stayed registered process-wide"
    )
    assert _auth_registry.list_session_providers() == []


def test_rediscovery_evicts_provider_when_plugin_gone():
    """Force re-discovery where the plugin does not come back → evicted."""
    manager, ctx = _real_ctx()
    ctx.register_dashboard_auth_provider(_Basic())

    # discover_and_load(force=True) step 1: unload-all parks the handle.
    manager.unload()
    assert get_provider("basic") is not None

    # Step 2: discovery ran, plugin did not re-register (disabled/removed).
    manager._evict_stale_persistent_registrations()

    assert get_provider("basic") is None, (
        "provider survived a re-discovery its plugin was dropped from"
    )


def test_rediscovery_keeps_provider_when_plugin_returns():
    """Force re-discovery where the plugin re-registers → new provider live."""
    manager, ctx = _real_ctx()
    old = _Basic("old")
    ctx.register_dashboard_auth_provider(old)

    manager.unload()
    # Plugin re-registers during discovery (upsert rotates in place).
    new = _Basic("new")
    ctx.register_dashboard_auth_provider(new)
    manager._evict_stale_persistent_registrations()

    assert get_provider("basic") is new
    # Eviction must be one-shot: the parked list is drained.
    assert manager._persistent_carryover == []


def test_rediscovery_same_object_reregistration_survives_eviction():
    """A plugin re-registering the SAME provider object must stay live."""
    manager, ctx = _real_ctx()
    provider = _Basic("same")
    ctx.register_dashboard_auth_provider(provider)

    manager.unload()
    ctx.register_dashboard_auth_provider(provider)
    manager._evict_stale_persistent_registrations()

    assert get_provider("basic") is provider


def test_persistent_dispose_is_idempotent_after_targeted_unload():
    """A handle disposed by a targeted unload never re-parks or re-releases."""
    manager, ctx = _real_ctx()
    ctx.register_dashboard_auth_provider(_Basic())

    manager.unload("basic")   # targeted unload disposes + forgets the handle
    assert get_provider("basic") is None

    # A later unload-all parks nothing (the handle is gone from the ledger),
    # and the eviction pass must not raise or double-release.
    manager.unload()
    assert manager._persistent_carryover == []
    manager._evict_stale_persistent_registrations()
    assert get_provider("basic") is None


# ---------------------------------------------------------------------------
# Idempotent re-registration: a multiplexed gateway builds one plugin manager
# per served profile, and each profile-scoped manager rediscovers the bundled
# auth backend and re-registers the SAME launch-owned provider every boot.
# The registry merges the global slot into every profile scope anyway, so the
# re-registration is a provable no-op — it must not spam a WARNING per profile
# per boot. The warning stays for genuinely first-time foreign-scope
# registrations (scope isolation is still enforced).
# ---------------------------------------------------------------------------


def _profile_ctx(manager: PluginManager) -> tuple[PluginContext, str, PluginManager]:
    profile_scope = hermes_home_key(Path(manager.scope_key) / "profiles" / "bot")
    profile_manager = PluginManager(scope_key=profile_scope)
    profile_ctx = PluginContext(
        PluginManifest(name="basic", version="0.0.1", kind="backend"),
        manager=profile_manager,
    )
    return profile_ctx, profile_scope, profile_manager


def test_profile_re_registering_launch_provider_is_quiet_no_op(caplog):
    """Same-name profile-scope re-registration of a launch-owned provider:
    debug log, no warning, no registry change."""
    import logging

    manager, ctx = _real_ctx()
    launch_provider = _Basic("launch-home")
    ctx.register_dashboard_auth_provider(launch_provider)

    profile_ctx, profile_scope, profile_manager = _profile_ctx(manager)

    with caplog.at_level(logging.DEBUG):
        registration = profile_ctx.register_dashboard_auth_provider(_Basic("profile"))

    assert registration is None
    # No behavior change: the launch provider still owns the name, and it is
    # resolvable from the profile scope (the registry merge covers it).
    assert get_provider("basic") is launch_provider
    assert get_provider("basic", scope=profile_scope) is launch_provider
    assert "basic" not in profile_manager._ownership_ledger

    assert not any(
        rec.levelno >= logging.WARNING and "dashboard-auth provider" in rec.message
        for rec in caplog.records
    )
    assert any(
        rec.levelno == logging.DEBUG and "already owns it" in rec.message
        for rec in caplog.records
    )


def test_first_foreign_scope_registration_still_warns(caplog):
    """A provider name nobody owns yet keeps the WARNING + refusal."""
    import logging

    profile_ctx, _scope, _manager = _profile_ctx(_real_ctx()[0])

    with caplog.at_level(logging.WARNING):
        registration = profile_ctx.register_dashboard_auth_provider(_Basic())

    assert registration is None
    assert get_provider("basic") is None
    assert any("owned by launch scope" in rec.message for rec in caplog.records)


def test_foreign_registration_warns_after_launch_provider_released(caplog):
    """Once the launch plugin releases the global slot (e.g. it is disabled),
    a later profile-scope registration is first-time again and still warns."""
    import logging

    manager, ctx = _real_ctx()
    ctx.register_dashboard_auth_provider(_Basic())
    manager.unload("basic")  # plugin disabled: the global slot is released
    assert get_provider("basic") is None

    profile_ctx, _scope, _manager = _profile_ctx(manager)

    with caplog.at_level(logging.WARNING):
        assert profile_ctx.register_dashboard_auth_provider(_Basic("late")) is None

    assert get_provider("basic") is None
    assert any("owned by launch scope" in rec.message for rec in caplog.records)
