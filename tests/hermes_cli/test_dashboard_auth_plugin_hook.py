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
from hermes_cli.plugins import PluginContext, PluginManifest


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


def test_plugin_ctx_register_dashboard_auth_provider_happy_path():
    ctx = _make_ctx()
    ctx.register_dashboard_auth_provider(_Stub())
    p = get_provider("stub")
    assert p is not None
    assert p.display_name == "Stub IdP"


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
