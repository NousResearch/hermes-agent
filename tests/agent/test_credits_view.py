"""Tests for the /credits command — shared view core + gateway handler.

`/credits` is the focused money surface (balance in, top-up out). These tests
exercise the surface-agnostic `build_credits_view()` core and assert the gateway
handler renders the block + tappable top-up URL + no-wait copy. The CLI panel is
a thin wrapper over the same view (interactive prompt_toolkit modal — covered by
the view-core tests plus manual verification).
"""

from __future__ import annotations

import asyncio

import pytest

import agent.account_usage as account_usage
from agent.account_usage import CreditsView, build_credits_view
from hermes_cli.nous_account import NousPortalAccountInfo, NousPaidServiceAccessInfo


def _account(**kwargs) -> NousPortalAccountInfo:
    kwargs.setdefault("logged_in", True)
    kwargs.setdefault("source", "account_api")
    kwargs.setdefault("fresh", True)
    kwargs.setdefault("portal_base_url", "https://portal.example.test")
    return NousPortalAccountInfo(**kwargs)


@pytest.fixture
def _logged_in_account(monkeypatch):
    """Stub the auth token + account fetch so build_credits_view runs offline."""
    monkeypatch.setattr(
        "hermes_cli.auth.get_provider_auth_state",
        lambda provider: {"access_token": "tok", "portal_base_url": "https://portal.example.test"},
    )

    def _install(account):
        monkeypatch.setattr(
            "hermes_cli.nous_account.get_nous_portal_account_info",
            lambda *a, **kw: account,
        )

    return _install


# ── build_credits_view core ─────────────────────────────────────────────────


def test_view_logged_out_when_no_token(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: {})
    view = build_credits_view()
    assert view == CreditsView(logged_in=False)


def test_view_built_with_org_pinned_url_and_identity(_logged_in_account):
    _logged_in_account(
        _account(
            org_slug="acme",
            org_name="Acme Inc",
            email="alice@example.test",
            paid_service_access=True,
            paid_service_access_info=NousPaidServiceAccessInfo(
                purchased_credits_remaining=30.0,
                total_usable_credits=30.0,
            ),
            subscription=None,
        )
    )

    view = build_credits_view()

    assert view.logged_in is True
    assert view.topup_url == "https://portal.example.test/orgs/acme/billing?topup=open"
    assert view.identity_line == "Topping up as alice@example.test / org Acme Inc"
    assert view.depleted is False
    # Balance lines carry the magnitudes but NOT the /usage affordance lines.
    blob = "\n".join(view.balance_lines)
    assert "Top-up credits: $30.00" in blob
    assert "Top up:" not in blob  # the trailing /usage affordance is stripped
    assert "(or run" not in blob


def test_view_depleted_flag(_logged_in_account):
    _logged_in_account(
        _account(
            org_slug="acme",
            email="alice@example.test",
            paid_service_access=False,
            paid_service_access_info=NousPaidServiceAccessInfo(
                total_usable_credits=0.0,
            ),
            subscription=None,
        )
    )

    view = build_credits_view()
    assert view.depleted is True


def test_view_falls_back_to_legacy_url_when_slug_null(_logged_in_account):
    _logged_in_account(
        _account(
            org_slug=None,
            email="alice@example.test",
            paid_service_access=True,
            paid_service_access_info=NousPaidServiceAccessInfo(
                purchased_credits_remaining=5.0,
                total_usable_credits=5.0,
            ),
            subscription=None,
        )
    )

    view = build_credits_view()
    assert view.topup_url == "https://portal.example.test/billing?topup=open"
    assert "/orgs/" not in view.topup_url


def test_view_fetch_failure_is_logged_out(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.get_provider_auth_state",
        lambda provider: {"access_token": "tok"},
    )

    def _boom(*a, **kw):
        raise RuntimeError("portal down")

    monkeypatch.setattr("hermes_cli.nous_account.get_nous_portal_account_info", _boom)

    view = build_credits_view()
    assert view.logged_in is False


# ── gateway _handle_topup_command (the messaging billing surface) ────────────


class _FakeEvent:
    pass


def _make_gateway_stub():
    """Minimal object exposing the mixin's _handle_topup_command."""
    from gateway.slash_commands import GatewaySlashCommandsMixin

    class _Stub(GatewaySlashCommandsMixin):
        def __init__(self):
            pass

    return _Stub()


def test_gateway_topup_renders_block_and_url(monkeypatch):
    view = CreditsView(
        logged_in=True,
        balance_lines=("📈 Nous credits", "Total usable: $52.50"),
        identity_line="Topping up as alice@example.test / org Acme",
        topup_url="https://portal.example.test/orgs/acme/billing?topup=open",
        depleted=False,
    )
    monkeypatch.setattr(account_usage, "build_credits_view", lambda *a, **kw: view)

    stub = _make_gateway_stub()
    out = asyncio.run(stub._handle_topup_command(_FakeEvent()))

    assert "💳" in out
    assert "Total usable: $52.50" in out
    assert "Topping up as alice@example.test / org Acme" in out
    assert "https://portal.example.test/orgs/acme/billing?topup=open" in out
    assert "Manage billing on the portal" in out
    # The helper's own 📈 header line is dropped (we render our own 💳 header).
    assert "📈 Nous credits" not in out


def test_gateway_topup_not_logged_in(monkeypatch):
    monkeypatch.setattr(
        account_usage, "build_credits_view", lambda *a, **kw: CreditsView(logged_in=False)
    )
    stub = _make_gateway_stub()
    out = asyncio.run(stub._handle_topup_command(_FakeEvent()))
    assert "Not logged into Nous Portal" in out


def test_gateway_topup_fetch_exception_is_not_logged_in(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(account_usage, "build_credits_view", _boom)
    stub = _make_gateway_stub()
    out = asyncio.run(stub._handle_topup_command(_FakeEvent()))
    assert "Not logged into Nous Portal" in out


# ── command registry ────────────────────────────────────────────────────────


def test_credits_command_fully_removed():
    """`/credits` and the old `/billing` are gone entirely — not commands, not
    aliases. Billing lives only on /topup, with NO aliases, on every platform."""
    from hermes_cli.commands import resolve_command, COMMAND_REGISTRY

    # Both old names resolve to nothing.
    assert resolve_command("credits") is None
    assert resolve_command("billing") is None
    # No standalone command for either remains in the registry.
    assert not any(c.name in ("credits", "billing") for c in COMMAND_REGISTRY)
    # And no command carries either as an alias.
    for c in COMMAND_REGISTRY:
        assert "credits" not in (c.aliases or ())
        assert "billing" not in (c.aliases or ())
    # /topup is the billing surface, on every surface, and carries no aliases.
    entry = next(c for c in COMMAND_REGISTRY if c.name == "topup")
    assert entry.cli_only is False
    assert entry.gateway_only is False
    assert not entry.aliases
