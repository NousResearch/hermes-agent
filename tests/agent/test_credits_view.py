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


# ── gateway _handle_credits_command ─────────────────────────────────────────


class _FakeEvent:
    pass


def _make_gateway_stub():
    """Minimal object exposing the mixin's _handle_credits_command."""
    from gateway.slash_commands import GatewaySlashCommandsMixin

    class _Stub(GatewaySlashCommandsMixin):
        def __init__(self):
            pass

    return _Stub()


def test_gateway_credits_renders_block_and_url(monkeypatch):
    view = CreditsView(
        logged_in=True,
        balance_lines=("📈 Nous credits", "Total usable: $52.50"),
        identity_line="Topping up as alice@example.test / org Acme",
        topup_url="https://portal.example.test/orgs/acme/billing?topup=open",
        depleted=False,
    )
    monkeypatch.setattr(account_usage, "build_credits_view", lambda *a, **kw: view)

    stub = _make_gateway_stub()
    out = asyncio.run(stub._handle_credits_command(_FakeEvent()))

    assert "💳" in out
    assert "Total usable: $52.50" in out
    assert "Topping up as alice@example.test / org Acme" in out
    assert "https://portal.example.test/orgs/acme/billing?topup=open" in out
    assert "credits will appear in /credits shortly" in out
    # The helper's own 📈 header line is dropped (we render our own 💳 header).
    assert "📈 Nous credits" not in out


def test_gateway_credits_not_logged_in(monkeypatch):
    monkeypatch.setattr(
        account_usage, "build_credits_view", lambda *a, **kw: CreditsView(logged_in=False)
    )
    stub = _make_gateway_stub()
    out = asyncio.run(stub._handle_credits_command(_FakeEvent()))
    assert "Not logged into Nous Portal" in out


def test_gateway_credits_fetch_exception_is_not_logged_in(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(account_usage, "build_credits_view", _boom)
    stub = _make_gateway_stub()
    out = asyncio.run(stub._handle_credits_command(_FakeEvent()))
    assert "Not logged into Nous Portal" in out


# ── command registry ────────────────────────────────────────────────────────


def test_credits_command_registered():
    from hermes_cli.commands import resolve_command, COMMAND_REGISTRY

    cmd = resolve_command("credits")
    assert cmd is not None and cmd.name == "credits"
    # Available on every surface (not cli_only / gateway_only).
    entry = next(c for c in COMMAND_REGISTRY if c.name == "credits")
    assert entry.cli_only is False
    assert entry.gateway_only is False


# ── CLI _show_credits non-interactive (TUI slash-worker) path ───────────────


def test_cli_show_credits_non_interactive_renders_text_not_modal(monkeypatch, capsys):
    """In the TUI slash-worker (no self._app), /credits must render the text
    variant — never invoke the prompt_toolkit modal, which would read the
    worker's JSON-RPC stdin and crash the command (only the depleted banner
    would survive). Regression for that exact failure.
    """
    import agent.account_usage as account_usage
    from cli import HermesCLI

    monkeypatch.setattr(
        account_usage,
        "build_credits_view",
        lambda *a, **k: CreditsView(
            logged_in=True,
            balance_lines=("📈 Nous credits", "Total usable: $0.00"),
            identity_line="Topping up as a@b.c / org Acme",
            topup_url="https://prev.test/orgs/acme/billing?topup=open",
            depleted=True,
        ),
    )

    cli = HermesCLI.__new__(HermesCLI)
    cli._app = None  # non-interactive, like the slash worker

    # Must NOT call the modal in this context.
    def _boom_modal(*a, **k):
        raise AssertionError("modal must not run without a live app")

    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _boom_modal, raising=False)

    cli._show_credits()

    out = capsys.readouterr().out
    assert "💳 Nous credits" in out
    assert "Total usable: $0.00" in out
    assert "Topping up as a@b.c / org Acme" in out
    assert "https://prev.test/orgs/acme/billing?topup=open" in out
    assert "credits will appear in /credits shortly" in out


def test_cli_show_credits_logged_out(monkeypatch, capsys):
    import agent.account_usage as account_usage
    from cli import HermesCLI

    monkeypatch.setattr(
        account_usage, "build_credits_view", lambda *a, **k: CreditsView(logged_in=False)
    )
    cli = HermesCLI.__new__(HermesCLI)
    cli._app = None
    cli._show_credits()
    assert "Not logged into Nous Portal" in capsys.readouterr().out
