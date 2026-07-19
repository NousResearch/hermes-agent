"""Tests for the /billing CLI handler (cli.py::_show_billing).

Focus on the non-interactive (no live prompt_toolkit app) path — the same
discipline as the /credits non-interactive test: it must render text, never
invoke the modal (which would read the slash-worker's JSON-RPC stdin and hang).
Plus role/kill-switch gating and logged-out handling.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

import agent.billing_view as bv
from agent.billing_view import BillingState, CardInfo, MonthlyCap
from cli import HermesCLI


@pytest.fixture
def cli():
    obj = HermesCLI.__new__(HermesCLI)  # bypass __init__ (no full app needed)
    obj._app = None  # non-interactive: forces the text path
    return obj


def _boom_modal(*a, **kw):
    raise AssertionError("modal must NOT be called in non-interactive mode")


def test_billing_logged_out(cli, monkeypatch, capsys):
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: BillingState(logged_in=False))
    cli._show_billing("/billing")
    out = capsys.readouterr().out
    assert "Not logged into Nous Portal" in out
    assert "hermes portal" in out


def test_billing_overview_non_interactive_renders_text_not_modal(cli, monkeypatch, capsys):
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _boom_modal, raising=False)
    state = BillingState(
        logged_in=True,
        org_name="Acme",
        role="OWNER",
        balance_usd=Decimal("142.5"),
        cli_billing_enabled=True,
        charge_presets=(Decimal("100"),),
        monthly_cap=MonthlyCap(limit_usd=Decimal("1000"), spent_this_month_usd=Decimal("180"),
                               is_default_ceiling=True),
        portal_url="https://portal/billing?topup=open",
    )
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: state)
    cli._show_billing("/billing")
    out = capsys.readouterr().out
    # Balance now leads in the title; dollars, never "credits".
    assert "Top up · balance $142.50" in out
    assert "credits" not in out.lower()
    # ZERO sub-commands: no /billing buy|auto-reload|limit advertising.
    assert "/billing buy" not in out
    assert "Actions:" not in out
    # Non-interactive funnels to the portal (the URL is the affordance).
    assert "Manage on portal:" in out


def test_billing_member_cannot_charge(cli, monkeypatch, capsys):
    state = BillingState(
        logged_in=True, role="MEMBER", balance_usd=Decimal("10"),
        cli_billing_enabled=True, portal_url="https://portal/billing",
    )
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: state)
    cli._show_billing("/billing")
    out = capsys.readouterr().out
    assert "require an org admin/owner" in out


def test_billing_killswitch_off_blocks(cli, monkeypatch, capsys):
    state = BillingState(
        logged_in=True, role="OWNER", balance_usd=Decimal("10"),
        cli_billing_enabled=False, portal_url="https://portal/billing",
    )
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: state)
    cli._show_billing("/billing")
    out = capsys.readouterr().out
    assert "turned off for this org" in out


def test_billing_limit_screen_readonly(cli, monkeypatch, capsys):
    state = BillingState(
        logged_in=True, role="OWNER", cli_billing_enabled=True,
        monthly_cap=MonthlyCap(limit_usd=Decimal("1000"), spent_this_month_usd=Decimal("250"),
                               is_default_ceiling=True),
        portal_url="https://portal/billing",
    )
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: state)
    # ZERO sub-commands: the limit screen is reached via the menu, never a
    # sub-command — call it directly the way the overview menu would.
    cli._billing_limit_screen(state)
    out = capsys.readouterr().out
    assert "Monthly spend limit" in out
    assert "$250 of $1000 used" in out
    assert "read-only" in out


def test_billing_sub_arg_ignored_opens_overview(cli, monkeypatch, capsys):
    # A stray sub-arg must NOT error and must NOT dispatch to a sub-screen —
    # it just opens the overview (spec §0.4: zero sub-commands).
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _boom_modal, raising=False)
    state = BillingState(
        logged_in=True, role="OWNER", balance_usd=Decimal("142.5"),
        cli_billing_enabled=True, charge_presets=(Decimal("25"),),
        portal_url="https://portal/billing",
    )
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: state)
    cli._show_billing("/billing buy")  # arg is ignored
    out = capsys.readouterr().out
    assert "Top up · balance" in out  # overview, NOT the buy screen
    # The buy screen's preset list isn't shown. (The overview's no-card hint may
    # legitimately mention the "Add funds" menu item, so key on presets instead.)
    assert "Presets:" not in out


def test_billing_buy_non_interactive_defers_to_portal(cli, monkeypatch, capsys):
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _boom_modal, raising=False)
    state = BillingState(
        logged_in=True, role="OWNER", cli_billing_enabled=True,
        charge_presets=(Decimal("25"), Decimal("50"), Decimal("100")),
        card=CardInfo(brand="visa", last4="4242"),
        portal_url="https://portal/billing",
    )
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: state)
    # Reached via the menu in real use; non-interactively it defers to the portal.
    cli._billing_buy_flow(state)
    out = capsys.readouterr().out
    assert "Add funds" in out
    assert "$25" in out and "$50" in out and "$100" in out
    assert "interactive CLI" in out  # defers; no charge attempted non-interactively


# ── Card visibility + the add-card path (inline w/ NAS card-resolver) ──


def _scripted(*responses):
    it = iter(responses)

    def _modal(self, **kw):
        return next(it)

    return _modal


def test_overview_shows_card_with_provenance(cli, monkeypatch, capsys):
    state = BillingState(
        logged_in=True, role="OWNER", balance_usd=Decimal("10"),
        cli_billing_enabled=True, charge_presets=(Decimal("25"),),
        card=CardInfo(brand="Visa", last4="4242", resolved_via="subPin"),
        portal_url="https://portal/billing",
    )
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: state)
    cli._show_billing("/topup")
    out = capsys.readouterr().out
    assert "Card: Visa ····4242 — the card on your subscription" in out


def test_overview_shows_no_card_hint(cli, monkeypatch, capsys):
    state = BillingState(
        logged_in=True, role="OWNER", balance_usd=Decimal("10"),
        cli_billing_enabled=True, charge_presets=(Decimal("25"),),
        card=None, portal_url="https://portal/billing",
    )
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: state)
    cli._show_billing("/topup")
    out = capsys.readouterr().out
    assert "No saved card on file" in out
    assert "Add funds" in out  # the hint names the path


def test_link_card_renders_brand_alone(cli, monkeypatch, capsys):
    # A Link payment method has no card number (last4 = "") — never "Link ····".
    state = BillingState(
        logged_in=True, role="OWNER", balance_usd=Decimal("10"),
        cli_billing_enabled=True, charge_presets=(Decimal("25"),),
        card=CardInfo(brand="Link", last4=""), portal_url="https://portal/billing",
    )
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: state)
    cli._show_billing("/topup")
    out = capsys.readouterr().out
    assert "Card: Link" in out
    assert "Link ····" not in out


def test_buy_flow_no_card_guides_then_continues_after_recheck(cli, monkeypatch, capsys):
    # No card → the guided add-card path; "check again" re-fetches state and,
    # once the card exists, continues straight into the preset menu.
    cli._app = object()
    common = dict(
        logged_in=True, role="OWNER", cli_billing_enabled=True,
        charge_presets=(Decimal("25"), Decimal("50")),
        min_usd=Decimal("5"), max_usd=Decimal("500"),
        portal_url="https://portal/billing",
    )
    nocard = BillingState(card=None, **common)
    withcard = BillingState(card=CardInfo(brand="Visa", last4="4242", resolved_via="customerDefault"), **common)
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: withcard)
    # add-card modal → "recheck"; preset modal → "cancel" (we only test the routing)
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted("recheck", "cancel"), raising=False)

    cli._billing_buy_flow(nocard)
    out = capsys.readouterr().out

    assert "Add a card first" in out
    assert "Card found: Visa ····4242 — your default card saved on the portal" in out
    assert "Cancelled. No funds added." in out  # reached the preset menu, then bailed


def test_buy_flow_no_card_back_abandons(cli, monkeypatch, capsys):
    cli._app = object()
    nocard = BillingState(
        logged_in=True, role="OWNER", cli_billing_enabled=True,
        charge_presets=(Decimal("25"),), card=None,
        portal_url="https://portal/billing",
    )
    calls = {"n": 0}

    def _no_fetch(*a, **kw):
        calls["n"] += 1
        return nocard

    monkeypatch.setattr(bv, "build_billing_state", _no_fetch)
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted("cancel"), raising=False)

    cli._billing_buy_flow(nocard)
    out = capsys.readouterr().out

    assert "Add a card first" in out
    assert "Cancelled. No funds added." in out
    assert calls["n"] == 0  # backed out before any re-check
