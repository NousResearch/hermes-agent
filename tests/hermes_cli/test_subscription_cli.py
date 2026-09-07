"""Tests for the /subscription CLI change flow (cli.py::_show_subscription).

Parity with the TUI overlay: the classic CLI now previews + applies a plan change
in-terminal (picker → preview → confirm → apply), allows remote spending inline on
insufficient_scope, and leads a scheduled downgrade/cancel with a prominent banner.
Interactive screens are driven by mocking `_prompt_text_input_modal`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

import agent.billing_usage as bu
import agent.subscription_view as sv
import hermes_cli.nous_billing as nb
from agent.subscription_view import CurrentSubscription, SubscriptionState, SubscriptionTier
from cli import HermesCLI


@pytest.fixture
def cli():
    obj = HermesCLI.__new__(HermesCLI)  # bypass __init__ (no full app needed)
    obj._app = None  # non-interactive by default; tests flip it on
    return obj


_TIERS = (
    SubscriptionTier(tier_id="free", name="Free", tier_order=0, dollars_per_month=Decimal("0"), monthly_credits=Decimal("0"), is_current=False, is_enabled=True),
    SubscriptionTier(tier_id="plus", name="Plus", tier_order=1, dollars_per_month=Decimal("20"), monthly_credits=Decimal("22"), is_current=False, is_enabled=True),
    SubscriptionTier(tier_id="ultra", name="Ultra", tier_order=3, dollars_per_month=Decimal("200"), monthly_credits=Decimal("220"), is_current=True, is_enabled=True),
)


def _sub_state(**current_over) -> SubscriptionState:
    current_fields = dict(tier_id="ultra", tier_name="Ultra", monthly_credits=Decimal("220"), cycle_ends_at="2026-07-28")
    current_fields.update(current_over)
    current = CurrentSubscription(**current_fields)
    return SubscriptionState(
        logged_in=True,
        org_name="Acme",
        org_id="org_1",
        role="OWNER",
        context="personal",
        current=current,
        tiers=_TIERS,
        portal_url="https://portal.example/billing",
    )


def _scripted_modal(*responses):
    it = iter(responses)

    def _modal(self, **kw):
        return next(it)

    return _modal


@pytest.fixture(autouse=True)
def _no_usage_model(monkeypatch):
    # The overview's usage model needs a live portal; None → the plan-field fallback.
    monkeypatch.setattr(bu, "build_usage_model", lambda *a, **kw: None, raising=False)


_FREE_TIERS = (
    SubscriptionTier(tier_id="free", name="Free", tier_order=0, dollars_per_month=Decimal("0"), monthly_credits=Decimal("0"), is_current=False, is_enabled=True),
    SubscriptionTier(tier_id="plus", name="Plus", tier_order=1, dollars_per_month=Decimal("20"), monthly_credits=Decimal("22"), is_current=False, is_enabled=True),
    SubscriptionTier(tier_id="ultra", name="Ultra", tier_order=3, dollars_per_month=Decimal("200"), monthly_credits=Decimal("220"), is_current=False, is_enabled=True),
)


def _free_state(tiers=_FREE_TIERS) -> SubscriptionState:
    return SubscriptionState(
        logged_in=True,
        org_name="Acme",
        org_id="org_1",
        role="OWNER",
        context="personal",
        current=None,  # Free = no plan
        tiers=tiers,
        portal_url="https://portal.example/billing",
    )


def _capture_opener(monkeypatch, *, opened_ok=False):
    """Patch the canonical browser opener to capture the URL; return whether a
    real browser 'opened' (default False → the printed-URL fallback fires)."""
    seen = {}

    def _open(self, url):
        seen["url"] = url
        return opened_ok

    monkeypatch.setattr(HermesCLI, "_open_url_in_browser", _open, raising=False)
    return seen


def test_free_prints_catalog_and_deep_links_with_plan(cli, monkeypatch, capsys):
    # (a)+(b): Free + admin + interactive prints the plan catalog (name · $/mo ·
    # $credits/mo) and a pick opens the portal deep-link with plan=<tier_id>.
    cli._app = object()  # interactive
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: _free_state())
    # catalog pick → "plus" (single modal; the pick opens the portal directly)
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("plus"), raising=False)
    opened = _capture_opener(monkeypatch)  # returns False → URL also printed

    cli._show_subscription()
    out = capsys.readouterr().out

    # Catalog rows — monthly credits render as DOLLARS ($22 credits/mo), never bare.
    assert "Choose a plan" in out
    assert "Plus · $20/mo · $22 credits/mo" in out
    assert "Ultra · $200/mo · $220 credits/mo" in out
    # Free (tier_order 0) is excluded from the paid catalog rows.
    assert "Free · $0" not in out
    # The pick opens the deep-link directly (no second open/copy menu); the URL
    # carries plan=<picked tier>, org_id first, plan second.
    assert opened.get("url") == "https://portal.example/manage-subscription?org_id=org_1&plan=plus"
    assert "/manage-subscription?org_id=org_1&plan=plus" in out
    assert "start Plus" in out




# ── canonical browser opener (guarded like the device-code auth flows) ──


def test_open_url_in_browser_refuses_remote_session(cli, monkeypatch):
    # (4): a remote/SSH session must NOT auto-open — webbrowser.open is never called.
    import webbrowser

    import hermes_cli.auth as auth
    import hermes_cli.auth_device_flow as auth_device_flow

    monkeypatch.setattr(auth, "_is_remote_session", lambda: True, raising=False)
    monkeypatch.setattr(auth_device_flow, "_is_remote_session", lambda: True, raising=False)
    called = {"n": 0}
    monkeypatch.setattr(webbrowser, "open", lambda url: called.update(n=called["n"] + 1) or True)

    assert cli._open_url_in_browser("https://x.example") is False
    assert called["n"] == 0  # guard short-circuits before webbrowser.open
















@pytest.fixture(autouse=True)
def _no_card_lookup(monkeypatch):
    # The charge_now confirm best-effort-fetches billing state to NAME the card
    # being charged; keep unit tests offline (generic line) unless overridden.
    import agent.billing_view as bv

    def _offline(*a, **kw):
        raise RuntimeError("offline")

    monkeypatch.setattr(bv, "build_billing_state", _offline, raising=False)


def test_upgrade_confirm_names_the_subscription_card(cli, monkeypatch, capsys):
    # Post-card-resolver NAS: the confirm names the exact card when it resolved
    # via the subscription rung (what the upgrade actually charges).
    cli._app = object()
    st = _sub_state(tier_id="plus", tier_name="Plus")
    tiers = tuple(
        SubscriptionTier(tier_id=t.tier_id, name=t.name, tier_order=t.tier_order, dollars_per_month=t.dollars_per_month, monthly_credits=t.monthly_credits, is_current=(t.tier_id == "plus"), is_enabled=True)
        for t in _TIERS
    )
    object.__setattr__(st, "tiers", tiers)
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: st)
    # picker → ultra; charge confirm → back out (we only assert the card line)
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("change", "ultra", "cancel"), raising=False)
    monkeypatch.setattr(nb, "post_subscription_preview", lambda **kw: {"effect": "charge_now", "targetTierName": "Ultra", "amountDueNowCents": 4630})
    import agent.billing_view as bv
    from agent.billing_view import BillingState as _BS
    from agent.billing_view import CardInfo as _CI

    _state = _BS(logged_in=True, card=_CI(brand="Visa", last4="4242", resolved_via="subPin"))
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: _state, raising=False)

    cli._show_subscription()
    out = capsys.readouterr().out

    assert "Visa ····4242 — the card on your subscription — will be charged." in out
