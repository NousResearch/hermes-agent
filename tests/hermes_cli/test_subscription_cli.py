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


def test_free_numbered_pick_stdin_fallback_opens_tier(cli, monkeypatch, capsys):
    # (1): the numbered contract — a bare digit through the modal's stdin fallback
    # maps to the Nth printed row and opens THAT tier's deep-link. The shared
    # normalizer only knows confirm-dialog digit aliases, so this path is bespoke.
    cli._app = object()
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: _free_state())
    # Row 1 = Plus (cheapest selectable). Feed a bare "1" as the stdin fallback.
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("1"), raising=False)
    opened = _capture_opener(monkeypatch)

    cli._show_subscription()
    out = capsys.readouterr().out

    # "1" resolved to Plus (row 1), not to the normalizer's alias → Plus URL opens.
    assert opened.get("url") == "https://portal.example/manage-subscription?org_id=org_1&plan=plus"
    assert "start Plus" in out


def test_free_catalog_cancel_builds_no_url(cli, monkeypatch, capsys):
    cli._app = object()
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: _free_state())
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("cancel"), raising=False)

    cli._show_subscription()
    out = capsys.readouterr().out

    assert "No plan started" in out
    assert "plan=" not in out  # nothing picked → no deep-link built


def test_free_no_paid_tiers_falls_back_to_plain_portal(cli, monkeypatch, capsys):
    # Only the Free tier exists → no catalog to show → plain portal hand-off.
    cli._app = object()
    only_free = (_FREE_TIERS[0],)
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: _free_state(only_free))
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("cancel"), raising=False)

    cli._show_subscription()
    out = capsys.readouterr().out

    assert "Choose a plan" not in out  # no catalog
    assert "plan=" not in out


# ── canonical browser opener (guarded like the device-code auth flows) ──


def test_open_url_in_browser_refuses_remote_session(cli, monkeypatch):
    # (4): a remote/SSH session must NOT auto-open — webbrowser.open is never called.
    import webbrowser

    import hermes_cli.auth as auth

    monkeypatch.setattr(auth, "_is_remote_session", lambda: True, raising=False)
    called = {"n": 0}
    monkeypatch.setattr(webbrowser, "open", lambda url: called.update(n=called["n"] + 1) or True)

    assert cli._open_url_in_browser("https://x.example") is False
    assert called["n"] == 0  # guard short-circuits before webbrowser.open


def test_open_url_in_browser_refuses_console_browser(cli, monkeypatch):
    # (4): a console/text-mode browser (w3m/lynx) must NOT hijack the TTY.
    import webbrowser

    import hermes_cli.auth as auth

    monkeypatch.setattr(auth, "_is_remote_session", lambda: False, raising=False)
    monkeypatch.setattr(auth, "_can_open_graphical_browser", lambda: False, raising=False)
    called = {"n": 0}
    monkeypatch.setattr(webbrowser, "open", lambda url: called.update(n=called["n"] + 1) or True)

    assert cli._open_url_in_browser("https://x.example") is False
    assert called["n"] == 0


def test_open_url_in_browser_opens_when_graphical(cli, monkeypatch):
    # (4): a real graphical browser → open and report True.
    import webbrowser

    import hermes_cli.auth as auth

    monkeypatch.setattr(auth, "_is_remote_session", lambda: False, raising=False)
    monkeypatch.setattr(auth, "_can_open_graphical_browser", lambda: True, raising=False)
    monkeypatch.setattr(webbrowser, "open", lambda url: True)

    assert cli._open_url_in_browser("https://x.example") is True


def test_open_url_in_browser_empty_is_false(cli):
    assert cli._open_url_in_browser("") is False


def test_blocked_upgrade_fallback_carries_plan_param(cli, monkeypatch, capsys):
    # (b): a blocked UPGRADE preview falls back to the portal with plan=<tier_id>.
    cli._app = object()
    st = _sub_state(tier_id="plus", tier_name="Plus")
    tiers = tuple(
        SubscriptionTier(tier_id=t.tier_id, name=t.name, tier_order=t.tier_order, dollars_per_month=t.dollars_per_month, monthly_credits=t.monthly_credits, is_current=(t.tier_id == "plus"), is_enabled=True)
        for t in _TIERS
    )
    object.__setattr__(st, "tiers", tiers)
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: st)
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("change", "ultra"), raising=False)
    monkeypatch.setattr(nb, "post_subscription_preview", lambda **kw: {"effect": "blocked", "reason": "Not available here."})

    cli._show_subscription()
    out = capsys.readouterr().out

    assert "Manage on portal:" in out
    assert "plan=ultra" in out  # the picked upgrade tier rides along


def test_blocked_downgrade_fallback_stays_generic(cli, monkeypatch, capsys):
    # (d): a blocked DOWNGRADE fallback stays native/generic — no plan= deep-link.
    cli._app = object()
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: _sub_state())  # current = Ultra
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("change", "plus"), raising=False)
    monkeypatch.setattr(nb, "post_subscription_preview", lambda **kw: {"effect": "blocked", "reason": "Nope."})

    cli._show_subscription()
    out = capsys.readouterr().out

    assert "Manage on portal:" in out
    assert "plan=" not in out  # downgrade never carries a portal plan= param


def test_overview_leads_with_scheduled_downgrade_banner(cli, monkeypatch, capsys):
    st = _sub_state(pending_downgrade_tier_name="Plus", pending_downgrade_at="2026-07-28")
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: st)

    cli._show_subscription()
    out = capsys.readouterr().out

    assert "Scheduled change" in out
    assert "──▶" in out
    assert "Plus" in out
    # the status line itself echoes the transition
    assert "Plan: Ultra → Plus" in out


def test_change_flow_schedules_a_downgrade(cli, monkeypatch, capsys):
    cli._app = object()  # interactive
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: _sub_state())
    # change menu → "change"; picker → "plus" (only selectable); confirm → "yes"
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("change", "plus", "yes"), raising=False)
    monkeypatch.setattr(
        nb, "post_subscription_preview",
        lambda **kw: {"effect": "scheduled", "targetTierName": "Plus", "effectiveAt": "2026-07-28T00:00:00Z", "monthlyCreditsDelta": "-198"},
    )
    seen = {}
    monkeypatch.setattr(nb, "put_subscription_pending_change", lambda **kw: seen.update(kw) or {"message": "Scheduled."})

    cli._show_subscription()
    out = capsys.readouterr().out

    assert seen.get("subscription_type_id") == "plus"
    assert "doesn't change today" in out
    # (d) Downgrades stay NATIVE — scheduled in-app, never a portal plan= deep-link.
    assert "plan=" not in out
    assert "manage-subscription" not in out


def test_change_flow_upgrade_charges_now(cli, monkeypatch, capsys):
    cli._app = object()
    # Current = Plus so Ultra is a selectable upgrade.
    st = _sub_state(tier_id="plus", tier_name="Plus")
    tiers = tuple(SubscriptionTier(tier_id=t.tier_id, name=t.name, tier_order=t.tier_order, dollars_per_month=t.dollars_per_month, monthly_credits=t.monthly_credits, is_current=(t.tier_id == "plus"), is_enabled=True) for t in _TIERS)
    object.__setattr__(st, "tiers", tiers)
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: st)
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("change", "ultra", "yes"), raising=False)
    monkeypatch.setattr(nb, "post_subscription_preview", lambda **kw: {"effect": "charge_now", "targetTierName": "Ultra", "amountDueNowCents": 4630})
    seen = {}
    monkeypatch.setattr(nb, "post_subscription_upgrade", lambda **kw: seen.update(kw) or {"status": "upgraded", "targetTierName": "Ultra"})

    cli._show_subscription()
    out = capsys.readouterr().out

    assert seen.get("subscription_type_id") == "ultra"
    assert seen.get("idempotency_key")  # minted
    assert "$46.30" in out
    assert "Upgraded to Ultra" in out


def test_change_menu_cancel_schedules_cancellation(cli, monkeypatch, capsys):
    cli._app = object()
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: _sub_state())
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("cancel_sub", "yes"), raising=False)
    seen = {}
    monkeypatch.setattr(nb, "put_subscription_pending_change", lambda **kw: seen.update(kw) or {"message": "Cancelled."})

    cli._show_subscription()
    out = capsys.readouterr().out

    assert seen.get("cancel") is True
    assert "cancels" in out.lower()


def test_pending_change_menu_offers_undo(cli, monkeypatch, capsys):
    cli._app = object()
    st = _sub_state(pending_downgrade_tier_name="Plus", pending_downgrade_at="2026-07-28")
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: st)
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("keep"), raising=False)
    called = {"n": 0}
    monkeypatch.setattr(nb, "delete_subscription_pending_change", lambda **kw: called.update(n=called["n"] + 1) or {"message": "Resumed."})

    cli._show_subscription()
    out = capsys.readouterr().out

    assert called["n"] == 1
    assert "Undone" in out


def test_insufficient_scope_triggers_stepup_then_replays(cli, monkeypatch, capsys):
    cli._app = object()
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: _sub_state())
    # menu → change; picker → plus; confirm → yes; step-up → yes
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("change", "plus", "yes", "yes"), raising=False)
    monkeypatch.setattr(nb, "post_subscription_preview", lambda **kw: {"effect": "scheduled", "targetTierName": "Plus", "effectiveAt": "2026-07-28"})
    calls = {"n": 0}

    def _put(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise nb.BillingScopeRequired("remote spending required")
        return {"message": "Scheduled."}

    monkeypatch.setattr(nb, "put_subscription_pending_change", _put)
    import hermes_cli.auth as auth

    monkeypatch.setattr(auth, "step_up_nous_billing_scope", lambda **kw: True, raising=False)

    cli._show_subscription()
    out = capsys.readouterr().out

    # applied once (scope-denied), granted, replayed → applied again
    assert calls["n"] == 2
    assert "Remote Spending allowed" in out


def test_stepup_declined_grant_does_not_replay(cli, monkeypatch, capsys):
    cli._app = object()
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: _sub_state())
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("change", "plus", "yes", "yes"), raising=False)
    monkeypatch.setattr(nb, "post_subscription_preview", lambda **kw: {"effect": "scheduled", "targetTierName": "Plus", "effectiveAt": "2026-07-28"})
    calls = {"n": 0}

    def _put(**kw):
        calls["n"] += 1
        raise nb.BillingScopeRequired("remote spending required")

    monkeypatch.setattr(nb, "put_subscription_pending_change", _put)
    import hermes_cli.auth as auth

    monkeypatch.setattr(auth, "step_up_nous_billing_scope", lambda **kw: False, raising=False)

    cli._show_subscription()
    out = capsys.readouterr().out

    assert calls["n"] == 1  # applied once, grant denied, no replay
    assert "Couldn't allow Remote Spending" in out


def test_unknown_preview_effect_fails_safe(cli, monkeypatch, capsys):
    # An unrecognized effect string must NOT schedule a real change (fail safe).
    cli._app = object()
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: _sub_state())
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("change", "plus"), raising=False)
    monkeypatch.setattr(nb, "post_subscription_preview", lambda **kw: {"effect": "weird_unknown", "targetTierName": "Plus"})
    put = {"n": 0}
    monkeypatch.setattr(nb, "put_subscription_pending_change", lambda **kw: put.update(n=put["n"] + 1) or {})

    cli._show_subscription()
    out = capsys.readouterr().out

    assert put["n"] == 0  # no mutation on an unknown effect
    assert "portal" in out.lower()


def test_bounded_stepup_does_not_loop_on_repeat_denial(cli, monkeypatch, capsys):
    # Grant "succeeds" but the scope stays denied → replay ONCE (allow_stepup=False),
    # then stop — no re-prompt / re-open-browser loop.
    cli._app = object()
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: _sub_state())
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("change", "plus", "yes", "yes"), raising=False)
    monkeypatch.setattr(nb, "post_subscription_preview", lambda **kw: {"effect": "scheduled", "targetTierName": "Plus", "effectiveAt": "2026-07-28"})
    calls = {"n": 0}

    def _put(**kw):
        calls["n"] += 1
        raise nb.BillingScopeRequired("still no scope")

    monkeypatch.setattr(nb, "put_subscription_pending_change", _put)
    import hermes_cli.auth as auth

    monkeypatch.setattr(auth, "step_up_nous_billing_scope", lambda **kw: True, raising=False)

    cli._show_subscription()
    out = capsys.readouterr().out

    assert calls["n"] == 2  # applied, granted, replayed once — no third attempt
    assert (
        "Remote Spending still isn't active for this terminal — the authorization "
        "didn't take. Retry, or make this change on the portal."
    ) in out


def test_upgrade_transport_failure_is_ambiguous_not_flat_failure(cli, monkeypatch, capsys):
    # BUG B: a charge-route failure must warn "may or may not have been charged"
    # (steer to a re-check), never a flat failure that invites a blind retry (which
    # would mint a fresh idempotency key the server can't dedup → a real 2nd charge).
    cli._app = object()
    st = _sub_state(tier_id="plus", tier_name="Plus")
    tiers = tuple(
        SubscriptionTier(tier_id=t.tier_id, name=t.name, tier_order=t.tier_order, dollars_per_month=t.dollars_per_month, monthly_credits=t.monthly_credits, is_current=(t.tier_id == "plus"), is_enabled=True)
        for t in _TIERS
    )
    object.__setattr__(st, "tiers", tiers)
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: st)
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("change", "ultra", "yes"), raising=False)
    monkeypatch.setattr(nb, "post_subscription_preview", lambda **kw: {"effect": "charge_now", "targetTierName": "Ultra", "amountDueNowCents": 4630})

    def _boom(**kw):
        raise nb.BillingError("Could not reach Nous Portal", error="endpoint_unavailable")

    monkeypatch.setattr(nb, "post_subscription_upgrade", _boom)

    cli._show_subscription()
    out = capsys.readouterr().out

    assert "may or may not have been charged" in out
    assert "Re-run /subscription" in out
    assert "could not be completed" not in out  # not the old flat failure


def test_upgrade_rate_limit_is_deterministic_not_ambiguous(cli, monkeypatch, capsys):
    # R2: a typed PRE-charge rejection (429 rate-limit) must NOT be mislabeled
    # "may or may not have been charged" — it never reached Stripe. It gets the
    # normal error copy, not the ambiguous one.
    cli._app = object()
    st = _sub_state(tier_id="plus", tier_name="Plus")
    tiers = tuple(
        SubscriptionTier(tier_id=t.tier_id, name=t.name, tier_order=t.tier_order, dollars_per_month=t.dollars_per_month, monthly_credits=t.monthly_credits, is_current=(t.tier_id == "plus"), is_enabled=True)
        for t in _TIERS
    )
    object.__setattr__(st, "tiers", tiers)
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: st)
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("change", "ultra", "yes"), raising=False)
    monkeypatch.setattr(nb, "post_subscription_preview", lambda **kw: {"effect": "charge_now", "targetTierName": "Ultra", "amountDueNowCents": 4630})

    def _rl(**kw):
        raise nb.BillingRateLimited("Slow down — too many requests.", error="rate_limited", status=429, retry_after=30)

    monkeypatch.setattr(nb, "post_subscription_upgrade", _rl)

    cli._show_subscription()
    out = capsys.readouterr().out

    assert "may or may not have been charged" not in out  # NOT the ambiguous copy
    assert "Slow down" in out  # the real, deterministic error


def test_upgrade_transport_failure_still_ambiguous_after_narrowing(cli, monkeypatch, capsys):
    # Regression floor for R2: a genuine transport failure (network_error, no status)
    # must STILL be ambiguous after the narrowing.
    cli._app = object()
    st = _sub_state(tier_id="plus", tier_name="Plus")
    tiers = tuple(
        SubscriptionTier(tier_id=t.tier_id, name=t.name, tier_order=t.tier_order, dollars_per_month=t.dollars_per_month, monthly_credits=t.monthly_credits, is_current=(t.tier_id == "plus"), is_enabled=True)
        for t in _TIERS
    )
    object.__setattr__(st, "tiers", tiers)
    monkeypatch.setattr(sv, "build_subscription_state", lambda *a, **kw: st)
    monkeypatch.setattr(HermesCLI, "_prompt_text_input_modal", _scripted_modal("change", "ultra", "yes"), raising=False)
    monkeypatch.setattr(nb, "post_subscription_preview", lambda **kw: {"effect": "charge_now", "targetTierName": "Ultra", "amountDueNowCents": 4630})

    def _net(**kw):
        raise nb.BillingError("Could not reach Nous Portal: timeout", error="network_error")

    monkeypatch.setattr(nb, "post_subscription_upgrade", _net)

    cli._show_subscription()
    out = capsys.readouterr().out

    assert "may or may not have been charged" in out


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
