"""/topup on a portal that bills purchases through Stripe Managed Payments.

The portal names a Managed Payments payer (``managedPayments.label``) only for a client that
sends ``Hermes-Billing-Payers: managed-payments``. The terminal then buys with that payer even
with no card on file, and says before the charge that Link bills it and adds any tax on top.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from agent.billing_view import billing_state_from_payload
from hermes_cli.cli_billing_mixin import CLIBillingMixin

LABEL = "Subscription payment method (billed by Link)"


class _Cli(CLIBillingMixin):
    """Answers each modal with the next scripted choice and records what it was shown."""

    def __init__(self, *choices: str):
        self.choices, self.asked = list(choices), []
        self._app = object()
        self.submitted: list[str] = []

    def _modal_choice(self, title, detail, choices):
        self.asked.append((title, detail, [c[0] for c in choices]))
        return self.choices.pop(0) if self.choices else None

    def _billing_submit_and_poll(self, state, amount, key, **_kw):
        self.submitted.append(str(amount))


def _state(**payload):
    return billing_state_from_payload({
        "org": {"id": "org_1", "role": "OWNER"}, "cliBillingEnabled": True, "chargePresets": ["10"],
        "bounds": {"minUsd": "5", "maxUsd": "1000"}, **payload})


def test_a_managed_payments_payer_buys_without_a_card(capsys):
    cli = _Cli("10", "pay")
    cli._billing_buy_flow(_state(managedPayments={"label": LABEL}))

    assert all(ids != ["portal", "recheck", "cancel"] for _, _, ids in cli.asked)  # no add-card detour
    assert cli.asked[0][1] == f"Payment: {LABEL}"
    assert cli.submitted == ["10"]
    out = capsys.readouterr().out
    assert f"Payment: {LABEL}" in out
    assert "allow Link to charge this payment method for this amount plus any sales tax or VAT" in out
    assert "Nous Research to charge your card" not in out
    # Link adds tax on top, so the amount is the credits bought, never a "Total" or "Pay $X".
    assert "Before tax: $10" in out and "Total:" not in out
    title, _, choices = cli.asked[-1]
    assert title == "Buy $10 of credits?" and choices[0] == "pay"


def test_the_managed_payments_payer_wins_over_the_card_on_file(capsys):
    cli = _Cli("pay")
    cli._billing_confirm_and_charge(
        _state(managedPayments={"label": LABEL}, card={"brand": "visa", "last4": "4242"}), Decimal(10))

    out = capsys.readouterr().out
    assert f"Payment: {LABEL}" in out
    assert "visa" not in out
    assert "Your card saved on the portal will be charged" not in out


def test_an_older_portal_keeps_the_card_on_file(capsys):
    cli = _Cli("pay")
    cli._billing_confirm_and_charge(_state(card={"brand": "visa", "last4": "4242"}), Decimal(10))

    out = capsys.readouterr().out
    assert "Payment: visa ····4242" in out
    assert "allow Nous Research to charge your card" in out


@pytest.mark.parametrize("payload", [{}, {"managedPayments": {"label": ""}}])
def test_no_payer_still_starts_with_adding_a_card(payload):
    cli = _Cli()  # cancels the add-card prompt
    cli._billing_buy_flow(_state(**payload))
    assert cli.submitted == []
    assert cli.asked and cli.asked[0][2] == ["portal", "recheck", "cancel"]


class _StepUpCli(_Cli):
    """Remote Spending is granted at once; billing state then reads back as ``fresh``."""

    def __init__(self, *choices: str):
        super().__init__(*choices)
        self.keys: list[str] = []

    def _step_up_remote_spending(self, **_kw):
        return True

    def _billing_submit_and_poll(self, state, amount, key, **_kw):
        self.submitted.append(str(amount))
        self.keys.append(key)


def test_a_managed_payments_purchase_resumes_after_remote_spending_with_its_key(monkeypatch):
    import agent.billing_view as bv
    monkeypatch.setattr(bv, "build_billing_state", lambda **_kw: _state(managedPayments={"label": LABEL}))

    cli = _StepUpCli("resume")
    cli._billing_handle_scope_required(_state(), amount=Decimal(10), idempotency_key="key-1")

    assert cli.submitted == ["10"] and cli.keys == ["key-1"]


def test_auto_reload_setup_still_needs_a_card_after_remote_spending(monkeypatch, capsys):
    import agent.billing_view as bv
    monkeypatch.setattr(bv, "build_billing_state", lambda **_kw: _state(managedPayments={"label": LABEL}))

    cli = _StepUpCli("resume")
    cli._billing_handle_scope_required(_state())  # no amount: auto-reload setup

    assert cli.submitted == []
    assert "no card" in capsys.readouterr().out.lower()

