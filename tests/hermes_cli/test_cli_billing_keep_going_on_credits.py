"""/topup "Keep going on credits" (window plans): the terminal flow for the account setting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hermes_cli.nous_billing as nb
from hermes_cli.cli_billing_mixin import CLIBillingMixin


class _Cli(CLIBillingMixin):
    def __init__(self, choice: str = "agree"):
        self.choice, self.asked = choice, []

    def _modal_choice(self, title, detail, choices):
        self.asked.append((title, [c[0] for c in choices]))
        return self.choice


@pytest.fixture
def sent(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(nb, "patch_keep_going_on_credits", lambda **kw: calls.append(kw) or {})
    return calls


def test_turning_on_asks_first_then_patches(sent, capsys):
    cli = _Cli("agree")
    cli._billing_keep_going_on_credits_flow(SimpleNamespace(keep_going_on_credits=False))
    assert cli.asked and cli.asked[0][0] == "Keep going on credits?"
    assert sent == [{"enabled": True}]
    assert "Keep going on credits: on" in capsys.readouterr().out


def test_declining_the_prompt_changes_nothing(sent):
    _Cli("cancel")._billing_keep_going_on_credits_flow(SimpleNamespace(keep_going_on_credits=False))
    assert sent == []


def test_turning_off_needs_no_prompt(sent, capsys):
    cli = _Cli("cancel")  # would cancel if asked
    cli._billing_keep_going_on_credits_flow(SimpleNamespace(keep_going_on_credits=True))
    assert cli.asked == [] and sent == [{"enabled": False}]
    assert "waits for the reset" in capsys.readouterr().out


def test_remote_spending_off_still_lets_an_admin_turn_it_off(sent):
    """The overview's kill-switch branch offers only "turn it off" (it stops spending)."""
    cli = _Cli("off")
    cli._app = object()
    cli._billing_offer_keep_going_off(SimpleNamespace(can_change_plan=True, keep_going_on_credits=True))
    assert cli.asked == [("Keep going on credits is on", ["off", "cancel"])]
    assert sent == [{"enabled": False}]
    # Never offered when off already, to a member, or with no interactive app.
    for state, app in ((SimpleNamespace(can_change_plan=True, keep_going_on_credits=False), object()),
                       (SimpleNamespace(can_change_plan=False, keep_going_on_credits=True), object()),
                       (SimpleNamespace(can_change_plan=True, keep_going_on_credits=True), None)):
        quiet = _Cli("off")
        quiet._app = app
        quiet._billing_offer_keep_going_off(state)
        assert quiet.asked == []
    assert sent == [{"enabled": False}]


def _deny_once(monkeypatch):
    calls: list[dict] = []

    def patch(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise nb.BillingScopeRequired("billing:manage required", status=403, error="insufficient_scope")
        return {}
    monkeypatch.setattr(nb, "patch_keep_going_on_credits", patch)
    return calls


def test_a_scope_denial_steps_up_then_sends_the_same_change_once(monkeypatch, capsys):
    calls = _deny_once(monkeypatch)
    cli = _Cli("cancel")
    monkeypatch.setattr(cli, "_step_up_remote_spending", lambda **_kw: True, raising=False)
    cli._billing_keep_going_on_credits_flow(SimpleNamespace(keep_going_on_credits=True))  # turning off
    assert calls == [{"enabled": False}, {"enabled": False}]
    out = capsys.readouterr().out
    assert "waits for the reset" in out and "charge" not in out.lower()


def test_a_declined_step_up_sends_nothing_more(monkeypatch):
    calls = _deny_once(monkeypatch)
    cli = _Cli("agree")
    monkeypatch.setattr(cli, "_step_up_remote_spending", lambda **_kw: False, raising=False)
    cli._billing_keep_going_on_credits_flow(SimpleNamespace(keep_going_on_credits=False))
    assert calls == [{"enabled": True}]


def test_turning_on_updates_the_session_state_at_once(sent, monkeypatch):
    """The "turn on Keep going" line must not wait for the next reply: the agent's state takes the new
    value and the notice policy re-runs; the account snapshot refreshes in the background."""
    import hermes_cli.nous_account as na
    from agent.credits_tracker import CreditsState
    from agent.usage_gauges import SCOPE_ALL, WindowGauge
    refreshed, emitted = [], []
    monkeypatch.setattr(na, "get_nous_portal_account_info", lambda force_fresh=False: refreshed.append(force_fresh))
    cli = _Cli("agree")
    cli.agent = SimpleNamespace(_credits_state=CreditsState(version=1, denominator_kind="usage_window",
                                                            gauges=(WindowGauge("day", SCOPE_ALL, "Today", 10_000),),
                                                            keep_going_on_credits=False),
                                _emit_credits_notices=lambda: emitted.append(cli.agent._credits_state.keep_going_on_credits))
    cli._billing_keep_going_on_credits_flow(SimpleNamespace(keep_going_on_credits=False))
    assert cli.agent._credits_state.keep_going_on_credits is True and emitted == [True]
    import threading
    for th in threading.enumerate():
        if th.name == "keep-going-refresh":
            th.join(timeout=5)
    assert refreshed == [True]
