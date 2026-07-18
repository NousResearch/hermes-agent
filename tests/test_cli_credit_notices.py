"""CLI REPL credit-notice gating: _on_notice queues, _flush_credit_notices renders.

Routine credit notices honor the per-platform show_credits toggle (resolved for
the "cli" platform key); depletion / restored always print.
"""

import cli
from cli import HermesCLI
from agent.credits_tracker import AgentNotice


def _make_cli(config):
    inst = HermesCLI.__new__(HermesCLI)
    inst.config = config
    return inst


def _flush_capturing(inst, monkeypatch):
    printed = []
    monkeypatch.setattr(cli, "_cprint", lambda text: printed.append(text))
    inst._flush_credit_notices()
    return printed


def test_routine_notice_suppressed_when_show_credits_off(monkeypatch):
    inst = _make_cli({"display": {"show_credits": False}})
    inst._on_notice(
        AgentNotice(text="• Grant spent · $5.00 top-up left", level="info", key="credits.grant_spent")
    )
    assert _flush_capturing(inst, monkeypatch) == []


def test_routine_notice_shown_by_default(monkeypatch):
    inst = _make_cli({})  # no config → default True for cli
    inst._on_notice(
        AgentNotice(text="• Grant spent · $5.00 top-up left", level="info", key="credits.grant_spent")
    )
    printed = _flush_capturing(inst, monkeypatch)
    assert len(printed) == 1 and "Grant spent" in printed[0]


def test_depletion_always_printed_even_when_off(monkeypatch):
    inst = _make_cli({"display": {"show_credits": False}})
    inst._on_notice(
        AgentNotice(text="✕ Credit access paused · run /usage for balance", level="error", key="credits.depleted")
    )
    printed = _flush_capturing(inst, monkeypatch)
    assert len(printed) == 1 and "Credit access paused" in printed[0]


def test_per_platform_cli_override_off(monkeypatch):
    inst = _make_cli({"display": {"platforms": {"cli": {"show_credits": False}}}})
    inst._on_notice(
        AgentNotice(text="⚠ Credits 90% used · $20.00 cap", level="warn", key="credits.usage")
    )
    assert _flush_capturing(inst, monkeypatch) == []
