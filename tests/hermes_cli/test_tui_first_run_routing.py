"""First-run routing between the classic CLI guard, the TUI and the shared-metrics offer.

A blank install (no provider) must reach the TUI's own "Setup Required" screen under
``hermes --tui``; the classic "Run setup now? [Y/n]" guard only fronts the classic CLI, and the
one-time shared-metrics offer waits until a provider exists (#128663).
"""

from __future__ import annotations

import sys
import types

import pytest


class _Launched(Exception):
    """Raised by the stubbed ``_launch_tui`` (the real one ``exec``s and never returns)."""


def _run_chat(monkeypatch, argv, *, provider_configured):
    import hermes_cli.main as main_mod
    from hermes_cli._parser import build_top_level_parser
    import hermes_cli.free_tier_bootstrap as bootstrap
    import hermes_cli.observability.shared_metrics_consent as consent

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=main_mod.cmd_chat)
    args = parser.parse_args(argv)
    calls: list[str] = []

    def launch_tui(*_a, **_k):
        calls.append("tui")
        raise _Launched

    def classic_main(**_k):
        calls.append("classic")

    monkeypatch.setattr(bootstrap, "run_bootstrap", lambda **_k: None)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: provider_configured)
    monkeypatch.setattr(main_mod, "_first_run_setup_guard", lambda _args: calls.append("guard"))
    monkeypatch.setattr(consent, "offer_consent_before_chat", lambda _args: calls.append("offer"))
    monkeypatch.setattr(main_mod, "_launch_tui", launch_tui)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_start_chat_background_prefetch", lambda: None)
    monkeypatch.setattr(main_mod, "_confirm_startup_expensive_model_override", lambda _args: None)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None, raising=False)
    fake_cli = types.ModuleType("cli")
    fake_cli.main = classic_main
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    from hermes_cli import gateway_chat

    # The classic CLI rides the gateway client: it must still front the guard and never reach
    # a session (no gateway, no admission) when no provider is configured.
    monkeypatch.setattr(gateway_chat, "run_gateway_chat", lambda *_a, **_k: calls.append("gateway"))
    monkeypatch.setattr(gateway_chat, "_register_terminal_process", lambda: None, raising=False)
    try:
        main_mod.cmd_chat(args)
    except _Launched:
        pass
    except SystemExit as exc:
        # launch_gateway_chat returns 0 after the guard; anything else is a refusal we did not expect.
        assert exc.code == 0, f"cmd_chat exited {exc.code} after {calls}"
    return calls


@pytest.fixture(autouse=True)
def _no_tui_env(monkeypatch):
    monkeypatch.delenv("HERMES_TUI", raising=False)


def test_blank_install_tui_skips_classic_guard_and_offer(monkeypatch):
    assert _run_chat(monkeypatch, ["chat", "--tui"], provider_configured=False) == ["tui"]


def test_blank_install_classic_cli_still_gets_the_guard(monkeypatch):
    assert _run_chat(monkeypatch, ["chat", "--cli"], provider_configured=False) == ["guard"]
