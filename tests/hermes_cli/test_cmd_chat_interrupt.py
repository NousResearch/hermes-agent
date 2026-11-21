"""Ctrl+C during chat startup exits cleanly with code 130 — contract for cmd_chat.

`_signal_handler_q` (installed for every chat invocation) raises KeyboardInterrupt
the moment SIGINT lands, and the whole `cli.run()` startup window (banner,
gateway-session ensure) sits outside run()'s own KI catch. cmd_chat only guarded
ValueError/ImportError, so a Ctrl+C during boot escaped as an unhandled traceback.
Fix: cmd_chat converts it to a one-line stderr notice and exit 130 (128+SIGINT).
"""
import argparse
import sys
import types

import pytest


def _args(**kw):
    base = dict(command="chat", query=None, model=None, provider=None, toolsets=None, skills=None,
                verbose=None, quiet=False, image=None, resume=None, worktree=False, checkpoints=False,
                pass_session_id=False, max_turns=None, output_format="text", tui=False, cli=True,
                safe_mode=False, ignore_user_config=False, continue_last=None, yolo=False,
                ignore_rules=False, source=None, query_file=None, oneshot_exit=False, compact=False,
                run_budget=None, reasoning=None, accept_hooks=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _fake_cli(raiser):
    fake = types.ModuleType("cli")
    fake.main = lambda **kw: raiser()
    return fake


@pytest.fixture
def chat_env(monkeypatch):
    import hermes_cli.main as m
    monkeypatch.setitem(sys.modules, "cli", _fake_cli(lambda: None))
    import hermes_cli.gateway_chat_startup as startup
    monkeypatch.setattr(startup, "launch_gateway_chat", lambda args: 0)
    for name in ("_resolve_chat_session_args", "_warn_retired_xai_models", "_start_chat_background_prefetch",
                 "_pin_kanban_board_env", "_confirm_startup_expensive_model_override",
                 "_guard_noninteractive_user_config"):
        monkeypatch.setattr(m, name, lambda *a, **k: None)
    monkeypatch.setattr(m, "_resolve_use_tui", lambda args: False)
    # Force the provider-configured gate: without it cmd_chat returns at the
    # first-run guard and cli.main (the mocked boot) never fires — the exact
    # non-hermetic trap that sank the prior attempt at this fix.
    monkeypatch.setattr(m, "_has_any_provider_configured", lambda: True)
    import hermes_cli.free_tier_bootstrap as boot
    monkeypatch.setattr(boot, "run_bootstrap", lambda **k: None)
    import hermes_cli.observability.shared_metrics_consent as consent
    monkeypatch.setattr(consent, "offer_consent_before_chat", lambda args: None)
    import hermes_cli.observability.shared_metrics_process as proc
    monkeypatch.setattr(proc, "begin_process", lambda name: None)
    return m


def test_cmd_chat_ctrl_c_during_startup_exits_130_with_notice(chat_env, monkeypatch, capsys):
    """The startup KI must become `Interrupted.` on stderr + exit 130, never a traceback."""
    m = chat_env
    monkeypatch.setitem(sys.modules, "cli", _fake_cli(lambda: (_ for _ in ()).throw(KeyboardInterrupt())))
    try:
        with pytest.raises(SystemExit) as exc:
            m.cmd_chat(_args())
    except KeyboardInterrupt:
        # Base behavior: the KI escapes cmd_chat unguarded. Convert it to a
        # normal failure with the asserted reason instead of letting pytest
        # abort the whole session on it.
        pytest.fail("KeyboardInterrupt escaped cmd_chat: Ctrl+C during chat startup is unguarded")
    assert exc.value.code == 130
    assert "Interrupted." in capsys.readouterr().err


def test_cmd_chat_error_handlers_survive_the_ctrl_c_guard(chat_env, monkeypatch, capsys):
    """The new KI clause must not shadow the existing ValueError/ImportError handling."""
    m = chat_env
    monkeypatch.setitem(sys.modules, "cli", _fake_cli(lambda: (_ for _ in ()).throw(ValueError("bad model"))))
    with pytest.raises(SystemExit) as exc:
        m.cmd_chat(_args())
    assert exc.value.code == 1
    assert "Error: bad model" in capsys.readouterr().out

    # ImportError with the partial-update hint firing exits 1 via the hint path.
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "emit_partial_update_hint", lambda e: True)
    monkeypatch.setitem(sys.modules, "cli", _fake_cli(lambda: (_ for _ in ()).throw(ImportError("resolve_turn_limit"))))
    with pytest.raises(SystemExit) as exc:
        m.cmd_chat(_args())
    assert exc.value.code == 1

    # ...and when the hint does not fire, the ImportError still re-raises.
    monkeypatch.setattr(hermes_constants, "emit_partial_update_hint", lambda e: False)
    with pytest.raises(ImportError):
        m.cmd_chat(_args())
