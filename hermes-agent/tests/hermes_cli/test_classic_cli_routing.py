"""The classic CLI is main's HermesCLI again: `hermes` / `hermes chat` (interactive and -q) reach
``cli.main``; only stream-json and the safe-mode bypass keep the gateway transport client."""
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


@pytest.fixture
def routed(monkeypatch):
    import hermes_cli.main as m
    calls = {}
    fake_cli = types.ModuleType("cli")
    fake_cli.main = lambda **kw: calls.setdefault("cli_main", kw)
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    import hermes_cli.gateway_chat_startup as startup
    monkeypatch.setattr(startup, "launch_gateway_chat", lambda args: calls.setdefault("gateway", args) and 0)
    for name in ("_resolve_chat_session_args", "_warn_retired_xai_models", "_start_chat_background_prefetch",
                 "_pin_kanban_board_env", "_confirm_startup_expensive_model_override",
                 "_guard_noninteractive_user_config"):
        monkeypatch.setattr(m, name, lambda *a, **k: None)
    monkeypatch.setattr(m, "_resolve_use_tui", lambda args: False)
    monkeypatch.setattr(m, "_has_any_provider_configured", lambda: True)
    import hermes_cli.free_tier_bootstrap as boot
    monkeypatch.setattr(boot, "run_bootstrap", lambda **k: None)
    import hermes_cli.observability.shared_metrics_consent as consent
    monkeypatch.setattr(consent, "offer_consent_before_chat", lambda args: None)
    import hermes_cli.observability.shared_metrics_process as proc
    monkeypatch.setattr(proc, "begin_process", lambda name: None)
    return m, calls


def test_interactive_chat_runs_main_hermes_cli(routed):
    m, calls = routed
    m.cmd_chat(_args())
    assert "cli_main" in calls and "gateway" not in calls


def test_query_runs_main_hermes_cli(routed):
    m, calls = routed
    m.cmd_chat(_args(query="hello"))
    assert calls["cli_main"]["query"] == "hello" and "gateway" not in calls


def test_cli_main_does_not_hand_off_to_thin_client():
    import inspect

    import cli
    assert "launch_from_kwargs" not in inspect.getsource(cli.main)
