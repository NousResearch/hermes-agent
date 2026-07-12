"""Regression tests for startup ``-m/--model`` direct aliases.

The slash-command ``/model MiniMax`` path already resolves ``model_aliases``
through ``hermes_cli.model_switch``.  Non-interactive startup used by
``hermes chat -m MiniMax`` must resolve the same alias before runtime provider
resolution; otherwise the session keeps the previous configured provider and
sends ``MiniMax`` to the wrong backend before falling through the fallback chain.
"""

from hermes_cli.model_switch import DirectAlias


def test_startup_model_arg_direct_alias_updates_provider(monkeypatch):
    import cli as cli_mod
    import hermes_cli.model_switch as ms

    monkeypatch.setattr(
        cli_mod,
        "CLI_CONFIG",
        {
            "display": {},
            "model": {"provider": "openai-codex", "default": "gpt-5.5"},
            "agent": {},
            "provider_routing": {},
            "openrouter": {},
            "checkpoints": {},
        },
    )
    monkeypatch.setattr(
        ms,
        "DIRECT_ALIASES",
        {"minimax": DirectAlias("MiniMax-M3", "minimax-cn", "")},
    )
    monkeypatch.setattr("hermes_state.SessionDB", lambda: None)
    monkeypatch.setattr(cli_mod, "_run_state_db_auto_maintenance", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "_run_checkpoint_auto_maintenance", lambda *a, **k: None)

    cli_obj = cli_mod.HermesCLI(model="MiniMax")

    assert cli_obj.model == "MiniMax-M3"
    assert cli_obj.requested_provider == "minimax-cn"
    assert cli_obj.provider == "minimax-cn"
