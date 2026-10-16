"""A consented launch reaches main's HermesCLI (``cli.main``) with its scoped options."""
import os
import sys
import types

import pytest


def test_gateway_launch_honors_interactive_consent_without_policy_environment_mutation(monkeypatch):
    from hermes_cli import main
    from tests.hermes_cli.test_cli_startup_model_cost_guard import _chat_args
    captured = []
    monkeypatch.setattr('hermes_cli.config.load_config', lambda: {
        'model': {'provider': 'openai-codex', 'default': 'gpt-5.5'}})
    monkeypatch.setattr(main, '_has_any_provider_configured', lambda: True)
    monkeypatch.setattr(main, '_pin_kanban_board_env', lambda: None)
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
    monkeypatch.setitem(sys.modules, 'cli', types.SimpleNamespace(main=lambda **kw: captured.append(dict(kw))))
    for answer, admitted in [('no', False), ('yes', True)]:
        monkeypatch.setattr('builtins.input', lambda prompt: answer)
        before = dict(os.environ)
        if admitted:
            main.cmd_chat(_chat_args(model='openai/gpt-5.5-pro', max_turns=3))
        else:
            with pytest.raises(SystemExit) as result:
                main.cmd_chat(_chat_args(model='openai/gpt-5.5-pro', max_turns=3))
            assert result.value.code == 1
            assert dict(os.environ) == before
        assert bool(captured) == admitted
    assert captured[0]['model'] == 'openai/gpt-5.5-pro' and captured[0]['max_turns'] == 3
