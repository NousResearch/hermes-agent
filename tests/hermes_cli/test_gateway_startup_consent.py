"""A consented launch reaches the canonical transport with its scoped options."""
import os
import sys

import pytest


def test_gateway_launch_honors_interactive_consent_without_policy_environment_mutation(monkeypatch):
    from hermes_cli import main
    from tests.hermes_cli.test_cli_startup_model_cost_guard import _chat_args
    import hermes_cli.gateway_chat as transport
    captured = []
    monkeypatch.setattr('hermes_cli.config.load_config', lambda: {
        'model': {'provider': 'openai-codex', 'default': 'gpt-5.5'}})
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(transport, 'launch_from_args', lambda args: captured.append(vars(args).copy()) or 0)
    before = dict(os.environ)
    for answer, admitted in [('no', False), ('yes', True)]:
        monkeypatch.setattr('builtins.input', lambda prompt: answer)
        with pytest.raises(SystemExit) as result:
            main.cmd_chat(_chat_args(model='openai/gpt-5.5-pro', ignore_rules=True, max_turns=3))
        assert result.value.code == (0 if admitted else 1)
        assert bool(captured) == admitted
    assert captured[0]['ignore_rules'] is True and captured[0]['max_turns'] == 3
    assert dict(os.environ) == before
