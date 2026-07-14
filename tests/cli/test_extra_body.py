from types import SimpleNamespace

import pytest

from agent.transports.chat_completions import ChatCompletionsTransport
from cli import _parse_extra_body
from hermes_cli._parser import build_top_level_parser
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


def _chat_args(**overrides):
    values = {
        "model": None,
        "provider": None,
        "toolsets": None,
        "skills": None,
        "verbose": None,
        "quiet": False,
        "query": "hello",
        "image": None,
        "resume": None,
        "worktree": False,
        "checkpoints": False,
        "pass_session_id": False,
        "max_turns": None,
        "ignore_rules": False,
        "ignore_user_config": False,
        "compact": False,
        "source": None,
        "tui": False,
        "cli": True,
        "yolo": False,
        "continue_last": None,
        "extra_body": '{"provider":{"only":["anthropic"]}}',
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_extra_body_argument_parses_json_object():
    _, _, parser = build_top_level_parser()
    args = parser.parse_args(["-x", '{"provider":{"only":["anthropic"]}}'])

    assert _parse_extra_body(args.extra_body) == {
        "provider": {"only": ["anthropic"]}
    }


@pytest.mark.parametrize("raw", ["[]", '"value"', "1", "true", "null"])
def test_extra_body_rejects_non_objects(raw):
    with pytest.raises(ValueError, match="JSON object"):
        _parse_extra_body(raw)


def test_extra_body_rejects_invalid_json():
    with pytest.raises(ValueError, match="valid JSON"):
        _parse_extra_body("{invalid}")


def test_cmd_chat_forwards_extra_body(monkeypatch):
    import cli
    from hermes_cli.main import cmd_chat

    received = {}
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda: True)
    monkeypatch.setattr(cli, "main", lambda **kwargs: received.update(kwargs))

    cmd_chat(_chat_args())

    assert received["extra_body"] == '{"provider":{"only":["anthropic"]}}'


def test_cmd_chat_rejects_extra_body_with_tui(monkeypatch, capsys):
    from hermes_cli.main import cmd_chat

    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda: True)

    with pytest.raises(SystemExit, match="1"):
        cmd_chat(_chat_args(tui=True, cli=False))

    assert "--extra-body is not supported with the TUI" in capsys.readouterr().err


def test_turn_route_composes_extra_body_with_fast_mode(monkeypatch):
    cli = SimpleNamespace(
        api_key="key",
        base_url="https://example.test/v1",
        provider="openrouter",
        api_mode="chat_completions",
        acp_command=None,
        acp_args=[],
        model="model",
        service_tier="priority",
        extra_body={"provider": {"only": ["anthropic"]}},
    )
    monkeypatch.setattr(
        "hermes_cli.models.resolve_fast_mode_overrides",
        lambda _model: {"service_tier": "priority"},
    )

    route = CLIAgentSetupMixin._resolve_turn_agent_config(cli, "hello")

    assert route["request_overrides"] == {
        "service_tier": "priority",
        "extra_body": {"provider": {"only": ["anthropic"]}},
    }


def test_legacy_transport_merges_extra_body_request_override():
    kwargs = ChatCompletionsTransport().build_kwargs(
        "model",
        [{"role": "user", "content": "hello"}],
        extra_body_additions={"reasoning": {"effort": "medium"}},
        request_overrides={
            "extra_body": {
                "provider": {"only": ["anthropic"]},
                "reasoning": {"effort": "high"},
            }
        },
    )

    assert kwargs["extra_body"] == {
        "provider": {"only": ["anthropic"]},
        "reasoning": {"effort": "high"},
    }
