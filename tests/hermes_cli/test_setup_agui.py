"""Tests for the AG-UI section of the interactive setup wizard."""

from argparse import ArgumentParser

from hermes_cli.setup import SETUP_SECTIONS, setup_agui
from hermes_cli.subcommands.setup import build_setup_parser


def _default_config():
    return {
        "agui": {
            "host": "127.0.0.1",
            "port": 8000,
            "toolsets": ["hermes-acp"],
            "provider": "",
            "model": "",
            "api_mode": "",
            "base_url": "",
        }
    }


def _run_setup(monkeypatch, config, answers, *, model_choice=0, existing_token=None):
    prompt_answers = iter(answers)
    saved_secrets = {}
    monkeypatch.setattr(
        "hermes_cli.setup.prompt",
        lambda *args, **kwargs: next(prompt_answers),
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        lambda *args, **kwargs: model_choice,
    )
    monkeypatch.setattr(
        "hermes_cli.setup.get_env_value",
        lambda key: existing_token if key == "HERMES_AGUI_SESSION_TOKEN" else None,
    )
    monkeypatch.setattr(
        "hermes_cli.setup.save_env_value",
        lambda key, value: saved_secrets.__setitem__(key, value),
    )

    setup_agui(config)
    return saved_secrets


def test_setup_agui_saves_behavioral_config_and_network_token(monkeypatch):
    config = _default_config()
    token = "a" * 32

    saved_secrets = _run_setup(
        monkeypatch,
        config,
        [
            "0.0.0.0",
            "9123",
            "hermes-acp, web",
            "custom",
            "gpt-test",
            "chat_completions",
            "http://localhost:4010/v1",
            token,
        ],
        model_choice=1,
    )

    assert config["agui"] == {
        "host": "0.0.0.0",
        "port": 9123,
        "toolsets": ["hermes-acp", "web"],
        "provider": "custom",
        "model": "gpt-test",
        "api_mode": "chat_completions",
        "base_url": "http://localhost:4010/v1",
    }
    assert saved_secrets == {"HERMES_AGUI_SESSION_TOKEN": token}


def test_setup_agui_keeps_current_port_on_invalid_input(monkeypatch, capsys):
    config = _default_config()

    saved_secrets = _run_setup(
        monkeypatch,
        config,
        ["127.0.0.1", "70000", "hermes-acp"],
    )

    assert config["agui"]["port"] == 8000
    assert config["agui"]["toolsets"] == ["hermes-acp"]
    assert saved_secrets == {}
    output = capsys.readouterr().out
    assert "Invalid port" in output


def test_setup_agui_rejects_weak_token_for_network_bind(monkeypatch, capsys):
    config = _default_config()

    saved_secrets = _run_setup(
        monkeypatch,
        config,
        ["0.0.0.0", "8000", "hermes-acp", "too-short"],
    )

    assert saved_secrets == {}
    assert "will refuse to start" in capsys.readouterr().out


def test_setup_agui_recovers_from_malformed_existing_values(monkeypatch):
    config = {
        "agui": {
            "host": None,
            "port": "not-a-port",
            "toolsets": "hermes-acp, web",
        }
    }

    _run_setup(
        monkeypatch,
        config,
        ["127.0.0.1", "8000", "hermes-acp, web"],
    )

    assert config["agui"]["host"] == "127.0.0.1"
    assert config["agui"]["port"] == 8000
    assert config["agui"]["toolsets"] == ["hermes-acp", "web"]


def test_setup_parser_accepts_agui_section():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    handler = object()
    build_setup_parser(subparsers, cmd_setup=handler)

    args = parser.parse_args(["setup", "agui"])

    assert args.section == "agui"
    assert args.func is handler
    registered_sections = {key: func for key, _label, func in SETUP_SECTIONS}
    assert registered_sections["agui"] is setup_agui
