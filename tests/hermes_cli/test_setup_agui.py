"""Tests for the AG-UI section of the interactive setup wizard."""

import importlib
from argparse import ArgumentParser

from hermes_cli.subcommands.setup import build_setup_parser


def _setup_module():
    """Resolve ``hermes_cli.setup`` fresh, rather than binding it at import.

    A module-level ``from hermes_cli.setup import setup_agui`` pins the function
    object at collection time. If anything later in a full-suite run evicts
    ``hermes_cli.setup`` from ``sys.modules`` and it gets re-imported, that
    creates a NEW module object: the pinned ``setup_agui`` still resolves its
    globals through the OLD dict, while ``monkeypatch`` patches the NEW one, so
    the patches silently miss and the real ``prompt()`` calls ``input()`` under
    capture. Resolving here keeps the patched namespace and the executed
    function the same object no matter what ``sys.modules`` currently holds.
    (``importlib.reload`` does not trigger this — it re-executes into the same
    module object — so only eviction plus re-import exposes it.)
    """
    return importlib.import_module("hermes_cli.setup")


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
    setup = _setup_module()
    prompt_answers = iter(answers)
    saved_secrets = {}
    monkeypatch.setattr(
        setup, "prompt",
        lambda *args, **kwargs: next(prompt_answers),
    )
    monkeypatch.setattr(
        setup, "prompt_choice",
        lambda *args, **kwargs: model_choice,
    )
    monkeypatch.setattr(
        setup, "get_env_value",
        lambda key: existing_token if key == "HERMES_AGUI_SESSION_TOKEN" else None,
    )
    monkeypatch.setattr(
        setup, "save_env_value",
        lambda key, value: saved_secrets.__setitem__(key, value),
    )

    setup.setup_agui(config)
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
    setup = _setup_module()
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    handler = object()
    build_setup_parser(subparsers, cmd_setup=handler)

    args = parser.parse_args(["setup", "agui"])

    assert args.section == "agui"
    assert args.func is handler
    # Both sides resolved from the same module object, so this compares the
    # registration against the live function rather than a stale import.
    registered_sections = {key: func for key, _label, func in setup.SETUP_SECTIONS}
    assert registered_sections["agui"] is setup.setup_agui
