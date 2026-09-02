"""Regression tests for numbered fallbacks when the interactive curses menu
cannot initialize (e.g. non-TTY, curses unavailable, terminal error)."""

import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli.config import get_env_value, load_config, read_raw_config, save_config


def _raise_menu(*args, **kwargs):
    # Mimic curses_radiolist hitting an unrecoverable terminal error so the
    # caller's except clause routes to the numbered-input fallback.
    raise subprocess.CalledProcessError(2, ["tput", "clear"])


def _set_edit_inputs(monkeypatch, values, *, api_key=""):
    responses = iter(values)
    monkeypatch.setattr(
        "hermes_cli.main.line_input",
        lambda _prompt="": next(responses),
    )
    monkeypatch.setattr(
        "hermes_cli.secret_prompt.masked_secret_prompt",
        lambda _prompt="": api_key,
    )


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("\x1b[27u", "cancel"),
        ("\x1b[D", "back"),
        ("\x1b[99;5u", "cancel"),
    ],
)
def test_scoped_numbered_input_handles_navigation_keys(sequence, expected):
    """The curses fallback stays escapable on POSIX and native Windows."""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from hermes_cli.curses_ui import (
        MenuNavigationStart,
        _NUMBERED_BACK_ENABLED,
        _NumberedNavigation,
        _read_numbered_input,
        reset_menu_navigation_handler,
        set_menu_navigation_handler,
    )

    def handler(event, *_args):
        return MenuNavigationStart(allow_back=True) if event == "begin" else None

    token = set_menu_navigation_handler(handler)
    back_token = _NUMBERED_BACK_ENABLED.set(True)
    try:
        with create_pipe_input() as pipe_input:
            pipe_input.send_text(sequence)
            with create_app_session(input=pipe_input, output=DummyOutput()):
                result = _read_numbered_input("Choice: ")
    finally:
        _NUMBERED_BACK_ENABLED.reset(back_token)
        reset_menu_navigation_handler(token)

    assert result is getattr(_NumberedNavigation, expected.upper())


def test_prompt_model_selection_requires_expensive_confirmation(monkeypatch, capsys):
    from hermes_cli.auth import _prompt_model_selection

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *_args, **_kwargs: SimpleNamespace(message="EXPENSIVE MODEL WARNING"),
    )
    responses = iter(["1", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    selected = _prompt_model_selection(
        ["openai/gpt-5.5-pro"],
        confirm_provider="nous",
    )

    out = capsys.readouterr().out
    assert selected is None
    assert "EXPENSIVE MODEL WARNING" in out


def test_prompt_model_selection_uses_line_editor_for_custom_model(monkeypatch):
    from hermes_cli.auth import _prompt_model_selection

    monkeypatch.setattr(
        "hermes_cli.curses_ui.curses_radiolist",
        lambda _title, choices, **_kwargs: len(choices) - 2,
    )
    monkeypatch.setattr(
        "hermes_cli.cli_output.line_input",
        lambda prompt_text: (
            "vendor/edited-model" if prompt_text == "Enter model name: " else ""
        ),
    )

    assert _prompt_model_selection(["vendor/default-model"]) == "vendor/edited-model"


def test_prompt_model_selection_fallback_uses_line_editor_for_custom_model(
    monkeypatch,
):
    from hermes_cli.auth import _prompt_model_selection

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "2")
    monkeypatch.setattr(
        "hermes_cli.cli_output.line_input",
        lambda prompt_text: (
            "vendor/edited-model" if prompt_text == "Enter model name: " else ""
        ),
    )

    assert _prompt_model_selection(["vendor/default-model"]) == "vendor/edited-model"


def test_remove_custom_provider_falls_back_on_menu_runtime_error(tmp_path, monkeypatch):
    from hermes_cli.main_provider_setup import _remove_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)

    cfg = load_config()
    cfg["custom_providers"] = [
        {"name": "Local A", "base_url": "http://localhost:8001/v1"},
        {"name": "Local B", "base_url": "http://localhost:8002/v1"},
    ]
    save_config(cfg)

    responses = iter(["1"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    _remove_custom_provider(cfg)

    reloaded = load_config()
    assert reloaded["custom_providers"] == [
        {"name": "Local B", "base_url": "http://localhost:8002/v1"},
    ]


def test_edit_legacy_custom_provider(tmp_path, monkeypatch):
    """Edit a legacy provider and keep its active model identity valid."""
    from hermes_cli.main import _edit_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)

    cfg = load_config()
    cfg["custom_providers"] = [
        {
            "name": "Local A",
            "base_url": "http://localhost:8001/v1",
            "model": "old-model",
        },
        {"name": "Local B", "base_url": "http://localhost:8002/v1"},
    ]
    cfg["model"] = {
        "provider": "custom:local-a",
        "default": "old-model",
        "base_url": "http://localhost:8001/v1",
    }
    save_config(cfg)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")
    _set_edit_inputs(monkeypatch, ["Renamed A", "", "new-model", ""])

    _edit_custom_provider(cfg)

    reloaded = load_config()
    edited = reloaded["custom_providers"][0]
    assert edited["name"] == "Renamed A"
    assert edited["base_url"] == "http://localhost:8001/v1"
    assert edited["model"] == "new-model"
    assert reloaded["model"]["provider"] == "custom:renamed-a"
    assert reloaded["model"]["default"] == "new-model"


def test_edit_v12_providers_entry(tmp_path, monkeypatch):
    """Editing a default never corrupts the provider's model catalog."""
    from hermes_cli.main import _edit_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)

    cfg = load_config()
    cfg["providers"] = {
        "my-local": {
            "name": "My Local",
            "api": "http://localhost:11434/v1",
            "model": "llama3",
            "default_model": "stale-default",
            "models": {
                "llama3": {"context_length": 8192, "supports_vision": False},
                "llama3.1": {
                    "context_length": 65536,
                    "supports_vision": True,
                },
            },
        },
    }
    save_config(cfg)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")
    _set_edit_inputs(
        monkeypatch,
        ["My Local Renamed", "", "llama3.1", "128k"],
    )

    _edit_custom_provider(cfg)

    reloaded = load_config()
    entry = reloaded["providers"]["my-local"]
    assert entry["name"] == "My Local Renamed"
    assert entry["api"] == "http://localhost:11434/v1"
    assert entry["default_model"] == "llama3.1"
    assert "model" not in entry
    assert entry["models"]["llama3"] == {
        "context_length": 8192,
        "supports_vision": False,
    }
    assert entry["models"]["llama3.1"]["context_length"] == 128000
    assert entry["models"]["llama3.1"]["supports_vision"] is True


def test_edit_custom_provider_preserves_model_metadata(tmp_path, monkeypatch):
    """Changing context length must not drop supports_vision or other metadata."""
    from hermes_cli.main import _edit_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)

    cfg = load_config()
    cfg["custom_providers"] = [
        {
            "name": "Local",
            "base_url": "http://localhost:8001/v1",
            "model": "qwen",
            "models": {"qwen": {"context_length": 32768, "supports_vision": True}},
        },
    ]
    save_config(cfg)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")
    _set_edit_inputs(monkeypatch, ["", "", "", "128k"])

    _edit_custom_provider(cfg)

    reloaded = load_config()
    model_meta = reloaded["custom_providers"][0]["models"]["qwen"]
    assert model_meta["context_length"] == 128000
    assert model_meta["supports_vision"] is True


def test_edit_custom_provider_cancel_selection(tmp_path, monkeypatch, capsys):
    from hermes_cli.main import _edit_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)

    cfg = load_config()
    cfg["custom_providers"] = [
        {"name": "Local A", "base_url": "http://localhost:8001/v1"},
    ]
    save_config(cfg)

    input_responses = iter(["2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(input_responses))

    _edit_custom_provider(cfg)

    captured = capsys.readouterr()
    assert "No change." in captured.out
    reloaded = load_config()
    assert reloaded["custom_providers"][0]["name"] == "Local A"


def test_edit_custom_provider_no_providers(tmp_path, monkeypatch, capsys):
    from hermes_cli.main import _edit_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    cfg = load_config()
    cfg["custom_providers"] = []
    save_config(cfg)

    _edit_custom_provider(cfg)

    captured = capsys.readouterr()
    assert "No custom providers configured." in captured.out


def test_edit_custom_provider_reprobes_changed_url(tmp_path, monkeypatch):
    from hermes_cli.main import _edit_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)

    cfg = load_config()
    cfg["providers"] = {
        "my-local": {
            "name": "My Local",
            "api": "http://localhost:7000/v1",
            "base_url": "http://localhost:8000/v1",
            "default_model": "llama3",
            "transport": "anthropic_messages",
            "extra_headers": {"X-Proxy-Key": "proxy-secret"},
        },
    }
    save_config(cfg)

    probe_calls = []

    def fake_probe(api_key, base_url, **kwargs):
        probe_calls.append((api_key, base_url, kwargs))
        return {
            "models": ["llama3"],
            "probed_url": f"{base_url}/v1/models",
            "used_fallback": True,
            "resolved_base_url": f"{base_url}/v1",
        }

    monkeypatch.setattr("hermes_cli.models.probe_api_models", fake_probe)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")
    _set_edit_inputs(
        monkeypatch,
        ["", "http://localhost:9000", "", ""],
    )

    _edit_custom_provider(cfg)

    assert probe_calls == [
        (
            "",
            "http://localhost:9000",
            {
                "api_mode": "anthropic_messages",
                "request_headers": {"X-Proxy-Key": "proxy-secret"},
            },
        )
    ]
    raw_entry = read_raw_config()["providers"]["my-local"]
    assert raw_entry["api"] == "http://localhost:9000/v1"
    assert "base_url" not in raw_entry


def test_edit_custom_provider_migrates_plaintext_key_to_env(tmp_path, monkeypatch):
    from hermes_cli.main import _edit_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)

    cfg = load_config()
    cfg["providers"] = {
        "my-local": {
            "name": "My Local",
            "api": "http://localhost:11434/v1",
            "api_key": "sk-old-plaintext",
            "default_model": "llama3",
        },
    }
    save_config(cfg)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")
    _set_edit_inputs(monkeypatch, ["My Local Renamed", "", "", ""])

    _edit_custom_provider(cfg)

    raw_entry = read_raw_config()["providers"]["my-local"]
    key_env = raw_entry["key_env"]
    assert raw_entry["name"] == "My Local Renamed"
    assert "api_key" not in raw_entry
    assert get_env_value(key_env) == "sk-old-plaintext"


def test_edit_custom_provider_preserves_external_key_reference(tmp_path, monkeypatch):
    from hermes_cli.main import _edit_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("EXTERNAL_PROVIDER_KEY", "sk-external")
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)

    cfg = load_config()
    cfg["providers"] = {
        "my-local": {
            "name": "My Local",
            "api": "http://localhost:11434/v1",
            "api_key": "${EXTERNAL_PROVIDER_KEY}",
            "default_model": "llama3",
        },
    }
    save_config(cfg)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")
    _set_edit_inputs(monkeypatch, ["My Local Renamed", "", "", ""])

    _edit_custom_provider(cfg)

    raw_entry = read_raw_config()["providers"]["my-local"]
    assert raw_entry["name"] == "My Local Renamed"
    assert raw_entry["api_key"] == "${EXTERNAL_PROVIDER_KEY}"
    assert "key_env" not in raw_entry


def test_edit_custom_provider_accepts_unresolved_url_template(
    tmp_path,
    monkeypatch,
):
    from hermes_cli.main import _edit_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)

    cfg = load_config()
    cfg["custom_providers"] = [
        {
            "name": "Templated",
            "baseUrl": "${MISSING_BASE_URL}",
            "model": "old-model",
        },
    ]
    save_config(cfg)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")
    _set_edit_inputs(monkeypatch, ["Renamed", "", "new-model", ""])

    _edit_custom_provider(cfg)

    raw_entry = read_raw_config()["custom_providers"][0]
    assert raw_entry["name"] == "Renamed"
    assert raw_entry["baseUrl"] == "${MISSING_BASE_URL}"
    assert raw_entry["model"] == "new-model"


def test_parse_context_length_input():
    from hermes_cli.main import _parse_context_length_input

    assert _parse_context_length_input("128k") == 128000
    assert _parse_context_length_input("32K") == 32000
    assert _parse_context_length_input("32,768") == 32768
    assert _parse_context_length_input("") is None
    assert _parse_context_length_input("", fallback=4096) == 4096
    assert _parse_context_length_input("abc", fallback=4096) == 4096
