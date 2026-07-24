"""Regression tests for numbered fallbacks when the interactive curses menu
cannot initialize (e.g. non-TTY, curses unavailable, terminal error)."""

import subprocess
from types import SimpleNamespace

from hermes_cli.config import load_config, save_config


def _raise_menu(*args, **kwargs):
    # Mimic curses_radiolist hitting an unrecoverable terminal error so the
    # caller's except clause routes to the numbered-input fallback.
    raise subprocess.CalledProcessError(2, ["tput", "clear"])


def test_prompt_model_selection_falls_back_on_menu_runtime_error(monkeypatch):
    from hermes_cli.auth import _prompt_model_selection

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)
    responses = iter(["2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    selected = _prompt_model_selection(["model-a", "model-b"])

    assert selected == "model-b"


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


def test_prompt_model_selection_allows_confirmed_expensive_model(monkeypatch):
    from hermes_cli.auth import _prompt_model_selection

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *_args, **_kwargs: SimpleNamespace(message="EXPENSIVE MODEL WARNING"),
    )
    responses = iter(["1", "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    selected = _prompt_model_selection(
        ["openai/gpt-5.5-pro"],
        confirm_provider="nous",
    )

    assert selected == "openai/gpt-5.5-pro"


def test_prompt_reasoning_effort_falls_back_on_menu_runtime_error(monkeypatch):
    from hermes_cli.main import _prompt_reasoning_effort_selection

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)
    responses = iter(["3"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    selected = _prompt_reasoning_effort_selection(["low", "medium", "high"], current_effort="")

    assert selected == "high"


def test_remove_custom_provider_falls_back_on_menu_runtime_error(tmp_path, monkeypatch):
    from hermes_cli.main import _remove_custom_provider

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
    """Edit a provider stored in the legacy custom_providers list."""
    from hermes_cli.main import _edit_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)

    cfg = load_config()
    cfg["custom_providers"] = [
        {"name": "Local A", "base_url": "http://localhost:8001/v1", "model": "old-model"},
        {"name": "Local B", "base_url": "http://localhost:8002/v1"},
    ]
    save_config(cfg)

    input_responses = iter(["1", "Renamed A", "", "new-model", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(input_responses))
    monkeypatch.setattr("getpass.getpass", lambda _prompt="": "")

    _edit_custom_provider(cfg)

    reloaded = load_config()
    edited = reloaded["custom_providers"][0]
    assert edited["name"] == "Renamed A"
    assert edited["base_url"] == "http://localhost:8001/v1"
    assert edited["model"] == "new-model"


def test_edit_v12_providers_entry(tmp_path, monkeypatch):
    """Edit a provider stored in the v12 keyed providers dict."""
    from hermes_cli.main import _edit_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)

    cfg = load_config()
    cfg["providers"] = {
        "my-local": {
            "name": "My Local",
            "api": "http://localhost:11434/v1",
            "default_model": "llama3",
            "models": {
                "llama3": {"context_length": 8192, "supports_vision": False},
            },
        },
    }
    save_config(cfg)

    input_responses = iter(["1", "My Local Renamed", "", "llama3.1", "128k"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(input_responses))
    monkeypatch.setattr("getpass.getpass", lambda _prompt="": "")

    _edit_custom_provider(cfg)

    reloaded = load_config()
    entry = reloaded["providers"]["my-local"]
    assert entry["name"] == "My Local Renamed"
    assert entry["api"] == "http://localhost:11434/v1"
    assert entry["default_model"] == "llama3.1"
    assert "llama3" not in entry["models"]
    assert entry["models"]["llama3.1"]["context_length"] == 128000
    assert entry["models"]["llama3.1"]["supports_vision"] is False


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

    input_responses = iter(["1", "", "", "", "128k"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(input_responses))
    monkeypatch.setattr("getpass.getpass", lambda _prompt="": "")

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


def test_parse_context_length_input():
    from hermes_cli.main import _parse_context_length_input

    assert _parse_context_length_input("128k") == 128000
    assert _parse_context_length_input("32K") == 32000
    assert _parse_context_length_input("32,768") == 32768
    assert _parse_context_length_input("") is None
    assert _parse_context_length_input("", fallback=4096) == 4096
    assert _parse_context_length_input("abc", fallback=4096) == 4096


def test_named_custom_provider_model_picker_falls_back_on_menu_runtime_error(tmp_path, monkeypatch):
    from hermes_cli.main import _model_flow_named_custom

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *args, **kwargs: ["model-a", "model-b"])
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)

    cfg = load_config()
    save_config(cfg)

    responses = iter(["2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    _model_flow_named_custom(
        cfg,
        {
            "name": "Local",
            "base_url": "http://localhost:8000/v1",
            "api_key": "",
            "model": "",
        },
    )

    reloaded = load_config()
    assert reloaded["model"]["provider"] == "custom"
    assert reloaded["model"]["base_url"] == "http://localhost:8000/v1"
    assert reloaded["model"]["default"] == "model-b"
