"""Regression tests for numbered fallbacks when the interactive curses menu
cannot initialize (e.g. non-TTY, curses unavailable, terminal error)."""

import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli.config import load_config, save_config


def _raise_menu(*args, **kwargs):
    # Mimic curses_radiolist hitting an unrecoverable terminal error so the
    # caller's except clause routes to the numbered-input fallback.
    raise subprocess.CalledProcessError(2, ["tput", "clear"])


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
