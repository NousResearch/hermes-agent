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


def test_remove_row_shown_for_v12_providers_only_config(tmp_path, monkeypatch):
    """The picker's Remove row must appear for migrated configs — entries live in the
    ``providers:`` map only, no raw legacy ``custom_providers`` list (the pre-v12 gate hid
    the row for every migrated config; the removal flow now covers both schemas)."""
    from hermes_cli.main_provider_setup import (
        _build_provider_picker_rows,
        _named_custom_provider_map,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: sample-api\n  default: some-model\n"
        "providers:\n  sample-api:\n    name: SampleApi\n    api: https://api.x.test/v1\n"
    )

    cfg = load_config()
    provider_map = _named_custom_provider_map(cfg)
    rows, _ = _build_provider_picker_rows(cfg, "sample-api", {}, provider_map)
    assert provider_map, "v12 map entries must render picker rows"
    assert any(key == "remove-custom" for key, _label, _members in rows)


def test_remove_custom_provider_covers_v12_map_prunes_twin_detaches_model(tmp_path, monkeypatch):
    """Removing a merged custom provider pops the ``providers:`` map entry, the legacy list
    twin, the main-model mirror, and the managed .env key — while leaving sibling entries
    (including raw ``${ENV}`` templates) byte-untouched."""
    from hermes_cli.main_provider_setup import _remove_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("HERMES_CUSTOM_TESTPROV_API_KEY=sk-secret\nSIBLING_KEY=sk-sib\n")
    raw_yaml = (
        "model:\n"
        "  provider: testprov\n"
        "  default: some-model\n"
        "  base_url: https://x.test/v1\n"
        "  api_key: sk-secret\n"
        "providers:\n"
        "  testprov:\n"
        "    name: TestProv\n"
        "    api: https://x.test/v1\n"
        "  sibling:\n"
        "    name: Sibling\n"
        "    api: ${SIBLING_URL}\n"
        "    api_key: ${SIBLING_KEY}\n"
        "  keepme:\n"
        "    name: KeepMe\n"
        "    api: https://keep.test/v1\n"
        "custom_providers:\n"
        "  - name: TestProv\n"
        "    base_url: https://x.test/v1\n"
        "  - name: Other\n"
        "    base_url: https://other.test/v1\n"
    )
    (tmp_path / "config.yaml").write_text(raw_yaml)

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _raise_menu)
    responses = iter(["1"])  # TestProv (merged from the legacy twin + map entry, listed first)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    _remove_custom_provider(load_config())

    saved = (tmp_path / "config.yaml").read_text()
    assert "testprov:" not in saved and "TestProv" not in saved, "map entry must be popped"
    assert "keepme:" in saved and "https://keep.test/v1" in saved, "unrelated map entry survives"
    assert "${SIBLING_URL}" in saved and "${SIBLING_KEY}" in saved, \
        "raw templates must not expand on save"
    assert "https://other.test/v1" in saved and "name: Other" in saved, \
        "legacy twin pruned, unrelated rows kept"
    model_block = saved.split("model:")[1].split("providers:")[0]
    assert "provider:" not in model_block and "base_url:" not in model_block \
        and "api_key:" not in model_block, \
        "main-model mirror of the deleted provider must detach"
    assert "default: some-model" in model_block, "the model name itself is not the provider mirror"
    env = (tmp_path / ".env").read_text()
    assert "HERMES_CUSTOM_TESTPROV_API_KEY" not in env, "managed .env slot must be scrubbed"
    assert "SIBLING_KEY=sk-sib" in env, "unrelated env vars survive"
