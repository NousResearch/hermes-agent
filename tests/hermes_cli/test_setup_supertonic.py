from types import SimpleNamespace

import pytest

from hermes_cli import setup


def _select_supertonic(question, choices, default=0):
    if question == "Select TTS provider:":
        return next(i for i, choice in enumerate(choices) if choice.startswith("Supertonic "))
    if question == "Select quality:":
        return 1
    return default


def test_supertonic_setup_uses_shared_post_setup(monkeypatch):
    probes = iter([None, object()])
    post_setup_calls = []
    config = {}

    monkeypatch.setattr(setup, "managed_nous_tools_enabled", lambda: False)
    monkeypatch.setattr(
        setup,
        "get_nous_subscription_features",
        lambda _config: SimpleNamespace(nous_auth_present=False),
    )
    monkeypatch.setattr(setup, "prompt_choice", _select_supertonic)
    monkeypatch.setattr(setup, "prompt_yes_no", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(setup, "prompt", lambda *_args, **kwargs: kwargs.get("default", ""))
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda _name: next(probes))
    monkeypatch.setattr("hermes_cli.tools_config._run_post_setup", post_setup_calls.append)
    monkeypatch.setattr(setup, "save_config", lambda _config: None)

    setup.setup_tts(config)

    assert post_setup_calls == ["supertonic"]
    assert config["tts"]["provider"] == "supertonic"


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        (True, "Text-to-Speech (Supertonic local)"),
        (False, "Text-to-Speech (Supertonic - not installed)"),
    ],
)
def test_setup_summary_reports_supertonic(tmp_path, monkeypatch, capsys, installed, expected):
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: object() if name == "supertonic" and installed else None)

    setup._print_setup_summary({"tts": {"provider": "supertonic"}}, tmp_path)

    assert expected in capsys.readouterr().out
