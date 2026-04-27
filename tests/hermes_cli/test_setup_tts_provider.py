"""Regression tests for setup wizard TTS provider handling."""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli import setup as setup_mod
from hermes_cli.config import load_config


def _feature_state(**overrides):
    values = {
        "managed_by_nous": False,
        "available": False,
        "current_provider": "",
        "direct_override": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _subscription_features():
    return SimpleNamespace(
        nous_auth_present=False,
        web=_feature_state(),
        browser=_feature_state(),
        image_gen=_feature_state(available=True),
        tts=_feature_state(),
        modal=_feature_state(),
    )


def _patch_common_setup_state(monkeypatch):
    monkeypatch.setattr(
        setup_mod,
        "get_nous_subscription_features",
        lambda _config: _subscription_features(),
    )
    monkeypatch.setattr(setup_mod, "managed_nous_tools_enabled", lambda: False)
    monkeypatch.setattr(setup_mod, "get_env_value", lambda _key: "")
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])


def _patch_neutts_installed(monkeypatch):
    def fake_find_spec(name, *args, **kwargs):
        return object() if name == "neutts" else None

    monkeypatch.setattr(setup_mod.importlib.util, "find_spec", fake_find_spec)


def test_setup_summary_reports_installed_neutts(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _patch_common_setup_state(monkeypatch)
    _patch_neutts_installed(monkeypatch)

    setup_mod._print_setup_summary({"tts": {"provider": "neutts"}}, tmp_path)
    output = capsys.readouterr().out

    assert "Text-to-Speech (NeuTTS local)" in output
    assert "not installed" not in output


def test_setup_tts_provider_keeps_installed_neutts(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _patch_common_setup_state(monkeypatch)
    _patch_neutts_installed(monkeypatch)

    def choose_neutts(_question, choices, _default=0, _description=None):
        return next(
            index
            for index, choice in enumerate(choices)
            if choice.startswith("NeuTTS ")
        )

    def fail_install_prompt(*args, **kwargs):
        raise AssertionError("installed NeuTTS should not prompt for installation")

    monkeypatch.setattr(setup_mod, "prompt_choice", choose_neutts)
    monkeypatch.setattr(setup_mod, "prompt_yes_no", fail_install_prompt)

    config = load_config()
    setup_mod._setup_tts_provider(config)

    assert load_config()["tts"]["provider"] == "neutts"
    assert "NeuTTS is already installed" in capsys.readouterr().out
