import pytest

from hermes_cli import setup, setup_summary


def _select_piper(question, choices, default=0):
    assert question == "Select TTS provider:"
    return next(i for i, choice in enumerate(choices) if choice.startswith("Piper "))


def test_setup_tts_lists_and_selects_installed_piper(monkeypatch):
    config = {}
    monkeypatch.setattr("hermes_cli.setup_tts.tool_backend_helpers.managed_nous_tools_enabled", lambda: False)
    monkeypatch.setattr(setup, "prompt_choice", _select_piper)
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: object() if name == "piper" else None)
    monkeypatch.setattr(setup, "save_config", lambda value: None)

    setup.setup_tts(config)

    assert config["tts"]["provider"] == "piper"


def test_setup_tts_uses_existing_piper_post_setup(monkeypatch):
    config = {}
    probes = iter([None, object()])
    post_setup_calls = []
    monkeypatch.setattr("hermes_cli.setup_tts.tool_backend_helpers.managed_nous_tools_enabled", lambda: False)
    monkeypatch.setattr(setup, "prompt_choice", _select_piper)
    monkeypatch.setattr(setup, "prompt_yes_no", lambda *args: True)
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: next(probes) if name == "piper" else None)
    monkeypatch.setattr(setup, "save_config", lambda value: None)
    monkeypatch.setattr("hermes_cli.tools_config._run_post_setup", post_setup_calls.append)

    setup.setup_tts(config)

    assert post_setup_calls == ["piper"]
    assert config["tts"]["provider"] == "piper"


@pytest.mark.parametrize(
    ("installed", "available"),
    [(True, True), (False, False)],
)
def test_setup_summary_reports_piper_availability(installed, available, monkeypatch):
    monkeypatch.setattr(setup, "_module_installed", lambda name: name == "piper" and installed)

    status = setup_summary._voice_provider_status(
        "Text-to-Speech", "piper", setup_summary._TTS_SUMMARY_ROWS, setup_summary._TTS_SUMMARY_DEFAULT)

    assert "Piper" in status[0]
    assert status[1] is available
    assert ("not installed" in status[0]) is not available
