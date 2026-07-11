from hermes_cli import setup


def _select_piper(question, choices, default=0):
    assert question == "Select TTS provider:"
    return next(i for i, choice in enumerate(choices) if choice.startswith("Piper "))


def test_setup_tts_lists_and_selects_installed_piper(monkeypatch):
    config = {}
    monkeypatch.setattr(setup, "managed_nous_tools_enabled", lambda: False)
    monkeypatch.setattr(setup, "prompt_choice", _select_piper)
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: object() if name == "piper" else None)
    monkeypatch.setattr(setup, "save_config", lambda value: None)

    setup._setup_tts_provider(config)

    assert config["tts"]["provider"] == "piper"


def test_setup_tts_uses_existing_piper_post_setup(monkeypatch):
    config = {}
    probes = iter([None, object()])
    post_setup_calls = []
    monkeypatch.setattr(setup, "managed_nous_tools_enabled", lambda: False)
    monkeypatch.setattr(setup, "prompt_choice", _select_piper)
    monkeypatch.setattr(setup, "prompt_yes_no", lambda *args: True)
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: next(probes) if name == "piper" else None)
    monkeypatch.setattr(setup, "save_config", lambda value: None)
    monkeypatch.setattr("hermes_cli.tools_config._run_post_setup", post_setup_calls.append)

    setup._setup_tts_provider(config)

    assert post_setup_calls == ["piper"]
    assert config["tts"]["provider"] == "piper"
