"""Regression tests for Edge TTS voice selection during setup."""

from hermes_cli import setup_tts


def test_edge_setup_persists_selected_voice(monkeypatch):
    config = {"tts": {"provider": "openai", "edge": {"voice": "en-US-AriaNeural"}}}
    prompts = iter([0, 1])  # Edge TTS, then Simplified Chinese.

    monkeypatch.setattr(setup_tts._setup, "prompt_choice", lambda *args: next(prompts))
    monkeypatch.setattr(setup_tts._setup, "save_config", lambda value: None)
    monkeypatch.setattr(setup_tts._setup, "print_header", lambda *args: None)
    monkeypatch.setattr(setup_tts._setup, "_info", lambda *args: None)
    monkeypatch.setattr(setup_tts._setup, "print_success", lambda *args: None)

    setup_tts._setup_tts_provider(config)

    assert config["tts"]["provider"] == "edge"
    assert config["tts"]["edge"]["voice"] == "zh-CN-XiaoxiaoNeural"


def test_edge_setup_keeps_unknown_configured_voice(monkeypatch):
    configured_voice = "de-DE-KatjaNeural"
    config = {"tts": {"provider": "openai", "edge": {"voice": configured_voice}}}
    prompts = iter([0, 4])  # Edge TTS, then the default Keep current entry.
    calls = []

    def prompt_choice(*args, **kwargs):
        calls.append((args, kwargs))
        return next(prompts)

    monkeypatch.setattr(setup_tts._setup, "prompt_choice", prompt_choice)
    monkeypatch.setattr(setup_tts._setup, "save_config", lambda value: None)
    monkeypatch.setattr(setup_tts._setup, "print_header", lambda *args: None)
    monkeypatch.setattr(setup_tts._setup, "_info", lambda *args: None)
    monkeypatch.setattr(setup_tts._setup, "print_success", lambda *args: None)

    setup_tts._setup_tts_provider(config)

    assert config["tts"]["provider"] == "edge"
    assert config["tts"]["edge"]["voice"] == configured_voice
    assert calls[1][0][2] == 4
    assert calls[1][0][1][-1] == f"Keep current ({configured_voice})"
