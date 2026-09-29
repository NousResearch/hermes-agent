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
