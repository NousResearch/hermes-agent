"""Edge TTS voice selection in setup keeps existing voices and allows custom IDs."""

import pytest

from hermes_cli import setup_tts


@pytest.fixture
def run_setup(monkeypatch):
    """Pick Edge TTS, then answer the voice menu (and optional ID prompt)."""
    for name in ("save_config", "print_header", "_info", "print_info", "print_success"):
        monkeypatch.setattr(setup_tts._setup, name, lambda *a, **k: None)

    def run(config, pick_voice, typed=None):
        calls = []

        def prompt_choice(question, choices, default=0, description=None):
            calls.append((question, list(choices), default, description))
            if "provider" in question:
                return next(i for i, label in enumerate(choices) if label.startswith("Edge TTS"))
            return pick_voice(choices, default)

        monkeypatch.setattr(setup_tts._setup, "prompt_choice", prompt_choice)
        monkeypatch.setattr(setup_tts._setup, "prompt", lambda *a, **k: typed)
        setup_tts._setup_tts_provider(config)
        return calls[-1]

    return run


def test_edge_setup_persists_selected_voice(run_setup):
    config = {"tts": {"provider": "openai", "edge": {"voice": "en-US-AriaNeural"}}}

    *_, description = run_setup(config, lambda choices, _: next(
        i for i, label in enumerate(choices) if label.startswith("Chinese (Simplified)")))

    assert "language" in description
    assert config["tts"]["provider"] == "edge"
    assert config["tts"]["edge"]["voice"] == "zh-CN-XiaoxiaoNeural"


def test_unlisted_current_voice_is_offered_first_and_kept(run_setup):
    config = {"tts": {"provider": "openai", "edge": {"voice": "fr-FR-DeniseNeural"}}}

    _, choices, default, _ = run_setup(config, lambda _, default: default)

    assert (default, choices[0]) == (0, "Current voice — fr-FR-DeniseNeural")
    assert config["tts"]["edge"]["voice"] == "fr-FR-DeniseNeural"


@pytest.mark.parametrize("typed, expected", [
    ("  de-DE-KatjaNeural ", "de-DE-KatjaNeural"),
    ("", "ja-JP-NanamiNeural"),
])
def test_other_voice_accepts_custom_id_and_blank_keeps_current(run_setup, typed, expected):
    config = {"tts": {"provider": "edge", "edge": {"voice": "ja-JP-NanamiNeural"}}}

    _, choices, _, _ = run_setup(config, lambda choices, _: len(choices) - 1, typed=typed)

    assert choices[-1] == "Other voice (enter voice ID)"
    assert config["tts"]["edge"]["voice"] == expected
