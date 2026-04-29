"""Tests for agent/tts_provider.py — ABC contract and default behavior."""

from __future__ import annotations

import pytest


def test_cannot_instantiate_without_name_and_synthesize():
    """TtsProvider must be abstract: name and synthesize required."""
    from agent.tts_provider import TtsProvider

    with pytest.raises(TypeError):
        TtsProvider()  # type: ignore[abstract]


def test_concrete_subclass_with_only_required_members_works():
    """A minimal concrete subclass only needs name + synthesize."""
    from agent.tts_provider import TtsProvider

    class Minimal(TtsProvider):
        @property
        def name(self) -> str:
            return "minimal"

        def synthesize(self, text, output_path, config):
            return {
                "success": True,
                "file_path": output_path,
                "format": "mp3",
                "native_opus": False,
                "voice_compatible": False,
            }

    p = Minimal()
    assert p.name == "minimal"
    assert p.display_name == "Minimal"  # default: name.title()
    assert p.is_available() is True  # default
    assert p.max_text_length() == 4000  # default
    assert p.list_voices() == []  # default
    assert p.default_voice() is None  # default
    schema = p.get_setup_schema()
    assert schema["name"] == "Minimal"
    assert schema["badge"] == ""
    assert schema["env_vars"] == []


def test_display_name_can_be_overridden():
    from agent.tts_provider import TtsProvider

    class Nice(TtsProvider):
        @property
        def name(self) -> str:
            return "nice"

        @property
        def display_name(self) -> str:
            return "Very Nice TTS"

        def synthesize(self, text, output_path, config):
            return {"success": True, "file_path": output_path, "format": "mp3",
                    "native_opus": False, "voice_compatible": False}

    assert Nice().display_name == "Very Nice TTS"


def test_synthesize_result_schema_is_documented():
    """The ABC docstring documents the expected result dict shape."""
    from agent.tts_provider import TtsProvider
    # The class-level docstring references file_path / format / native_opus / voice_compatible
    doc = TtsProvider.synthesize.__doc__ or ""
    for key in ("file_path", "format", "native_opus", "voice_compatible",
                "success", "error_type"):
        assert key in doc, f"synthesize docstring missing '{key}' — contract drift"


def test_subclass_missing_name_is_abstract():
    """Forgetting `name` property keeps the class abstract."""
    from agent.tts_provider import TtsProvider

    class NoName(TtsProvider):
        def synthesize(self, text, output_path, config):
            return {}

    with pytest.raises(TypeError):
        NoName()  # type: ignore[abstract]


def test_subclass_missing_synthesize_is_abstract():
    """Forgetting `synthesize` keeps the class abstract."""
    from agent.tts_provider import TtsProvider

    class NoSynth(TtsProvider):
        @property
        def name(self) -> str:
            return "nosynth"

    with pytest.raises(TypeError):
        NoSynth()  # type: ignore[abstract]
