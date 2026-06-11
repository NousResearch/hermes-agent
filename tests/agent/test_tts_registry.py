"""Tests for agent/tts_registry.py and agent/tts_provider.py.

Covers:
- Registration happy path
- Registration rejection: non-TTSProvider type
- Registration rejection: empty/whitespace name
- Built-in name shadowing: warning + silent ignore (no exception)
- Re-registration: overwrites + logs at debug
- Case + whitespace insensitivity on lookup
- ABC contract: default implementations work
- ABC contract: synthesize() must be implemented
- ABC contract: stream() raises NotImplementedError by default
- resolve_output_format helper coerces invalid input
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pytest

from agent import tts_registry
from agent.tts_provider import (
    DEFAULT_OUTPUT_FORMAT,
    VALID_OUTPUT_FORMATS,
    TTSProvider,
    resolve_output_format,
)


class _FakeProvider(TTSProvider):
    def __init__(
        self,
        name: str = "fake",
        display: Optional[str] = None,
        voice_compat: bool = False,
        synthesize_impl: Optional[Any] = None,
    ):
        self._name = name
        self._display = display
        self._voice_compat = voice_compat
        self._synthesize_impl = synthesize_impl

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._display if self._display is not None else super().display_name

    @property
    def voice_compatible(self) -> bool:
        return self._voice_compat

    def synthesize(self, text: str, output_path: str, **kw):
        if self._synthesize_impl is not None:
            return self._synthesize_impl(text, output_path, **kw)
        return output_path


@pytest.fixture(autouse=True)
def _reset_registry():
    tts_registry._reset_for_tests()
    yield
    tts_registry._reset_for_tests()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_happy_path(self):
        p = _FakeProvider(name="cartesia")
        tts_registry.register_provider(p)
        assert tts_registry.get_provider("cartesia") is p
        assert [r.name for r in tts_registry.list_providers()] == ["cartesia"]

    def test_rejects_non_provider_type(self):
        with pytest.raises(TypeError, match="expects a TTSProvider instance"):
            tts_registry.register_provider("not a provider")  # type: ignore[arg-type]
        assert tts_registry.list_providers() == []

    def test_rejects_empty_name(self):
        p = _FakeProvider(name="")
        with pytest.raises(ValueError, match="non-empty string"):
            tts_registry.register_provider(p)
        assert tts_registry.list_providers() == []

    def test_rejects_whitespace_name(self):
        p = _FakeProvider(name="   ")
        with pytest.raises(ValueError, match="non-empty string"):
            tts_registry.register_provider(p)
        assert tts_registry.list_providers() == []

    @pytest.mark.parametrize(
        "builtin",
        ["edge", "openai", "elevenlabs", "minimax", "gemini",
         "mistral", "xai", "piper", "kittentts", "neutts"],
    )
    def test_rejects_builtin_shadow_with_warning(self, builtin, caplog):
        """Built-in names always win — plugin registration is silently ignored
        but a warning is logged so the operator can see what happened.
        """
        p = _FakeProvider(name=builtin)
        with caplog.at_level(logging.WARNING, logger="agent.tts_registry"):
            tts_registry.register_provider(p)
        assert "shadows a built-in name" in caplog.text
        assert builtin in caplog.text
        assert tts_registry.get_provider(builtin) is None
        assert tts_registry.list_providers() == []

    def test_builtin_shadow_case_insensitive(self, caplog):
        """``EDGE``/``Edge``/``  edge  `` all collide with the ``edge`` built-in."""
        for variant in ("EDGE", "Edge", "  edge  ", "eDgE"):
            tts_registry._reset_for_tests()
            with caplog.at_level(logging.WARNING, logger="agent.tts_registry"):
                tts_registry.register_provider(_FakeProvider(name=variant))
            assert tts_registry.list_providers() == [], (
                f"variant {variant!r} should have been rejected as a built-in shadow"
            )

    def test_reregistration_overwrites(self, caplog):
        p1 = _FakeProvider(name="cartesia")
        p2 = _FakeProvider(name="cartesia")
        tts_registry.register_provider(p1)
        with caplog.at_level(logging.DEBUG, logger="agent.tts_registry"):
            tts_registry.register_provider(p2)
        assert tts_registry.get_provider("cartesia") is p2
        assert "re-registered" in caplog.text


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestLookup:
    def test_get_provider_missing_returns_none(self):
        assert tts_registry.get_provider("nonexistent") is None

    def test_get_provider_non_string_returns_none(self):
        assert tts_registry.get_provider(None) is None  # type: ignore[arg-type]
        assert tts_registry.get_provider(123) is None  # type: ignore[arg-type]

    def test_get_provider_case_insensitive(self):
        p = _FakeProvider(name="cartesia")
        tts_registry.register_provider(p)
        assert tts_registry.get_provider("CARTESIA") is p
        assert tts_registry.get_provider("Cartesia") is p

    def test_get_provider_whitespace_tolerant(self):
        p = _FakeProvider(name="cartesia")
        tts_registry.register_provider(p)
        assert tts_registry.get_provider("  cartesia  ") is p

    def test_list_providers_sorted(self):
        tts_registry.register_provider(_FakeProvider(name="zylo"))
        tts_registry.register_provider(_FakeProvider(name="alpha"))
        tts_registry.register_provider(_FakeProvider(name="middle"))
        names = [p.name for p in tts_registry.list_providers()]
        assert names == ["alpha", "middle", "zylo"]


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


class TestABCContract:
    def test_must_implement_synthesize(self):
        class Incomplete(TTSProvider):
            @property
            def name(self) -> str:
                return "incomplete"
            # synthesize NOT implemented

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_must_implement_name(self):
        class Incomplete(TTSProvider):
            def synthesize(self, text, output_path, **kw):
                return output_path
            # name NOT implemented

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_display_name_defaults_to_title(self):
        p = _FakeProvider(name="cartesia")
        assert p.display_name == "Cartesia"

    def test_display_name_override_respected(self):
        p = _FakeProvider(name="cartesia", display="Cartesia AI")
        assert p.display_name == "Cartesia AI"

    def test_is_available_default_true(self):
        p = _FakeProvider(name="cartesia")
        assert p.is_available() is True

    def test_list_voices_default_empty(self):
        p = _FakeProvider(name="cartesia")
        assert p.list_voices() == []

    def test_list_models_default_empty(self):
        p = _FakeProvider(name="cartesia")
        assert p.list_models() == []

    def test_default_model_none_when_no_models(self):
        p = _FakeProvider(name="cartesia")
        assert p.default_model() is None

    def test_default_voice_none_when_no_voices(self):
        p = _FakeProvider(name="cartesia")
        assert p.default_voice() is None

    def test_default_model_first_listed(self):
        class WithModels(_FakeProvider):
            def list_models(self):
                return [{"id": "sonic-2"}, {"id": "sonic-1"}]

        p = WithModels(name="cartesia")
        assert p.default_model() == "sonic-2"

    def test_default_voice_first_listed(self):
        class WithVoices(_FakeProvider):
            def list_voices(self):
                return [{"id": "voice-aria"}, {"id": "voice-jasper"}]

        p = WithVoices(name="cartesia")
        assert p.default_voice() == "voice-aria"

    def test_get_setup_schema_default_minimal(self):
        p = _FakeProvider(name="cartesia")
        schema = p.get_setup_schema()
        assert schema["name"] == "Cartesia"
        assert schema["env_vars"] == []

    def test_stream_raises_not_implemented_by_default(self):
        p = _FakeProvider(name="cartesia")
        with pytest.raises(NotImplementedError, match="does not implement streaming"):
            next(p.stream("hello"))

    def test_voice_compatible_default_false(self):
        p = _FakeProvider(name="cartesia")
        assert p.voice_compatible is False

    def test_voice_compatible_override(self):
        p = _FakeProvider(name="cartesia", voice_compat=True)
        assert p.voice_compatible is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestResolveOutputFormat:
    @pytest.mark.parametrize("valid", sorted(VALID_OUTPUT_FORMATS))
    def test_valid_passes_through(self, valid):
        assert resolve_output_format(valid) == valid

    def test_uppercase_normalized(self):
        assert resolve_output_format("MP3") == "mp3"
        assert resolve_output_format("Opus") == "opus"

    def test_whitespace_stripped(self):
        assert resolve_output_format("  wav  ") == "wav"

    def test_invalid_returns_default(self):
        assert resolve_output_format("aiff") == DEFAULT_OUTPUT_FORMAT
        assert resolve_output_format("") == DEFAULT_OUTPUT_FORMAT

    def test_none_returns_default(self):
        assert resolve_output_format(None) == DEFAULT_OUTPUT_FORMAT

    def test_non_string_returns_default(self):
        assert resolve_output_format(123) == DEFAULT_OUTPUT_FORMAT  # type: ignore[arg-type]
        assert resolve_output_format([]) == DEFAULT_OUTPUT_FORMAT  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sync invariant: registry's built-in list vs dispatcher's built-in list
# ---------------------------------------------------------------------------


class TestBuiltinSync:
    """``_BUILTIN_NAMES`` in agent/tts_registry.py is duplicated from
    ``BUILTIN_TTS_PROVIDERS`` in tools/tts_tool.py (importing directly
    would create a circular dependency). This test fails loudly if the
    two lists drift — a new built-in added to tts_tool.py MUST also be
    added to tts_registry.py's _BUILTIN_NAMES or the registry will
    accept a name the dispatcher will silently route to the wrong
    handler.
    """

    def test_registry_builtins_match_dispatcher_builtins(self):
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS

        assert tts_registry._BUILTIN_NAMES == BUILTIN_TTS_PROVIDERS, (
            "agent.tts_registry._BUILTIN_NAMES and "
            "tools.tts_tool.BUILTIN_TTS_PROVIDERS have drifted!\n"
            f"  Registry only: {sorted(tts_registry._BUILTIN_NAMES - BUILTIN_TTS_PROVIDERS)}\n"
            f"  Dispatcher only: {sorted(BUILTIN_TTS_PROVIDERS - tts_registry._BUILTIN_NAMES)}\n"
            "Add the missing names to whichever list is incomplete. "
            "These two lists exist as a circular-import workaround and "
            "MUST be kept in sync manually."
        )
