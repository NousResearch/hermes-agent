"""Tests for agent/transcription_registry.py and agent/transcription_provider.py.

Covers:
- Registration happy path
- Registration rejection: non-TranscriptionProvider type
- Registration rejection: empty/whitespace name
- Built-in name shadowing: warning + silent ignore (no exception)
- Re-registration: overwrites + logs at debug
- Case + whitespace insensitivity on lookup
- ABC contract: default implementations work
- ABC contract: transcribe() must be implemented
- Sync invariant: registry built-ins match tools/transcription_tools.py
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pytest

from agent import transcription_registry
from agent.transcription_provider import TranscriptionProvider


class _FakeProvider(TranscriptionProvider):
    def __init__(
        self,
        name: str = "fake",
        display: Optional[str] = None,
        available: bool = True,
        transcribe_impl: Optional[Any] = None,
    ):
        self._name = name
        self._display = display
        self._available = available
        self._transcribe_impl = transcribe_impl

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._display if self._display is not None else super().display_name

    def is_available(self) -> bool:
        return self._available

    def transcribe(self, file_path: str, **kw):
        if self._transcribe_impl is not None:
            return self._transcribe_impl(file_path, **kw)
        return {"success": True, "transcript": f"fake({file_path})", "provider": self._name}


@pytest.fixture(autouse=True)
def _reset_registry():
    transcription_registry._reset_for_tests()
    yield
    transcription_registry._reset_for_tests()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_happy_path(self):
        p = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(p)
        assert transcription_registry.get_provider("openrouter") is p
        assert [r.name for r in transcription_registry.list_providers()] == ["openrouter"]

    def test_rejects_non_provider_type(self):
        with pytest.raises(TypeError, match="expects a TranscriptionProvider instance"):
            transcription_registry.register_provider("not a provider")  # type: ignore[arg-type]
        assert transcription_registry.list_providers() == []

    def test_rejects_empty_name(self):
        p = _FakeProvider(name="")
        with pytest.raises(ValueError, match="non-empty string"):
            transcription_registry.register_provider(p)
        assert transcription_registry.list_providers() == []

    def test_rejects_whitespace_name(self):
        p = _FakeProvider(name="   ")
        with pytest.raises(ValueError, match="non-empty string"):
            transcription_registry.register_provider(p)
        assert transcription_registry.list_providers() == []

    @pytest.mark.parametrize(
        "builtin",
        ["local", "local_command", "local_server", "groq", "openai", "mistral", "xai"],
    )
    def test_rejects_builtin_shadow_with_warning(self, builtin, caplog):
        p = _FakeProvider(name=builtin)
        with caplog.at_level(logging.WARNING, logger="agent.transcription_registry"):
            transcription_registry.register_provider(p)
        assert "shadows a built-in name" in caplog.text
        assert builtin in caplog.text
        assert transcription_registry.get_provider(builtin) is None
        assert transcription_registry.list_providers() == []

    def test_builtin_shadow_case_insensitive(self, caplog):
        for variant in ("OPENAI", "OpenAi", "  openai  ", "oPeNaI"):
            transcription_registry._reset_for_tests()
            with caplog.at_level(logging.WARNING, logger="agent.transcription_registry"):
                transcription_registry.register_provider(_FakeProvider(name=variant))
            assert transcription_registry.list_providers() == [], (
                f"variant {variant!r} should have been rejected as a built-in shadow"
            )

    def test_reregistration_overwrites(self, caplog):
        p1 = _FakeProvider(name="openrouter")
        p2 = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(p1)
        with caplog.at_level(logging.DEBUG, logger="agent.transcription_registry"):
            transcription_registry.register_provider(p2)
        assert transcription_registry.get_provider("openrouter") is p2
        assert "re-registered" in caplog.text


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestLookup:
    def test_get_provider_missing_returns_none(self):
        assert transcription_registry.get_provider("nonexistent") is None

    def test_get_provider_non_string_returns_none(self):
        assert transcription_registry.get_provider(None) is None  # type: ignore[arg-type]
        assert transcription_registry.get_provider(123) is None  # type: ignore[arg-type]

    def test_get_provider_case_insensitive(self):
        p = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(p)
        assert transcription_registry.get_provider("OPENROUTER") is p
        assert transcription_registry.get_provider("OpenRouter") is p

    def test_get_provider_whitespace_tolerant(self):
        p = _FakeProvider(name="openrouter")
        transcription_registry.register_provider(p)
        assert transcription_registry.get_provider("  openrouter  ") is p

    def test_list_providers_sorted(self):
        transcription_registry.register_provider(_FakeProvider(name="zylo"))
        transcription_registry.register_provider(_FakeProvider(name="alpha"))
        transcription_registry.register_provider(_FakeProvider(name="middle"))
        names = [p.name for p in transcription_registry.list_providers()]
        assert names == ["alpha", "middle", "zylo"]


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


class TestABCContract:
    def test_must_implement_transcribe(self):
        class Incomplete(TranscriptionProvider):
            @property
            def name(self) -> str:
                return "incomplete"
            # transcribe NOT implemented

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_must_implement_name(self):
        class Incomplete(TranscriptionProvider):
            def transcribe(self, file_path, **kw):
                return {"success": True, "transcript": "", "provider": "incomplete"}
            # name NOT implemented

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_display_name_defaults_to_title(self):
        p = _FakeProvider(name="openrouter")
        assert p.display_name == "Openrouter"

    def test_display_name_override_respected(self):
        p = _FakeProvider(name="openrouter", display="OpenRouter STT")
        assert p.display_name == "OpenRouter STT"

    def test_is_available_default_true(self):
        p = _FakeProvider(name="openrouter")
        assert p.is_available() is True

    def test_list_models_default_empty(self):
        p = _FakeProvider(name="openrouter")
        assert p.list_models() == []

    def test_default_model_none_when_no_models(self):
        p = _FakeProvider(name="openrouter")
        assert p.default_model() is None

    def test_default_model_first_listed(self):
        class WithModels(_FakeProvider):
            def list_models(self):
                return [{"id": "whisper-large-v3-turbo"}, {"id": "whisper-large-v3"}]

        p = WithModels(name="openrouter")
        assert p.default_model() == "whisper-large-v3-turbo"

    def test_get_setup_schema_default_minimal(self):
        p = _FakeProvider(name="openrouter")
        schema = p.get_setup_schema()
        assert schema["name"] == "Openrouter"
        assert schema["env_vars"] == []


# ---------------------------------------------------------------------------
# Sync invariant: registry built-ins vs dispatcher built-ins
# ---------------------------------------------------------------------------


class TestBuiltinSync:
    """``_BUILTIN_NAMES`` in agent/transcription_registry.py is duplicated
    from ``BUILTIN_STT_PROVIDERS`` in tools/transcription_tools.py
    (importing directly would create a circular dependency). This test
    fails loudly if the two lists drift — a new built-in added to
    transcription_tools.py MUST also be added to
    transcription_registry.py's ``_BUILTIN_NAMES`` or the registry will
    accept a name the dispatcher will silently route to the wrong
    handler.
    """

    def test_registry_builtins_match_dispatcher_builtins(self):
        from tools.transcription_tools import BUILTIN_STT_PROVIDERS

        assert transcription_registry._BUILTIN_NAMES == BUILTIN_STT_PROVIDERS, (
            "agent.transcription_registry._BUILTIN_NAMES and "
            "tools.transcription_tools.BUILTIN_STT_PROVIDERS have drifted!\n"
            f"  Registry only: {sorted(transcription_registry._BUILTIN_NAMES - BUILTIN_STT_PROVIDERS)}\n"
            f"  Dispatcher only: {sorted(BUILTIN_STT_PROVIDERS - transcription_registry._BUILTIN_NAMES)}\n"
            "Add the missing names to whichever list is incomplete. "
            "These two lists exist as a circular-import workaround and "
            "MUST be kept in sync manually."
        )
