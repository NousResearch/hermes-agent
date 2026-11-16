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
- Sync invariant: registry built-ins match speech/transcription/transcriber.py
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




    @pytest.mark.parametrize("builtin", sorted(transcription_registry._BUILTIN_NAMES))
    def test_rejects_builtin_shadow_with_warning(self, builtin, caplog):
        p = _FakeProvider(name=builtin)
        with caplog.at_level(logging.WARNING, logger="agent.transcription_registry"):
            transcription_registry.register_provider(p)
        assert builtin in caplog.text
        assert transcription_registry.get_provider(builtin) is None
        assert transcription_registry.list_providers() == []




# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestLookup:

    def test_get_provider_non_string_returns_none(self):
        assert transcription_registry.get_provider(None) is None  # type: ignore[arg-type]
        assert transcription_registry.get_provider(123) is None  # type: ignore[arg-type]



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
