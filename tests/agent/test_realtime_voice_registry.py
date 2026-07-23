"""Behavior tests for the realtime voice provider contract and registry."""

from __future__ import annotations

import logging

import pytest

from agent import realtime_voice_registry
from agent.realtime_voice_provider import (
    RealtimeVoiceProvider,
    RealtimeVoiceSession,
)


class _FakeSession(RealtimeVoiceSession):
    def __init__(self):
        self.closed = False

    async def send_audio(self, audio, *, mime_type=None):
        return None

    async def send_text(self, text, *, end_of_turn=True):
        return None

    async def send_tool_result(self, call_id, output, *, name=None):
        return None

    def events(self):
        async def _stream():
            yield {"type": "session.started"}

        return _stream()

    async def close(self):
        self.closed = True


class _FakeProvider(RealtimeVoiceProvider):
    def __init__(self, name="fake", display=None):
        self._name = name
        self._display = display

    @property
    def name(self):
        return self._name

    @property
    def display_name(self):
        return self._display or super().display_name

    async def open_session(self, **kwargs):
        return _FakeSession()


@pytest.fixture(autouse=True)
def _clean_registry():
    realtime_voice_registry._reset_for_tests()
    yield
    realtime_voice_registry._reset_for_tests()


class TestRegistration:
    def test_rejects_non_provider_type(self):
        with pytest.raises(TypeError, match="RealtimeVoiceProvider"):
            realtime_voice_registry.register_provider(object())  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["", " ", "\t"])
    def test_rejects_empty_name(self, name):
        with pytest.raises(ValueError, match="non-empty"):
            realtime_voice_registry.register_provider(_FakeProvider(name=name))

    def test_rejects_incompatible_api_version(self, caplog):
        provider = _FakeProvider(name="future")
        provider.api_version = 999

        with caplog.at_level(logging.WARNING, logger="agent.realtime_voice_registry"):
            accepted = realtime_voice_registry.register_provider(provider)

        assert accepted is False
        assert realtime_voice_registry.get_provider("future") is None
        assert "targets API v999" in caplog.text

    def test_plugin_reregistration_replaces_plugin(self):
        first = _FakeProvider(name="custom")
        second = _FakeProvider(name="custom")

        assert realtime_voice_registry.register_provider(first) is True
        assert realtime_voice_registry.register_provider(second) is True
        assert realtime_voice_registry.get_provider("custom") is second

    def test_builtin_replaces_plugin_and_cannot_be_shadowed(self, caplog):
        plugin = _FakeProvider(name="openai")
        built_in = _FakeProvider(name="openai")
        shadow = _FakeProvider(name=" OPENAI ")

        assert realtime_voice_registry.register_provider(plugin) is True
        assert realtime_voice_registry.register_provider(built_in, built_in=True) is True
        assert realtime_voice_registry.get_provider("openai") is built_in
        assert realtime_voice_registry.is_builtin_provider(" OpenAI ") is True

        with caplog.at_level(logging.WARNING, logger="agent.realtime_voice_registry"):
            accepted = realtime_voice_registry.register_provider(shadow)

        assert accepted is False
        assert realtime_voice_registry.get_provider("openai") is built_in
        assert "Built-in providers always win" in caplog.text


class TestLookup:
    def test_lookup_normalizes_case_and_whitespace(self):
        provider = _FakeProvider(name="gemini")
        realtime_voice_registry.register_provider(provider)

        assert realtime_voice_registry.get_provider(" GEMINI ") is provider

    def test_non_string_lookup_is_missing(self):
        assert realtime_voice_registry.get_provider(None) is None  # type: ignore[arg-type]
        assert realtime_voice_registry.is_builtin_provider(None) is False  # type: ignore[arg-type]

    def test_list_is_sorted_by_normalized_registry_name(self):
        realtime_voice_registry.register_provider(_FakeProvider(name="zylo"))
        realtime_voice_registry.register_provider(_FakeProvider(name="Alpha"))
        realtime_voice_registry.register_provider(_FakeProvider(name="middle"))

        assert [provider.name for provider in realtime_voice_registry.list_providers()] == [
            "Alpha",
            "middle",
            "zylo",
        ]


class TestProviderContract:
    def test_requires_name(self):
        class Incomplete(RealtimeVoiceProvider):
            async def open_session(self, **kwargs):
                return _FakeSession()

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_requires_open_session(self):
        class Incomplete(RealtimeVoiceProvider):
            @property
            def name(self):
                return "incomplete"

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_defaults_are_safe_and_provider_neutral(self):
        provider = _FakeProvider(name="openai-realtime")

        assert provider.display_name == "Openai-Realtime"
        assert provider.is_available() is True
        assert provider.default_model() is None
        assert provider.default_voice() is None
        assert provider.get_capabilities()["tool_calling"] is False
        assert provider.get_capabilities()["transports"] == []
        assert provider.get_setup_schema()["env_vars"] == []

    def test_defaults_follow_provider_catalog_order(self):
        class CatalogProvider(_FakeProvider):
            def list_models(self):
                return [{"id": "primary"}, {"id": "fallback"}]

            def list_voices(self):
                return [{"id": "alloy"}, {"id": "verse"}]

        provider = CatalogProvider()
        assert provider.default_model() == "primary"
        assert provider.default_voice() == "alloy"


class TestSessionContract:
    def test_requires_core_lifecycle_methods(self):
        class Incomplete(RealtimeVoiceSession):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    @pytest.mark.asyncio
    async def test_async_context_closes_session(self):
        session = _FakeSession()

        async with session as active:
            assert active is session

        assert session.closed is True

    @pytest.mark.asyncio
    async def test_optional_audio_commit_is_noop(self):
        session = _FakeSession()
        assert await session.commit_audio() is None

    @pytest.mark.asyncio
    async def test_explicit_interruption_is_opt_in(self):
        session = _FakeSession()

        with pytest.raises(NotImplementedError, match="explicit interruption"):
            await session.interrupt()

    @pytest.mark.asyncio
    async def test_event_stream_uses_normalized_envelope(self):
        session = _FakeSession()
        events = [event async for event in session.events()]

        assert events == [{"type": "session.started"}]
