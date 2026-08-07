"""Behavior tests for the realtime voice provider contract and registry."""

from __future__ import annotations

import logging

import pytest

from agent import realtime_voice_registry
from agent.realtime_voice_provider import (
    RealtimeCapability,
    RealtimeToolResult,
    RealtimeVoiceProvider,
    RealtimeVoiceSession,
    SessionReady,
    UnsupportedRealtimeCapability,
)


class _FakeSession(RealtimeVoiceSession):
    def __init__(self, capabilities=()):
        super().__init__(capabilities)
        self.close_calls = 0
        self.result_batches = []
        self.continuations = []

    async def send_audio(self, audio, *, mime_type=None):
        return None

    async def _submit_tool_results(self, batch_id, results):
        self.result_batches.append((batch_id, tuple(results)))

    def events(self):
        async def _stream():
            yield SessionReady(session_id="session")

        return _stream()

    async def _close(self):
        self.close_calls += 1

    async def _continue_response(self, batch_id):
        self.continuations.append(batch_id)


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

    async def open_session(self, setup):
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
            async def open_session(self, setup):
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
        assert provider.capabilities == frozenset()
        assert provider.list_models() == ()
        assert provider.get_setup_schema()["env_vars"] == ()

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

        assert session.close_calls == 1

    @pytest.mark.asyncio
    async def test_optional_audio_commit_is_capability_gated(self):
        session = _FakeSession()
        with pytest.raises(UnsupportedRealtimeCapability):
            await session.commit_audio()

    @pytest.mark.asyncio
    async def test_explicit_interruption_is_opt_in(self):
        session = _FakeSession()

        with pytest.raises(UnsupportedRealtimeCapability, match="explicit_interruption"):
            await session.interrupt()

    @pytest.mark.asyncio
    async def test_event_stream_uses_normalized_envelope(self):
        session = _FakeSession()
        events = [event async for event in session.events()]

        assert events == [SessionReady(session_id="session")]

    @pytest.mark.asyncio
    async def test_tool_results_and_continuation_are_explicit_separate_operations(self):
        session = _FakeSession(
            {RealtimeCapability.TOOL_CALLING, RealtimeCapability.CONTINUATION}
        )
        results = [
            RealtimeToolResult("call-1", "batch", "first", {"n": 1}),
            RealtimeToolResult("call-2", "batch", "second", {"n": 2}),
        ]

        await session.submit_tool_results("batch", results)
        assert session.continuations == []

        await session.continue_response("batch")
        assert session.result_batches == [("batch", tuple(results))]
        assert session.continuations == ["batch"]

    @pytest.mark.asyncio
    async def test_tool_result_batch_rejects_mixed_or_empty_identity(self):
        session = _FakeSession()
        mixed = [RealtimeToolResult("call", "other", "tool", "output")]

        with pytest.raises(ValueError, match="batch_id"):
            await session.submit_tool_results("batch", mixed)
        with pytest.raises(ValueError, match="at least one"):
            await session.submit_tool_results("batch", [])

        assert session.result_batches == []
