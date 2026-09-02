"""Behavior tests for the realtime voice provider registry and provider contract."""

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

    async def send_audio(self, audio):
        return None

    async def _submit_tool_results(self, results, continue_response):
        self.result_batches.append((tuple(results), continue_response))

    def _events(self):
        async def _stream():
            yield SessionReady(session_id="session")

        return _stream()

    async def _close(self):
        self.close_calls += 1


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

    @pytest.mark.parametrize("name", ["", " ", "\t", " padded", "padded "])
    def test_rejects_empty_name(self, name):
        with pytest.raises(ValueError, match="trimmed identifier"):
            realtime_voice_registry.register_provider(_FakeProvider(name=name))

    def test_rejects_oversized_name(self):
        with pytest.raises(ValueError, match="provider name"):
            realtime_voice_registry.register_provider(_FakeProvider(name="x" * 513))

    def test_rejects_incompatible_api_version(self, caplog):
        provider = _FakeProvider(name="future")
        provider.api_version = 999

        with caplog.at_level(logging.WARNING, logger="agent.realtime_voice_registry"):
            accepted = realtime_voice_registry.register_provider(provider)

        assert accepted is False
        assert realtime_voice_registry.get_provider("future") is None
        assert "targets API v999" in caplog.text

    def test_reregistration_replaces_previous(self):
        first = _FakeProvider(name="custom")
        second = _FakeProvider(name="custom")

        assert realtime_voice_registry.register_provider(first) is True
        assert realtime_voice_registry.register_provider(second) is True
        assert realtime_voice_registry.get_provider("custom") is second

    def test_scoped_registrations_are_isolated_and_fall_back_to_global(self):
        global_provider = _FakeProvider(name="custom")
        first = _FakeProvider(name="custom")
        second = _FakeProvider(name="custom")

        assert realtime_voice_registry.register_provider(global_provider) is True
        assert realtime_voice_registry.register_provider(first, scope="profile-a") is True
        assert realtime_voice_registry.register_provider(second, scope="profile-b") is True

        assert realtime_voice_registry.get_provider("custom", scope="profile-a") is first
        assert realtime_voice_registry.get_provider("custom", scope="profile-b") is second
        assert realtime_voice_registry.get_provider("custom", scope="profile-c") is global_provider

    def test_restore_registration_is_exact_and_scope_local(self):
        previous = _FakeProvider(name="custom")
        current = _FakeProvider(name="custom")
        replacement = _FakeProvider(name="custom")

        assert realtime_voice_registry.register_provider(previous, scope="profile") is True
        assert realtime_voice_registry.snapshot_registration("custom", scope="profile") is previous
        assert realtime_voice_registry.register_provider(current, scope="profile") is True
        assert (
            realtime_voice_registry.restore_registration(
                "custom", current, previous, scope="profile"
            )
            is True
        )
        assert realtime_voice_registry.get_provider("custom", scope="profile") is previous

        assert realtime_voice_registry.register_provider(replacement, scope="profile") is True
        assert (
            realtime_voice_registry.restore_registration(
                "custom", current, previous, scope="profile"
            )
            is False
        )
        assert realtime_voice_registry.get_provider("custom", scope="profile") is replacement

    def test_restore_without_previous_removes_current(self):
        current = _FakeProvider(name="custom")

        assert realtime_voice_registry.register_provider(current) is True
        assert realtime_voice_registry.restore_registration("custom", current, None) is True
        assert realtime_voice_registry.get_provider("custom") is None
        assert realtime_voice_registry.list_providers() == []

    def test_failed_scoped_restore_does_not_create_empty_scope(self):
        stale = _FakeProvider(name="custom")

        assert realtime_voice_registry.restore_registration(
            "custom", stale, None, scope="missing-profile"
        ) is False
        assert "missing-profile" not in realtime_voice_registry._scoped_providers

    def test_scoped_lookup_preserves_falsey_provider(self):
        class _FalseyProvider(_FakeProvider):
            def __bool__(self):
                return False

        global_provider = _FakeProvider(name="custom")
        scoped_provider = _FalseyProvider(name="custom")
        assert realtime_voice_registry.register_provider(global_provider) is True
        assert realtime_voice_registry.register_provider(
            scoped_provider, scope="profile"
        ) is True

        assert (
            realtime_voice_registry.get_provider("custom", scope="profile")
            is scoped_provider
        )


class TestLookup:
    def test_lookup_normalizes_case_and_whitespace(self):
        provider = _FakeProvider(name="gemini")
        realtime_voice_registry.register_provider(provider)

        assert realtime_voice_registry.get_provider(" GEMINI ") is provider

    def test_non_string_lookup_is_missing(self):
        assert realtime_voice_registry.get_provider(None) is None  # type: ignore[arg-type]

    def test_list_is_sorted_by_normalized_registry_name(self):
        realtime_voice_registry.register_provider(_FakeProvider(name="zylo"))
        realtime_voice_registry.register_provider(_FakeProvider(name="Alpha"))
        realtime_voice_registry.register_provider(_FakeProvider(name="middle"))

        assert [provider.name for provider in realtime_voice_registry.list_providers()] == [
            "Alpha",
            "middle",
            "zylo",
        ]

    def test_list_merges_scope_over_global(self):
        global_provider = _FakeProvider(name="shared")
        scoped_provider = _FakeProvider(name="shared")
        only_global = _FakeProvider(name="global-only")
        realtime_voice_registry.register_provider(global_provider)
        realtime_voice_registry.register_provider(only_global)
        realtime_voice_registry.register_provider(scoped_provider, scope="profile")

        listed = realtime_voice_registry.list_providers(scope="profile")

        assert listed == [only_global, scoped_provider]


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
        assert session.closed is True

    @pytest.mark.asyncio
    async def test_optional_operations_are_capability_gated(self):
        session = _FakeSession()

        with pytest.raises(UnsupportedRealtimeCapability, match="input_commit_events"):
            await session.commit_audio()
        with pytest.raises(UnsupportedRealtimeCapability, match="response_cancellation"):
            await session.cancel_response()
        with pytest.raises(UnsupportedRealtimeCapability, match="output_truncation"):
            await session.truncate_output("item", 10)
        with pytest.raises(UnsupportedRealtimeCapability, match="explicit_response"):
            await session.create_response()
        with pytest.raises(UnsupportedRealtimeCapability, match="dynamic_context"):
            await session.add_context("item", "text")

    @pytest.mark.asyncio
    async def test_event_stream_uses_normalized_envelope(self):
        session = _FakeSession()
        events = [event async for event in session.events()]

        assert events == [SessionReady(session_id="session")]

    @pytest.mark.asyncio
    async def test_tool_results_require_tool_calling(self):
        session = _FakeSession()

        with pytest.raises(UnsupportedRealtimeCapability, match="tool_calling"):
            await session.submit_tool_results([RealtimeToolResult("call", "done")])

        assert session.result_batches == []

    @pytest.mark.asyncio
    async def test_tool_results_are_delivered_as_one_ordered_batch(self):
        session = _FakeSession({RealtimeCapability.TOOL_CALLING})
        results = [
            RealtimeToolResult("call-1", "first"),
            RealtimeToolResult("call-2", "second"),
        ]

        await session.submit_tool_results(results, continue_response=False)
        await session.submit_tool_results(results[:1])

        assert session.result_batches == [
            (tuple(results), False),
            ((results[0],), True),
        ]

    @pytest.mark.asyncio
    async def test_tool_result_batch_rejects_empty_or_duplicate_identity(self):
        session = _FakeSession({RealtimeCapability.TOOL_CALLING})
        duplicate = [
            RealtimeToolResult("call", "first"),
            RealtimeToolResult("call", "second"),
        ]

        with pytest.raises(ValueError, match="at least one"):
            await session.submit_tool_results([])
        with pytest.raises(ValueError, match="repeat a call_id"):
            await session.submit_tool_results(duplicate)
        with pytest.raises(TypeError, match="RealtimeToolResult"):
            await session.submit_tool_results(["not-a-result"])  # type: ignore[list-item]

        assert session.result_batches == []
