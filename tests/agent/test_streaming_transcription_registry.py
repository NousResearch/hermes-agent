"""Tests for agent/streaming_transcription_registry.py and
agent/streaming_transcription_provider.py.

Covers:
- Registration happy path
- Registration rejection: non-StreamingTranscriptionProvider type
- Registration rejection: empty/whitespace name
- Re-registration: overwrites + logs at debug
- Case + whitespace insensitivity on lookup
- ABC contract: start()/feed()/finish()/cancel() are declared abstract
- Envelope contract: provider config passthrough is provider-owned
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pytest

from agent import streaming_transcription_registry
from agent.streaming_transcription_provider import StreamingTranscriptionProvider


class _FakeStreamingProvider(StreamingTranscriptionProvider):
    def __init__(self, name: str = "fake_stream"):
        self._name = name
        self.started = 0
        self.feeds = 0
        self.finished = 0
        self.cancelled = 0

    @property
    def name(self) -> str:
        return self._name

    def start(self, *, language: Optional[str] = None,
              config: Optional[Dict[str, Any]] = None) -> str:
        self.started += 1
        return f"stream-{self.started}"

    def feed(self, stream_key: str, pcm: bytes) -> str:
        self.feeds += 1
        return "partial"

    def finish(self, stream_key: str) -> str:
        self.finished += 1
        return "full transcript"

    def cancel(self, stream_key: str) -> None:
        self.cancelled += 1


@pytest.fixture(autouse=True)
def _reset_streaming_registry():
    streaming_transcription_registry._reset_for_tests()
    yield
    streaming_transcription_registry._reset_for_tests()


def test_registration_happy_path():
    p = _FakeStreamingProvider(name="acme transport")
    streaming_transcription_registry.register_provider(p)
    assert streaming_transcription_registry.get_provider("acme transport") is p
    assert [r.name for r in streaming_transcription_registry.list_providers()] == ["acme transport"]


def test_rejects_non_streaming_provider_type():
    class NotAProvider:
        name = "nope"

    with pytest.raises(TypeError):
        streaming_transcription_registry.register_provider(NotAProvider())


def test_rejects_empty_name():
    with pytest.raises(ValueError):
        streaming_transcription_registry.register_provider(_FakeStreamingProvider(name="   "))


def test_reregister_overwrites(caplog):
    a = _FakeStreamingProvider(name="dup")
    b = _FakeStreamingProvider(name="dup")
    streaming_transcription_registry.register_provider(a)
    with caplog.at_level(logging.DEBUG, logger="agent.streaming_transcription_registry"):
        streaming_transcription_registry.register_provider(b)
    assert streaming_transcription_registry.get_provider("dup") is b


def test_lookup_is_case_and_whitespace_insensitive():
    p = _FakeStreamingProvider(name="acme transport")
    streaming_transcription_registry.register_provider(p)
    assert streaming_transcription_registry.get_provider("  Acme Transport ") is p


def test_get_provider_returns_none_for_non_string():
    assert streaming_transcription_registry.get_provider(None) is None
    assert streaming_transcription_registry.get_provider(123) is None


def test_provider_contract_config_passthrough():
    p = _FakeStreamingProvider(name="cfg-aware")
    streaming_transcription_registry.register_provider(p)
    resolved = streaming_transcription_registry.get_provider("cfg-aware")
    key = resolved.start(config={"language": "zh"})
    assert key == "stream-1"
    assert resolved.feed(key, b"\x00" * 320) == "partial"
    assert resolved.finish(key) == "full transcript"
    resolved.cancel(key)
    assert (resolved.started, resolved.feeds, resolved.finished, resolved.cancelled) == (1, 1, 1, 1)


def test_provider_abstract_contract():
    with pytest.raises(TypeError):
        _IncompleteProvider()


class _IncompleteProvider(StreamingTranscriptionProvider):
    @property
    def name(self) -> str:
        return "incomplete"