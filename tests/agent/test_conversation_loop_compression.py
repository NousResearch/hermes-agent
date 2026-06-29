"""Tests for compression decisions in the conversation loop."""

import pytest

from agent.conversation_loop import _should_compress_with_messages


def test_should_compress_with_messages_passes_transcript_context():
    class Recorder:
        def __init__(self):
            self.calls = []

        def should_compress(self, prompt_tokens=None, messages=None):
            self.calls.append((prompt_tokens, messages))
            return True

    compressor = Recorder()
    messages = [{"role": "assistant", "content": "summary"}]

    assert _should_compress_with_messages(compressor, 90_000, messages) is True
    assert compressor.calls == [(90_000, messages)]


def test_should_compress_with_messages_falls_back_for_legacy_engines():
    class Legacy:
        def __init__(self):
            self.calls = []

        def should_compress(self, prompt_tokens=None):
            self.calls.append(prompt_tokens)
            return False

    compressor = Legacy()

    assert _should_compress_with_messages(compressor, 90_000, []) is False
    assert compressor.calls == [90_000]


def test_should_compress_with_messages_does_not_hide_internal_type_errors():
    class Broken:
        def should_compress(self, prompt_tokens=None, messages=None):
            raise TypeError("internal bug unrelated to keyword handling")

    with pytest.raises(TypeError, match="internal bug"):
        _should_compress_with_messages(Broken(), 90_000, [])