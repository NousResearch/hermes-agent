"""Regression tests for context-engine compression signature compatibility.

Added to ``ContextEngine.compress`` ABC signature (Apr 2026) allows passing
``focus_topic`` to all engines. Older plugins written against the prior ABC
(no focus_topic kwarg) must keep working, while current plugins must not lose
their manual focus merely because they omit the built-in-only ``force`` kwarg.
"""

import pytest

from agent.conversation_compression import _call_context_engine_compress


def test_compress_context_supports_engine_without_focus_topic():
    """Older plugins without focus_topic in compress() signature don't crash."""
    captured_kwargs = []

    class _StrictOldPluginEngine:
        """Mimics a plugin written against the pre-focus_topic ABC."""

        compression_count = 0

        def compress(self, messages, current_tokens=None):
            # NOTE: no focus_topic kwarg — TypeError if caller passes one.
            captured_kwargs.append({"current_tokens": current_tokens})
            return [messages[0], messages[-1]]

    engine = _StrictOldPluginEngine()

    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]

    compressed = _call_context_engine_compress(
        engine,
        messages,
        current_tokens=100,
        focus_topic="foo",
        force=True,
    )

    # The older engine is called once with only the keyword it declares.
    assert compressed == [messages[0], messages[-1]]
    assert captured_kwargs == [{"current_tokens": 100}]


def test_compress_context_preserves_focus_for_engine_without_force_keyword():
    captured_kwargs = []

    class _DocumentedPluginEngine:
        def compress(self, messages, current_tokens=None, focus_topic=None):
            captured_kwargs.append(
                {"current_tokens": current_tokens, "focus_topic": focus_topic}
            )
            return messages

    messages = [{"role": "user", "content": "keep the release decision"}]
    compressed = _call_context_engine_compress(
        _DocumentedPluginEngine(),
        messages,
        current_tokens=100,
        focus_topic="release decision",
        force=True,
    )

    assert compressed is messages
    assert captured_kwargs == [
        {"current_tokens": 100, "focus_topic": "release decision"}
    ]


def test_compress_context_does_not_mask_internal_type_error():
    calls = 0

    class _BrokenPluginEngine:
        def compress(self, messages, current_tokens=None, focus_topic=None):
            del messages, current_tokens, focus_topic
            nonlocal calls
            calls += 1
            raise TypeError("engine implementation bug")

    with pytest.raises(TypeError, match="implementation bug"):
        _call_context_engine_compress(
            _BrokenPluginEngine(),
            [],
            current_tokens=100,
            focus_topic="release decision",
            force=False,
        )

    assert calls == 1
