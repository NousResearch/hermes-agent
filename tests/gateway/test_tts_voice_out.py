"""Tests for the ``voice_out_carries_text`` adapter contract.

Platforms whose voice messages inherently render the spoken text
(e.g. Carbon Voice runs server-side STT and shows the transcript
inside the voice-memo bubble) opt into suppressing the follow-up text
bubble that the auto-TTS dispatch would otherwise send.

This module pins the class-attribute contract:

  - ``BasePlatformAdapter.voice_out_carries_text`` exists, defaults to
    False, and subclasses can override without affecting the base.

The dispatch behavior itself (reply threading via ``reply_to``,
opt-in suppression, failed-TTS fallback, and retaining the text when
``prepare_tts_text`` truncates or strips formatting) is covered by
behavioral tests in ``tests/gateway/test_base_topic_sessions.py``
(``TestVoiceOutCarriesTextDelivery``), which drive
``_process_message_background`` end-to-end.
"""

from __future__ import annotations

from gateway.platforms.base import BasePlatformAdapter


class TestVoiceOutCarriesTextDefault:
    """Default is False — every existing adapter is unaffected."""

    def test_default_is_false_on_base_class(self):
        """``BasePlatformAdapter.voice_out_carries_text`` defaults to False."""
        assert BasePlatformAdapter.voice_out_carries_text is False

    def test_subclass_can_override_to_true(self):
        """Subclasses opt in by setting the class attribute to True."""

        class _OptIn(BasePlatformAdapter):
            voice_out_carries_text = True

            # BasePlatformAdapter is abstract; stub the abstracts.
            async def connect(self): ...  # pragma: no cover
            async def disconnect(self): ...  # pragma: no cover
            async def send(self, *a, **kw): ...  # pragma: no cover
            async def get_chat_info(self, *a, **kw): ...  # pragma: no cover

        assert _OptIn.voice_out_carries_text is True
        assert BasePlatformAdapter.voice_out_carries_text is False

    def test_subclass_default_inherits_false(self):
        """A subclass that doesn't override still reads False (inherited)."""

        class _NoOverride(BasePlatformAdapter):
            async def connect(self): ...  # pragma: no cover
            async def disconnect(self): ...  # pragma: no cover
            async def send(self, *a, **kw): ...  # pragma: no cover
            async def get_chat_info(self, *a, **kw): ...  # pragma: no cover

        assert _NoOverride.voice_out_carries_text is False
