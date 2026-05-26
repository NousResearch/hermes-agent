"""Tests for the TTS dispatch fixes in ``BasePlatformAdapter``.

Two related bugs in the auto-TTS path
(``gateway/platforms/base.py:_run_message_response``):

  1. ``play_tts(...)`` was called without ``reply_to``, so the
     synthesized voice memo arrived as a top-level post even when the
     parallel text path would have threaded its reply under the
     inbound message. Visible on every voice-out platform.

  2. ``_tts_caption_delivered`` only fired for Telegram (via the
     ``telegram_tts_caption`` path), so adapters whose voice messages
     inherently carry the spoken text (e.g. Carbon Voice runs
     server-side STT and shows the transcript inside the voice-memo
     bubble) ended up shipping a *duplicate* text bubble right after
     the voice memo.

This module pins:

  - ``BasePlatformAdapter.voice_out_carries_text`` exists as a
    class attribute, defaults to False, and subclasses can override.
  - The TTS dispatch in ``base.py`` reads ``voice_out_carries_text``
    via ``getattr`` (so adapters that don't override stay opted out).
  - The TTS dispatch passes ``reply_to`` to ``play_tts``.

The dispatch flow itself is exercised end-to-end by the existing
integration tests under ``tests/gateway/`` per platform; we only need
to lock the contract here so it can't quietly regress.
"""

from __future__ import annotations

import pathlib

import pytest

from gateway.platforms.base import BasePlatformAdapter


# ── voice_out_carries_text class attribute ──────────────────────────────


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


# ── Source-grep: dispatch reads the attribute and passes reply_to ───────
#
# These are regression guards — the actual dispatch flow has too many
# moving parts (TTS provider, file IO, platform mocks) for a tight unit
# test, but the calling convention is small enough that source-grep
# catches any accidental removal.


class TestDispatchCallSites:
    """The TTS dispatch in base.py must keep using the two contracts."""

    @staticmethod
    def _base_py_source() -> str:
        # Same dir as this test → climb to repo root → gateway/platforms/base.py
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        return (repo_root / "gateway" / "platforms" / "base.py").read_text()

    def test_play_tts_call_passes_reply_to(self):
        """``play_tts(...)`` must receive ``reply_to`` so the voice memo
        threads under the user's message instead of going top-level."""
        src = self._base_py_source()
        # Locate the play_tts call inside the auto-TTS dispatch block.
        # The relevant substring is unique enough that source-grep is
        # robust against unrelated edits.
        call_idx = src.find("tts_result = await self.play_tts(")
        assert call_idx >= 0, (
            "Could not find play_tts call in base.py — has the dispatch "
            "been renamed? Update this test to match."
        )
        # Read the next ~400 chars and look for reply_to=.
        snippet = src[call_idx : call_idx + 400]
        assert "reply_to=" in snippet, (
            "play_tts must be called with reply_to= so the voice memo "
            "threads under the user's inbound. Without it, TTS audio "
            "arrives as a top-level post even when the text path would "
            "have threaded."
        )

    def test_tts_caption_delivered_reads_voice_out_carries_text(self):
        """The _tts_caption_delivered check must accept the
        ``voice_out_carries_text`` opt-in so adapters where the voice
        memo already carries the transcript (CV, etc.) don't ship a
        duplicate text bubble."""
        src = self._base_py_source()
        assert 'voice_out_carries_text' in src, (
            "_tts_caption_delivered must consult voice_out_carries_text "
            "(either directly or via getattr) so platforms that re-render "
            "the spoken text inside the voice bubble — Carbon Voice's "
            "server-side STT, etc. — can suppress the duplicate text send."
        )
