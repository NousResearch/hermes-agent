"""Streaming-TTS adapter contract for ``BasePlatformAdapter``.

Extracted from ``gateway/platforms/base.py`` (god-file decomposition, wave 1,
shard s4, cluster c1) following the mechanical mixin lift that produced
``plugins/platforms/telegram/authz_mixin.py`` (#75742).  This mixin holds the
streaming-TTS cluster: the adapter-contract entry points
(``supports_streaming_tts`` .. ``abort_streaming_tts``) plus the per-turn
whole-file suppression helpers.

Behavior-neutral: every method is lifted verbatim from ``BasePlatformAdapter``.
``self.*`` calls resolve unchanged via the MRO, and ``StreamingTTSMixin``
precedes ``BasePlatformAdapter`` in the bases so resolution order is what it
was when these methods sat on the class.

Two details keep the lift observationally identical:

* ``logger`` is bound by explicit name rather than ``__name__``, so records
  emitted from these methods keep the logger name
  ``"gateway.platforms.base"``.
* The module-level helpers ``streaming_tts_turn_key`` and
  ``streaming_tts_should_skip_whole_file`` remain defined in ``base.py``
  (tests import them from there); this module imports them back in.
"""

import logging
from typing import Any, Dict, Optional

from gateway.platforms.base import (
    AudioFormat,
    StreamingTTSHandle,
    streaming_tts_should_skip_whole_file,
    streaming_tts_turn_key,
)

# Bind the adapter's logger by name so log records lifted with these methods
# are emitted under exactly the name they were before.
logger = logging.getLogger("gateway.platforms.base")


class StreamingTTSMixin:
    """Streaming-TTS cluster lifted verbatim from ``BasePlatformAdapter``."""

    # ------------------------------------------------------------------
    # Streaming TTS adapter contract (#60671)
    # ------------------------------------------------------------------
    # Voice-capable adapters (LiveKit, Discord voice, …) override these to
    # accept PCM audio chunks while the LLM is still generating.  The default
    # implementations report "unsupported" so existing adapters are
    # source-compatible and keep the whole-file auto-TTS fallback.

    def supports_streaming_tts(self, chat_id: str, audio_format: AudioFormat) -> bool:
        """Return True when this adapter can accept streaming PCM for *chat_id*.

        Default: False (whole-file auto-TTS path remains).  Override to opt in.
        """
        return False

    async def begin_streaming_tts(
        self,
        chat_id: str,
        audio_format: AudioFormat,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[StreamingTTSHandle]:
        """Open a streaming-audio session for *chat_id*.

        Returns an opaque handle passed to subsequent ``write_streaming_tts``
        / ``finish_streaming_tts`` / ``abort_streaming_tts`` calls, or
        ``None`` to decline (caller falls back to whole-file TTS).
        """
        return None

    async def write_streaming_tts(self, handle: StreamingTTSHandle, chunk: bytes) -> None:
        """Write one PCM chunk to the adapter's outbound audio track."""
        pass

    async def finish_streaming_tts(self, handle: StreamingTTSHandle, *, interrupted: bool = False) -> None:
        """Signal normal end of the audio stream."""
        pass

    async def abort_streaming_tts(self, handle: StreamingTTSHandle, error: Optional[str] = None) -> None:
        """Abort the stream due to an error or cancellation.

        Must be idempotent: late producer chunks after abort must be silently
        dropped, not raise.  Restores adapter state to "not streaming".
        """
        pass

    def _streaming_tts_turn_key(
        self,
        session_key: str | None,
        turn_marker: Any = None,
        *,
        event: Any = None,
    ) -> str | None:
        return streaming_tts_turn_key(session_key, turn_marker, event=event)

    def _mark_streaming_tts_completed_turn(
        self,
        session_key: str | None,
        turn_marker: Any = None,
        *,
        event: Any = None,
    ) -> None:
        turn_key = self._streaming_tts_turn_key(session_key, turn_marker, event=event)
        if turn_key is not None:
            completed = getattr(self, "_streaming_tts_completed_turns", None)
            if completed is None:
                completed = set()
                self._streaming_tts_completed_turns = completed
            completed.add(turn_key)

    def _streaming_tts_turn_completed(
        self,
        session_key: str | None,
        turn_marker: Any = None,
        *,
        event: Any = None,
    ) -> bool:
        return streaming_tts_should_skip_whole_file(
            getattr(self, "_streaming_tts_completed_turns", set()),
            session_key,
            turn_marker,
            event=event,
        )
