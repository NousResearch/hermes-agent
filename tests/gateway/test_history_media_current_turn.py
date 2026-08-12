"""Regression: current-turn TTS media must not be dedup-stripped (#B, staging 2026-07-29).

_history_media_paths_for_session feeds the delivery-time dedup that stops the
model re-sending files it already delivered in PRIOR turns. The agent persists
transcript rows as it produces them, so by delivery time the CURRENT turn's
tool results (e.g. text_to_speech's MEDIA-tagged JSON) are already in the
persisted transcript. The old implementation excluded only the trailing
assistant entry — leaving this turn's tool rows in "history", so a fresh TTS
path deduped against ITSELF and the attachment was silently stripped
(`response_delivery_dropped`: user saw TTS succeed but no audio arrive).

Contract: everything from the LAST USER MESSAGE onward is the current turn and
must be excluded from the dedup set; genuinely-prior turns must stay in it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gateway.config import PlatformConfig, Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _StubStore:
    def __init__(self, transcript: List[Dict[str, Any]]) -> None:
        self._transcript = transcript

    def load_transcript(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._transcript)


class _StubAdapter(BasePlatformAdapter):
    """Minimal concrete adapter (BasePlatformAdapter is abstract)."""

    def __init__(self, transcript: List[Dict[str, Any]]) -> None:
        super().__init__(PlatformConfig(), Platform.API_SERVER)
        self._session_store = _StubStore(transcript)

    async def start(self) -> None:  # pragma: no cover - unused
        pass

    async def stop(self) -> None:  # pragma: no cover - unused
        pass

    async def connect(self, *, is_reconnect: bool = False) -> bool:  # pragma: no cover - unused
        return True

    async def disconnect(self) -> None:  # pragma: no cover - unused
        pass

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:  # pragma: no cover - unused
        return SendResult(success=True)


def _tts_tool_row(path: str) -> Dict[str, Any]:
    return {
        "role": "tool",
        "content": (
            '{"success": true, "file_path": "%s", "media_tag": "MEDIA:%s"}'
            % (path, path)
        ),
    }


def test_current_turn_tts_media_not_treated_as_history():
    """This turn's TTS output (persisted before delivery) must NOT be deduped."""
    current = "/opt/data/cache/audio/tts_now.mp3"
    transcript = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        # ── current turn ──
        {"role": "user", "content": "say it out loud"},
        {"role": "assistant", "content": "", "tool_calls": [{}]},
        _tts_tool_row(current),
        {"role": "assistant", "content": f"MEDIA:{current}"},
    ]
    adapter = _StubAdapter(transcript)
    paths: Optional[set] = adapter._history_media_paths_for_session("k")
    assert not paths or current not in paths


def test_prior_turn_media_still_deduped():
    """A file delivered in a PRIOR turn stays in the dedup set."""
    old = "/opt/data/cache/audio/tts_old.mp3"
    transcript = [
        {"role": "user", "content": "speak"},
        {"role": "assistant", "content": "", "tool_calls": [{}]},
        _tts_tool_row(old),
        {"role": "assistant", "content": f"MEDIA:{old}"},
        # ── current turn ──
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "you're welcome"},
    ]
    adapter = _StubAdapter(transcript)
    paths = adapter._history_media_paths_for_session("k")
    assert paths and old in paths


def test_no_user_row_falls_back_to_trailing_assistant_exclusion():
    """Unusual store shape (no user rows): keep the old safe behavior."""
    old = "/opt/data/cache/audio/tts_only.mp3"
    transcript = [
        _tts_tool_row(old),
        {"role": "assistant", "content": f"MEDIA:{old}"},
    ]
    adapter = _StubAdapter(transcript)
    # Only guarantee: it does not crash and returns a set-or-None; the tool
    # row (not excludable without a user anchor) may keep the path present.
    result = adapter._history_media_paths_for_session("k")
    assert result is None or isinstance(result, set)
