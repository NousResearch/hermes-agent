"""Behaviour contract for session tagging in auto-TTS output paths.

One gateway serves many concurrent sessions, and every auto-TTS artifact lands
in one shared ``hermes_voice`` directory. Without a session tag in the filename
an artifact cannot be attributed after the fact — log correlation, per-session
cleanup, and any external tooling watching the directory have to guess, and a
watcher guessing by mtime can pick up another session's audio.

The value becomes a path segment, so these tests also pin that a hostile or
malformed session key cannot escape the directory.
"""

import os

from gateway.platforms.base import build_auto_tts_output_path


class _Platform:
    """Minimal stand-in: only the platform NAME is read for the extension."""

    def __init__(self, value: str) -> None:
        self.value = value


def test_session_tag_appears_and_paths_stay_unique():
    """The tag is in the filename, and two calls never collide."""
    key = "agent:main:discord:thread:123:123"
    first = os.path.basename(build_auto_tts_output_path(_Platform("discord"), key))
    second = os.path.basename(build_auto_tts_output_path(_Platform("discord"), key))

    assert "discord" in first and "123" in first
    assert first != second, "uuid must still guarantee uniqueness within a session"


def test_untagged_path_still_works():
    """Omitting the session key keeps the original anonymous form."""
    name = os.path.basename(build_auto_tts_output_path(_Platform("discord")))
    assert name.startswith("tts_reply_") and name.endswith(".mp3")


def test_hostile_session_keys_cannot_escape_the_directory():
    """A session key is data, not a path fragment.

    ``..`` traversal, separators and an over-long key must all resolve to a
    plain filename inside the intended directory.
    """
    for key in ("../../etc/passwd", "a/b/c", "..", "." * 300, "x" * 500, "", "   "):
        path = build_auto_tts_output_path(_Platform("discord"), key)
        directory, name = os.path.split(path)
        assert os.path.basename(directory) == "hermes_voice"
        assert ".." not in name and os.sep not in name
        assert name.startswith("tts_reply_")
        assert len(name) < 200
