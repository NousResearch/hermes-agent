# Tests for agent.title_placeholders (spec §4, ticket t_a642ab85) and its
# re-arm consumer in agent.title_generator (ticket t_f0122159).

from agent.title_generator import _has_upgraded_title
from agent.title_placeholders import is_rejected_placeholder
from hermes_state import SessionDB

# --- Real strings captured from the live DB, 2026-09-30 (map card t_15b3bcbd) ---

REAL_REJECTED = [
    # STT-failure note with drifting cache path (gateway/run_inbound.py:2047)
    "[voice message could not be transcribed automatically; audio is available at: /path/audio_xxx.ogg]",
    # Raw STT JSON leak — provider result swallowed whole
    '{"text":" Go to all my recent projects, check what changed in the deploy config, and summarize the open threads"}',
]

REAL_KEPT = [
    # Real opener typed by the owner (spec §4.3) — must NOT match
    "Transcribe voice message audio_308eb1310457.ogg",
]


# --- Every other note family that gateway/run_inbound.py can emit ---

ALL_BRACKET_NOTES = [
    # :2047 — transcription failed
    "[voice message could not be transcribed automatically; the audio is available at: /home/andre/.cache/hermes/audio/audio_9f2c.ogg]",
    # :2066 — transcript empty/inaudible
    "[The user sent a voice message but it came through empty or inaudible — speech-to-text returned no words. Do not guess at the content; ask the user to resend or type it out.]",
    # :2089 — STT disabled
    "[The user sent a voice message: /abs/path/audio_1.ogg (duration: 0:42)]",
    "[The user sent a voice message: /abs/path/audio_1.ogg]",
    # :2098 — transcription module unavailable
    "[voice message could not be transcribed]",
    # Ticket-required family: [audio message …]
    "[audio message could not be transcribed]",
]


class TestRealCapturedStrings:
    def test_real_rejected_notes_rejected(self):
        for s in REAL_REJECTED:
            assert is_rejected_placeholder(s), s

    def test_real_transcribe_opener_kept(self):
        for s in REAL_KEPT:
            assert not is_rejected_placeholder(s), s

    def test_all_bracket_note_families_rejected(self):
        for s in ALL_BRACKET_NOTES:
            assert is_rejected_placeholder(s), s


class TestExclusions:
    def test_empty_and_none_are_not_placeholders(self):
        assert not is_rejected_placeholder(None)
        assert not is_rejected_placeholder("")

    def test_whitespace_only_not_placeholder(self):
        assert not is_rejected_placeholder("   ")
        assert not is_rejected_placeholder("\n\t ")

    def test_normal_subjects_kept(self):
        for s in (
            "Deploy cron pipeline",
            "Fix the FTS5 search regression",
            "[no marker here]",  # brackets alone are not enough
            "Voice message transcription ideas",  # words, no brackets
            "Question about voice message routing",
        ):
            assert not is_rejected_placeholder(s), s

    def test_transcribe_variants_kept(self):
        # Real opener shape: no brackets, no JSON — survives by form (§4.3)
        for s in (
            "Transcribe voice message audio_308eb1310457.ogg",
            "transcribe voice message audio_abc123.ogg please",
            "Transcribe voice message ",
        ):
            assert not is_rejected_placeholder(s), s


class TestEdgeCases:
    def test_surrounding_whitespace_and_newlines_still_rejected(self):
        assert is_rejected_placeholder("  [voice message could not be transcribed]  \n")
        assert is_rejected_placeholder("\t[audio message: /tmp/a.ogg]\n")

    def test_whitespace_inside_brackets_still_rejected(self):
        assert is_rejected_placeholder("[  voice message  ]")
        assert is_rejected_placeholder("[ The user sent a voice message ]")

    def test_partial_json_not_rejected(self):
        # Not a {"text" leak: space after the brace, different key, or prose.
        assert not is_rejected_placeholder('{ "text": "x" }')
        assert not is_rejected_placeholder('{"other":"y"}')
        assert not is_rejected_placeholder('a {"text":"x"} prefix')

    def test_json_leak_prefix_boundaries(self):
        assert is_rejected_placeholder('{"text":""}')
        assert is_rejected_placeholder('  {"text":" Go to all my recent projects…')
        assert not is_rejected_placeholder('{ "text": "x" }')  # space after brace
        assert not is_rejected_placeholder('{"other":"y"}')

    def test_unclosed_bracket_not_rejected(self):
        # Whole-title bracket shape is required; prose wrapping a note survives.
        assert not is_rejected_placeholder("[voice message could not be transcribed")
        assert not is_rejected_placeholder("See [voice message] notes below")

    def test_mixed_case_markers_rejected(self):
        assert is_rejected_placeholder("[VOICE MESSAGE could not be transcribed]")
        assert is_rejected_placeholder("[The User Sent A Voice Message: /a.ogg]")

    def test_rejected_still_yields_tag_only_compose_legal(self):
        # §4.4: rejection suppresses the subject, never the rename itself —
        # the predicate only classifies; empty-subject composition is legal.
        assert is_rejected_placeholder("[voice message could not be transcribed]")


class TestHasUpgradedTitleRearms:
    """The terminal-state guard re-arms over a rejected placeholder: an STT note is not
    a name the lane should be stuck behind, so a late real title can still replace it."""

    def _titled(self, tmp_path, title, *, source):
        db = SessionDB(tmp_path / "state.db")
        try:
            db.create_session("s1", source="telegram")
            db.set_auto_title("s1", title, source=source) if source != "user" \
                else db.set_session_title("s1", title)
            return db
        except Exception:
            db.close()
            raise

    def test_rejected_placeholder_rearms_even_at_llm_authority(self, tmp_path):
        db = self._titled(tmp_path, "[voice message could not be transcribed]", source="llm")
        try:
            assert not _has_upgraded_title(db, "s1")
        finally:
            db.close()

    def test_normal_llm_title_is_terminal(self, tmp_path):
        db = self._titled(tmp_path, "Deploy cron pipeline", source="llm")
        try:
            assert _has_upgraded_title(db, "s1")
        finally:
            db.close()

    def test_manual_user_title_is_terminal(self, tmp_path):
        db = self._titled(tmp_path, "Deploy cron pipeline", source="user")
        try:
            assert _has_upgraded_title(db, "s1")
        finally:
            db.close()

    def test_unreadable_store_fails_open(self):
        class Broken:
            def get_session_title_source(self, _sid):
                raise RuntimeError("store unreadable")

        assert _has_upgraded_title(Broken(), "s1")
