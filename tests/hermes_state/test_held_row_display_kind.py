"""W1 cycle 1e: held-row display-kind exclusion from model-restore reads.

A desktop-intake-screen hold stops the turn AFTER its submit-time user row
was persisted. Unmarked, that row rides into every later turn's model
context through the transcript (caught live 19 Nov: two held rows were
ingested by the following turn). ``HELD_ROW_DISPLAY_KIND`` marks such rows:
``get_messages_as_conversation`` (every model-restore variant) skips them,
display reads keep them so the UI still shows the send with its failed
bubble.
"""

import unittest

from hermes_state import SessionDB
from hermes_state_messages import HELD_ROW_DISPLAY_KIND


def _store(tmpdir):
    db = SessionDB(tmpdir / "state.db")
    db.create_session("s", "cli")
    return db


class TestHeldRowExclusion(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.tmp = Path(tempfile.mkdtemp())
        self.db = _store(self.tmp)
        self.addCleanup(self._close)

    def _close(self):
        try:
            self.db.close()
        except Exception:
            pass

    def _seed(self):
        first = self.db.append_message("s", role="user", content="before")
        held = self.db.append_message("s", role="user", content="held attack")
        after = self.db.append_message("s", role="user", content="after hold")
        reply = self.db.append_message("s", role="assistant", content="ok")
        return first, held, after, reply

    def _row_id(self, appended):
        # append_message returns the durable dict; fall back to the projection.
        if isinstance(appended, dict) and isinstance(appended.get("id"), int):
            return appended["id"]
        rows = self.db.get_messages_as_conversation("s", include_row_ids=True)
        for row in rows:
            if row.get("content") == "held attack":
                row_id = row.get("_row_id")
                assert isinstance(row_id, int)
                return row_id
        raise AssertionError("row not found")

    def test_model_restore_skips_held_rows_display_keeps_them(self):
        _, held, _, _ = self._seed()
        self.assertEqual(
            self.db.set_message_display_kind(
                "s", self._row_id(held), HELD_ROW_DISPLAY_KIND), 1)
        model_rows = self.db.get_messages_as_conversation("s")
        contents = [r.get("content") for r in model_rows]
        self.assertIn("before", contents)
        self.assertIn("after hold", contents)
        self.assertIn("ok", contents)
        self.assertNotIn("held attack", contents)
        # Display reads (include_compacted) keep the held row for the UI.
        display_rows = self.db.get_messages_as_conversation(
            "s", include_compacted=True)
        display_contents = [r.get("content") for r in display_rows]
        self.assertIn("held attack", display_contents)
        self.assertIn("after hold", display_contents)

    def test_repair_alternation_restore_also_skips_held_rows(self):
        _, held, _, _ = self._seed()
        self.db.set_message_display_kind(
            "s", self._row_id(held), HELD_ROW_DISPLAY_KIND)
        rows = self.db.get_messages_as_conversation("s", repair_alternation=True)
        contents = [r.get("content") for r in rows]
        self.assertNotIn("held attack", contents)
        self.assertFalse(any("held attack" in str(c) for c in contents))
        # The two remaining user rows glue under repair_alternation when the
        # seed puts them adjacent (pre-existing behaviour); in real transcripts
        # every user row follows an assistant reply, so this is seed-only.
        self.assertTrue(any("after hold" in str(c) for c in contents))

    def test_other_display_kinds_unaffected(self):
        _, held, _, _ = self._seed()
        self.db.set_message_display_kind("s", self._row_id(held), "steer")
        rows = self.db.get_messages_as_conversation("s")
        contents = [r.get("content") for r in rows]
        self.assertIn("held attack", contents)  # only the HELD kind is skipped

    def test_get_resume_conversations_model_excludes_display_keeps(self):
        # Round-1 F1: the cold-resume loader must not hydrate held text into
        # model_history (it feeds conversation_history on restart).
        _, held, _, _ = self._seed()
        self.db.set_message_display_kind(
            "s", self._row_id(held), HELD_ROW_DISPLAY_KIND)
        model_history, display_history = self.db.get_resume_conversations("s")
        model_contents = [str(m.get("content")) for m in model_history]
        display_contents = [str(m.get("content")) for m in display_history]
        self.assertFalse(any("held attack" in c for c in model_contents))
        self.assertTrue(any("held attack" in c for c in display_contents))

    def test_include_held_optin_restores_row(self):
        # Round-1 F3: display-side surfaces (session.history RPC, rewind
        # picker, trace export) opt in and still see the held send.
        _, held, _, _ = self._seed()
        self.db.set_message_display_kind(
            "s", self._row_id(held), HELD_ROW_DISPLAY_KIND)
        rows = self.db.get_messages_as_conversation("s", include_held=True)
        contents = [r.get("content") for r in rows]
        self.assertIn("held attack", contents)
        rows = self.db.get_messages_as_conversation("s")
        self.assertNotIn("held attack",
                         [r.get("content") for r in rows])

    def test_setter_guards(self):
        _, held, _, _ = self._seed()
        row_id = self._row_id(held)
        self.assertEqual(self.db.set_message_display_kind("", row_id, "x"), 0)
        self.assertEqual(self.db.set_message_display_kind("s", 0, "x"), 0)
        self.assertEqual(
            self.db.set_message_display_kind("s", True, "x"), 0)
        self.assertEqual(
            self.db.set_message_display_kind("s", row_id, "   "), 0)
        self.assertEqual(
            self.db.set_message_display_kind("other", row_id, "x"), 0)
        # idempotent restamp
        self.assertEqual(
            self.db.set_message_display_kind(
                "s", row_id, HELD_ROW_DISPLAY_KIND), 1)


if __name__ == "__main__":
    unittest.main()
