"""`hermes --resume <id>` must accept the IDs the CLI itself prints.

`hermes sessions list` renders session IDs truncated to fit its fixed
column (18 chars of a 22+ char ID). The resume resolver used an exact
`get_session` lookup only, so the ID shown by our own listing could not
be pasted back. `SessionDB.resolve_session_id` (exact or unambiguous
prefix) already existed and the gateway, console engine, and
`hermes sessions export/delete` all use it. This file pins the plain
CLI resume path to the same resolver.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_state import SessionDB


def _new_id(prefix: str) -> str:
    """A realistic 22-char id: YYYYmmdd_HHMMSS_hex."""
    return f"20260829_120000_{prefix}"


class TestResolveSessionByNameOrId(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="resolve_sid_t_")
        os.environ["HERMES_HOME"] = self.home
        import importlib

        import hermes_cli.main as m

        importlib.reload(m)
        self.m = m

    def tearDown(self):
        os.environ.pop("HERMES_HOME", None)

    def _db(self):
        return SessionDB()

    def test_exact_id_resolves(self):
        db = self._db()
        db.create_session(_new_id("aabbcc"), source="cli")
        self.assertEqual(
            self.m._resolve_session_by_name_or_id(_new_id("aabbcc")),
            _new_id("aabbcc"),
        )

    def test_truncated_id_from_listing_resolves(self):
        """The 18-char prefix `hermes sessions list` prints must resolve."""
        sid = _new_id("aabbcc")
        db = self._db()
        db.create_session(sid, source="cli")
        truncated = sid[:18]
        self.assertEqual(len(truncated), 18)
        self.assertEqual(self.m._resolve_session_by_name_or_id(truncated), sid)

    def test_ambiguous_prefix_returns_none(self):
        db = self._db()
        db.create_session(_new_id("aa1000"), source="cli")
        db.create_session(_new_id("aa2000"), source="cli")
        # First 17 chars are shared, so the prefix is ambiguous.
        prefix = _new_id("aa1000")[:17]
        self.assertIsNone(self.m._resolve_session_by_name_or_id(prefix))

    def test_title_fallback_still_works(self):
        sid = _new_id("aabbcc")
        db = self._db()
        db.create_session(sid, source="cli")
        db.set_session_title(sid, "my project")
        self.assertEqual(self.m._resolve_session_by_name_or_id("my project"), sid)

    def test_unknown_input_returns_none(self):
        self.assertIsNone(self.m._resolve_session_by_name_or_id("no such session"))


if __name__ == "__main__":
    unittest.main()
