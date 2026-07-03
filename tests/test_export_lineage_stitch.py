"""E2E: export reassembles compression-split conversations.

Drives the REAL SessionDB against a temp HERMES_HOME. Builds an actual
compression chain (parent end_reason='compression' -> continuation child)
and asserts export_session stitches it and export_all folds continuations
into their root instead of emitting fragments.
"""
import importlib, os, tempfile, unittest, json
from pathlib import Path


class ExportLineageStitchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["HERMES_HOME"] = self.tmp.name
        import hermes_state
        importlib.reload(hermes_state)
        self.hs = hermes_state
        self.db = hermes_state.SessionDB(db_path=Path(self.tmp.name) / "state.db")

    def tearDown(self):
        try:
            self.db.close()
        except Exception:
            pass
        self.tmp.cleanup()
        os.environ.pop("HERMES_HOME", None)

    def _chain(self):
        """root -[compression]-> mid -[compression]-> tip ; 2 msgs each."""
        self.db.create_session("root", source="cli"); self.db.set_session_title("root", "Long Convo")
        self.db.append_message("root", "user", "u1")
        self.db.append_message("root", "assistant", "a1")
        self.db.end_session("root", "compression")

        self.db.create_session("mid", source="cli", parent_session_id="root")
        self.db.append_message("mid", "user", "u2")
        self.db.append_message("mid", "assistant", "a2")
        self.db.end_session("mid", "compression")

        self.db.create_session("tip", source="cli", parent_session_id="mid")
        self.db.append_message("tip", "user", "u3")
        self.db.append_message("tip", "assistant", "a3")

    def test_export_session_stitches_full_chain_from_any_member(self):
        self._chain()
        for entry in ("root", "mid", "tip"):
            exported = self.db.export_session(entry)
            self.assertIsNotNone(exported, entry)
            self.assertEqual(exported["id"], "root", f"{entry}: root metadata expected")
            contents = [m["content"] for m in exported["messages"]]
            self.assertEqual(contents, ["u1", "a1", "u2", "a2", "u3", "a3"],
                             f"{entry}: chain not stitched in order")
            self.assertEqual(exported["_lineage_session_ids"], ["root", "mid", "tip"])

    def test_legacy_flag_returns_single_segment(self):
        self._chain()
        exported = self.db.export_session("mid", stitch_lineage=False)
        self.assertEqual([m["content"] for m in exported["messages"]], ["u2", "a2"])
        self.assertNotIn("_lineage_session_ids", exported)

    def test_export_all_folds_continuations_into_root(self):
        self._chain()
        # standalone unrelated session
        self.db.create_session("solo", source="cli"); self.db.set_session_title("solo", "Solo")
        self.db.append_message("solo", "user", "s1")
        rows = self.db.export_all()
        ids = sorted(r["id"] for r in rows)
        self.assertEqual(ids, ["root", "solo"],
                         "continuations must not appear as their own export rows")
        root_row = next(r for r in rows if r["id"] == "root")
        self.assertEqual([m["content"] for m in root_row["messages"]],
                         ["u1", "a1", "u2", "a2", "u3", "a3"])

    def test_branch_and_delegate_children_are_not_folded(self):
        """A branch/delegate child is a distinct conversation, not a continuation."""
        self.db.create_session("p", source="cli"); self.db.set_session_title("p", "Parent")
        self.db.append_message("p", "user", "p1")
        self.db.end_session("p", "compression")
        # real continuation
        self.db.create_session("cont", source="cli", parent_session_id="p")
        self.db.append_message("cont", "user", "c1")
        # branch child of p (NOT a continuation) — carries _branched_from marker
        self.db.create_session("br", source="cli", parent_session_id="p",
                               model_config={"_branched_from": "p"})
        self.db.append_message("br", "user", "b1")

        rows = self.db.export_all()
        ids = sorted(r["id"] for r in rows)
        # p folds cont; br stands alone
        self.assertIn("br", ids)
        self.assertIn("p", ids)
        self.assertNotIn("cont", ids)
        p_row = next(r for r in rows if r["id"] == "p")
        self.assertEqual([m["content"] for m in p_row["messages"]], ["p1", "c1"])


if __name__ == "__main__":
    unittest.main()
