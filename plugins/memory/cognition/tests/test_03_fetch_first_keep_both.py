"""Acceptance: fetch-first push with keep-both conflicts.

Push fetches committed state before writing; an unchanged base applies, a
changed base keeps both versions. Refs are deterministic:
``cog-{sessionId}-{seq}`` for pushes, ``{ref}-conflict-{unix_ts}`` for the
conflicting side.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load  # noqa: E402

merge_mod = load("merge")
provenance_mod = load("provenance")
queue_mod = load("queue")

CommitLog = merge_mod.CommitLog


def prov(repo="default", session="sess-1", author="Ada", ts=1_700_000_000.0):
    return provenance_mod.make_provenance(
        source_session=session, author=author, origin_repo=repo,
        trace_id="trace-1", created_at=ts)


class FetchFirstKeepBothTest(unittest.TestCase):
    def test_refs_are_deterministic(self):
        directory = tempfile.mkdtemp(prefix="cog-q-")
        queue = queue_mod.DurablePushQueue(os.path.join(directory, "q.db"))
        self.assertEqual(queue.enqueue({"a": 1}, session_id="abc", now=0.0),
                         "cog-abc-1")
        self.assertEqual(queue.enqueue({"a": 2}, session_id="abc", now=0.0),
                         "cog-abc-2")
        self.assertEqual(merge_mod.conflict_ref("cog-abc-1", now=1_700_000_000.0),
                         "cog-abc-1-conflict-1700000000")
        queue.close()

    def test_fetch_first_applies_on_unchanged_base(self):
        log = CommitLog(os.path.join(tempfile.mkdtemp(prefix="cog-log-"), "c.json"))
        entry = {"ref": "cog-s-1", "content": "v1", "salience": 0.5,
                 "provenance": prov()}
        status, landed = merge_mod.fetch_first_push(log, entry, None, now=100.0)
        self.assertEqual((status, landed), ("applied", "cog-s-1"))
        # A writer that saw v1 may overwrite with v2 (base matches committed).
        committed = log.get("cog-s-1")
        entry2 = dict(entry, content="v2")
        status, landed = merge_mod.fetch_first_push(log, entry2, committed, now=101.0)
        self.assertEqual((status, landed), ("applied", "cog-s-1"))
        self.assertEqual(log.get("cog-s-1")["content"], "v2")

    def test_stale_base_keeps_both_with_conflict_ref(self):
        log = CommitLog(os.path.join(tempfile.mkdtemp(prefix="cog-log-"), "c.json"))
        original = {"ref": "cog-s-1", "content": "original", "salience": 0.5,
                    "provenance": prov()}
        merge_mod.fetch_first_push(log, original, None, now=100.0)
        stale_base = dict(original)  # writer A read before writer B committed
        newer = dict(original, content="writer-B")
        merge_mod.fetch_first_push(log, newer, stale_base, now=101.0)
        incoming = dict(original, content="writer-A-late")
        status, landed = merge_mod.fetch_first_push(log, incoming, stale_base, now=102.0)
        self.assertEqual(status, "conflict")
        self.assertEqual(landed, "cog-s-1-conflict-102")
        # Keep-both: committed winner untouched, loser preserved under conflict ref.
        self.assertEqual(log.get("cog-s-1")["content"], "writer-B")
        conflict = log.get("cog-s-1-conflict-102")
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["content"], "writer-A-late")
        self.assertEqual(conflict["conflict_with"], "cog-s-1")
        self.assertIn("cog-s-1", log.refs())
        self.assertIn("cog-s-1-conflict-102", log.refs())


if __name__ == "__main__":
    unittest.main()
