"""Acceptance: mandatory per-entry provenance and eviction-preserving archive.

Every entry carries source_session, author, created_at, origin_repo, and
trace_id — a missing or empty field rejects the entry before it can commit.
Eviction archives first: the removed entry is always recoverable with a reason.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load  # noqa: E402

provider_mod = load("provider")
merge_mod = load("merge")
provenance_mod = load("provenance")


def make_provider(home, **config):
    full = dict(provider_mod._DEFAULTS)
    full.update(config)
    provider = provider_mod.CognitionMemoryProvider(config=full)
    provider.initialize("sess-1", hermes_home=home, platform="cli",
                        agent_context="primary")
    return provider


def good_entry(ref="cog-s-1"):
    return {
        "ref": ref,
        "content": "a durable fact",
        "salience": 0.5,
        "provenance": provenance_mod.make_provenance(
            source_session="sess-1", author="Ada", origin_repo="default",
            trace_id="t1", created_at=1_700_000_000.0),
    }


class ProvenanceArchiveTest(unittest.TestCase):
    def test_each_mandatory_field_is_required(self):
        for field in ("source_session", "author", "created_at",
                      "origin_repo", "trace_id"):
            entry = good_entry()
            del entry["provenance"][field]
            with self.assertRaises(provenance_mod.ProvenanceError, msg=field):
                provenance_mod.validate_provenance(entry)
            # Empty strings count as missing too.
            entry = good_entry()
            entry["provenance"][field] = ""
            with self.assertRaises(provenance_mod.ProvenanceError, msg=field):
                provenance_mod.validate_provenance(entry)

    def test_entry_without_provenance_record_is_rejected(self):
        with self.assertRaises(provenance_mod.ProvenanceError):
            provenance_mod.validate_provenance({"ref": "cog-s-1"})

    def test_unprovable_entry_cannot_commit(self):
        log = merge_mod.CommitLog(
            os.path.join(tempfile.mkdtemp(prefix="cog-log-"), "c.json"))
        with self.assertRaises(provenance_mod.ProvenanceError):
            merge_mod.fetch_first_push(log, {"ref": "cog-s-9"}, None, now=1.0)
        self.assertEqual(log.refs(), [])

    def test_eviction_archives_first_then_removes(self):
        home = tempfile.mkdtemp(prefix="cog-home-")
        provider = make_provider(home)
        entry = good_entry()
        status, _ = merge_mod.fetch_first_push(provider._log, entry, None, now=1.0)
        self.assertEqual(status, "applied")
        self.assertTrue(provider.archive_and_evict("cog-s-1", "test-evict"))
        # Gone from committed state...
        self.assertNotIn("cog-s-1", provider._log.refs())
        # ...but preserved in the archive with its reason and provenance.
        archive_path = os.path.join(home, "cognition", "archive.jsonl")
        records = provenance_mod.load_archive(archive_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["ref"], "cog-s-1")
        self.assertEqual(records[0]["reason"], "test-evict")
        self.assertEqual(records[0]["entry"]["provenance"]["author"], "Ada")
        provider.shutdown()

    def test_evicting_unknown_ref_reports_false(self):
        home = tempfile.mkdtemp(prefix="cog-home-")
        provider = make_provider(home)
        self.assertFalse(provider.archive_and_evict("cog-nope", "test"))
        provider.shutdown()


if __name__ == "__main__":
    unittest.main()
