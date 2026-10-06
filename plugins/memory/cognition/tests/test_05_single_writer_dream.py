"""Acceptance: single-writer lock per entry id; Dream reads committed only.

Concurrent pushes to the same ref serialize — exactly one wins the fast path
and every loser is preserved as a keep-both conflict, never silently dropped.
Dream (read-only synthesis) observes committed state only: it can never see
the unpushed queue, in-flight merges, or mutate anything.
"""

import json
import os
import sys
import tempfile
import threading
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


def entry_for(ref, content, i=0):
    return {
        "ref": ref,
        "content": content,
        "salience": 0.5,
        "provenance": provenance_mod.make_provenance(
            source_session="sess-%d" % i, author="Ada", origin_repo="default",
            trace_id="t%d" % i, created_at=1_700_000_000.0 + i),
    }


class SingleWriterDreamTest(unittest.TestCase):
    def test_concurrent_pushes_to_one_ref_serialize_with_keep_both(self):
        log = merge_mod.CommitLog(
            os.path.join(tempfile.mkdtemp(prefix="cog-log-"), "c.json"))
        base = entry_for("cog-s-1", "seed", i=0)
        merge_mod.fetch_first_push(log, base, None, now=1.0)
        seen_base = log.get("cog-s-1")  # every racer saw this base
        outcomes, barrier = [], threading.Barrier(8)

        def race(i):
            barrier.wait()  # release all racers at once
            status, landed = merge_mod.fetch_first_push(
                log, entry_for("cog-s-1", "racer-%d" % i, i=i),
                dict(seen_base), now=2.0 + i)
            outcomes.append((status, landed))

        threads = [threading.Thread(target=race, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        applied = [o for o in outcomes if o[0] == "applied"]
        conflicts = [o for o in outcomes if o[0] == "conflict"]
        # Exactly one racer wins the fast path; the other seven keep both.
        self.assertEqual(len(applied), 1)
        self.assertEqual(len(conflicts), 7)
        # Eight committed entries total: the seed ref (overwritten once by the
        # fast-path winner) plus seven keep-both conflicts — all content
        # preserved, refs unique.
        self.assertEqual(len(log.refs()), 8)
        self.assertEqual(len(set(log.refs())), 8)

    def test_dream_reads_committed_state_only(self):
        home = tempfile.mkdtemp(prefix="cog-home-")
        provider = make_provider(home)
        provider.on_session_end([{"role": "user",
                                  "content": "I prefer concise answers, remember that"}])
        queued = provider._queue.pending_count()
        self.assertGreaterEqual(queued, 1)
        # Dream sees committed state only: the unpushed queue is invisible.
        dream = json.loads(provider._tool_dream({}))
        self.assertEqual(dream["committed"], 0)
        # Draining commits; now Dream sees it. Snapshot isolation: mutating the
        # returned snapshot cannot corrupt committed state.
        provider._drain_due()
        dream = json.loads(provider._tool_dream({}))
        self.assertEqual(dream["committed"], queued)
        snapshot = merge_mod.committed_snapshot(provider._log)
        snapshot.clear()
        self.assertEqual(len(provider._log.refs()), queued)
        provider.shutdown()

    def test_dream_digest_surfaces_keep_both_conflicts(self):
        home = tempfile.mkdtemp(prefix="cog-home-")
        provider = make_provider(home)
        base = entry_for("cog-s-1", "seed")
        merge_mod.fetch_first_push(provider._log, base, None, now=1.0)
        seen = provider._log.get("cog-s-1")
        merge_mod.fetch_first_push(provider._log, entry_for("cog-s-1", "b"), seen, now=2.0)
        merge_mod.fetch_first_push(provider._log, entry_for("cog-s-1", "a-late"),
                                   dict(base), now=3.0)
        dream = json.loads(provider._tool_dream({}))
        self.assertEqual(dream["conflicts"], 1)
        self.assertIn("Unresolved conflicts", dream["digest"])
        provider.shutdown()


if __name__ == "__main__":
    unittest.main()
