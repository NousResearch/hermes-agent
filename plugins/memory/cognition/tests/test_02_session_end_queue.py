"""Acceptance: session-end durable push queue.

FIFO order, SQLite durability across reopen, retry schedule 1m/5m/30m with
jitter and a 24h dead-letter cap, the 1000-entry bound with
drop-oldest-with-tombstone, and backlog surfacing — wired through the real
provider hooks (push fixed on session end, sync_turn persists nothing).
"""

import json
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load  # noqa: E402

provider_mod = load("provider")
queue_mod = load("queue")

DurablePushQueue = queue_mod.DurablePushQueue

MESSAGES = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello!"},
    {"role": "user", "content": "I prefer concise answers with no fluff, remember that"},
    {"role": "assistant", "content": "noted."},
    {"role": "user", "content": "we decided to ship the queue first and polish later"},
]


def make_provider(home, **config):
    full = dict(provider_mod._DEFAULTS)
    full.update(config)
    provider = provider_mod.CognitionMemoryProvider(config=full)
    provider.initialize("sess-9", hermes_home=home, platform="cli",
                        agent_context="primary")
    return provider


class SessionEndQueueTest(unittest.TestCase):
    def test_push_happens_on_session_end_not_on_sync_turn(self):
        home = tempfile.mkdtemp(prefix="cog-home-")
        provider = make_provider(home)
        provider.sync_turn("I prefer tea", "noted", session_id="sess-9")
        self.assertEqual(provider._queue.pending_count(), 0)
        provider.on_session_end(MESSAGES)
        pending = provider._queue.pending_count()
        self.assertGreaterEqual(pending, 2)  # prefer + decided
        provider.shutdown()

    def test_fifo_order_and_durability_across_reopen(self):
        directory = tempfile.mkdtemp(prefix="cog-q-")
        path = os.path.join(directory, "queue.db")
        queue = DurablePushQueue(path)
        refs = [queue.enqueue({"n": i}, session_id="s", now=1000.0 + i)
                for i in range(3)]
        self.assertEqual(refs, ["cog-s-1", "cog-s-2", "cog-s-3"])
        queue.close()
        reopened = DurablePushQueue(path)  # durability: survives restart
        due = reopened.due(now=2000.0)
        self.assertEqual([d["ref"] for d in due], refs)  # FIFO
        self.assertEqual([d["payload"]["n"] for d in due], [0, 1, 2])
        reopened.close()

    def test_retry_schedule_1m_5m_30m_then_dead_letter_at_24h(self):
        directory = tempfile.mkdtemp(prefix="cog-q-")
        queue = DurablePushQueue(os.path.join(directory, "queue.db"))
        rng = random.Random(1234)
        ref = queue.enqueue({"n": 1}, session_id="s", now=0.0)
        t0 = 0.0
        # attempt 1 fails -> due again ~60s later
        self.assertEqual(queue.fail(ref, now=t0, rng=rng), "retry")
        due_at_1 = queue.due(now=10.0)
        self.assertEqual(due_at_1, [])
        # Query past the worst-case jitter ceiling (60s + 10%): the seeded
        # jitter here (+5.6s) lands next_due after the nominal 61s mark.
        item = queue.due(now=67.0)
        self.assertEqual(len(item), 1)
        delay1 = item[0]["next_due_at"] - t0
        self.assertTrue(54.0 <= delay1 <= 66.0, delay1)  # 60s +/- 10% jitter
        # attempt 2 fails -> ~300s later
        self.assertEqual(queue.fail(ref, now=61.0, rng=rng), "retry")
        item = queue.due(now=61.0 + 331.0)
        self.assertEqual(len(item), 1)
        delay2 = item[0]["next_due_at"] - 61.0
        self.assertTrue(270.0 <= delay2 <= 330.0, delay2)
        # attempt 3 fails -> ~1800s later
        self.assertEqual(queue.fail(ref, now=362.0, rng=rng), "retry")
        item = queue.due(now=362.0 + 1981.0)
        self.assertEqual(len(item), 1)
        delay3 = item[0]["next_due_at"] - 362.0
        self.assertTrue(1620.0 <= delay3 <= 1980.0, delay3)
        # past 24h -> dead-letter with tombstone, never retried again
        self.assertEqual(queue.fail(ref, now=24 * 3600 + 1.0, rng=rng),
                         "dead-letter")
        self.assertEqual(queue.pending_count(), 0)
        self.assertEqual(queue.due(now=10 * 24 * 3600), [])
        tombs = queue.tombstones()
        self.assertEqual(len(tombs), 1)
        self.assertEqual(tombs[0]["reason"], "expired-max-24h")
        queue.close()

    def test_bound_drops_oldest_with_tombstone(self):
        directory = tempfile.mkdtemp(prefix="cog-q-")
        queue = DurablePushQueue(os.path.join(directory, "queue.db"),
                                 max_entries=3)
        refs = [queue.enqueue({"n": i}, session_id="s", now=float(i))
                for i in range(5)]
        self.assertEqual(queue.pending_count(), 3)
        due = queue.due(now=100.0)
        self.assertEqual([d["payload"]["n"] for d in due], [2, 3, 4])
        tombs = queue.tombstones()
        self.assertEqual([t["ref"] for t in tombs], refs[:2])
        self.assertTrue(all(t["reason"] == "evicted-drop-oldest" for t in tombs))
        queue.close()

    def test_backlog_surfaces_pending_oldest_and_next_retry(self):
        directory = tempfile.mkdtemp(prefix="cog-q-")
        queue = DurablePushQueue(os.path.join(directory, "queue.db"))
        queue.enqueue({"n": 1}, session_id="s", now=1000.0)
        queue.enqueue({"n": 2}, session_id="s", now=1100.0)
        summary = queue.backlog(now=2000.0)
        self.assertEqual(summary["pending"], 2)
        self.assertAlmostEqual(summary["oldest_age_secs"], 1000.0)
        status = json.loads(
            make_provider(tempfile.mkdtemp(prefix="cog-home-"))._tool_status({}))
        self.assertIn("pending_unpushed", status)
        queue.close()


if __name__ == "__main__":
    unittest.main()
