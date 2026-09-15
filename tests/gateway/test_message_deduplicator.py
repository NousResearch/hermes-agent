"""Tests for MessageDeduplicator TTL enforcement (#10306) and restart survival.

TTL enforcement (#10306): is_duplicate() must check age at query time, not only
purge on cache overflow — otherwise entries stay "duplicate" forever.

Restart survival: the cache is per-process memory, so a gateway restart empties
it. Slack re-delivers un-acked Socket Mode events on reconnect AND re-routes
events that were in flight when the process stopped, so a redelivered event
whose original receipt died with the old process would run a second turn.
``persist_path`` keeps the seen-set on disk so the mark outlives the process.
"""

import json
import time

from gateway.platforms.helpers import MessageDeduplicator
from hermes_constants import get_hermes_home


class TestMessageDeduplicatorTTL:
    """TTL-based expiration must work regardless of cache size."""

    def test_duplicate_within_ttl(self):
        """Same message within TTL window is duplicate."""
        dedup = MessageDeduplicator(ttl_seconds=60)
        assert dedup.is_duplicate("msg-1") is False
        assert dedup.is_duplicate("msg-1") is True

    def test_not_duplicate_after_ttl_expires(self):
        """Same message AFTER TTL expires should NOT be duplicate."""
        dedup = MessageDeduplicator(ttl_seconds=5)
        assert dedup.is_duplicate("msg-1") is False

        # Fast-forward time past TTL
        dedup._seen["msg-1"] = time.time() - 10  # 10s ago, TTL is 5s
        assert dedup.is_duplicate("msg-1") is False, \
            "Expired entry should not be treated as duplicate"


    def test_contains_expires_stale_message_without_refreshing_it(self):
        dedup = MessageDeduplicator(ttl_seconds=5)
        dedup._seen["msg-1"] = time.time() - 10

        assert dedup.contains("msg-1") is False
        assert "msg-1" not in dedup._seen

    def test_max_size_eviction_prunes_expired(self):
        """Cache pruning on overflow removes expired entries."""
        dedup = MessageDeduplicator(max_size=5, ttl_seconds=60)
        # Add 6 entries, with the first 3 expired
        now = time.time()
        for i in range(3):
            dedup._seen[f"old-{i}"] = now - 120  # expired (2 min ago, TTL 60s)
        for i in range(3):
            dedup.is_duplicate(f"new-{i}")
        # Now we have 6 entries. Next insert triggers pruning.
        dedup.is_duplicate("trigger")
        # The 3 expired entries should be gone, leaving 4 fresh ones
        assert len(dedup._seen) == 4
        assert "old-0" not in dedup._seen
        assert "new-0" in dedup._seen


class TestMessageDeduplicatorRestartSurvival:
    """The seen-set must outlive the process when a persist_path is given."""

    KEY = "T0BURRXNHQB:1789134549.911449"

    def test_seen_event_is_duplicate_in_a_fresh_instance(self, tmp_path):
        """The invariant: a fresh process, sharing only the file, suppresses the replay."""
        journal = tmp_path / "seen.json"
        first = MessageDeduplicator(ttl_seconds=1800, persist_path=journal)
        assert first.is_duplicate(self.KEY) is False

        # A restart: brand-new instance, nothing shared with the old one but the file.
        restarted = MessageDeduplicator(ttl_seconds=1800, persist_path=journal)
        assert restarted.is_duplicate(self.KEY) is True

    def test_ttl_still_expires_across_restart(self, tmp_path):
        """Surviving the restart must not make a mark immortal."""
        journal = tmp_path / "seen.json"
        journal.write_text(
            json.dumps({"seen": {"T1:ancient": time.time() - 7200}}), encoding="utf-8")
        dedup = MessageDeduplicator(ttl_seconds=1800, persist_path=journal)
        assert dedup.is_duplicate("T1:ancient") is False

    def test_corrupt_journal_degrades_to_memory_only(self, tmp_path):
        """A damaged file must never break inbound delivery."""
        journal = tmp_path / "seen.json"
        journal.write_text("{not json", encoding="utf-8")
        dedup = MessageDeduplicator(ttl_seconds=60, persist_path=journal)
        assert dedup.is_duplicate("m1") is False
        assert dedup.is_duplicate("m1") is True

    def test_unwritable_journal_degrades_without_raising(self, tmp_path):
        """An unusable path (a file where a directory belongs) is not fatal."""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        dedup = MessageDeduplicator(ttl_seconds=60, persist_path=blocker / "seen.json")
        assert dedup.is_duplicate("m1") is False
        assert dedup.is_duplicate("m1") is True

    def test_persisted_state_is_bounded(self, tmp_path):
        """The file cannot grow without limit; the newest marks are the ones kept."""
        journal = tmp_path / "seen.json"
        dedup = MessageDeduplicator(max_size=10, ttl_seconds=1800, persist_path=journal)
        for i in range(50):
            dedup.is_duplicate(f"m{i}")
        reloaded = MessageDeduplicator(max_size=10, ttl_seconds=1800, persist_path=journal)
        assert len(reloaded._seen) <= 10
        assert reloaded.is_duplicate("m49") is True, "the newest mark must survive"

    def test_discard_releases_the_mark_across_restart(self, tmp_path):
        """discard() releases a claim after a failed handoff — including on disk."""
        journal = tmp_path / "seen.json"
        dedup = MessageDeduplicator(ttl_seconds=1800, persist_path=journal)
        assert dedup.is_duplicate("m1") is False
        dedup.discard("m1")
        assert MessageDeduplicator(ttl_seconds=1800, persist_path=journal).is_duplicate("m1") is False

    def test_clear_removes_persisted_state(self, tmp_path):
        journal = tmp_path / "seen.json"
        dedup = MessageDeduplicator(ttl_seconds=1800, persist_path=journal)
        assert dedup.is_duplicate("m1") is False
        dedup.clear()
        assert MessageDeduplicator(ttl_seconds=1800, persist_path=journal).is_duplicate("m1") is False

    def test_default_remains_memory_only(self, tmp_path):
        """Adapters that pass no persist_path keep today's behaviour and write nothing."""
        mine = tmp_path / "mine"
        mine.mkdir()
        dedup = MessageDeduplicator(ttl_seconds=60)
        assert dedup._persist_path is None
        assert dedup.is_duplicate("m1") is False
        assert list(mine.iterdir()) == []
