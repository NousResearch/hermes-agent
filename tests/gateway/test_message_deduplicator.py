"""Tests for MessageDeduplicator TTL enforcement (#10306).

Previously, is_duplicate() returned True for any previously seen ID without
checking its age — expired entries were only purged when cache size exceeded
max_size.  Normal workloads never overflowed, so messages stayed "duplicate"
forever.

The fix checks TTL at query time: if the entry's timestamp plus TTL is in
the past, the entry is treated as expired and the message is allowed through.
"""

import time

from gateway.platforms.helpers import MessageDeduplicator


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


