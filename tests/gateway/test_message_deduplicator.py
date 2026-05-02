"""Tests for MessageDeduplicator TTL enforcement (#10306).

Previously, is_duplicate() returned True for any previously seen ID without
checking its age — expired entries were only purged when cache size exceeded
max_size.  Normal workloads never overflowed, so messages stayed "duplicate"
forever.

The fix checks TTL at query time: if the entry's timestamp plus TTL is in
the past, the entry is treated as expired and the message is allowed through.
"""

import time
from unittest.mock import patch

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

    def test_expired_entry_gets_refreshed(self):
        """After an expired entry is allowed through, it should be re-tracked."""
        dedup = MessageDeduplicator(ttl_seconds=5)
        assert dedup.is_duplicate("msg-1") is False

        # Expire the entry
        dedup._seen["msg-1"] = time.time() - 10

        # Should be allowed through (expired)
        assert dedup.is_duplicate("msg-1") is False
        # Now should be duplicate again (freshly tracked)
        assert dedup.is_duplicate("msg-1") is True

    def test_different_messages_not_confused(self):
        """Different message IDs are independent."""
        dedup = MessageDeduplicator(ttl_seconds=60)
        assert dedup.is_duplicate("msg-1") is False
        assert dedup.is_duplicate("msg-2") is False
        assert dedup.is_duplicate("msg-1") is True
        assert dedup.is_duplicate("msg-2") is True

    def test_empty_id_never_duplicate(self):
        """Empty/None message IDs are never treated as duplicate."""
        dedup = MessageDeduplicator(ttl_seconds=60)
        assert dedup.is_duplicate("") is False
        assert dedup.is_duplicate("") is False

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

    def test_max_size_eviction_caps_fresh_entries(self):
        """Fresh entries must still be capped to max_size on overflow."""
        dedup = MessageDeduplicator(max_size=2, ttl_seconds=60)

        dedup.is_duplicate("msg-1")
        dedup.is_duplicate("msg-2")
        dedup.is_duplicate("msg-3")

        assert len(dedup._seen) == 2
        assert "msg-1" not in dedup._seen
        assert "msg-2" in dedup._seen
        assert "msg-3" in dedup._seen

    def test_ttl_zero_means_no_dedup(self):
        """With TTL=0, all entries expire immediately."""
        dedup = MessageDeduplicator(ttl_seconds=0)
        assert dedup.is_duplicate("msg-1") is False
        # Entry was just added at time.time(), and TTL is 0,
        # so now - seen_time >= 0 = ttl, meaning it's expired
        # But time.time() might be the exact same float, so
        # the check is `now - ts < ttl` which is `0 < 0` = False
        # This means TTL=0 effectively disables dedup
        assert dedup.is_duplicate("msg-1") is False
