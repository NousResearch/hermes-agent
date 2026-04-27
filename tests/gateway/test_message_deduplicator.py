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


class TestMessageDeduplicatorContent:
    """Secondary (sender, content, time-bucket) dedup for #16182.

    Weixin ilinkai redelivers the same logical message with a fresh
    message_id within a few seconds, slipping past the primary check.
    """

    def test_same_sender_same_content_same_bucket_is_duplicate(self):
        dedup = MessageDeduplicator(ttl_seconds=300)
        assert dedup.is_duplicate_content("user-1", "你是谁？", time_bucket_seconds=60) is False
        assert dedup.is_duplicate_content("user-1", "你是谁？", time_bucket_seconds=60) is True

    def test_different_sender_same_content_not_duplicate(self):
        dedup = MessageDeduplicator(ttl_seconds=300)
        assert dedup.is_duplicate_content("user-1", "hello", time_bucket_seconds=60) is False
        assert dedup.is_duplicate_content("user-2", "hello", time_bucket_seconds=60) is False

    def test_same_sender_different_content_not_duplicate(self):
        dedup = MessageDeduplicator(ttl_seconds=300)
        assert dedup.is_duplicate_content("user-1", "hello", time_bucket_seconds=60) is False
        assert dedup.is_duplicate_content("user-1", "world", time_bucket_seconds=60) is False

    def test_normalization_collapses_whitespace_and_case(self):
        """Trivial reformatting of the same content still hashes equal."""
        dedup = MessageDeduplicator(ttl_seconds=300)
        assert dedup.is_duplicate_content("user-1", "Hello  world", 60) is False
        # uppercase, trailing whitespace, double-space - same normalized form
        assert dedup.is_duplicate_content("user-1", "HELLO WORLD ", 60) is True

    def test_content_dedup_independent_of_message_id_dedup(self):
        """Primary and secondary caches are separate dicts."""
        dedup = MessageDeduplicator(ttl_seconds=300)
        # Two different msg_ids, same content from same sender (the ilinkai bug)
        assert dedup.is_duplicate("msg-aaa") is False
        assert dedup.is_duplicate_content("user-1", "hi", 60) is False
        assert dedup.is_duplicate("msg-bbb") is False  # different id, primary lets it through
        assert dedup.is_duplicate_content("user-1", "hi", 60) is True  # secondary catches it

    def test_empty_sender_or_content_never_duplicate(self):
        dedup = MessageDeduplicator(ttl_seconds=300)
        assert dedup.is_duplicate_content("", "hello", 60) is False
        assert dedup.is_duplicate_content("user-1", "", 60) is False
        # Neither call should have populated the cache
        assert dedup._seen_content == {}

    def test_zero_or_negative_bucket_disables_dedup(self):
        dedup = MessageDeduplicator(ttl_seconds=300)
        assert dedup.is_duplicate_content("user-1", "hello", 0) is False
        assert dedup.is_duplicate_content("user-1", "hello", 0) is False
        assert dedup.is_duplicate_content("user-1", "hello", -5) is False

    def test_different_buckets_not_duplicate(self):
        """A repeat in a *later* bucket falls through, so deliberate retries are not lost."""
        dedup = MessageDeduplicator(ttl_seconds=300)
        # Pin the wall clock to a value comfortably mid-bucket so the
        # neighbour-bucket fallback (within the first second of a bucket)
        # can't fire. 1_000_000 mod 60 = 40s, well past the 1s window.
        with patch("gateway.platforms.helpers.time.time", return_value=1_000_000.0):
            assert dedup.is_duplicate_content("user-1", "hi", 60) is False
        # Advance 90s -> next bucket. Same content, same sender, but the
        # bucket component of the composite key differs, so it's not a dup.
        with patch("gateway.platforms.helpers.time.time", return_value=1_000_090.0):
            assert dedup.is_duplicate_content("user-1", "hi", 60) is False
            # And dedup within the new bucket still works.
            assert dedup.is_duplicate_content("user-1", "hi", 60) is True

    def test_clear_wipes_both_caches(self):
        dedup = MessageDeduplicator(ttl_seconds=300)
        dedup.is_duplicate("msg-1")
        dedup.is_duplicate_content("user-1", "hi", 60)
        assert dedup._seen and dedup._seen_content
        dedup.clear()
        assert dedup._seen == {}
        assert dedup._seen_content == {}

    def test_max_size_caps_content_cache(self):
        dedup = MessageDeduplicator(max_size=3, ttl_seconds=300)
        for i in range(5):
            dedup.is_duplicate_content(f"user-{i}", "hi", 60)
        assert len(dedup._seen_content) <= 3

    def test_content_hash_is_stable_across_instances(self):
        """Hash is deterministic so two adapter instances would agree."""
        a = MessageDeduplicator._content_hash("Hello World")
        b = MessageDeduplicator._content_hash("hello  world")
        assert a == b
        c = MessageDeduplicator._content_hash("hello universe")
        assert a != c
