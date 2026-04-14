import time

from gateway.platforms.helpers import MessageDeduplicator


def test_message_deduplicator_rejects_duplicate_within_ttl():
    dedup = MessageDeduplicator(ttl_seconds=60)

    assert dedup.is_duplicate("msg-1") is False
    assert dedup.is_duplicate("msg-1") is True


def test_message_deduplicator_accepts_message_after_ttl_expires():
    dedup = MessageDeduplicator(ttl_seconds=0.1)

    assert dedup.is_duplicate("msg-1") is False
    time.sleep(0.2)

    assert dedup.is_duplicate("msg-1") is False


def test_message_deduplicator_replaces_expired_timestamp_on_reuse():
    dedup = MessageDeduplicator(ttl_seconds=0.1)

    assert dedup.is_duplicate("msg-1") is False
    first_seen = dedup._seen["msg-1"]
    time.sleep(0.2)

    assert dedup.is_duplicate("msg-1") is False
    assert dedup._seen["msg-1"] > first_seen
