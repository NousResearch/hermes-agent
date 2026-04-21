import time

from gateway.platforms.helpers import MessageDeduplicator


def test_message_deduplicator_deduplicates_within_ttl():
    dedup = MessageDeduplicator(ttl_seconds=0.5, max_size=1000)

    assert dedup.is_duplicate("msg-1", "chat-1") is False
    assert dedup.is_duplicate("msg-1", "chat-1") is True


def test_message_deduplicator_allows_same_message_after_ttl_expires():
    dedup = MessageDeduplicator(ttl_seconds=0.1, max_size=1000)

    assert dedup.is_duplicate("msg-1", "chat-1") is False

    time.sleep(0.2)

    assert dedup.is_duplicate("msg-1", "chat-1") is False


def test_message_deduplicator_max_size_still_caps_non_expired_entries():
    dedup = MessageDeduplicator(ttl_seconds=60, max_size=3)

    assert dedup.is_duplicate("msg-1", "chat-1") is False
    assert dedup.is_duplicate("msg-2", "chat-1") is False
    assert dedup.is_duplicate("msg-3", "chat-1") is False
    assert dedup.is_duplicate("msg-4", "chat-1") is False

    assert len(dedup._seen) <= 3


def test_message_deduplicator_different_chats_are_independent():
    dedup = MessageDeduplicator(ttl_seconds=300, max_size=1000)

    assert dedup.is_duplicate("msg-1", "chat-1") is False
    assert dedup.is_duplicate("msg-1", "chat-2") is False
    assert dedup.is_duplicate("msg-1", "chat-1") is True
    assert dedup.is_duplicate("msg-1", "chat-2") is True


def test_message_deduplicator_backward_compatible_without_chat_id():
    dedup = MessageDeduplicator(ttl_seconds=0.5, max_size=1000)

    assert dedup.is_duplicate("msg-1") is False
    assert dedup.is_duplicate("msg-1") is True
