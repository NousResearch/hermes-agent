"""MessageEvent retains IDs when a busy session coalesces text inputs."""

from gateway.platforms.base import merge_pending_message_event
from gateway.platforms.event import MessageEvent, MessageType


def test_text_merge_retains_all_absorbed_message_ids():
    first = MessageEvent(text="first", message_type=MessageType.TEXT, message_id="a2a-task-1")
    second = MessageEvent(text="second", message_type=MessageType.TEXT, message_id="a2a-task-2")
    third = MessageEvent(text="third", message_type=MessageType.TEXT, message_id="a2a-task-3")
    pending = {}

    merge_pending_message_event(pending, "a2a-context", first, merge_text=True)
    merge_pending_message_event(pending, "a2a-context", second, merge_text=True)
    merge_pending_message_event(pending, "a2a-context", third, merge_text=True)

    merged = pending["a2a-context"]
    assert merged.message_id == "a2a-task-1"
    assert merged.text.endswith("third")
    assert merged.merged_message_ids == ["a2a-task-1", "a2a-task-2", "a2a-task-3"]
