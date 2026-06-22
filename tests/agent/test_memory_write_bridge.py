import json

import pytest

from agent.memory_write_bridge import collect_memory_write_notifications


def test_collect_notifications_includes_remove_with_old_text_after_success():
    notifications = collect_memory_write_notifications(
        json.dumps({"success": True}),
        {
            "action": "remove",
            "target": "memory",
            "old_text": "stale preference entry",
        },
    )

    assert notifications == [
        {
            "action": "remove",
            "target": "memory",
            "content": "",
            "old_text": "stale preference entry",
        }
    ]


def test_collect_notifications_skips_failed_memory_write():
    notifications = collect_memory_write_notifications(
        json.dumps({"success": False, "error": "No entry matched"}),
        {
            "action": "remove",
            "target": "memory",
            "old_text": "stale preference entry",
        },
    )

    assert notifications == []


def test_collect_notifications_skips_staged_memory_write():
    notifications = collect_memory_write_notifications(
        json.dumps({"success": True, "staged": True, "pending_id": "abc123"}),
        {
            "action": "remove",
            "target": "memory",
            "old_text": "stale preference entry",
        },
    )

    assert notifications == []


@pytest.mark.parametrize("tool_result", [None, [], object()])
def test_collect_notifications_skips_unrecognized_tool_result_shape(tool_result):
    notifications = collect_memory_write_notifications(
        tool_result,
        {
            "action": "add",
            "target": "memory",
            "content": "new fact",
        },
    )

    assert notifications == []


def test_collect_notifications_preserves_old_text_for_replace_and_remove_batch():
    notifications = collect_memory_write_notifications(
        json.dumps({"success": True}),
        {
            "target": "user",
            "operations": [
                {"action": "replace", "old_text": "old preference", "content": "updated"},
                {"action": "remove", "old_text": "obsolete preference"},
                {"action": "add", "content": "new fact"},
            ],
        },
    )

    assert notifications == [
        {
            "action": "replace",
            "target": "user",
            "content": "updated",
            "old_text": "old preference",
        },
        {
            "action": "remove",
            "target": "user",
            "content": "",
            "old_text": "obsolete preference",
        },
        {
            "action": "add",
            "target": "user",
            "content": "new fact",
            "old_text": "",
        },
    ]
