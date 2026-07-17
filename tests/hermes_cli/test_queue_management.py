"""Shared contracts for snapshot-bound queue management."""

from __future__ import annotations

import queue
import threading

import pytest

from hermes_cli.queue_management import (
    ManagedPromptQueue,
    QueueSnapshotStore,
    format_queue_snapshot,
)


def _items(*ids: str):
    return [
        {
            "id": item_id,
            "preview": f"prompt {index}",
            "has_media": index % 2 == 0,
        }
        for index, item_id in enumerate(ids, start=1)
    ]


def test_snapshot_resolves_positions_to_frozen_opaque_ids_without_rendering_ids():
    now = [100.0]
    store = QueueSnapshotStore(ttl_seconds=300, clock=lambda: now[0])

    snapshot = store.open(
        control_key="telegram:dm:alice",
        owner_key="alice",
        target_key="telegram:group:source",
        items=_items("opaque-a", "opaque-b"),
        source_label="Team / release",
    )

    one = store.resolve("telegram:dm:alice", "alice", "2")
    all_items = store.resolve("telegram:dm:alice", "alice", "all")
    rendered = format_queue_snapshot(snapshot)

    assert one.status == "ok"
    assert one.queue_ids == ("opaque-b",)
    assert one.target_key == "telegram:group:source"
    assert all_items.queue_ids == ("opaque-a", "opaque-b")
    assert "1. prompt 1" in rendered
    assert "2. prompt 2" in rendered
    assert "Team / release" in rendered
    assert "/dequeue 2" in rendered
    assert "/dequeue all" in rendered
    assert "opaque-a" not in rendered
    assert "opaque-b" not in rendered


def test_new_snapshot_replaces_old_snapshot_for_same_owner_and_control_session():
    store = QueueSnapshotStore(ttl_seconds=300, clock=lambda: 1.0)
    first = store.open("dm", "alice", "group-a", _items("a"))
    second = store.open("dm", "alice", "group-b", _items("b"))

    assert first.snapshot_id != second.snapshot_id
    assert store.resolve("dm", "alice", "1", snapshot_id=first.snapshot_id).status == "superseded"
    current = store.resolve("dm", "alice", "1")
    assert current.status == "ok"
    assert current.queue_ids == ("b",)
    assert current.target_key == "group-b"


def test_snapshot_owner_isolation_and_expiry_are_fail_closed():
    now = [10.0]
    store = QueueSnapshotStore(ttl_seconds=300, clock=lambda: now[0])
    store.open("dm", "alice", "group", _items("a"))

    assert store.resolve("dm", "bob", "1").status == "missing"
    now[0] = 311.0
    assert store.resolve("dm", "alice", "1").status == "expired"


def test_snapshot_resolves_opaque_ids_only_from_its_current_frozen_view():
    store = QueueSnapshotStore(ttl_seconds=300, clock=lambda: 1.0)
    first = store.open("dm", "alice", "group", _items("a", "b"))

    selected = store.resolve_ids(
        "dm", "alice", ["b"], snapshot_id=first.snapshot_id
    )
    forged = store.resolve_ids(
        "dm", "alice", ["other"], snapshot_id=first.snapshot_id
    )
    replacement = store.open("dm", "alice", "group", _items("c"))
    stale = store.resolve_ids(
        "dm", "alice", ["a"], snapshot_id=first.snapshot_id
    )

    assert selected.status == "ok"
    assert selected.queue_ids == ("b",)
    assert forged.status == "out_of_range"
    assert replacement.snapshot_id != first.snapshot_id
    assert stale.status == "superseded"


def test_snapshot_target_invalidation_invalidates_every_operator_view_for_session():
    store = QueueSnapshotStore(ttl_seconds=300, clock=lambda: 1.0)
    first = store.open("control-a", "alice", "session-old", _items("a"))
    second = store.open("control-b", "bob", "session-old", _items("b"))
    unaffected = store.open("control-c", "alice", "session-other", _items("c"))

    store.invalidate_target("session-old")

    assert store.resolve("control-a", "alice", "1", snapshot_id=first.snapshot_id).status == "missing"
    assert store.resolve("control-b", "bob", "1", snapshot_id=second.snapshot_id).status == "missing"
    assert store.resolve("control-c", "alice", "1", snapshot_id=unaffected.snapshot_id).status == "ok"


def test_empty_snapshot_is_valid_but_has_no_removable_items():
    store = QueueSnapshotStore(ttl_seconds=300, clock=lambda: 1.0)
    snapshot = store.open("dm", "alice", "dm", [])

    assert store.resolve("dm", "alice", "1").status == "empty"
    assert store.resolve("dm", "alice", "all").status == "empty"
    assert "no queued turns" in format_queue_snapshot(snapshot).lower()


@pytest.mark.parametrize("selector", ["", "0", "-1", "1,2", "first", "confirm"])
def test_invalid_dequeue_selectors_do_not_resolve(selector):
    store = QueueSnapshotStore(ttl_seconds=300, clock=lambda: 1.0)
    store.open("dm", "alice", "dm", _items("a"))

    assert store.resolve("dm", "alice", selector).status == "invalid_selector"


def test_out_of_range_position_does_not_drift_to_another_item():
    store = QueueSnapshotStore(ttl_seconds=300, clock=lambda: 1.0)
    store.open("dm", "alice", "dm", _items("a"))

    result = store.resolve("dm", "alice", "2")

    assert result.status == "out_of_range"
    assert result.queue_ids == ()


def test_managed_prompt_queue_preserves_queue_api_and_removes_stable_ids():
    pending = ManagedPromptQueue()
    pending.put("first")
    pending.put(("second", ["image.png"]))
    pending.put_system("[Continuing toward your standing goal]")

    visible = pending.snapshot_items()
    assert [item.preview for item in visible] == ["first", "second"]
    assert len({item.queue_id for item in visible}) == 2
    assert pending.qsize() == 3

    assert pending.remove_ids([visible[0].queue_id]) == 1
    assert pending.qsize() == 2
    assert pending.get_nowait() == ("second", ["image.png"])
    assert pending.get_nowait() == "[Continuing toward your standing goal]"
    assert pending.empty()

    with pytest.raises(queue.Empty):
        pending.get_nowait()


def test_managed_prompt_queue_can_snapshot_empty_state_and_clear_frozen_ids_only():
    pending = ManagedPromptQueue()
    pending.put("old one")
    pending.put("old two")
    frozen = pending.snapshot_items()
    pending.put("new arrival")

    assert pending.remove_ids([item.queue_id for item in frozen]) == 2
    remaining = pending.snapshot_items()
    assert [item.preview for item in remaining] == ["new arrival"]
    assert pending.get_nowait() == "new arrival"


def test_managed_prompt_queue_removal_releases_join_waiters():
    pending = ManagedPromptQueue()
    pending.put("remove before execution")
    queue_id = pending.snapshot_items()[0].queue_id
    finished = threading.Event()
    waiter = threading.Thread(
        target=lambda: (pending.join(), finished.set()),
        daemon=True,
    )
    waiter.start()

    assert pending.remove_ids([queue_id]) == 1
    assert finished.wait(timeout=0.5)
    waiter.join(timeout=0.5)
    assert not waiter.is_alive()
