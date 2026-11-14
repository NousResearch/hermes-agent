"""``kanban.wake_event_kinds`` narrows which terminal events wake the origin.

Contract under test (the reason the knob exists): a stack can keep the passive
ping for every terminal event while only waking a full agent turn for the kinds
that need a decision. Absent/empty/unknown-only configuration must keep the
historical behaviour (every _WAKE_KINDS entry may wake).
"""

from types import SimpleNamespace

from gateway.kanban_watchers_notifier import (
    _WAKE_KINDS,
    _KanbanNotification,
    wake_kinds_from_config,
)


def test_absent_key_keeps_every_default_wake_kind():
    assert wake_kinds_from_config({}) == _WAKE_KINDS
    assert wake_kinds_from_config({"kanban": {}}) == _WAKE_KINDS
    assert wake_kinds_from_config(None) == _WAKE_KINDS


def test_unknown_only_value_falls_back_instead_of_muting_wakes():
    # A typo must not silently turn every wake off.
    assert wake_kinds_from_config({"kanban": {"wake_event_kinds": ["nope"]}}) == _WAKE_KINDS
    assert wake_kinds_from_config({"kanban": {"wake_event_kinds": []}}) == _WAKE_KINDS


def test_narrows_to_the_listed_kinds_and_tolerates_a_comma_string():
    assert wake_kinds_from_config(
        {"kanban": {"wake_event_kinds": ["blocked", "block_loop_detected"]}}
    ) == ("blocked", "block_loop_detected")
    assert wake_kinds_from_config(
        {"kanban": {"wake_event_kinds": "blocked, block_loop_detected"}}
    ) == ("blocked", "block_loop_detected")


def test_bookkeeping_kinds_cannot_be_promoted_into_a_wake():
    # "status"/"archived"/"unblocked" have no formatter: allowing them would
    # build a wake text with no event behind it.
    assert wake_kinds_from_config(
        {"kanban": {"wake_event_kinds": ["blocked", "archived", "status"]}}
    ) == ("blocked",)


def _notification(kinds, allowed, delivery_mode="notify+wake"):
    events = [SimpleNamespace(kind=kind, id=index + 1, payload={}) for index, kind in enumerate(kinds)]
    task = SimpleNamespace(
        title="task title", assignee="dev", session_id="sess-1", status="done", result=None,
    )
    sub = {
        "task_id": "t_00000001", "platform": "discord", "chat_id": "chan-1",
        "delivery_mode": delivery_mode, "last_event_id": 0,
    }
    d = {"sub": sub, "task": task, "events": events, "cursor": max(e.id for e in events), "board": None}
    notif = _KanbanNotification(None, d, platform_cls=object, sub_fail_counts={})
    notif.wake_kinds_allowed = allowed
    notif.build_wake_text()
    return notif


def test_completed_no_longer_wakes_when_only_blocks_may():
    notif = _notification(["completed"], allowed=("blocked", "block_loop_detected"))
    assert notif.send_passive is True          # the passive ping still goes out
    assert notif.wake_kinds == set()           # ... but no agent turn
    assert notif.synth == ""


def test_blocked_still_wakes_under_the_same_narrowing():
    notif = _notification(["blocked"], allowed=("blocked", "block_loop_detected"))
    assert notif.send_passive is True
    assert notif.wake_kinds == {"blocked"}
    assert "t_00000001" in notif.synth


def test_default_kinds_still_wake_on_completion():
    # No regression: without the new key, a completion keeps waking the origin.
    notif = _notification(["completed"], allowed=_WAKE_KINDS)
    assert notif.wake_kinds == {"completed"}
