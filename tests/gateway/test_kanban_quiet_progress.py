"""``kanban.quiet_progress_notifications`` drops automatically-continuing progress events.

Contract under test: opting in must remove a redundant progress event (a retry the
dispatcher performs itself, an automated review handoff) from BOTH delivery legs —
the passive ping and the synthetic wake — while completion, real blockers,
exhausted retries and anything whose continuation cannot be established stay
visible. The filter is per delivery only; the durable history and cursor are
untouched.
"""

import asyncio
from types import SimpleNamespace

import pytest

from agent.i18n import t
import gateway.kanban_watchers_common as watchers_common
import hermes_cli.kanban_db_dispatch as dispatch_mod
from gateway.kanban_watchers_notifier import (
    _KanbanNotification,
    continuation_established,
    quiet_progress_enabled,
)


class _RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
        return SimpleNamespace(success=True)


class _StubRunner:
    """Capability legs satisfied: this process owns the dispatcher and it may spawn."""

    def __init__(self, *, lock_handle=object()):
        self._kanban_dispatcher_lock_handle = lock_handle
        self.pings = []

    def _kanban_sub_op(self, board, op, sub, *, event_id=None):
        self.pings.append((op, event_id))

    async def _deliver_kanban_artifacts(self, *, adapter, chat_id, metadata, event_payload, task):
        return None


@pytest.fixture
def capable(monkeypatch):
    monkeypatch.setattr(watchers_common, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(dispatch_mod, "_profile_exists_fn", lambda: (lambda name: name != "ghost"))
    monkeypatch.setattr(dispatch_mod, "review_dispatch_enabled", lambda: True)


def _notification(
    kinds,
    *,
    status="review",
    assignee="dev",
    quiet=True,
    runner=None,
    cursor=41,
    payloads=None,
):
    events = [
        SimpleNamespace(kind=kind, id=index + 1, payload=(payloads or {}).get(kind, {}))
        for index, kind in enumerate(kinds)
    ]
    task = SimpleNamespace(
        title="quiet card", assignee=assignee, session_id="sess-1", status=status, result=None,
    )
    sub = {
        "task_id": "t_00000001", "platform": "discord", "chat_id": "chan-1",
        "delivery_mode": "notify+wake", "last_event_id": 0,
    }
    d = {"sub": sub, "task": task, "events": events, "cursor": cursor, "board": None}
    notif = _KanbanNotification(runner or _StubRunner(), d, platform_cls=object, sub_fail_counts={})
    notif.quiet_progress = quiet
    notif.adapter = _RecordingAdapter()
    notif.is_push_adapter = True
    return notif


def test_absent_key_keeps_the_passive_ping_and_the_wake(capable):
    enabled = quiet_progress_enabled({"kanban": {}})
    assert enabled is False
    notif = _notification(["review_requested"], quiet=enabled)

    notif.apply_quiet_progress_policy()
    assert [ev.kind for ev in notif.d["events"]] == ["review_requested"]

    notif.build_wake_text()
    assert notif.wake_kinds == {"review_requested"}

    assert asyncio.run(notif._send_pings()) is True
    assert len(notif.adapter.sent) == 1


def test_only_a_literal_true_opts_in():
    assert quiet_progress_enabled({"kanban": {"quiet_progress_notifications": "true"}}) is False
    assert quiet_progress_enabled({"kanban": {"quiet_progress_notifications": 1}}) is False
    assert quiet_progress_enabled({"kanban": {"quiet_progress_notifications": "yes"}}) is False
    assert quiet_progress_enabled({"kanban": {"quiet_progress_notifications": True}}) is True
    assert quiet_progress_enabled({"kanban": {}}) is False
    assert quiet_progress_enabled({}) is False


def test_redundant_review_handoff_is_quieted_from_both_legs(capable):
    notif = _notification(["review_requested"], status="review")

    notif.apply_quiet_progress_policy()
    assert notif.d["events"] == []

    notif.build_wake_text()
    assert notif.wake_kinds == set()
    assert notif.synth == ""

    assert asyncio.run(notif._send_pings()) is True
    assert notif.adapter.sent == []


def test_manual_review_stays_visible_when_review_dispatch_is_off(capable, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "review_dispatch_enabled", lambda: False)
    notif = _notification(["review_requested"], status="review")

    notif.apply_quiet_progress_policy()
    assert [ev.kind for ev in notif.d["events"]] == ["review_requested"]

    notif.build_wake_text()
    assert notif.wake_kinds == {"review_requested"}


def _estop(monkeypatch):
    monkeypatch.setattr(watchers_common, "_kanban_dispatch_allowed", lambda: False)


def _no_predicate(monkeypatch):
    monkeypatch.setattr(dispatch_mod, "_profile_exists_fn", lambda: None)


# One unreadable capability leg per case: every one must keep the notification.
_UNCERTAIN_CASES = {
    "no_lock": {"runner": {"lock_handle": None}},
    "estop": {"patch": _estop},
    "blank_assignee": {"notification": {"assignee": "   "}},
    "unknown_assignee": {"notification": {"assignee": "ghost"}},
    "predicate_none": {"patch": _no_predicate},
    "not_claimable_lane": {"notification": {"status": "todo"}},
}


@pytest.mark.parametrize("case", sorted(_UNCERTAIN_CASES))
def test_uncertain_continuation_preserves_the_notification(capable, monkeypatch, case):
    spec = _UNCERTAIN_CASES[case]
    spec.get("patch", lambda _mp: None)(monkeypatch)

    notif = _notification(
        ["review_requested"], runner=_StubRunner(**spec.get("runner", {})), **spec.get("notification", {}),
    )
    notif.apply_quiet_progress_policy()
    assert [ev.kind for ev in notif.d["events"]] == ["review_requested"]


def test_task_none_is_not_established(capable):
    assert continuation_established(_StubRunner(), None, "review_requested") is False


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("completed", {}),
        ("gave_up", {}),
        ("blocked", {}),
        ("blocked", {"kind": "needs_input"}),
        ("block_loop_detected", {}),
    ],
)
def test_terminal_and_blocker_kinds_are_never_quieted(capable, kind, payload):
    # Status "ready": every capability leg is satisfied, so only the kind decides.
    notif = _notification([kind], status="ready", payloads={kind: payload})

    notif.apply_quiet_progress_policy()
    assert [ev.kind for ev in notif.d["events"]] == [kind]


def test_mixed_batch_keeps_only_the_visible_event(capable):
    notif = _notification(["review_requested", "blocked"], status="review")

    notif.apply_quiet_progress_policy()
    assert [ev.kind for ev in notif.d["events"]] == ["blocked"]

    for ev in notif.d["events"]:
        notif.format_event(ev)
    notif.build_wake_text()
    assert notif.wake_kinds == {"blocked"}
    assert notif.wake_review_detail == ""
    assert t("gateway.kanban.wake.blocked") in notif.synth
    assert "review_requested" not in notif.wake_kinds


@pytest.mark.parametrize("kind", ["crashed", "timed_out"])
def test_retry_transitions_are_quieted_only_in_the_claimable_lane(capable, kind):
    quieted = _notification([kind], status="ready")
    quieted.apply_quiet_progress_policy()
    assert quieted.d["events"] == []

    # Breaker tripped: the dispatcher will not retry, a `gave_up` decision follows.
    kept = _notification([kind], status="blocked")
    kept.apply_quiet_progress_policy()
    assert [ev.kind for ev in kept.d["events"]] == [kind]


def test_filter_leaves_the_cursor_and_claim_for_advance(capable):
    notif = _notification(["review_requested"], cursor=41)

    notif.apply_quiet_progress_policy()
    assert notif.d["cursor"] == 41
    assert notif.d["events"] == []
    # The claim itself is untouched: advance() still consumes the same cursor.
    assert notif.sub["task_id"] == "t_00000001"
