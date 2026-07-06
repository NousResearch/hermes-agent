"""Behavior-contract tests for the persistent cross-turn todo plan card.

Prove: ONE living card edited in place (no stacking) across turns and within a
turn, a brand-new plan after a finished one seeds a fresh card, an edit failure
reseeds once, identical text skips, and thread metadata is routed.
"""

import asyncio

from gateway.todo_card import (
    decide_todo_card_action,
    todo_list_finished,
    publish_todo_card,
    TODO_CARD_MAX_AGE_S,
)


# ── pure: todo_list_finished ────────────────────────────────────────────────
def test_finished_true_when_all_terminal():
    items = [
        {"id": "a", "status": "completed"},
        {"id": "b", "status": "cancelled"},
    ]
    assert todo_list_finished(items) is True


def test_finished_false_with_pending():
    items = [{"id": "a", "status": "completed"}, {"id": "b", "status": "pending"}]
    assert todo_list_finished(items) is False


def test_finished_false_when_empty():
    assert todo_list_finished([]) is False
    assert todo_list_finished(None) is False


def test_finished_accepts_raw_payload():
    import json
    payload = json.dumps({"todos": [{"id": "a", "status": "completed"}]})
    assert todo_list_finished(payload) is True


# ── pure: decide_todo_card_action ───────────────────────────────────────────
def test_decide_seed_when_no_state():
    assert decide_todo_card_action(
        None, thread_sig="t", new_plan=False, card_text="x", now=1000.0
    ) == "seed"


def test_decide_edit_when_live_and_changed():
    state = {"message_id": "m1", "last_text": "old", "finished": False,
             "thread_sig": "t", "seeded_at": 1000.0}
    assert decide_todo_card_action(
        state, thread_sig="t", new_plan=False, card_text="new", now=1001.0
    ) == "edit"


def test_decide_skip_when_text_unchanged():
    state = {"message_id": "m1", "last_text": "same", "finished": False,
             "thread_sig": "t", "seeded_at": 1000.0}
    assert decide_todo_card_action(
        state, thread_sig="t", new_plan=False, card_text="same", now=1001.0
    ) == "skip"


def test_decide_seed_on_new_plan_after_finished():
    state = {"message_id": "m1", "last_text": "done", "finished": True,
             "thread_sig": "t", "seeded_at": 1000.0}
    assert decide_todo_card_action(
        state, thread_sig="t", new_plan=True, card_text="fresh", now=1001.0
    ) == "seed"


def test_decide_edit_on_reopen_finished_plan():
    # merge=true / read after a finished plan is NOT a new plan -> edit in place.
    state = {"message_id": "m1", "last_text": "done", "finished": True,
             "thread_sig": "t", "seeded_at": 1000.0}
    assert decide_todo_card_action(
        state, thread_sig="t", new_plan=False, card_text="reopened", now=1001.0
    ) == "edit"


def test_decide_seed_on_thread_change():
    state = {"message_id": "m1", "last_text": "x", "finished": False,
             "thread_sig": "topicA", "seeded_at": 1000.0}
    assert decide_todo_card_action(
        state, thread_sig="topicB", new_plan=False, card_text="y", now=1001.0
    ) == "seed"


def test_decide_seed_when_stale():
    state = {"message_id": "m1", "last_text": "x", "finished": False,
             "thread_sig": "t", "seeded_at": 1000.0}
    assert decide_todo_card_action(
        state, thread_sig="t", new_plan=False, card_text="y",
        now=1000.0 + TODO_CARD_MAX_AGE_S + 1,
    ) == "seed"


# ── async IO: publish_todo_card with a fake adapter ─────────────────────────
class _Result:
    def __init__(self, success=True, message_id=None, error="", retryable=False):
        self.success = success
        self.message_id = message_id
        self.error = error
        self.retryable = retryable


class FakeAdapter:
    REQUIRES_EDIT_FINALIZE = False

    def __init__(self, *, edit_ok=True, edit_error="", edit_retryable=False):
        self.sends = []
        self.edits = []
        self._edit_ok = edit_ok
        self._edit_error = edit_error
        self._edit_retryable = edit_retryable
        self._n = 0

    async def send(self, *, chat_id, content, reply_to=None, metadata=None):
        self._n += 1
        mid = f"m{self._n}"
        self.sends.append(
            {"chat_id": chat_id, "content": content, "reply_to": reply_to,
             "metadata": metadata, "message_id": mid}
        )
        return _Result(success=True, message_id=mid)

    async def edit_message(self, *, chat_id, message_id, content,
                           finalize=False, metadata=None):
        self.edits.append(
            {"chat_id": chat_id, "message_id": message_id, "content": content,
             "metadata": metadata, "finalize": finalize}
        )
        return _Result(
            success=self._edit_ok, message_id=message_id,
            error=self._edit_error, retryable=self._edit_retryable,
        )


def _publish(adapter, store, **kw):
    base = dict(
        adapter=adapter, store=store, session_key="s1",
        chat_id="c1", metadata={"thread_id": "9"}, reply_to=None,
        edit_accepts_metadata=True, thread_sig="9",
    )
    base.update(kw)
    asyncio.run(publish_todo_card(**base))


def test_first_call_seeds_then_second_edits_same_message():
    """THE core contract: across two turns the card is sent ONCE then edited."""
    adapter = FakeAdapter()
    store = {}
    _publish(adapter, store, card_text="Plan v1", finished=False, new_plan=True)
    _publish(adapter, store, card_text="Plan v2", finished=False, new_plan=False)
    assert len(adapter.sends) == 1            # seeded once, not twice (no stacking)
    assert len(adapter.edits) == 1            # second call edited in place
    assert adapter.edits[0]["message_id"] == adapter.sends[0]["message_id"]
    assert adapter.edits[0]["content"] == "Plan v2"
    assert store["s1"]["message_id"] == "m1"


def test_three_calls_one_turn_no_stacking():
    adapter = FakeAdapter()
    store = {}
    _publish(adapter, store, card_text="A", finished=False, new_plan=True)
    _publish(adapter, store, card_text="B", finished=False, new_plan=False)
    _publish(adapter, store, card_text="C", finished=True, new_plan=False)
    assert len(adapter.sends) == 1
    assert [e["content"] for e in adapter.edits] == ["B", "C"]


def test_skip_when_unchanged_does_not_edit():
    adapter = FakeAdapter()
    store = {}
    _publish(adapter, store, card_text="same", finished=False, new_plan=True)
    _publish(adapter, store, card_text="same", finished=True, new_plan=False)
    assert len(adapter.sends) == 1
    assert len(adapter.edits) == 0            # identical text -> no edit
    assert store["s1"]["finished"] is True    # finished flag still refreshed


def test_new_plan_after_finished_seeds_fresh_card():
    adapter = FakeAdapter()
    store = {}
    _publish(adapter, store, card_text="done", finished=True, new_plan=True)
    _publish(adapter, store, card_text="brand new plan", finished=False, new_plan=True)
    assert len(adapter.sends) == 2            # a SECOND card for the new task
    assert store["s1"]["message_id"] == "m2"


def test_edit_failure_reseeds_once():
    adapter = FakeAdapter(edit_ok=False, edit_error="message to edit not found")
    store = {}
    _publish(adapter, store, card_text="v1", finished=False, new_plan=True)
    _publish(adapter, store, card_text="v2", finished=False, new_plan=False)
    assert len(adapter.edits) == 1            # tried to edit
    assert len(adapter.sends) == 2            # edit failed -> reseed
    assert store["s1"]["message_id"] == "m2"  # store points at the new card


def test_flood_edit_failure_keeps_state_no_reseed():
    adapter = FakeAdapter(edit_ok=False, edit_error="flood_control:30", edit_retryable=False)
    store = {}
    _publish(adapter, store, card_text="v1", finished=False, new_plan=True)
    _publish(adapter, store, card_text="v2", finished=False, new_plan=False)
    assert len(adapter.edits) == 1
    assert len(adapter.sends) == 1            # flood -> NO reseed (avoid spam)
    assert store["s1"]["message_id"] == "m1"  # stale id kept for next retry


def test_retryable_edit_failure_keeps_state_no_reseed():
    adapter = FakeAdapter(edit_ok=False, edit_error="connection reset", edit_retryable=True)
    store = {}
    _publish(adapter, store, card_text="v1", finished=False, new_plan=True)
    _publish(adapter, store, card_text="v2", finished=False, new_plan=False)
    assert len(adapter.sends) == 1            # transient -> keep state
    assert store["s1"]["message_id"] == "m1"


def test_seed_routes_thread_metadata():
    adapter = FakeAdapter()
    store = {}
    _publish(adapter, store, card_text="x", finished=False, new_plan=True)
    assert adapter.sends[0]["metadata"] == {"thread_id": "9"}


def test_edit_routes_thread_metadata_when_accepted():
    adapter = FakeAdapter()
    store = {}
    _publish(adapter, store, card_text="x", finished=False, new_plan=True)
    _publish(adapter, store, card_text="y", finished=False, new_plan=False)
    assert adapter.edits[0]["metadata"] == {"thread_id": "9"}


def test_thread_change_seeds_fresh_card():
    adapter = FakeAdapter()
    store = {}
    _publish(adapter, store, card_text="x", finished=False, new_plan=False, thread_sig="topicA")
    _publish(adapter, store, card_text="y", finished=False, new_plan=False, thread_sig="topicB")
    assert len(adapter.sends) == 2
