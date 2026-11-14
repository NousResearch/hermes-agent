"""Contract cases for ``messages.retain_refused_turn_for_display``: a safety-refused turn leaves the model read and its user row stays in the display read, tagged as refused, in one write.

Every case runs each registered backend through public operations only. Whole-transcript reads are compared as exact sequences; the refused write is proven non-mutating against a re-read.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from persistence import TurnLeaseLost
from persistence.conversation.messages import (
    CONTENT_REFUSED_DISPLAY_KIND,
    CONTENT_REFUSED_METADATA_KEY,
    ConversationRequest,
    NewMessage,
    TranscriptView,
)
from persistence.conversation.sessions import NewSession
from persistence.core.identity import SessionId
from session_rules.turns import HISTORICAL_TASK_HEADING, SUMMARY_PREFIX, _SUMMARY_END_MARKER, project_compaction_message_for_display

S = SessionId("refused-turn-s1")
NOTICE = "refused: try a different query"
CALL = ({"id": "t1", "type": "function", "function": {"name": "shell", "arguments": "{}"}},)


def ts(i: int) -> float:
    return datetime(2026, 9, 11, 10, 0, i, tzinfo=UTC).timestamp()


def _store(backend, scope, *rows):
    """A fresh session holding ``rows`` (``(role, content, extra)``); returns the store and the row ids."""
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=S, source="cli"))
    ids = [store.messages.append(S, NewMessage(role=role, content=content, timestamp=ts(i), **extra))
           for i, (role, content, extra) in enumerate(rows)]
    return store, ids


def _model(store):
    return [(m.role, m.content) for m in store.messages.read_conversation(S)]


def _display(store):
    return [(m.role, m.content, m.display_kind)
            for m in store.messages.read_conversation(S, ConversationRequest(view=TranscriptView.DISPLAY))]


PRIOR = [("user", "hello", {}), ("assistant", "hi there", {})]
REFUSED_RUN = [("user", "the blocked ask", {}), ("assistant", "", {"tool_calls": CALL}), ("tool", "ran", {"tool_call_id": "t1"})]


def test_the_refused_turn_leaves_the_model_and_stays_on_screen_tagged(backend, scope_a):
    store, _ids = _store(backend, scope_a, *PRIOR, *REFUSED_RUN)
    before = store.sessions.get(S).rewind_count

    assert store.messages.retain_refused_turn_for_display(S, notice=NOTICE) is True

    assert _model(store) == [("user", "hello"), ("assistant", "hi there")]
    assert _display(store) == [("user", "hello", None), ("assistant", "hi there", None),
                               ("user", "the blocked ask", CONTENT_REFUSED_DISPLAY_KIND)]
    refused = store.messages.read_conversation(S, ConversationRequest(view=TranscriptView.DISPLAY))[-1]
    assert refused.display_metadata[CONTENT_REFUSED_METADATA_KEY] == {"notice": NOTICE}
    record = store.sessions.get(S)
    assert (record.rewind_count, record.message_count, record.tool_call_count) == (before + 1, 2, 0)


def test_the_boundary_row_is_the_target_not_the_newest_user_row(backend, scope_a):
    store, ids = _store(backend, scope_a, *PRIOR, *REFUSED_RUN, ("user", "actually, do this instead", {}))

    assert store.messages.retain_refused_turn_for_display(S, notice=NOTICE, boundary_row_id=ids[2]) is True

    assert _model(store) == [("user", "hello"), ("assistant", "hi there")]
    assert _display(store) == [("user", "hello", None), ("assistant", "hi there", None),
                               ("user", "the blocked ask", CONTENT_REFUSED_DISPLAY_KIND)]


def test_without_a_boundary_the_newest_user_turn_is_the_target(backend, scope_a):
    store, _ids = _store(backend, scope_a, *PRIOR, *REFUSED_RUN, ("user", "actually, do this instead", {}))

    assert store.messages.retain_refused_turn_for_display(S, notice=NOTICE) is True

    assert _model(store) == [("user", "hello"), ("assistant", "hi there"), ("user", "the blocked ask"),
                             ("assistant", ""), ("tool", "ran")]
    assert _display(store)[-1] == ("user", "actually, do this instead", CONTENT_REFUSED_DISPLAY_KIND)


@pytest.mark.parametrize("boundary", [999999, "assistant", "rewound"])
def test_a_boundary_that_is_not_an_active_user_row_falls_back_to_the_newest_turn(backend, scope_a, boundary):
    store, ids = _store(backend, scope_a, *PRIOR, ("user", "the blocked ask", {}))
    if boundary == "rewound":
        store.messages.rewind_user_turn(S, -1)
        store.messages.append(S, NewMessage(role="user", content="the blocked ask", timestamp=ts(9)))
    target = {"assistant": ids[1], "rewound": ids[2]}.get(boundary, boundary)

    assert store.messages.retain_refused_turn_for_display(S, notice=NOTICE, boundary_row_id=target) is True

    assert _model(store) == [("user", "hello"), ("assistant", "hi there")]
    assert _display(store)[-1] == ("user", "the blocked ask", CONTENT_REFUSED_DISPLAY_KIND)


def test_an_internal_boundary_row_leaves_the_model_with_its_run(backend, scope_a):
    store, ids = _store(backend, scope_a, *PRIOR, ("user", "[internal] background work finished", {"display_kind": "hidden"}),
                        ("assistant", "", {"tool_calls": CALL}), ("tool", "ran", {"tool_call_id": "t1"}))

    assert store.messages.retain_refused_turn_for_display(S, notice=NOTICE, boundary_row_id=ids[2]) is True

    assert _model(store) == [("user", "hello"), ("assistant", "hi there")]


def test_nothing_to_retain_without_a_user_turn(backend, scope_a):
    store, _ids = _store(backend, scope_a, ("assistant", "a greeting", {}))
    before = store.sessions.get(S).rewind_count

    assert store.messages.retain_refused_turn_for_display(S, notice=NOTICE) is False

    assert _model(store) == [("assistant", "a greeting")]
    assert store.sessions.get(S).rewind_count == before


def test_a_live_turn_lease_refuses_and_changes_nothing(backend, scope_a):
    store, ids = _store(backend, scope_a, *PRIOR, ("user", "the blocked ask", {}))
    assert store.coordination.try_acquire_turn_lease(session_id=S, holder="live", ttl_seconds=300.0)
    model, display = _model(store), _display(store)

    with pytest.raises(TurnLeaseLost):
        store.messages.retain_refused_turn_for_display(S, notice=NOTICE, boundary_row_id=ids[2])

    assert (_model(store), _display(store)) == (model, display)


def test_a_carrier_head_keeps_its_handoff_and_shows_once_as_refused(backend, scope_a):
    carrier = f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n{_SUMMARY_END_MARKER}\n\nthe refused ask"
    store, ids = _store(backend, scope_a, ("user", "older ask", {}), ("assistant", "done", {}), ("user", carrier, {}),
                        ("assistant", "failed", {}))

    assert store.messages.retain_refused_turn_for_display(S, notice=NOTICE, boundary_row_id=ids[2]) is True

    model = store.messages.read_conversation(S)
    assert [m.role for m in model] == ["user", "assistant", "user"]
    assert "the refused ask" not in str(model[-1].content)
    display = store.messages.read_conversation(S, ConversationRequest(view=TranscriptView.DISPLAY))
    assert [m.content for m in display if m.display_kind == CONTENT_REFUSED_DISPLAY_KIND] == [carrier]


def test_a_refused_carrier_unwrapped_for_display_keeps_its_refused_tag(backend, scope_a):
    carrier = f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n{_SUMMARY_END_MARKER}\n\nthe refused ask"
    store, ids = _store(backend, scope_a, ("user", "older ask", {}), ("assistant", "done", {}), ("user", carrier, {}),
                        ("assistant", "failed", {}))
    assert store.messages.retain_refused_turn_for_display(S, notice=NOTICE, boundary_row_id=ids[2]) is True
    display = store.messages.read_conversation(S, ConversationRequest(view=TranscriptView.DISPLAY))
    (row,) = [m for m in display if m.display_kind == CONTENT_REFUSED_DISPLAY_KIND]

    shown = project_compaction_message_for_display({"role": row.role, "content": row.content, "display_kind": row.display_kind})
    other = project_compaction_message_for_display({"role": "user", "content": carrier, "display_kind": "hidden"})

    assert shown is not None and other is not None
    assert (shown["content"], shown["display_kind"]) == ("the refused ask", CONTENT_REFUSED_DISPLAY_KIND)
    assert (other["content"], "display_kind" in other) == ("the refused ask", False)
