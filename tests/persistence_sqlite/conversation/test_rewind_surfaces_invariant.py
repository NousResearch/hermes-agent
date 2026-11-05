"""Invariant: the user-turn rewind leaves the IDENTICAL active message set whichever way a surface calls it (with
the warm in-memory history /undo holds, with that history and durable row-id adoption, or from the durable
transcript alone), and an out-of-range target changes nothing in any call shape."""

from __future__ import annotations

import pytest

from persistence.sqlite.conversation._store.messages import (
    append_message, get_messages_as_conversation, get_resume_conversations,
)
from persistence.sqlite.conversation._store.rewind import RewindTargetUnavailableError, rewind_user_turn
from persistence.sqlite.conversation._store.sessions import create_session

CALLS = ("warm", "warm_adopting_row_ids", "durable")


_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}


def _seed(holder, sid: str, turns: int = 3, rich: bool = False) -> None:
    """``rich``: every turn is an image-bearing user ask answered through a tool_call/tool_result pair."""
    create_session(holder, sid, source="cli")
    for i in range(1, turns + 1):
        if not rich:
            append_message(holder, sid, "user", f"q{i}")
            append_message(holder, sid, "assistant", f"a{i}")
            continue
        append_message(holder, sid, "user", [{"type": "text", "text": f"q{i}"}, dict(_IMAGE)])
        append_message(holder, sid, "assistant", None, tool_calls=[
            {"id": f"call-{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
        append_message(holder, sid, "tool", f"out{i}", tool_call_id=f"call-{i}", tool_name="terminal")
        append_message(holder, sid, "assistant", f"a{i}")


def _active_rows(holder, sid: str):
    return [tuple(r) for r in holder._read_all(
        "SELECT id, role, content, active FROM messages WHERE session_id = ? ORDER BY id", (sid,))]


def _rewind_via(call: str, holder, sid: str, n: int):
    """Back up ``n`` user turns through one call shape; returns the rewind outcome or raises on refusal."""
    if call == "durable":
        return rewind_user_turn(holder, sid, -n)
    warm = get_resume_conversations(holder, sid)[0]  # the live process holds the alternation-repaired projection
    user_turns = sum(1 for m in warm if m.get("role") == "user")
    return rewind_user_turn(holder, sid, user_turns - n, warm_history=warm,
                            adopt_row_ids=call == "warm_adopting_row_ids")


@pytest.fixture()
def holder(backend, scope_a):
    return backend.open(scope_a).holder()


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("rich", [False, True], ids=["text", "tool_calls+image"])
def test_every_call_persists_the_same_active_set(holder, n, rich):
    expected = None
    for call in CALLS:
        sid = f"rewind-{call}-{n}-{rich}"
        _seed(holder, sid, rich=rich)
        assert _rewind_via(call, holder, sid, n).rewound_count
        rows = [(role, content, active) for _id, role, content, active in _active_rows(holder, sid)]
        expected = expected or rows
        assert rows == expected, call
    active = [(r, c) for r, c, a in expected if a]
    if not rich:
        assert [c for _r, c in active] == [f"q{i}" if k == 0 else f"a{i}" for i in range(1, 4 - n) for k in (0, 1)]
    else:
        # The whole tool exchange of a rewound turn goes with it (no orphan tool_result), and the
        # surviving image-bearing user rows stay byte-identical to what was stored.
        assert [r for r, _c in active] == ["user", "assistant", "tool", "assistant"] * (3 - n)
        assert [c for r, c in active if r == "user"] == [
            c for r, c, _a in expected if r == "user"][: 3 - n]
        assert all("QUJD" in c for r, c in active if r == "user")


def test_warm_rewind_leaves_the_warm_history_shape_alone_while_adoption_takes_row_ids(holder):
    """``_row_id`` adoption is asked for by a surface whose clients address follow-ups by durable row; a warm
    history that had no ids before the rewind must not grow them otherwise."""
    for sid in ("shape-cli", "shape-tui"):
        _seed(holder, sid)
    warm = get_messages_as_conversation(holder, "shape-cli")
    assert not any("_row_id" in m for m in warm)
    prefix = rewind_user_turn(holder, "shape-cli", 2, warm_history=warm).prefix
    assert len(prefix) == 4 and not any("_row_id" in m for m in prefix)

    installed = _rewind_via("warm_adopting_row_ids", holder, "shape-tui", 1).prefix
    assert [m["_row_id"] for m in installed] == [row[0] for row in _active_rows(holder, "shape-tui") if row[3]]


@pytest.mark.parametrize("call", CALLS)
def test_out_of_range_target_changes_nothing_in_every_call(holder, call):
    sid = f"oob-{call}"
    _seed(holder, sid, turns=1)
    before = _active_rows(holder, sid)
    # A transcript with no user turn at all is refused without touching the store.
    create_session(holder, sid + "-empty", source="cli")
    with pytest.raises(RewindTargetUnavailableError):
        _rewind_via(call, holder, sid + "-empty", 1)
    assert _active_rows(holder, sid + "-empty") == []
    assert _active_rows(holder, sid) == before


@pytest.mark.parametrize("call", CALLS)
def test_undo_after_an_unanswered_turn_rewinds_the_merged_turn_in_every_call(holder, call):
    """#115493: a turn that ended with no assistant reply leaves a stored ``user;user`` pair that live replay
    merges into one turn. /undo 1 takes back that merged turn (both stored rows) instead of refusing with
    "history changed" forever, and /undo 1 again reaches the turn before it."""
    sid = f"wedged-{call}"
    create_session(holder, sid, source="cli")
    for role, content in (("user", "q1"), ("assistant", "a1"), ("user", "q2_failed"), ("user", "q3"),
                          ("assistant", "a3")):
        append_message(holder, sid, role, content)
    assert len([m for m in get_resume_conversations(holder, sid)[0] if m["role"] == "user"]) == 2
    assert _rewind_via(call, holder, sid, 1) is not None
    assert [c for _i, _r, c, a in _active_rows(holder, sid) if a] == ["q1", "a1"]
    assert _rewind_via(call, holder, sid, 1) is not None
    assert [c for _i, _r, c, a in _active_rows(holder, sid) if a] == []
