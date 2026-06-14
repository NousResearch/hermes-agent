"""Regression: identity-based SQLite flushing must survive ``id()`` reuse.

``_flush_messages_to_session_db`` tracks already-written messages so repeated
flushes (the turn-start and turn-end persist paths both run) don't duplicate
rows. When that tracking keys on ``id(msg)`` it is unsafe: ``repair_message_
sequence`` can drop a this-turn message that was already flushed (e.g. merging
consecutive user turns), after which CPython may recycle that freed dict's
``id()`` onto the freshly-appended assistant reply. The reply then collides with
a *retained* "already flushed" id and is silently skipped — the turn's answer
never reaches ``state.db``. The loss is self-reinforcing: a missing assistant
reply leaves a consecutive-user violation that makes the next repair shrink even
more.

Fix: track written messages by object identity while holding a strong reference
(so a tracked message's ``id()`` cannot be recycled), compare with ``is``, and
prune the pin to messages still present on each flush so memory stays bounded.
"""
import types

from hermes_state import SessionDB
from run_agent import AIAgent
from agent.agent_runtime_helpers import repair_message_sequence


def _make_flusher(db, session_id):
    """A minimal stand-in exposing the real flush method against a real DB."""
    stub = types.SimpleNamespace(
        _session_db=db,
        _session_db_created=True,
        _last_flushed_db_idx=0,
        session_id=session_id,
    )
    stub._apply_persist_user_message_override = lambda messages: None
    stub._ensure_db_session = lambda: None
    stub._flush_messages_to_session_db = types.MethodType(
        AIAgent._flush_messages_to_session_db, stub
    )
    return stub


def _assistant_texts(db, session_id):
    return [
        m.get("content")
        for m in db.get_messages_as_conversation(session_id)
        if m.get("role") == "assistant" and (m.get("content") or "").strip()
    ]


def test_assistant_reply_persists_when_repair_recycles_a_flushed_id(tmp_path):
    """A repair-freed flushed message's recycled id() must not skip the reply."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = db.create_session(session_id="s1", source="test")

    # Loaded history with a role-alternation violation already in the DB:
    # two consecutive user turns (the shape interrupted/voice-burst turns leave).
    loaded = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2a"},
        {"role": "user", "content": "q2b"},  # consecutive-user violation
    ]
    for m in loaded:
        db.append_message(sid, role=m["role"], content=m["content"])

    conversation_history = [dict(m) for m in loaded]
    messages = list(conversation_history)
    messages.append({"role": "user", "content": "q3"})  # this turn's user input

    flusher = _make_flusher(db, sid)

    # Turn-start persist (mirrors the pre-loop session persist).
    flusher._flush_messages_to_session_db(messages, conversation_history)

    # Defensive role-alternation repair runs before the API call and shrinks
    # `messages` in place (merges the consecutive user turns + q3), freeing the
    # just-flushed q3 dict whose id() can then be recycled.
    repaired = repair_message_sequence(flusher, messages)
    assert repaired >= 1, "expected the consecutive-user violation to be repaired"

    # Model produces its answer; appended to the live list (may reuse a freed id).
    messages.append({"role": "assistant", "content": "THE_REPLY"})

    # Turn-end persist.
    flusher._flush_messages_to_session_db(messages, conversation_history)

    texts = _assistant_texts(db, sid)
    assert "THE_REPLY" in texts, (
        "assistant reply was dropped — a recycled id() collided with a retained "
        f"'already flushed' entry. assistant rows={texts}"
    )


def test_no_duplicate_writes_on_normal_turns(tmp_path):
    """A clean turn (no repair) writes each new message exactly once across the
    multiple persist calls of a turn."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = db.create_session(session_id="s2", source="test")

    loaded = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    for m in loaded:
        db.append_message(sid, role=m["role"], content=m["content"])

    conversation_history = [dict(m) for m in loaded]
    messages = list(conversation_history)

    flusher = _make_flusher(db, sid)

    # New user turn + assistant reply, persisted via two flush calls (as the
    # turn-start and turn-end paths both do).
    messages.append({"role": "user", "content": "how are you"})
    flusher._flush_messages_to_session_db(messages, conversation_history)
    messages.append({"role": "assistant", "content": "doing well"})
    flusher._flush_messages_to_session_db(messages, conversation_history)
    # An extra redundant flush must be a no-op.
    flusher._flush_messages_to_session_db(messages, conversation_history)

    rows = db.get_messages_as_conversation(sid)
    contents = [m.get("content") for m in rows]
    assert contents.count("how are you") == 1, contents
    assert contents.count("doing well") == 1, contents
    assert contents == ["hello", "hi there", "how are you", "doing well"], contents
