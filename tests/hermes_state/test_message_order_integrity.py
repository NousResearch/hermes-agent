"""Regression guards: a resumed session must always be a provider-valid sequence.

Two independent ways the ``messages`` table can hold a conversation that the
provider rejects with::

    Messages with role 'tool' must be a response to a preceding message with
    'tool_calls'                       (HTTP 400, DeepSeek; non-retryable)

1. **Ordering.** ``get_messages_as_conversation()`` replays the transcript
   ``ORDER BY timestamp, id``.  ``timestamp`` is written by ``append_message()``
   as ``time.time()`` — a wall clock, not a monotonic counter — and nothing
   enforces that it increases per session.  A backwards wall-clock step (NTP
   correction, WSL2 VM resume, two flush batches from different writers) puts a
   later batch of rows *before* an earlier one, so a ``tool`` result can be
   replayed immediately after an ``assistant`` message that carries no
   ``tool_calls``.  Insertion order (``id``) is the order the messages were
   actually produced and is never ambiguous.

2. **Persisted pair damage.** A session can hold a ``tool`` result whose
   ``assistant``/``tool_calls`` row was never written (or a tool_call whose
   results were dropped).  ``ContextCompressor._sanitize_tool_pairs()`` repairs
   this shape but only runs inside the compression path — never on resume — and
   it compares *set membership*, so it is blind to case (1) where every call_id
   exists but in the wrong position.

Both shapes are unrecoverable from the user's side: the history is rebuilt from
the DB on every resume, so the 400 repeats forever.
"""
import json
import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _tc(call_id, name="terminal"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": "{}"}}


def _build_tool_turn(db, sid, prefix="call"):
    """Persist: user → assistant(tool_calls) → tool results → assistant.

    Two identical turns, the second one with three parallel tool calls.
    """
    db.ensure_session(sid, source="cli", model="test-model")
    db.append_message(sid, "user", "mira el archivo")
    db.append_message(sid, "assistant", "voy a mirarlo",
                      tool_calls=[_tc(f"{prefix}_A")])
    db.append_message(sid, "tool", json.dumps({"content": "ok"}),
                      tool_call_id=f"{prefix}_A", tool_name="read_file")
    db.append_message(sid, "assistant", "listo")
    db.append_message(sid, "user", "ahora cambia el layout")
    db.append_message(sid, "assistant", "necesito ver la estructura",
                      tool_calls=[_tc(f"{prefix}_B0"), _tc(f"{prefix}_B1")])
    db.append_message(sid, "tool", json.dumps({"diff": "--- a\n+++ b"}),
                      tool_call_id=f"{prefix}_B0", tool_name="patch")
    db.append_message(sid, "tool", json.dumps({"diff": "--- a\n+++ b"}),
                      tool_call_id=f"{prefix}_B1", tool_name="patch")
    db.append_message(sid, "assistant", "hecho")


def _assert_provider_valid(messages):
    """The invariant every OpenAI-compatible provider enforces.

    A ``tool`` message must either follow an ``assistant`` message that declares
    its ``tool_call_id``, or follow another ``tool`` message from that same
    group.  Additionally every declared ``tool_call_id`` must have a result.
    """
    problems = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")

        if role == "tool" and (i == 0 or messages[i - 1].get("role") != "tool"):
            previous = messages[i - 1] if i else None
            declared = {
                tc.get("id") for tc in ((previous or {}).get("tool_calls") or [])
            }
            if msg.get("tool_call_id") not in declared:
                problems.append(
                    f"tool result {msg.get('tool_call_id')!r} at index {i} follows "
                    f"role={previous.get('role') if previous else None!r} which does "
                    f"not declare it"
                )

        if role == "assistant" and msg.get("tool_calls"):
            call_ids = [tc.get("id") for tc in msg["tool_calls"]]
            seen, j = [], i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                seen.append(messages[j].get("tool_call_id"))
                j += 1
            missing = [c for c in call_ids if c not in seen]
            if missing:
                problems.append(
                    f"assistant at index {i} declares {missing} with no following result"
                )
            i = j
            continue
        i += 1

    assert not problems, "\n".join(problems)


def _ids_in_order(db, sid):
    rows = db._conn.execute(
        "SELECT id FROM messages WHERE session_id = ? ORDER BY id", (sid,)
    ).fetchall()
    return [r["id"] for r in rows]


# --------------------------------------------------------------------------
# 1. ordering must not depend on the wall clock
# --------------------------------------------------------------------------

def test_resume_is_valid_when_wall_clock_steps_backwards(db):
    """A later flush batch stamped *earlier* must not be replayed out of order."""
    sid = "s-inverted"
    _build_tool_turn(db, sid)

    # Simulate a backwards wall-clock step (~2.1 s, as observed in the wild) for
    # the tool *results* of the last turn and everything after them — the
    # assistant message that declares those call_ids stays where it was, which
    # is exactly the shape a mid-batch clock step produces.
    conn = db._conn
    ids = _ids_in_order(db, sid)
    late = ids[-3:]                     # 2 tool results + trailing assistant
    first_late = conn.execute(
        "SELECT timestamp FROM messages WHERE id = ?", (late[0],)
    ).fetchone()[0]
    declaring = conn.execute(
        "SELECT id FROM messages WHERE id = ?", (ids[-4],)
    ).fetchone()[0]
    conn.execute("BEGIN IMMEDIATE")
    for mid in late:
        conn.execute(
            "UPDATE messages SET timestamp = timestamp - 2.138 WHERE id = ?", (mid,)
        )
    conn.commit()
    assert (
        conn.execute(
            "SELECT timestamp FROM messages WHERE id = ?", (late[0],)
        ).fetchone()[0]
        == pytest.approx(first_late - 2.138)
    )
    assert declaring not in late        # the declaring assistant is NOT moved

    loaded = db.get_messages_as_conversation(sid)
    _assert_provider_valid(loaded)


def test_load_order_follows_insertion_order_not_timestamp(db):
    """The replayed order must equal the order the messages were appended in."""
    sid = "s-order"
    _build_tool_turn(db, sid)

    conn = db._conn
    ids = _ids_in_order(db, sid)
    conn.execute("BEGIN IMMEDIATE")
    for mid in ids[-3:]:
        conn.execute(
            "UPDATE messages SET timestamp = timestamp - 5.0 WHERE id = ?", (mid,)
        )
    conn.commit()

    loaded = db.get_messages_as_conversation(sid)
    # Compare role + tool_call_id signatures against the insertion order.
    rows = conn.execute(
        "SELECT role, content, tool_call_id FROM messages "
        "WHERE session_id = ? ORDER BY id",
        (sid,),
    ).fetchall()
    expected = [(r["role"], r["content"], r["tool_call_id"]) for r in rows]
    actual = [
        (m.get("role"), m.get("content"), m.get("tool_call_id")) for m in loaded
    ]
    assert actual == expected


# --------------------------------------------------------------------------
# 2. damaged tool pairs must be repaired on load, not only on compression
# --------------------------------------------------------------------------

def test_orphan_tool_result_without_assistant_call_is_repaired_on_load(db):
    """`tool` row whose assistant/tool_calls row was never persisted."""
    sid = "s-orphan-result"
    _build_tool_turn(db, sid)

    # Insert the shape seen in the wild: a tool result directly after a user
    # message, with no assistant declaring its call_id.
    db.append_message(sid, "user", "hola?")
    db.append_message(sid, "tool", json.dumps({"output": "[Command interrupted]",
                                               "exit_code": 130}),
                      tool_call_id="call_00_ghost",
                      tool_name="terminal")

    loaded = db.get_messages_as_conversation(sid)
    _assert_provider_valid(loaded)


def test_assistant_tool_call_without_result_is_repaired_on_load(db):
    """`assistant` row declaring a tool_call whose result was never persisted."""
    sid = "s-orphan-call"
    _build_tool_turn(db, sid)

    db.append_message(sid, "user", "sigue")
    db.append_message(sid, "assistant", "voy a ejecutarlo",
                      tool_calls=[_tc("call_00_never_ran")])

    loaded = db.get_messages_as_conversation(sid)
    _assert_provider_valid(loaded)


def test_repaired_result_keeps_message_count_stable(db):
    """Repair must not silently drop unrelated messages."""
    sid = "s-count"
    _build_tool_turn(db, sid)
    db.append_message(sid, "user", "hola?")
    db.append_message(sid, "tool", json.dumps({"output": "x"}),
                      tool_call_id="call_00_ghost", tool_name="terminal")

    loaded = db.get_messages_as_conversation(sid)
    total = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)
    ).fetchone()[0]
    # only the unusable orphan may be dropped; the rest must survive
    assert len(loaded) in (total, total - 1)
    assert any(m.get("role") == "user" and m.get("content") == "hola?"
               for m in loaded)
