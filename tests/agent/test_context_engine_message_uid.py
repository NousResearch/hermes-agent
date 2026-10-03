"""A context engine sees ``message_uid`` on every message the host hands it.

The engine-facing surfaces are ``compress()`` (the live list), ``post_llm_call``'s
``conversation_history`` (the same live dicts after the turn flush) and ``on_turn_complete``
(structural clones). The id must reach all three after a cold restore that never asked for
``_row_id``, survive the persist override and an in-place compaction commit, and never reach the
provider. A consecutive-user merge keeps the first constituent's uid and records the absorbed ones.
"""

from __future__ import annotations

import logging
import os
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.context_engine import ContextEngine
from hermes_state import SessionDB

UID_LEN = 32


class _CapturingEngine(ContextEngine):
    """Records what compress() and on_turn_complete() receive; compress() returns marker-swept copies."""

    last_prompt_tokens = 0

    def __init__(self) -> None:
        self.compress_input = None
        self.turn_complete_messages = None
        self._last_compress_aborted = False
        self._last_summary_error = None
        self.compression_count = 1

    @property
    def name(self) -> str:
        return "capturing"

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
        self.compress_input = [dict(m) for m in messages]
        summary = {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"}
        return [summary] + [dict(m) for m in messages[-2:]]

    def on_turn_complete(self, messages, usage=None, **kwargs):
        self.turn_complete_messages = messages

    def _record_compression_failure_cooldown(self, *a, **k):
        pass


def _make_agent(session_db, session_id):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
            session_db=session_db, session_id=session_id, skip_context_files=True, skip_memory=True,
        )
    agent.compression_in_place = True
    agent.context_compressor = _CapturingEngine()
    return agent


def _seed(db, sid, n=6):
    db.create_session(sid, "cli", model="test/model")
    for i in range(n):
        db.append_message(session_id=sid, role="user" if i % 2 == 0 else "assistant", content=f"seed msg {i}")
    return [dict(r) for r in db._conn.execute(
        "SELECT id, content, message_uid FROM messages WHERE session_id = ? AND active = 1 ORDER BY id",
        (sid,)).fetchall()]


@pytest.fixture()
def db(tmp_path):
    with closing(SessionDB(db_path=Path(tmp_path) / "state.db")) as handle:
        yield handle


def test_compress_input_after_a_cold_restore_carries_every_rows_uid(db):
    """The ACP/gateway restore shape (no ``include_row_ids``) → compress() input → committed generation."""
    from agent.conversation_compression import compress_context

    sid = "20260928_120000_uid"
    stored = _seed(db, sid)
    restored = db.get_messages_as_conversation(sid, repair_alternation=True)
    assert all("_row_id" not in m for m in restored)
    agent = _make_agent(db, sid)
    agent._session_db_created = True
    agent._last_flushed_db_idx = len(restored)

    compressed, _sp = compress_context(agent, restored, approx_tokens=100_000, system_message="sys")

    engine = agent.context_compressor
    assert [m.get("message_uid") for m in engine.compress_input] == [r["message_uid"] for r in stored]
    # The committed generation: the copied tail keeps its uids, the summary got a fresh one.
    active = [dict(r) for r in db._conn.execute(
        "SELECT content, message_uid FROM messages WHERE session_id = ? AND active = 1 ORDER BY id", (sid,))]
    assert [r["content"] for r in active] == [
        "[CONTEXT COMPACTION] summary of prior turns", "seed msg 4", "seed msg 5"]
    assert [r["message_uid"] for r in active[1:]] == [stored[4]["message_uid"], stored[5]["message_uid"]]
    assert len(active[0]["message_uid"]) == UID_LEN
    # And the live list the caller keeps carries the same uids (what the next compress()/post_llm_call sees).
    assert [m.get("message_uid") for m in compressed] == [r["message_uid"] for r in active]


@pytest.mark.parametrize("in_place", [True, False], ids=["in_place", "rotation"])
def test_an_engine_authored_uid_survives_the_commit(db, in_place):
    """An engine that pre-stamps ``message_uid`` on the rows it emits gets that value written through the
    commit (in place and on rotation), so it can recognise its own rows by uid afterwards."""
    from agent.conversation_compression import compress_context

    class _PreStamping(_CapturingEngine):
        def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
            summary = {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns",
                       "message_uid": "e" * UID_LEN}
            return [summary] + [dict(m) for m in messages[-2:]]

    sid = f"20260928_120050_{'inplace' if in_place else 'rotate'}"
    stored = _seed(db, sid)
    restored = db.get_messages_as_conversation(sid)
    agent = _make_agent(db, sid)
    agent.context_compressor = _PreStamping()
    agent.compression_in_place = in_place
    agent._session_db_created = True
    agent._last_flushed_db_idx = len(restored)

    compressed, _sp = compress_context(agent, restored, approx_tokens=100_000, system_message="sys")

    committed_sid = agent.session_id
    if not in_place:
        assert committed_sid != sid  # rotation published a child session
    active = db.get_messages_as_conversation(committed_sid)
    assert [m["content"] for m in active][:1] == ["[CONTEXT COMPACTION] summary of prior turns"]
    assert active[0]["message_uid"] == "e" * UID_LEN
    assert [m["message_uid"] for m in active[1:3]] == [stored[4]["message_uid"], stored[5]["message_uid"]]
    assert [m.get("message_uid") for m in compressed[:3]] == [m["message_uid"] for m in active[:3]]


def test_turn_flush_stamps_uids_on_the_live_dicts_post_llm_call_hands_over(db):
    """``post_llm_call(conversation_history=list(messages))`` passes the live dicts; after the turn flush
    every one of them carries the durable uid, including the current-turn user row."""
    sid = "20260928_120100_flush"
    db.create_session(sid, "cli", model="test/model")
    agent = _make_agent(db, sid)
    agent._session_db_created = True
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    agent._persist_user_message_idx = 0

    agent._persist_session(messages, conversation_history=None)

    assert all(len(m.get("message_uid", "")) == UID_LEN for m in messages)
    stored = [r[0] for r in db._conn.execute(
        "SELECT message_uid FROM messages WHERE session_id = ? AND active = 1 ORDER BY id", (sid,))]
    assert [m["message_uid"] for m in messages] == stored


def test_persist_override_keeps_the_uid(db):
    """The ACP/gateway persist override rewrites the current-turn user CONTENT in place; the id is the row's."""
    sid = "20260928_120200_override"
    db.create_session(sid, "cli", model="test/model")
    agent = _make_agent(db, sid)
    agent._session_db_created = True
    messages = [{"role": "user", "content": "api-only variant with injected context"}]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "what the user typed"
    agent._persist_session(messages, conversation_history=None)
    uid = messages[0]["message_uid"]

    agent._apply_persist_user_message_override(messages)

    assert messages[0]["content"] == "what the user typed"
    assert messages[0]["message_uid"] == uid
    row = db._conn.execute(
        "SELECT content, message_uid FROM messages WHERE session_id = ? AND active = 1", (sid,)).fetchone()
    assert (row[0], row[1]) == ("what the user typed", uid)


def test_consecutive_user_merge_keeps_the_first_uid_and_records_the_absorbed_ones():
    from agent.agent_runtime_helpers import _merge_consecutive_users

    a = {"role": "user", "content": "first", "message_uid": "a" * UID_LEN}
    b = {"role": "user", "content": "second", "message_uid": "b" * UID_LEN}
    c = {"role": "user", "content": "third", "message_uid": "c" * UID_LEN}

    merged, repairs = _merge_consecutive_users([a, b, c])

    assert repairs == 2 and merged == [a]
    assert a["content"] == "first\n\nsecond\n\nthird"
    assert a["message_uid"] == "a" * UID_LEN
    assert a["_absorbed_message_uids"] == ["b" * UID_LEN, "c" * UID_LEN]
    # A restored (no ``_row_id``) absorbed dict still leaves its uid on the survivor.
    assert "_absorbed_row_ids" not in a


def test_restart_after_a_merge_still_sees_the_composites_constituents(db):
    """A dangling user row, a restart, the next prompt merged into it, a turn flush, another restart: the
    restored survivor names the absorbed row by uid instead of leaving the engine to parse ``\\n\\n``."""
    sid = "20260928_120300_witness"
    db.create_session(sid, "cli", model="test/model")
    db.append_message(session_id=sid, role="user", content="unanswered before the crash")
    dangling_uid = db.get_messages_as_conversation(sid)[0]["message_uid"]
    # Restart: the ACP/gateway restore shape, then the next prompt lands and the pre-request repair merges.
    history = db.get_messages_as_conversation(sid, repair_alternation=True, include_row_ids=True)
    agent = _make_agent(db, sid)
    agent._session_db_created = True
    prompt = {"role": "user", "content": "next prompt"}
    messages = history + [prompt]
    agent._persist_user_message_idx = 1
    agent._persist_session(messages, conversation_history=history)  # the turn-start flush of the prompt
    from agent.agent_runtime_helpers import repair_message_sequence
    repair_message_sequence(agent, messages)
    assert len(messages) == 1 and messages[0]["message_uid"] == dangling_uid
    assert messages[0]["_absorbed_message_uids"] == [prompt["message_uid"]]
    messages.append({"role": "assistant", "content": "reply"})
    agent._persist_session(messages, conversation_history=None)
    # Second restart.
    again = db.get_messages_as_conversation(sid, repair_alternation=True)
    survivor = next(m for m in again if m["message_uid"] == dangling_uid)
    assert survivor["content"].startswith("unanswered before the crash")
    assert prompt["message_uid"] in survivor["_absorbed_message_uids"]


def test_tool_result_flushed_after_its_call_pairs_through_the_live_list(db):
    """Tool round shape: the assistant(tool_calls) row is flushed first, the result in a later flush."""
    sid = "20260928_120400_tool"
    db.create_session(sid, "cli", model="test/model")
    agent = _make_agent(db, sid)
    agent._session_db_created = True
    assistant = {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "call_x", "type": "function", "function": {"name": "t", "arguments": "{}"}}]}
    messages = [{"role": "user", "content": "q"}, assistant]
    agent._persist_user_message_idx = 0
    agent._persist_session(messages, conversation_history=None)
    uid = assistant["_tool_call_uids"]["call_x"]
    result = {"role": "tool", "content": "r", "tool_call_id": "call_x", "tool_name": "t"}
    messages.append(result)
    agent._persist_session(messages, conversation_history=None)
    assert result["_tool_call_uid"] == uid
    stored = db._conn.execute(
        "SELECT tool_call_uid FROM messages WHERE session_id = ? AND role = 'tool'", (sid,)).fetchone()[0]
    assert stored == uid


def test_the_live_list_walk_binds_a_result_to_the_nearest_call_that_names_it():
    from agent.message_metadata import tool_call_uid_from_history

    def call(uid=None):
        msg = {"role": "assistant", "content": "",
               "tool_calls": [{"id": "call_x", "type": "function", "function": {"name": "t", "arguments": "{}"}}]}
        if uid:
            msg["_tool_call_uids"] = {"call_x": uid}
        return msg

    result = {"role": "tool", "content": "r", "tool_call_id": "call_x"}
    # Nearest call wins over an older one with the same provider id.
    assert tool_call_uid_from_history([call("1" * UID_LEN), result, call("2" * UID_LEN), result], 3, {}) == "2" * UID_LEN
    # A nearer call without a map (legacy) owns the result: no uid, never the older occurrence's.
    assert tool_call_uid_from_history([call("1" * UID_LEN), result, call(), result], 3, {}) is None
    # Never across a user turn.
    assert tool_call_uid_from_history([call("1" * UID_LEN), {"role": "user", "content": "q"}, result], 2, {}) is None


def test_within_one_row_the_last_occurrence_of_an_id_owns_the_results_after_it():
    """A folded row holds several occurrences of one id; like separate rows, the later occurrence shadows the
    earlier one, and an occurrence persisted without a uid (``None`` slot) pairs its results with nothing."""
    from agent.message_metadata import tool_call_uid_from_history

    def folded(slots):
        return {"role": "assistant", "content": "", "tool_calls": [_reused_call("")["tool_calls"][0]] * len(slots),
                "_tool_call_uids": {"call_x": slots}}

    result = {"role": "tool", "content": "r", "tool_call_id": "call_x"}
    one, two = "1" * UID_LEN, "2" * UID_LEN
    assert tool_call_uid_from_history([folded([one, None]), result], 1, {}) is None
    assert tool_call_uid_from_history([folded([None, one]), result], 1, {}) == one
    assert tool_call_uid_from_history([folded([one, two]), result], 1, {}) == two

    # Responses-style composite ids: each occurrence keeps its own spelling, so the result that names the
    # mapped occurrence's full id still pairs with it while the bare id follows the (unmapped) last one.
    composite = {"role": "assistant", "content": "", "_tool_call_uids": {"call_x": [one, None]}, "tool_calls": [
        {"id": f"call_x|fc_{n}", "type": "function", "function": {"name": "t", "arguments": "{}"}} for n in (1, 2)]}
    for tool_call_id, expected in (("call_x|fc_1", one), ("call_x", None)):
        named = {"role": "tool", "content": "r", "tool_call_id": tool_call_id}
        assert tool_call_uid_from_history([composite, named], 1, {}) == expected


@pytest.mark.parametrize("mapped_first", [True, False], ids=["mapped-then-legacy", "legacy-then-mapped"])
def test_a_mixed_era_fold_keeps_one_slot_per_occurrence(mapped_first):
    """Folding a turn this build wrote (uid map) with one an older build wrote (none) that reuse a provider
    id: the survivor's map spells out every occurrence, ``None`` for the unmapped one, so a following result
    pairs with its nearest occurrence's uid, or with nothing, never with the other occurrence's."""
    from agent.agent_runtime_helpers import _merge_consecutive_assistants
    from agent.message_metadata import tool_call_uid_from_history

    uid = "1" * UID_LEN
    turns = [_reused_call("", uid), _reused_call("")]
    if not mapped_first:
        turns.reverse()
    merged, repairs = _merge_consecutive_assistants(turns)
    survivor = merged[0]
    assert repairs == 1 and [tc["id"] for tc in survivor["tool_calls"]] == ["call_x", "call_x"]
    assert survivor["_tool_call_uids"] == {"call_x": [uid, None] if mapped_first else [None, uid]}
    result = {"role": "tool", "content": "r", "tool_call_id": "call_x"}
    assert tool_call_uid_from_history([survivor, result], 1, {}) == (None if mapped_first else uid)


def test_a_stale_key_on_the_absorbed_turn_never_overwrites_the_real_occurrences_uid():
    """Repair can prune a call and leave its ``_tool_call_uids`` key behind. When the absorbed turn carries a
    stale key for an id only the survivor still calls, the fold keeps the survivor's uid for that id: the
    stale key names no call, so it pairs nothing and is dropped."""
    from agent.agent_runtime_helpers import _merge_assistant_into
    from agent.message_metadata import tool_call_uid_from_history

    def turn(call_id, uids):
        return {"role": "assistant", "content": "",
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "t", "arguments": "{}"}}],
                "_tool_call_uids": uids}

    real, stale, other = "1" * UID_LEN, "2" * UID_LEN, "3" * UID_LEN
    prev = turn("call_x", {"call_x": real})
    _merge_assistant_into(prev, turn("call_y", {"call_x": stale, "call_y": other}))
    assert prev["_tool_call_uids"] == {"call_x": real, "call_y": other}
    result = {"role": "tool", "content": "r", "tool_call_id": "call_x"}
    assert tool_call_uid_from_history([prev, result], 1, {}) == real


def test_a_fold_that_maps_nothing_leaves_no_map():
    from agent.agent_runtime_helpers import _merge_assistant_into

    prev, new = _reused_call(""), _reused_call("")
    prev["_tool_call_uids"] = {}
    _merge_assistant_into(prev, new)
    assert len(prev["tool_calls"]) == 2 and "_tool_call_uids" not in prev


@pytest.mark.parametrize("mapped_first", [True, False], ids=["mapped-then-legacy", "legacy-then-mapped"])
def test_a_mixed_era_store_flushes_the_pairing_restore_derives(db, mapped_first):
    """A store spanning the upgrade: turn A written by this build (uid map) and turn B by an older build (no
    map), adjacent and reusing one provider id, then a result the older build left unpaired. Restore folds
    them; the next result for that id is flushed with the NEAREST occurrence's uid (none when that is B's),
    and a fresh restore derives the same pairing for every result row."""
    import json

    sid = "20260929_120000_mixed"
    db.create_session(sid, "cli", model="test/model")
    calls = _reused_call("")["tool_calls"]
    db.append_message(session_id=sid, role="user", content="q")

    def older_build(role, **cols):
        cols = {"session_id": sid, "role": role, "content": "", "timestamp": 1.0, **cols}
        db._conn.execute(f"INSERT INTO messages ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                         tuple(cols.values()))
        db._conn.commit()

    for mapped in ((True, False) if mapped_first else (False, True)):
        if mapped:
            db.append_message(session_id=sid, role="assistant", content="", tool_calls=calls)
        else:
            older_build("assistant", tool_calls=json.dumps(calls))
    older_build("tool", content="r-old", tool_call_id="call_x", tool_name="t")
    uid = json.loads(db._conn.execute(
        "SELECT tool_call_uids FROM messages WHERE tool_call_uids IS NOT NULL").fetchone()[0])["call_x"]
    assert db._conn.execute("SELECT COUNT(*) FROM messages WHERE tool_call_uid IS NOT NULL").fetchone()[0] == 0

    agent = _make_agent(db, sid)
    agent._session_db_created = True
    live = db.get_messages_as_conversation(sid, repair_alternation=True, include_row_ids=True)
    survivor = next(m for m in live if m.get("tool_calls"))
    live.append({"role": "tool", "content": "r-new", "tool_call_id": "call_x", "tool_name": "t"})
    agent._persist_session(live, conversation_history=None)  # the real _db_flush_collect + _db_flush_write

    expected = None if mapped_first else uid
    stored = dict(db._conn.execute(
        "SELECT content, tool_call_uid FROM messages WHERE session_id = ? AND role = 'tool' AND active = 1",
        (sid,)).fetchall())
    assert stored == {"r-old": None, "r-new": expected}
    # Every stored result row (unfolded, so none is dropped as a repeat) restores with the flushed pairing.
    again = db.get_messages_as_conversation(sid, repair_alternation=False)
    assert {m["content"]: m.get("_tool_call_uid") for m in again if m["role"] == "tool"} == {
        "r-old": expected, "r-new": expected}
    refolded = db.get_messages_as_conversation(sid, repair_alternation=True)
    slots = [uid, None] if mapped_first else [None, uid]
    assert survivor["_tool_call_uids"] == next(m for m in refolded if m.get("tool_calls"))["_tool_call_uids"] == {
        "call_x": slots}


def test_a_superseded_verification_candidate_is_not_a_witness_constituent():
    """Alternation repair DISCARDS a provisional verification candidate in favour of the final answer; its
    row is retired like an absorbed row, but its uid is not a constituent of anything."""
    from agent.agent_runtime_helpers import _merge_consecutive_assistants

    candidate = {"role": "assistant", "content": "provisional", "finish_reason": "verification_required",
                 "message_uid": "p" * UID_LEN, "_row_id": 7}
    final = {"role": "assistant", "content": "verified answer", "message_uid": "f" * UID_LEN}

    merged, repairs = _merge_consecutive_assistants([candidate, final])

    assert repairs == 1 and merged == [final]
    assert final["message_uid"] == "f" * UID_LEN
    assert "_absorbed_message_uids" not in final
    assert final["_absorbed_row_ids"] == [7]


def test_assistant_merge_keeps_the_absorbed_turns_tool_call_uids():
    from agent.agent_runtime_helpers import _merge_consecutive_assistants

    first = {"role": "assistant", "content": "a",
             "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
             "_tool_call_uids": {"call_1": "1" * UID_LEN}}
    second = {"role": "assistant", "content": "b",
              "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
              "_tool_call_uids": {"call_2": "2" * UID_LEN}}
    merged, repairs = _merge_consecutive_assistants([first, second])
    assert repairs == 1 and merged == [first]
    assert first["_tool_call_uids"] == {"call_1": "1" * UID_LEN, "call_2": "2" * UID_LEN}


def _reused_call(content, occurrence_uid=None):
    msg = {"role": "assistant", "content": content,
           "tool_calls": [{"id": "call_x", "type": "function", "function": {"name": "t", "arguments": "{}"}}]}
    if occurrence_uid:
        msg["_tool_call_uids"] = {"call_x": occurrence_uid}
    return msg


def test_assistant_merge_keeps_every_occurrence_of_a_reused_provider_id():
    """Two folded turns that reuse a provider id (llama.cpp) both keep their call AND their occurrence uid;
    the ids on the wire are untouched, and a following result pairs with the nearest (later) call."""
    from agent.agent_runtime_helpers import _merge_consecutive_assistants
    from agent.message_metadata import index_tool_call_uids, resolve_tool_call_uid

    first, second = _reused_call("", "1" * UID_LEN), _reused_call("", "2" * UID_LEN)
    merged, repairs = _merge_consecutive_assistants([first, second])
    assert repairs == 1 and merged == [first]
    assert [tc["id"] for tc in first["tool_calls"]] == ["call_x", "call_x"]
    assert first["_tool_call_uids"] == {"call_x": ["1" * UID_LEN, "2" * UID_LEN]}
    index: dict = {}
    index_tool_call_uids(index, first)
    assert resolve_tool_call_uid(index, "call_x") == "2" * UID_LEN


def test_a_reused_id_merge_survives_rewrite_and_restore(db):
    """The per-occurrence map a fold produces is what the survivor's rewrite persists and a restore returns;
    the result keeps the uid of the call it answered."""
    from agent.agent_runtime_helpers import repair_message_sequence

    sid = "20260928_120600_reused"
    db.create_session(sid, "cli", model="test/model")
    agent = _make_agent(db, sid)
    agent._session_db_created = True
    first, second = _reused_call("checking"), _reused_call("")
    result = {"role": "tool", "content": "r", "tool_call_id": "call_x", "tool_name": "t"}
    messages = [{"role": "user", "content": "q"}, first, second, result]
    agent._persist_user_message_idx = 0
    agent._persist_session(messages, conversation_history=None)
    occurrences = [first["_tool_call_uids"]["call_x"], second["_tool_call_uids"]["call_x"]]
    assert len(set(occurrences)) == 2 and result["_tool_call_uid"] == occurrences[1]

    repair_message_sequence(agent, messages)
    assert first["_tool_call_uids"] == {"call_x": occurrences}
    agent._persist_session(messages, conversation_history=None)

    again = db.get_messages_as_conversation(sid, repair_alternation=False)
    stored = next(m for m in again if m["role"] == "assistant" and m.get("tool_calls"))
    assert stored["_tool_call_uids"] == {"call_x": occurrences}
    assert next(m for m in again if m["role"] == "tool")["_tool_call_uid"] == occurrences[1]


@pytest.mark.parametrize("kept, dropped", [
    ([{"type": "text", "text": "kept"}], "UNIQUE-DROPPED-TEXT"),
    ("kept", [{"type": "text", "text": "UNIQUE-DROPPED-PART"}]),
    ([{"type": "text", "text": "kept"}], [{"type": "text", "text": "UNIQUE-DROPPED-PART"}]),
])
def test_an_assistant_fold_that_discards_the_later_text_is_no_witness(kept, dropped):
    """Multimodal content is never joined: the later turn's row is retired, but its uid must not be
    claimed as a constituent of text it does not contain."""
    from agent.agent_runtime_helpers import _merge_consecutive_assistants

    survivor = {"role": "assistant", "content": kept, "message_uid": "a" * UID_LEN}
    later = {"role": "assistant", "content": dropped, "message_uid": "b" * UID_LEN, "_row_id": 9}
    merged, repairs = _merge_consecutive_assistants([survivor, later])
    assert repairs == 1 and merged == [survivor] and survivor["content"] == kept
    assert "_absorbed_message_uids" not in survivor
    assert survivor["_absorbed_row_ids"] == [9]


def test_a_fold_ignores_a_non_dict_uid_map_on_either_turn():
    """A plugin-built dict can carry a malformed map; the fold keeps the well-formed side instead of raising."""
    from agent.agent_runtime_helpers import _merge_assistant_into

    def turn(call_id, uids):
        return {"role": "assistant", "content": "",
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "t", "arguments": "{}"}}],
                "_tool_call_uids": uids}

    prev, new = turn("X", "junk"), turn("Y", {"Y": "b" * 32})
    _merge_assistant_into(prev, new)
    assert prev["_tool_call_uids"] == {"Y": "b" * 32}
    prev, new = turn("X", {"X": "a" * 32}), turn("Y", ["junk"])
    _merge_assistant_into(prev, new)
    assert prev["_tool_call_uids"] == {"X": "a" * 32}


def test_a_select_context_selection_is_stripped_before_the_provider():
    """The request copy is stripped BEFORE ``select_context``; an engine that hands back the
    ``conversation_messages`` clones must not put the ids back on the wire."""
    from agent.conversation_loop import _apply_context_engine_selection
    from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS

    class _Selecting(_CapturingEngine):
        def select_context(self, api_messages, conversation_messages=None, incoming_message=None, budget_tokens=0):
            return list(conversation_messages)

    class _Agent:
        context_compressor = _Selecting()
        session_id = "s"

    history = [
        {"role": "user", "content": "q", "message_uid": "a" * UID_LEN, "timestamp": 1.0, "_row_id": 1,
         "_absorbed_message_uids": ["b" * UID_LEN]},
        {"role": "assistant", "content": "", "message_uid": "c" * UID_LEN, "timestamp": 2.0,
         "tool_calls": [{"id": "call_x", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
         "_tool_call_uids": {"call_x": "d" * UID_LEN}},
        {"role": "tool", "content": "r", "tool_call_id": "call_x", "message_uid": "e" * UID_LEN, "timestamp": 3.0,
         "_tool_call_uid": "d" * UID_LEN},
    ]
    api = [{k: v for k, v in m.items() if k not in PERSISTENCE_ONLY_MESSAGE_FIELDS} for m in history]

    out = _apply_context_engine_selection(_Agent(), api, history, history[0], logger=logging.getLogger("t"))

    assert [m["content"] for m in out] == ["q", "", "r"]
    assert out[1]["tool_calls"][0]["id"] == "call_x"
    assert not any(key in m for m in out for key in PERSISTENCE_ONLY_MESSAGE_FIELDS)
    assert all(m["message_uid"] for m in history)  # history untouched


def test_the_inflight_task_restated_onto_the_carrier_records_its_uid():
    from agent.context_compressor import (
        COMPRESSED_SUMMARY_METADATA_KEY, SUMMARY_PREFIX, _INFLIGHT_TASK_REPLAY_HEADER, _SUMMARY_END_MARKER,
        ContextCompressor,
    )

    compressor = object.__new__(ContextCompressor)
    compressor.quiet_mode = True
    carrier = {"role": "user", "content": f"{SUMMARY_PREFIX}\n\nwhat happened\n\n{_SUMMARY_END_MARKER}",
               COMPRESSED_SUMMARY_METADATA_KEY: True, "message_uid": "s" * UID_LEN}
    inflight = {"role": "user", "content": "finish the task", "message_uid": "a" * UID_LEN}

    out = compressor._reappend_inflight_user_task([carrier], inflight)

    assert out == [carrier] and _INFLIGHT_TASK_REPLAY_HEADER in carrier["content"]
    assert carrier["message_uid"] == "s" * UID_LEN
    assert carrier["_absorbed_message_uids"] == ["a" * UID_LEN]


def test_the_real_user_anchor_folded_into_scaffolding_keeps_the_anchors_uid():
    """The anchor's text leads the composite, so its uid is the composite's; the scaffolding row's is absorbed."""
    from agent.conversation_compression import _insert_real_user_anchor, _merge_anchor_into_user_message

    target = {"role": "user", "content": "[todo snapshot]", "message_uid": "t" * UID_LEN}
    anchor = {"role": "user", "content": "the real ask", "message_uid": "a" * UID_LEN,
              "_absorbed_message_uids": ["b" * UID_LEN]}
    _merge_anchor_into_user_message(target, anchor)
    assert target["content"] == "the real ask\n\n[todo snapshot]"
    assert target["message_uid"] == "a" * UID_LEN
    assert target["_absorbed_message_uids"] == ["b" * UID_LEN, "t" * UID_LEN]

    # Never-persisted scaffolding (no uid) absorbs nothing but still takes the anchor's identity.
    compressed = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"},
                  {"role": "user", "content": "[todo snapshot]"}]
    assert _insert_real_user_anchor(compressed, dict(anchor, _absorbed_message_uids=[])) == "merged"
    assert compressed[-1]["message_uid"] == "a" * UID_LEN
    assert "_absorbed_message_uids" not in compressed[-1]


def test_micro_compactions_adjacent_user_merge_records_the_witness():
    from types import SimpleNamespace

    from agent.micro_compaction import MicroCompactionMixin

    first = {"role": "user", "content": "one", "message_uid": "1" * UID_LEN}
    second = {"role": "user", "content": "two", "message_uid": "2" * UID_LEN}
    merged = MicroCompactionMixin._merge_adjacent_user_turns(SimpleNamespace(), [first, second])

    assert merged == [first] and first["content"] == "one\n\ntwo"
    assert first["message_uid"] == "1" * UID_LEN
    assert first["_absorbed_message_uids"] == ["2" * UID_LEN]


def test_an_assistant_merge_rewrite_persists_the_unioned_tool_call_uids(db):
    """Alternation repair unions two flushed assistant turns' tool calls; the survivor's row-addressed
    rewrite must carry the unioned uid map (it follows ``tool_calls``), and a restore must return it."""
    from agent.agent_runtime_helpers import repair_message_sequence

    sid = "20260928_120500_union"
    db.create_session(sid, "cli", model="test/model")
    agent = _make_agent(db, sid)
    agent._session_db_created = True
    survivor = {"role": "assistant", "content": "let me check"}
    caller = {"role": "assistant", "content": "",
              "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "t", "arguments": "{}"}}]}
    result = {"role": "tool", "content": "r", "tool_call_id": "call_2", "tool_name": "t"}
    messages = [{"role": "user", "content": "q"}, survivor, caller, result]
    agent._persist_user_message_idx = 0
    agent._persist_session(messages, conversation_history=None)  # both assistants flushed with a digest
    uids = dict(caller["_tool_call_uids"])
    assert set(uids) == {"call_2"} and "_tool_call_uids" not in survivor and result["_tool_call_uid"] == uids["call_2"]

    repair_message_sequence(agent, messages)
    assert messages == [messages[0], survivor, result] and survivor["_tool_call_uids"] == uids
    agent._persist_session(messages, conversation_history=None)

    assert survivor["_tool_call_uids"] == uids
    again = db.get_messages_as_conversation(sid, repair_alternation=False)
    stored = next(m for m in again if m["role"] == "assistant" and m["content"].startswith("let me check"))
    assert [tc["id"] for tc in stored["tool_calls"]] == ["call_2"]
    assert stored["_tool_call_uids"] == uids
    assert next(m for m in again if m["role"] == "tool")["_tool_call_uid"] == uids["call_2"]
