"""Compaction must not deactivate the just-delivered assistant reply (#118900).

Measured cause (Desktop macOS client, Hermes 0.21.4): a reply that had just
finished streaming was folded into the compaction summary by engine-driven
preflight maintenance and its row archived (active=0). The Desktop renders the
active set, so the reply vanished from the surface on next render while its
content stayed on disk.

The built-in compressor keeps the latest visible reply in the tail
(``_ensure_last_assistant_message_in_tail``), but plugin engines (e.g. LCM)
implement their own ``compress()`` without that guard — and nothing enforces
the invariant at the durable commit layer. ``compress_context`` is the one
choke point every engine and every path (threshold preflight, engine
maintenance, manual /compress; in-place and rotation) routes through, so the
guard lives there, next to ``_ensure_compressed_has_user_turn``.
"""

import pytest

from agent.context_compressor import SUMMARY_PREFIX
from agent.conversation_compression_reply_anchor import (
    _ensure_compressed_keeps_last_assistant_reply,
    _reply_insertion_index,
)
import itertools

# Trailing whitespace on purpose: a conforming engine may hand the row back stripped.
REPLY = "x" * 200 + " just-delivered verdict report the user is still reading.  "

SUMMARY = "[Recent Summary (d0, node 490)]\n\nEarlier turns summarized."
FOLLOWER = "Thanks, one more question."


def _assert_no_same_role_adjacency(messages):
    """Strict alternation, tool rows included: a tool-call assistant is an assistant row
    too, so never assistant;assistant (or user;user / tool;tool) anywhere in the view."""
    roles = [m.get("role") for m in messages]
    assert not any(a == b for a, b in itertools.pairwise(roles)), f"same-role adjacency in {roles}"


def _count_reply(messages):
    return sum(
        1 for m in messages
        if m.get("role") == "assistant" and (m.get("content") or "").strip() == REPLY.strip()
    )


@pytest.mark.parametrize(
    "kept_shape",
    [
        pytest.param(lambda: REPLY.strip(), id="stripped"),
        pytest.param(lambda: [{"type": "text", "text": REPLY}], id="parts_list"),
    ],
)
def test_normalized_twin_of_reply_counts_as_present(kept_shape):
    """A conforming engine that keeps the reply whitespace-stripped or with parts-list
    content is NOT a dropped reply: exactly one copy stays, and the reinsertion must never
    manufacture assistant;assistant adjacency (strict-alternation invariant)."""
    original = [
        {"role": "user", "content": "Audit the auth module."},
        {"role": "assistant", "content": "Short ack."},
        {"role": "user", "content": "Go deeper, full report."},
        {"role": "assistant", "content": REPLY, "finish_reason": "stop"},
        {"role": "user", "content": FOLLOWER},
    ]
    compressed = [
        {"role": "user", "content": SUMMARY},
        {"role": "assistant", "content": kept_shape()},
        {"role": "user", "content": FOLLOWER},
    ]

    inserted = _ensure_compressed_keeps_last_assistant_reply(original, compressed)

    assert inserted is None
    assert len(compressed) == 3
    assistant_rows = [m for m in compressed if m.get("role") == "assistant"]
    assert len(assistant_rows) == 1
    _assert_no_same_role_adjacency(compressed)



def _tool_chain(*call_ids):
    """Content-less tool-call assistants + their ``tool`` results, one round per id."""
    rows = []
    for call_id in call_ids:
        calls = [{"id": call_id, "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        rows += [
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "content": "ok", "tool_call_id": call_id},
        ]
    return rows


def test_later_text_twin_does_not_move_reply_past_tool_followers(caplog):
    """Regression for #138173: a dropped early text follower must not make the
    reply cross a later tool round whose ids still locate its original slot."""
    def tool_call(call_id):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        }

    def tool_result(call_id):
        return {"role": "tool", "tool_call_id": call_id, "content": f"result {call_id}"}

    reply = {"role": "assistant", "content": "Starting the work now."}
    original = [
        {"role": "user", "content": "start"}, reply, {"role": "user", "content": "continue"},
        tool_call("call_1"), tool_result("call_1"), {"role": "user", "content": "continue"},
        tool_call("call_2"), tool_result("call_2"),
    ]
    compressed = [
        {"role": "user", "content": SUMMARY}, tool_call("call_1"), tool_result("call_1"),
        {"role": "user", "content": "continue"}, tool_call("call_2"), tool_result("call_2"),
    ]

    assert _reply_insertion_index(original[2:], compressed) == 1
    with caplog.at_level("WARNING", logger="agent.conversation_compression_reply_anchor"):
        inserted = _ensure_compressed_keeps_last_assistant_reply(original, compressed)

    assert inserted is None
    assert any("strict role alternation" in record.getMessage() for record in caplog.records)
    assert [message["role"] for message in compressed] == ["user", "assistant", "tool", "user", "assistant", "tool"]


@pytest.mark.parametrize("shape", ["S1", "S2", "S3"])
@pytest.mark.parametrize(
    ("identity_key", "identity_side"),
    [(None, None), ("_row_id", "original"), ("_row_id", "compressed"),
     ("message_uid", "original"), ("message_uid", "compressed")],
)
def test_first_surviving_follower_retains_reply(shape, identity_key, identity_side):
    """S1: steer after tools; S2: two text followers; S3: metadata-stripped tools."""
    followers = [{"role": "user", "content": "please do X"}]
    if shape != "S2":
        followers += _tool_chain("call_1")
    if shape != "S3":
        followers += [{"role": "user", "content": "and Y"}]
    compressed = [{"role": "user", "content": SUMMARY}, *[dict(m) for m in followers]]
    if identity_key:
        rows = followers if identity_side == "original" else compressed[1:]
        for i, row in enumerate(rows, 1):
            row[identity_key] = i if identity_key == "_row_id" else f"uid-{i}"
    original = [
        {"role": "user", "content": "start the task " * 100},
        {"role": "assistant", "content": REPLY}, *followers,
    ]

    assert _reply_insertion_index(followers, compressed) == 1
    inserted = _ensure_compressed_keeps_last_assistant_reply(original, compressed)

    assert inserted is compressed[1]
    assert inserted["content"] == REPLY
    assert [m["content"] for m in compressed[2:]] == [m["content"] for m in followers]
    assert _count_reply(compressed) == 1


@pytest.mark.parametrize("identity_key", ["_row_id", "message_uid"])
@pytest.mark.parametrize("stripped_twin", [False, True])
def test_durable_follower_bounds_later_text_twin(identity_key, stripped_twin):
    """A durable user follower bounds a text twin even without any tool round."""
    identity = 7 if identity_key == "_row_id" else "follower-7"
    followers = [
        {"role": "user", "content": "continue"},
        {"role": "user", "content": "identified", identity_key: identity},
    ]
    if stripped_twin:
        followers = [{"role": "user", "content": "continue", identity_key: identity}]
    compressed = [
        {"role": "user", "content": SUMMARY}, dict(followers[-1]),
        {"role": "user", "content": "continue"},
    ]
    original = [
        {"role": "user", "content": "start " * 300},
        {"role": "assistant", "content": REPLY}, *followers,
    ]
    assert _reply_insertion_index(followers, compressed) == 1
    assert _ensure_compressed_keeps_last_assistant_reply(original, compressed) is compressed[1]


def test_first_text_follower_uses_last_normalized_occurrence():
    follower = {"role": "user", "content": [{"type": "text", "text": " continue  "}]}
    compressed = [
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "older reply"},
        {"role": "user", "content": "continue"},
        {"role": "user", "content": "and Y"},
    ]
    original = [
        {"role": "user", "content": "start " * 300},
        {"role": "assistant", "content": REPLY}, follower, compressed[-1],
    ]
    assert _reply_insertion_index(original[2:], compressed) == 2
    # The last occurrence is next to an older assistant: recovery must still yield.
    assert _ensure_compressed_keeps_last_assistant_reply(original, compressed) is None
    assert _count_reply(compressed) == 0


def test_incomparable_identity_kinds_allow_text_fallback():
    follower = {"role": "user", "content": "continue", "_row_id": 3}
    compressed = [
        {"role": "user", "content": SUMMARY},
        {"role": "user", "content": "continue", "message_uid": "uid-3"},
        *_tool_chain("call_1"),
    ]
    assert _reply_insertion_index([follower], compressed) == 1


@pytest.mark.parametrize("identity_key", ["_row_id", "message_uid"])
@pytest.mark.parametrize("kind", ["text", "call", "result"])
def test_conflicting_durable_identities_do_not_match(identity_key, kind):
    follower = {"role": "user", "content": "continue"}
    if kind != "text":
        follower = _tool_chain("call_1")[0 if kind == "call" else 1]
    follower[identity_key] = 1 if identity_key == "_row_id" else "uid-1"
    twin = dict(follower)
    twin[identity_key] = 2 if identity_key == "_row_id" else "uid-2"
    compressed = [{"role": "user", "content": SUMMARY}, twin,
                  {"role": "tool", "content": "unrelated", "tool_call_id": "other"}]
    assert _reply_insertion_index([follower], compressed) == len(compressed)


@pytest.mark.parametrize("kind", ["call", "result"])
@pytest.mark.parametrize("identity_key", ["_row_id", "message_uid"])
@pytest.mark.parametrize("side", ["original", "compressed"])
def test_one_sided_identity_preserves_tool_bound(kind, identity_key, side):
    follower = _tool_chain("call_1")[0 if kind == "call" else 1]
    kept = dict(follower)
    (follower if side == "original" else kept)[identity_key] = (
        1 if identity_key == "_row_id" else "uid-1"
    )
    compressed = [{"role": "user", "content": SUMMARY}, kept]
    assert _reply_insertion_index([follower], compressed) == 1


@pytest.mark.parametrize("kind", ["text", "call", "result"])
@pytest.mark.parametrize("shared_key", ["_row_id", "message_uid"])
@pytest.mark.parametrize("both_shared", [False, True])
def test_mixed_identity_conflict_does_not_reinsert_before_history(kind, shared_key, both_shared):
    follower = {"role": "user", "content": "continue"}
    if kind != "text":
        follower = _tool_chain("call_1")[0 if kind == "call" else 1]
    follower.update(_row_id=8, message_uid="new")
    older = dict(follower)
    older[shared_key] = 7 if shared_key == "_row_id" else "old"
    if not both_shared:
        older.pop("message_uid" if shared_key == "_row_id" else "_row_id")
    tail = {"role": "user", "content": "next task"}
    compressed = [{"role": "user", "content": SUMMARY}, older, *_tool_chain("older"), tail]
    original = [
        {"role": "user", "content": "old history " * 300},
        older, *_tool_chain("older"),
        {"role": "assistant", "content": REPLY}, follower, tail,
    ]

    inserted = _ensure_compressed_keeps_last_assistant_reply(original, compressed)

    assert inserted is compressed[-2]
    assert inserted["content"] == REPLY
    assert compressed[1] is older
    assert compressed[-1] is tail
    assert _count_reply(compressed) == 1


@pytest.mark.parametrize("identity", [{}, {"_row_id": 8}, {"message_uid": "new"},
                                      {"_row_id": 8, "message_uid": "new"}])
def test_safe_first_follower_survives_later_reused_tool_ids(identity):
    followers = [
        {"role": "user", "content": FOLLOWER, **identity}, *_tool_chain("call_0"),
        {"role": "user", "content": "steer"}, *_tool_chain("call_0"),
    ]
    compressed = [{"role": "user", "content": SUMMARY}, *[dict(m) for m in followers]]
    original = [
        {"role": "user", "content": "old history " * 300},
        {"role": "assistant", "content": REPLY}, *followers,
    ]

    inserted = _ensure_compressed_keeps_last_assistant_reply(original, compressed)

    assert inserted is compressed[1]
    assert inserted["content"] == REPLY
    assert compressed[2:] == followers
    assert _count_reply(compressed) == 1
    _assert_no_same_role_adjacency(compressed)


@pytest.mark.parametrize("identity_keys", [("_row_id",), ("message_uid",), ("_row_id", "message_uid")])
@pytest.mark.parametrize("identified_roles", [("assistant",), ("tool",), ("assistant", "tool")])
def test_durable_reused_tool_follower_bounds_dropped_text_twin(identity_keys, identified_roles, caplog):
    """Reused pairing ids must not hide a durable bound before a later text twin."""
    followers = [
        {"role": "user", "content": "continue"}, *_tool_chain("call_0"),
        {"role": "user", "content": "continue"}, *_tool_chain("call_0"),
    ]
    for i, row in enumerate(followers, 1):
        if row["role"] in identified_roles:
            row.update({key: i if key == "_row_id" else f"uid-{i}" for key in identity_keys})
    original = [
        {"role": "user", "content": "old history " * 300},
        {"role": "assistant", "content": REPLY}, *followers,
    ]
    compressed = [{"role": "user", "content": SUMMARY}, *[dict(m) for m in followers[1:]]]
    before = list(compressed)

    # A call locates index 1; its result alone locates index 2. Both seams
    # border the first assistant and must yield to the existing neighbor guard.
    expected_index = 1 if "assistant" in identified_roles else 2
    assert _reply_insertion_index(followers, compressed, reused_ids=frozenset({"call_0"})) == expected_index
    with caplog.at_level("WARNING", logger="agent.conversation_compression_reply_anchor"):
        inserted = _ensure_compressed_keeps_last_assistant_reply(original, compressed)

    assert inserted is None
    assert compressed == before
    assert _count_reply(compressed) == 0
    assert any("strict role alternation" in r.getMessage() for r in caplog.records)
    _assert_no_same_role_adjacency(compressed)


@pytest.mark.parametrize("identity_key", ["_row_id", "message_uid"])
@pytest.mark.parametrize("side", ["original", "compressed"])
def test_one_sided_identity_does_not_disambiguate_reused_tools(identity_key, side):
    followers = _tool_chain("call_0", "call_0")
    compressed = [{"role": "user", "content": SUMMARY}, *[dict(m) for m in followers]]
    for i, row in enumerate(followers if side == "original" else compressed[1:], 1):
        row[identity_key] = i if identity_key == "_row_id" else f"uid-{i}"
    original = [
        {"role": "user", "content": "old history " * 300},
        {"role": "assistant", "content": REPLY}, *followers,
    ]
    before = list(compressed)

    assert _reply_insertion_index(followers, compressed, reused_ids=frozenset({"call_0"})) is None
    assert _ensure_compressed_keeps_last_assistant_reply(original, compressed) is None
    assert compressed == before


def _seed(tmp_path, tail_rows, fold, follower):
    """Real SessionDB with the persisted history, a stub engine returning ``fold``, and the
    live list ending on the just-typed ``follower`` turn (unpersisted, like production)."""
    import os
    from unittest.mock import MagicMock, patch

    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "E2E_118900_REPLY"
    db.create_session(session_id, source="cli")
    rows = [
        ("user", "Audit the auth module."),
        ("assistant", "Short ack."),
        ("user", "Go deeper, full report."),
    ]
    for i in range(6):
        rows += [("user", f"bulk q {i} " + "q" * 80), ("assistant", f"bulk a {i} " + "a" * 80)]
    for role, content in [*rows, *tail_rows, ("assistant", REPLY)]:
        db.append_message(session_id, role, content)
    # Production live dicts carry _row_id + timestamp (session flush stamps both).
    messages = [
        *db.get_messages_as_conversation(session_id, include_row_ids=True),
        {"role": "user", "content": follower},
    ]
    assert isinstance(messages[-2].get("_row_id"), int) and messages[-2].get("timestamp") is not None

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform="telegram",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    engine = MagicMock()
    engine.compress.return_value = fold
    engine.compression_count = 1
    engine.last_prompt_tokens = 0
    engine.last_completion_tokens = 0
    engine._last_summary_error = None
    engine._last_compress_aborted = False
    engine._last_summary_auth_failure = False
    engine._last_aux_model_failure_model = None
    engine._last_aux_model_failure_error = None
    agent.context_compressor = engine
    return db, session_id, messages, agent


def _compress(agent, messages):
    from agent.conversation_compression import CompressionCommitFence

    agent._persist_user_message_idx = len(messages) - 1
    out_messages, _ = agent._compress_context(
        messages, "sys", approx_tokens=120_000,
        commit_fence=CompressionCommitFence(),
    )
    return out_messages


class TestEngineDropsReplyEndToEnd:
    """Full ``_compress_context`` with a stub engine that mimics the measured
    LCM fold (long just-delivered reply summarized away, next-turn user row
    kept): the reply row must stay ACTIVE in state.db after commit, while
    genuinely old rows are still archived (compression itself keeps working)."""

    @pytest.mark.parametrize("reused_ids", [False, True])
    def test_first_follower_order_is_committed_and_reloaded(self, tmp_path, reused_ids):
        from agent.conversation_compression import _message_text
        from hermes_state import SessionDB

        followers = [{"role": "user", "content": FOLLOWER}, *_tool_chain("call_1"),
                     {"role": "user", "content": "and Y"}]
        fold = [{"role": "user", "content": SUMMARY}, *[dict(m) for m in followers]]
        db, session_id, _, agent = _seed(tmp_path, [], fold, FOLLOWER)
        if reused_ids:
            followers = [{"role": "user", "content": FOLLOWER}, *_tool_chain("call_0"),
                         {"role": "user", "content": "steer"}, *_tool_chain("call_0"),
                         {"role": "user", "content": "and Y"}]
        for row in followers[:-1]:
            db.append_message(
                session_id, row["role"], row["content"],
                tool_calls=row.get("tool_calls"), tool_call_id=row.get("tool_call_id"),
            )
        messages = [*db.get_messages_as_conversation(session_id, include_row_ids=True), followers[-1]]
        if reused_ids:
            first_follower = messages[-len(followers)]
            first_follower["message_uid"] = "first-follower"
            fold = [{"role": "user", "content": SUMMARY}, dict(first_follower),
                    *[dict(m) for m in followers[1:]]]
            agent.context_compressor.compress.return_value = fold

        out = _compress(agent, messages)
        live = db.get_messages_as_conversation(session_id)
        db.close()
        reloaded_db = SessionDB(db_path=tmp_path / "state.db")
        try:
            reloaded = reloaded_db.get_messages_as_conversation(session_id)
        finally:
            reloaded_db.close()
        expected = [fold[0], {"role": "assistant", "content": REPLY}, *followers]
        for view in (out, live, reloaded):
            # Persistence normalizes whitespace and empty tool-call content;
            # compare visible rows in order, with tool pairing checked below.
            assert [(m["role"], _message_text(m).strip()) for m in view] == [
                (m["role"], _message_text(m).strip()) for m in expected
            ]
            assert _count_reply(view) == 1
            _assert_no_same_role_adjacency(view)
            call_id = "call_0" if reused_ids else "call_1"
            assert view[3]["tool_calls"][0]["id"] == view[4]["tool_call_id"] == call_id

    @pytest.mark.parametrize("identity_keys", [("_row_id",), ("message_uid",), ("_row_id", "message_uid")])
    def test_dropped_follower_reused_tool_order_is_committed_and_reloaded(self, tmp_path, identity_keys):
        from agent.conversation_compression import _message_text
        from hermes_state import SessionDB

        db, session_id, _, agent = _seed(tmp_path, [], [], "continue")
        followers = [
            {"role": "user", "content": "continue"}, *_tool_chain("call_0"),
            {"role": "user", "content": "continue"}, *_tool_chain("call_0"),
        ]
        followers[2]["content"] = "first round result"
        followers[5]["content"] = "second round result"
        for row in followers:
            db.append_message(
                session_id, row["role"], row["content"],
                tool_calls=row.get("tool_calls"), tool_call_id=row.get("tool_call_id"),
            )
        messages = db.get_messages_as_conversation(session_id, include_row_ids=True)
        kept = [dict(m) for m in messages[-len(followers) + 1:]]
        for row in kept:
            for key in ("_row_id", "message_uid"):
                if key not in identity_keys:
                    row.pop(key, None)
        fold = [{"role": "user", "content": SUMMARY}, *kept]
        agent.context_compressor.compress.return_value = fold
        # Snapshot the engine's order before the real compression path can mutate it.
        expected = [(m["role"], _message_text(m).strip()) for m in fold]

        try:
            out = _compress(agent, messages)
            live = db.get_messages_as_conversation(session_id, repair_alternation=False)
        finally:
            db.close()
        reloaded_db = SessionDB(db_path=tmp_path / "state.db")
        try:
            reloaded = reloaded_db.get_messages_as_conversation(session_id, repair_alternation=False)
        finally:
            reloaded_db.close()
        for view in (out, live, reloaded):
            assert [(m["role"], _message_text(m).strip()) for m in view] == expected
            assert _count_reply(view) == 0
            _assert_no_same_role_adjacency(view)
            assert [m["tool_call_id"] for m in view if m["role"] == "tool"] == ["call_0", "call_0"]

    def _compress_and_read(self, db, session_id, agent, messages, tail_role):
        out_messages = _compress(agent, messages)
        live = db.get_messages_as_conversation(session_id)
        live_contents = [m.get("content") for m in live]
        # Compression still folds: the superseded early rows are archived, not live.
        assert "Short ack." not in live_contents
        assert "Go deeper, full report." not in live_contents
        # The model-facing list and the durable active set both alternate strictly and
        # neither ends on an assistant row: the trailing turn is the one the loop is about
        # to answer, so the old reply must never read as its answer.
        for view in (out_messages, live):
            _assert_no_same_role_adjacency(view)
            assert view[-1].get("role") == tail_role
        return live, live_contents

    @pytest.mark.parametrize("todo_store", [False, True], ids=["dropped", "dropped_with_todo_store"])
    def test_reply_row_stays_active_after_commit(self, tmp_path, todo_store):
        """LCM-shaped fold: summary + next-turn tail only; the long reply gone. With an
        active todo store the snapshot fold rewrites the trailing user row, so the reply
        must be back in place BEFORE it runs."""
        fold = [{"role": "user", "content": SUMMARY}, {"role": "user", "content": FOLLOWER}]
        db, session_id, messages, agent = _seed(tmp_path, [], fold, FOLLOWER)
        if todo_store:
            agent._todo_store._items = [{"id": "t1", "content": "inspect image", "status": "pending"}]

        live, live_contents = self._compress_and_read(db, session_id, agent, messages, tail_role="user")

        assert _count_reply(live) == 1, "just-delivered reply left the active set after compaction"
        assert any(str(c).startswith(FOLLOWER) for c in live_contents)
        # Display projection (include_compacted): the reply renders once, not as archived
        # original + fresh twin.
        display = db.get_messages_as_conversation(
            session_id, include_ancestors=True, include_row_ids=True, include_compacted=True,
        )
        assert _count_reply(display) == 1

    @pytest.mark.parametrize(
        ("call_ids", "skip_reason"),
        [
            # Unique ids: the only slot is assistant-adjacent to the first call row.
            pytest.param(("call_1", "call_2"), "strict role alternation", id="unique_ids"),
            # A provider that reuses one id for every call (llama.cpp, see
            # ``_dedupe_tool_call_ids``): the ids cannot locate the slot, so the reply must
            # not be bound to a later round and land mid-chain.
            pytest.param(("call_0", "call_0", "call_0"), "reuse tool-call ids", id="constant_ids"),
            # Bridge ids ``call_id|item_id`` pair on the call half (coalesce_tool_call_id),
            # so distinct item ids behind one call id are still a reused id.
            pytest.param(
                ("call_0|fc_1", "call_0|fc_2", "call_0|fc_3"), "reuse tool-call ids", id="composite_ids",
            ),
        ],
    )
    def test_tool_chain_slot_is_skipped(self, tmp_path, call_ids, skip_reason, caplog):
        """Mid-turn compaction: the engine folded the new user turn into the summary but kept
        the current tool chain (content-less tool-call assistants + tool results). The
        recovery is SKIPPED (logged) rather than committed as assistant;assistant or placed
        inside the chain: strict alternation beats recovery."""
        # Canonical summary prefix so the user-turn anchor can tell the summary from a
        # real turn and restore the dropped user turn behind the reply.
        fold = [{"role": "user", "content": SUMMARY_PREFIX + " earlier turns."}, *_tool_chain(*call_ids)]
        db, session_id, messages, agent = _seed(tmp_path, [], fold, FOLLOWER)
        # The tool rounds already live after the new user turn; the engine keeps them.
        messages += fold[1:]

        with caplog.at_level("WARNING", logger="agent.conversation_compression_reply_anchor"):
            live, _ = self._compress_and_read(db, session_id, agent, messages, tail_role="tool")

        # No legal slot: the reply is not recovered, the chain is untouched, and the skip is
        # logged so the operator can find the reply on disk. (The user-turn anchor's own
        # placement inside the chain is pre-existing and not asserted.)
        assert _count_reply(live) == 0
        assert all(m.get("tool_calls") for m in live if m.get("role") == "assistant")
        assert len([m for m in live if m.get("tool_calls")]) == len(call_ids)
        assert any(
            "not reinserting" in r.getMessage() and skip_reason in r.getMessage() for r in caplog.records
        )

    def test_older_assistant_at_slot_is_not_displaced(self, tmp_path):
        """The engine kept an older assistant row ("noted") right where the reply would go,
        with "ok" twins around it: reinsertion would either create assistant;assistant or
        put the newest reply before an older one. Invariant beats recovery."""
        fold = [
            {"role": "user", "content": SUMMARY},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "ok"},
        ]
        db, session_id, messages, agent = _seed(
            tmp_path, [("user", "ok"), ("assistant", "noted"), ("user", "ok")], fold, "ok",
        )

        out_messages = _compress(agent, messages)

        live = db.get_messages_as_conversation(session_id)
        live_contents = [m.get("content") for m in live]
        for view in (out_messages, live):
            _assert_no_same_role_adjacency(view)
            assert view[-1].get("role") == "user"
        # The reply is not forced in beside "noted", and chronology holds ("noted" still
        # precedes the last "ok" that followed it).
        assert _count_reply(live) == 0
        assert live_contents.index("noted") < len(live_contents) - 1 == live_contents.index("ok")
