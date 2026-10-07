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
)

# Trailing whitespace on purpose: a conforming engine may hand the row back stripped.
REPLY = "x" * 200 + " just-delivered verdict report the user is still reading.  "

SUMMARY = "[Recent Summary (d0, node 490)]\n\nEarlier turns summarized."
FOLLOWER = "Thanks, one more question."


def _assert_no_same_role_adjacency(messages):
    """Strict alternation, tool rows included: a tool-call assistant is an assistant row
    too, so never assistant;assistant (or user;user / tool;tool) anywhere in the view."""
    roles = [m.get("role") for m in messages]
    assert not any(a == b for a, b in zip(roles, roles[1:])), f"same-role adjacency in {roles}"


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
