"""Compaction must not deactivate the just-delivered assistant reply (#118900).

Measured cause (Desktop macOS client, Hermes 0.21.4): a reply that had just
finished streaming was folded into the compaction summary by engine-driven
preflight maintenance and its row archived (active=0). The Desktop renders the
active set, so the reply vanished from the surface on next render while its
content stayed on disk.

The built-in compressor keeps the latest visible reply in the tail
(``_ensure_last_assistant_message_in_tail``), but plugin engines (e.g. LCM)
implement their own ``compress()`` without that guard — and nothing enforces
the invariant at the durable commit layer. ``_commit_compaction`` is the one
choke point every engine and every path (threshold preflight, engine
maintenance, manual /compress; in-place and rotation) routes through, so the
guard lives here, mirroring ``_ensure_compressed_has_user_turn``.
"""

from agent.conversation_compression import (
    _ensure_compressed_keeps_last_assistant_reply,
)

REPLY = "x" * 200 + " just-delivered verdict report the user is still reading."

SUMMARY = "[Recent Summary (d0, node 490)]\n\nEarlier turns summarized."


def _transcript(*, with_follower=True):
    original = [
        {"role": "user", "content": "Audit the auth module."},
        {"role": "assistant", "content": "Short ack."},
        {"role": "user", "content": "Go deeper, full report."},
        {"role": "assistant", "content": REPLY, "finish_reason": "stop"},
    ]
    if with_follower:
        original.append({"role": "user", "content": "Thanks, one more question."})
    return original


def test_s1_dropped_reply_reinserted_before_surviving_follower():
    """S1 (the measured bug): engine folds the just-delivered long reply into
    the summary; the next-turn user message survives. The reply must come back
    ahead of that follower, not appended after it."""
    original = _transcript(with_follower=True)
    compressed = [
        {"role": "user", "content": SUMMARY},
        {"role": "user", "content": "Thanks, one more question."},
    ]

    outcome = _ensure_compressed_keeps_last_assistant_reply(original, compressed)

    assert outcome == "reinserted"
    kept = [m for m in compressed if m.get("role") == "assistant" and m.get("content") == REPLY]
    assert len(kept) == 1
    follower = next(m for m in compressed if m.get("content") == "Thanks, one more question.")
    assert compressed.index(kept[0]) < compressed.index(follower)
    roles = [m.get("role") for m in compressed]
    assert not any(a == b == "assistant" for a, b in zip(roles, roles[1:]))


def test_s2_reply_already_kept_is_noop():
    """S2: engine (built-in path) kept the reply in the tail — transcript must
    not be touched."""
    original = _transcript(with_follower=True)
    compressed = [
        {"role": "user", "content": SUMMARY},
        {"role": "assistant", "content": REPLY, "finish_reason": "stop"},
        {"role": "user", "content": "Thanks, one more question."},
    ]
    before = [dict(m) for m in compressed]

    outcome = _ensure_compressed_keeps_last_assistant_reply(original, compressed)

    assert outcome == "already_present"
    assert compressed == before


def test_s3_trailing_reply_without_follower_appends_at_end():
    """S3: compression ran before the next user turn; the reply is the last
    row. It must be appended back at the end."""
    original = _transcript(with_follower=False)
    compressed = [{"role": "user", "content": SUMMARY}]

    outcome = _ensure_compressed_keeps_last_assistant_reply(original, compressed)

    assert outcome == "reinserted"
    assert compressed[-1].get("role") == "assistant"
    assert compressed[-1].get("content") == REPLY


def test_s4_empty_reasoning_only_reply_not_reinserted():
    """S4: the #118755/#118738 family (text stranded in reasoning, empty
    content) is a different bug with its own fix — this guard stays out."""
    original = [
        {"role": "user", "content": "Think hard."},
        {"role": "assistant", "content": "", "reasoning": "long chain of thought"},
    ]
    compressed = [{"role": "user", "content": SUMMARY}]
    before = [dict(m) for m in compressed]

    outcome = _ensure_compressed_keeps_last_assistant_reply(original, compressed)

    assert outcome == "no_reply"
    assert compressed == before


def test_s5_tool_call_assistant_not_reinserted():
    """S5: an assistant row carrying tool_calls must keep its atomic tool
    group — reinserting it alone would corrupt pairing."""
    original = [
        {"role": "user", "content": "Run it."},
        {
            "role": "assistant",
            "content": "Working.",
            "tool_calls": [{"id": "call-1", "function": {"name": "terminal", "arguments": "{}"}}],
        },
    ]
    compressed = [{"role": "user", "content": SUMMARY}]
    before = [dict(m) for m in compressed]

    outcome = _ensure_compressed_keeps_last_assistant_reply(original, compressed)

    assert outcome == "no_reply"
    assert compressed == before


def test_s6_summary_rows_never_count_as_the_reply():
    """S6: a compaction handoff must not satisfy the presence check — the live
    reply text must survive verbatim, not merely inside a summary."""
    original = _transcript(with_follower=False)
    compressed = [{"role": "user", "content": SUMMARY + "\n\n" + REPLY}]

    outcome = _ensure_compressed_keeps_last_assistant_reply(original, compressed)

    assert outcome == "reinserted"
    assert sum(1 for m in compressed if m.get("role") == "assistant" and m.get("content") == REPLY) == 1


class TestEngineDropsReplyEndToEnd:
    """Full ``_compress_context`` with a stub engine that mimics the measured
    LCM fold (long just-delivered reply summarized away, next-turn user row
    kept): the reply row must stay ACTIVE in state.db after commit, while
    genuinely old rows are still archived (compression itself keeps working)."""

    def _agent(self, db, session_id):
        import os
        from unittest.mock import MagicMock, patch

        from agent.conversation_compression import CompressionCommitFence  # noqa: F401 (re-export check)
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
        # LCM-shaped fold: summary + next-turn tail only; the long reply gone.
        engine.compress.return_value = [
            {"role": "user", "content": SUMMARY},
            {"role": "user", "content": "Thanks, one more question."},
        ]
        engine.compression_count = 1
        engine.last_prompt_tokens = 0
        engine.last_completion_tokens = 0
        engine._last_summary_error = None
        engine._last_compress_aborted = False
        engine._last_summary_auth_failure = False
        engine._last_aux_model_failure_model = None
        engine._last_aux_model_failure_error = None
        agent.context_compressor = engine
        return agent

    def test_reply_row_stays_active_after_commit(self, tmp_path):
        from agent.conversation_compression import CompressionCommitFence
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "E2E_118900_REPLY"
        db.create_session(session_id, source="cli")
        for role, content in [
            ("user", "Audit the auth module."),
            ("assistant", "Short ack."),
            ("user", "Go deeper, full report."),
            ("assistant", REPLY),
        ]:
            db.append_message(session_id, role, content)

        messages = [*db.get_messages_as_conversation(session_id),
                    {"role": "user", "content": "Thanks, one more question."}]
        agent = self._agent(db, session_id)
        agent._persist_user_message_idx = len(messages) - 1

        agent._compress_context(
            messages, "sys", approx_tokens=120_000,
            commit_fence=CompressionCommitFence(),
        )

        live = [m.get("content") for m in db.get_messages_as_conversation(session_id)]
        assert REPLY in live, "just-delivered reply left the active set after compaction"
        assert "Thanks, one more question." in live
        # Compression still folds: the superseded early rows are archived, not live.
        assert "Short ack." not in live
        assert "Go deeper, full report." not in live
