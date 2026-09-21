"""Regression: a detached background-review fork must never own a compression pass (#118438).

Production incident (issue #118438): a background memory/skill review fork replaying a long
snapshot crossed the compression threshold inside its OWN turn and started a full LLM summary
pass. When the next live user turn arrived, ``cancel_background_review_for_live_turn``
hard-interrupted the fork (``tool_reason="background review superseded"``), discarding the
in-flight summary after minutes of streaming (``commit_status="aborted"``,
``failure_class="explicit_interrupt"``). The next turn's preflight then restarted the same
compression from zero — with the default 600s ceiling one "Summarizing thread" cycle can burn
10+ minutes and produce nothing, and the attempt → abort → cooldown → re-fire loop repeats.

Ownership boundary: compression owns the conversation lifecycle. A review fork never runs a
compression pass at all; the fork's replayed snapshot stays bounded by the aggregate
input-token budget (``_review_input_budget_exhausted``) and the deterministic tool-result
prune, neither of which needs an LLM call. Foreground priority is unchanged — the fix removes
the discardable work, not the supersede.

These tests drive the REAL ``_run_review_in_thread`` + ``run_conversation`` path (the fork is
built by the real fork-construction code) with only the compressor's LLM call stubbed, and the
real ``turn_context_compaction._preflight_compression`` for the live-agent guard.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def _build_parent_agent(db: SessionDB, session_id: str):
    """Real AIAgent pinned to ``session_id`` (mirrors the #93057 harness)."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._compression_feasibility_checked = True
    return agent


def _tool_response(prompt_tokens: int) -> SimpleNamespace:
    message = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="web_search", arguments='{"query": "x"}'),
            )
        ],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="test/model",
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=1, total_tokens=prompt_tokens + 1
        ),
    )


def _final_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant",
                    content="review complete",
                    tool_calls=None,
                    reasoning_content=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        model="test/model",
    )


def _drive_review(parent, snapshot, captured):
    """Run the REAL review-thread driver; inside the fork's real ``run_conversation`` stub
    only the compressor's LLM summary call and the provider client. Returns the review result."""
    import agent.background_review as br
    from run_agent import AIAgent

    real_run_conversation = AIAgent.run_conversation

    def _run_review_with_stubs(self, *args, **kwargs):
        captured["marker"] = getattr(self, "_review_fork_compression_disallowed", "missing")
        captured["compression_enabled"] = self.compression_enabled
        captured["input_budget"] = getattr(self, "_review_input_token_budget", "missing")
        # Threshold crossing is real (1 < any pressure); only the summary LLM call is stubbed.
        self.context_compressor.threshold_tokens = 1
        self.context_compressor.protect_first_n = 1
        self.context_compressor.protect_last_n = 1
        self.context_compressor.compress = MagicMock(
            return_value=[
                {"role": "user", "content": "[CONTEXT COMPACTION] review summary"},
                {"role": "assistant", "content": "summary acknowledged"},
            ]
        )
        self.context_compressor.should_compress = MagicMock(return_value=True)
        self.context_compressor.should_compress_info = MagicMock(
            return_value=(True, "over threshold")
        )
        self.context_compressor.should_compress_preflight = MagicMock(return_value=True)
        self.context_compressor.should_defer_preflight_to_real_usage = MagicMock(
            return_value=False
        )
        self.context_compressor.get_active_compression_failure_cooldown = MagicMock(
            return_value=None
        )
        self.context_compressor.select_context = MagicMock(return_value=None)
        self._compression_feasibility_checked = True
        self.client = MagicMock()
        self.client.chat.completions.create.side_effect = [
            _tool_response(100),
            _final_response(),
        ]
        self._disable_streaming = True
        self._use_prompt_caching = False

        def _fake_execute_tool_calls(assistant_message, messages, *_args):
            tool_call = assistant_message.tool_calls[0]
            messages.append(
                {
                    "role": "tool",
                    "name": tool_call.function.name,
                    "tool_call_id": tool_call.id,
                    "content": "ok",
                }
            )

        self._execute_tool_calls = _fake_execute_tool_calls
        result = real_run_conversation(self, *args, **kwargs)
        captured["compression_calls"] = self.context_compressor.compress.call_count
        create = self.client.chat.completions.create
        captured["create_calls"] = create.call_count
        captured["outbound"] = [call.kwargs.get("messages") for call in create.call_args_list]
        return result

    with patch.object(AIAgent, "run_conversation", _run_review_with_stubs):
        return br._run_review_in_thread(parent, snapshot, "review this conversation")


def test_review_fork_never_owns_a_compression_pass(tmp_path: Path) -> None:
    """#118438 gap test: threshold-crossing pressure inside the fork's turn must NOT start
    a compression pass — the fork replays its snapshot uncompressed and the review still
    completes. On the pre-fix tree the post-tool / pre-API gates fire ``compress()`` on the
    fork (the pass a later supersede discards whole), so this assertion is RED there.
    """
    parent_sid = "REVIEW_FORK_COMPRESSION_DISALLOWED_118438"

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(parent_sid, source="discord")
    db.append_message(parent_sid, role="user", content="durable parent turn")
    durable_before = db.get_messages(parent_sid)
    parent = _build_parent_agent(db, parent_sid)
    parent._cached_system_prompt = "stable parent prompt"

    snapshot = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"review turn {i} " + "x" * 200,
        }
        for i in range(24)
    ]

    captured: dict = {}
    try:
        _drive_review(parent, snapshot, captured)

        assert captured["compression_calls"] == 0, (
            "#118438: the review fork owned a compression pass "
            f"(compress() called {captured['compression_calls']}x). A live turn superseding "
            "the fork discards that pass whole after minutes of streaming; the next turn's "
            "preflight restarts it from zero. A fork must never carry compression work — "
            "its snapshot stays bounded by the aggregate input budget and the deterministic "
            "tool-result prune."
        )
        # The review itself is unharmed: tool call + final answer, snapshot replayed whole.
        assert captured["create_calls"] == 2, (
            f"expected a 2-request review (tool call + final), got {captured['create_calls']}"
        )
        second_contents = [str(m.get("content", "")) for m in captured["outbound"][1]]
        assert any("review turn 12" in text for text in second_contents), (
            "the fork's follow-up request must replay its snapshot uncompressed — "
            "no summary pass may rewrite it"
        )
        assert not any("[CONTEXT COMPACTION]" in text for text in second_contents)
        # The fork's other bounds stay armed (this is what replaces the compaction bound).
        assert isinstance(captured["input_budget"], int) and captured["input_budget"] > 0
        # Parent transcript untouched.
        assert db.get_messages(parent_sid) == durable_before
    finally:
        db.close()


def test_detach_marks_fork_compression_disallowed(tmp_path: Path) -> None:
    """The detachment seam (``_detach_fork_compression``) must mark the fork so every
    automatic compression gate can recognize it for the fork's whole lifetime."""
    parent_sid = "REVIEW_FORK_COMPRESSION_MARKER_118438"

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(parent_sid, source="discord")
    parent = _build_parent_agent(db, parent_sid)
    parent._cached_system_prompt = "stable parent prompt"

    snapshot = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"review turn {i}"}
        for i in range(8)
    ]

    captured: dict = {}
    try:
        _drive_review(parent, snapshot, captured)
        assert captured["marker"] is True, (
            "#118438: a detached review fork must carry "
            "_review_fork_compression_disallowed=True for its whole lifetime "
            f"(got {captured['marker']!r})"
        )
    finally:
        db.close()


def test_live_agent_preflight_still_compresses_over_threshold(tmp_path: Path) -> None:
    """Behaviour guard (green before AND after): the gate is marker-based and dormant for
    normal agents — a LIVE turn over threshold still runs its preflight compression pass.
    Drives the real ``turn_context_compaction._preflight_compression``.
    """
    from agent import turn_context_compaction as tcc

    session_sid = "LIVE_PREFLIGHT_STILL_COMPRESSES_118438"
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_sid, source="cli")
    agent = _build_parent_agent(db, session_sid)

    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "x" * 200}
        for i in range(24)
    ]
    out = tcc.CompactionOutcome(
        messages=list(messages),
        active_system_prompt="sys",
        conversation_history=None,
        current_turn_user_idx=len(messages) - 1,
    )
    compressor = agent.context_compressor
    compressor.threshold_tokens = 1
    compressor.protect_first_n = 1
    compressor.protect_last_n = 1
    compressor.compress = MagicMock(
        return_value=[
            {"role": "user", "content": "[CONTEXT COMPACTION] live summary"},
            {"role": "assistant", "content": "ack"},
        ]
    )
    compressor.get_active_compression_failure_cooldown = MagicMock(return_value=None)
    compressor.should_defer_preflight_to_real_usage = MagicMock(return_value=False)
    agent._compress_context = MagicMock(return_value=(compressor.compress.return_value, "sys"))

    try:
        tcc._preflight_compression(agent, out, "sys", "hello", effective_task_id="t")
        assert agent._compress_context.call_count >= 1, (
            "live-turn preflight compression must still fire over threshold — "
            "the #118438 gate is marker-based and must stay dormant without the marker"
        )
    finally:
        db.close()
