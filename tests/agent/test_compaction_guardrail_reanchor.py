"""Wiring regression for #109683.

Mid-turn REAL compaction (``AIAgent._compress_context`` returning a genuinely new
message list) replaces the visible history; a legitimate re-read afterward must not be
counted against a no-progress streak warmed against the PRE-compaction transcript. A
same-list no-op/aborted attempt must leave that streak untouched — the distinguishing
signal is result-list identity at the ``compression_facade.py`` choke point, not any
other side channel. Complements the cross-turn tracker in open PR #85352.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from hermes_state import SessionDB


def _build_agent(tmp_path: Path, session_id: str):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="cli")
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
    agent._cached_system_prompt = "sys"
    return agent


def _warm_guardrail_streaks(agent):
    """Warm an idempotent no-progress streak (read_file) plus an unrelated
    exact-failure streak (terminal) that must never be touched by a compaction reset."""
    guard = agent._tool_guardrails
    read_args = {"path": "notes.md"}
    guard.before_call("read_file", read_args)
    guard.after_call("read_file", read_args, "same content", failed=False)
    fail_args = {"command": "pytest -k thing"}
    guard.before_call("terminal", fail_args)
    guard.after_call("terminal", fail_args, "boom", failed=True)
    return read_args, fail_args


def test_real_compaction_resets_no_progress_streak_but_not_exact_failure_109683(tmp_path):
    from agent.conversation_compression import CompressionCommitFence
    from agent.tool_guardrails import ToolCallSignature

    agent = _build_agent(tmp_path, "COMPACTION_REANCHOR_REAL")
    read_args, fail_args = _warm_guardrail_streaks(agent)
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]

    def _fake_real_compaction(agent_arg, msgs, system_message, **kwargs):
        # Real compaction: a brand-new list object, never the caller's own list.
        return [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}], "sys"

    with patch("agent.conversation_compression.compress_context", side_effect=_fake_real_compaction):
        agent._compress_context(messages, "sys", commit_fence=CompressionCommitFence())

    read_sig = ToolCallSignature.from_call("read_file", read_args)
    fail_sig = ToolCallSignature.from_call("terminal", fail_args)
    guard = agent._tool_guardrails
    assert read_sig not in guard._no_progress, "no-progress streak must reset after a real compaction"
    assert guard._exact_failure_counts[fail_sig] == 1, "exact-failure counter must survive a real compaction"
    assert guard._same_tool_failure_counts["terminal"] == 1, "same-tool-failure counter must survive"


def test_noop_compaction_leaves_no_progress_streak_untouched_109683(tmp_path):
    from agent.conversation_compression import CompressionCommitFence
    from agent.tool_guardrails import ToolCallSignature

    agent = _build_agent(tmp_path, "COMPACTION_REANCHOR_NOOP")
    read_args, _fail_args = _warm_guardrail_streaks(agent)
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]

    def _fake_noop_compaction(agent_arg, msgs, system_message, **kwargs):
        # Aborted/no-op attempt: hands back the caller's OWN list, unchanged.
        return msgs, system_message

    with patch("agent.conversation_compression.compress_context", side_effect=_fake_noop_compaction):
        agent._compress_context(messages, "sys", commit_fence=CompressionCommitFence())

    read_sig = ToolCallSignature.from_call("read_file", read_args)
    guard = agent._tool_guardrails
    assert read_sig in guard._no_progress, "a same-list no-op must not reset the no-progress streak"
    assert guard._no_progress[read_sig][1] == 1
