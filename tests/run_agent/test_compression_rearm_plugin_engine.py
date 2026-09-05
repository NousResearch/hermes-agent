"""Regression test for #103355: a successful compaction performed by a PLUGIN
context engine must arm the completed-compaction boundary latch so the next
provider-confirmed prompt count can re-arm the per-turn compression budget.

Plugin engines selected via ``context.engine`` become ``agent.context_compressor``
(agent/agent_init.py) but implement only the ContextEngine contract
(agent/context_engine.py). They never set the built-in compressor's private
progress latch (``_last_compression_made_progress``) and do not define
``record_completed_compaction``. The host therefore falls back to structural
proof of progress: a committed rewrite that differs from the input arms the
boundary latch (``_verify_compaction_cleared_threshold``) that the next real
prompt count consumes to reset ``compression_attempts``.

See https://github.com/NousResearch/hermes-agent/issues/103355
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _seed(db, sid, title, n=8):
    db.create_session(sid, "cli", model="test/model")
    db.set_session_title(sid, title)
    for i in range(n):
        db.append_message(
            session_id=sid,
            role="user" if i % 2 == 0 else "assistant",
            content=f"msg {i}",
        )


def _make_agent(session_db, session_id):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_in_place = True
    return agent


class TestPluginEngineCompressionRearm:
    def test_plugin_shaped_rewrite_arms_boundary_latch(self):
        """A committed rewrite by an engine that never sets the built-in private
        progress latch must still arm ``_verify_compaction_cleared_threshold`` —
        the latch the next provider prompt count consumes to re-arm the budget.

        Regression for #103355: plugin context engines (context.engine, e.g.
        hermes-lcm) cannot know about ``_last_compression_made_progress`` (it is
        not part of the ContextEngine contract), so every successful compaction
        used to burn an attempt and long turns died at ``max_attempts``.
        """
        from agent.conversation_compression import compress_context
        from hermes_state import SessionDB

        def _rewrite_compress(messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
            # A plugin engine rewrites the transcript WITHOUT touching the built-in
            # private latches.
            return [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"},
                {"role": "assistant", "content": "recent reply"},
            ]

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260904_120000_plugin01"
            _seed(db, sid, "plugin-engine")
            agent = _make_agent(db, sid)
            agent.context_compressor.compress = _rewrite_compress
            # The plugin contract has no record_completed_compaction: the boundary
            # must fall back to arming the latch directly (getattr on the class).
            with patch.object(ContextCompressor, "record_completed_compaction", None):
                messages = [{"role": "user", "content": f"m{i}"} for i in range(8)]
                compressed, _sp = compress_context(
                    agent, messages, approx_tokens=100_000, system_message="sys"
                )

        assert len(compressed) == 2
        assert agent.context_compressor._verify_compaction_cleared_threshold is True, (
            "a committed compaction must arm the completed-compaction latch even when "
            "the engine never set the built-in _last_compression_made_progress flag"
        )

    def test_noop_rewrite_does_not_arm(self):
        """A no-op (transcript unchanged) must NOT arm the latch: the structural
        fallback mirrors _candidate_rejected's no-op comparison, so it cannot
        turn ineffective compactions into budget re-arms."""
        from agent.conversation_compression import compress_context
        from hermes_state import SessionDB

        def _noop_compress(messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
            return list(messages)

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260904_120100_noop0001"
            _seed(db, sid, "noop")
            agent = _make_agent(db, sid)
            agent.context_compressor.compress = _noop_compress
            with patch.object(ContextCompressor, "record_completed_compaction", None):
                messages = [{"role": "user", "content": f"m{i}"} for i in range(8)]
                compressed, _sp = compress_context(
                    agent, messages, approx_tokens=100_000, system_message="sys"
                )

        # Unchanged input -> rejected before the boundary; nothing armed.
        assert len(compressed) == 8
        assert agent.context_compressor._verify_compaction_cleared_threshold is False
