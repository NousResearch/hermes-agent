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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


class _PluginEngineShim:
    """ContextEngine-contract compressor: update_from_response() never touches the
    built-in's private boundary latch (a real plugin engine cannot know about
    ``_verify_compaction_cleared_threshold`` — it is not part of the contract)."""

    threshold_tokens = 2000

    def __init__(self, armed: bool = False):
        # A successful plugin compaction arms the latch via the host else-leg
        # (conversation_compression boundary); nothing else here.
        self._verify_compaction_cleared_threshold = armed
        self.awaiting_real_usage_after_compression = False
        self._context_probed = False
        self._context_probe_persistable = False

    def update_from_response(self, usage):
        """Contract method only — deliberately does NOT clear the private latch."""

    def record_completed_compaction(self, **kwargs):
        """Plugins do not define this; presence here is only to mirror getattr shape."""


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


class TestPluginEngineLatchIsOneShot:
    """Review follow-up (strzhao on PR #103373): a PLUGIN engine's successful
    compaction arms the host's boundary latch, but nothing on the plugin side ever
    consumes it — the built-in clears it inside ``update_from_response`` while a
    plugin's contract method does not. Without a host-side consume the latch is
    STICKY: one successful compaction would re-arm the budget on every later
    below-threshold response, including after an ineffective attempt that should
    stay burnt (the anti-thrash backstop #11529 semantics)."""

    def _agent(self, compressor):
        agent = SimpleNamespace()
        agent.context_compressor = compressor
        agent.model = "test/model"
        agent.provider = "custom"
        agent.base_url = "http://local/v1"
        agent.api_mode = "chat_completions"
        agent.api_key = "test-key"
        agent.client = None
        agent.session_api_calls = 0
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0
        agent.session_input_tokens = 0
        agent.session_output_tokens = 0
        agent.session_cache_read_tokens = 0
        agent.session_cache_write_tokens = 0
        agent.session_reasoning_tokens = 0
        agent.session_estimated_cost_usd = 0.0
        agent.session_cost_status = None
        agent.session_cost_source = None
        agent._session_db = None
        agent.session_id = None
        agent.verbose_logging = False
        agent._usage_anchor = None
        agent._turn_base_usage_anchor = None
        agent._last_turn_usage = {}
        return agent

    def _response(self, prompt_tokens=500):
        return SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=50,
                total_tokens=prompt_tokens + 50,
                input_tokens=prompt_tokens,
                output_tokens=50,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            )
        )

    def test_plugin_latch_is_consumed_after_first_real_usage(self):
        """The armed latch must be one-shot for plugin engines: after the first
        provider-confirmed real usage consumes it, a LATER below-threshold
        response must NOT re-arm the budget from the stale latch (only a fresh
        successful compaction may arm it again)."""
        from agent.turn_usage import record_response_usage

        comp = _PluginEngineShim(armed=True)  # a compaction just succeeded
        agent = self._agent(comp)
        outcome = record_response_usage(
            agent, self._response(prompt_tokens=500), messages=[{"role": "user", "content": "x"}],
            api_call_count=1, api_duration=0.1, compression_attempts=1,
            max_compression_attempts=3,
        )
        # First response after the successful compaction: budget re-arms (below threshold).
        assert outcome.rearmed is True
        assert outcome.compression_attempts == 0
        # And the latch is now CONSUMED — one-shot, not sticky.
        assert comp._verify_compaction_cleared_threshold is False, (
            "host must consume the boundary latch after update_from_response for "
            "plugin engines (their contract method never clears it)"
        )

    def test_stale_latch_does_not_rearm_after_ineffective_attempt(self):
        """Sticky-latch regression: after the consuming response, an ineffective
        compaction (no progress -> latch NOT re-armed) must leave the budget burnt.
        With a sticky latch the next below-threshold response would wrongly re-arm."""
        from agent.turn_usage import record_response_usage

        comp = _PluginEngineShim(armed=True)
        agent = self._agent(comp)
        # First response consumes the armed latch and re-arms (attempts -> 0).
        record_response_usage(
            agent, self._response(prompt_tokens=500), messages=[{"role": "user", "content": "x"}],
            api_call_count=1, api_duration=0.1, compression_attempts=1,
            max_compression_attempts=3,
        )
        assert comp._verify_compaction_cleared_threshold is False
        # Ineffective compaction burns an attempt and arms nothing (latch stays False).
        outcome = record_response_usage(
            agent, self._response(prompt_tokens=500), messages=[{"role": "user", "content": "y"}],
            api_call_count=2, api_duration=0.1, compression_attempts=1,
            max_compression_attempts=3,
        )
        assert outcome.rearmed is False, (
            "stale latch must not re-arm the budget after an ineffective attempt "
            "(one-shot semantics, #11529)"
        )
        assert outcome.compression_attempts == 1  # attempt stays burnt

