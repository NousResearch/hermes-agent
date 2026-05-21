"""Tests: compression boundary hook and memory provider context chain.

Tests that:
1. The context engine is notified of a compression-boundary rollover.
2. on_pre_compress return value is forwarded to compress() as provider_context
   (issue #23367 — the chain was broken: return value was discarded).

When _compress_context rotates session_id (compression split), the active
context engine receives on_session_start(new_sid, boundary_reason="compression",
old_session_id=<old>). This lets plugin engines (e.g. hermes-lcm) preserve
DAG lineage across the split instead of treating it as a fresh /new.

See hermes-lcm#68: after Hermes compresses and mints a new physical session,
LCM was losing continuity (compression_count: 1, store_messages: 0,
dag_nodes: 0). With boundary_reason="compression" plugins can distinguish
this from a real user-initiated /new.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCompressionBoundaryHook:
    def _make_agent(self, session_db):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            return AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )

    def test_on_session_start_called_with_compression_boundary(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)

            # Stub the context compressor: we only need to observe the hook.
            compressor = MagicMock()
            compressor.compress.return_value = [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": "tail question"},
            ]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            # Avoid the summary-error warning path
            compressor._last_summary_error = None
            # MagicMock auto-creates truthy attrs; explicitly clear the abort
            # flag so the post-compress abort branch in
            # conversation_compression.py does not short-circuit before the
            # session-id rotation we are asserting on.
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor

            original_sid = agent.session_id
            messages = [
                {"role": "user", "content": f"m{i}"} for i in range(10)
            ]

            agent._compress_context(messages, "sys", approx_tokens=10_000)

            # Session_id rotated
            assert agent.session_id != original_sid, \
                "compression should rotate session_id when session_db is set"

            # Hook fired with boundary_reason="compression" and old_session_id
            calls = [
                c for c in compressor.on_session_start.call_args_list
            ]
            assert calls, "on_session_start was never called on the context engine"
            # Find the compression boundary call (there may be others from init)
            comp_calls = [
                c for c in calls
                if c.kwargs.get("boundary_reason") == "compression"
            ]
            assert comp_calls, (
                f"Expected an on_session_start call with "
                f"boundary_reason='compression', got {calls!r}"
            )
            call = comp_calls[-1]
            # Positional new session_id
            assert call.args and call.args[0] == agent.session_id, \
                f"Expected new session_id as first positional arg, got {call!r}"
            assert call.kwargs.get("old_session_id") == original_sid, \
                f"Expected old_session_id={original_sid!r}, got {call.kwargs!r}"

    def test_no_hook_when_no_session_db(self):
        """Without session_db, session_id does not rotate and the hook is not fired."""
        from run_agent import AIAgent
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=None,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )

        compressor = MagicMock()
        compressor.compress.return_value = [{"role": "user", "content": "x"}]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        agent.context_compressor = compressor

        original_sid = agent.session_id
        agent._compress_context([{"role": "user", "content": "m"}], "sys", approx_tokens=100)

        # No DB => no rotation => no compression-boundary hook
        assert agent.session_id == original_sid
        comp_calls = [
            c for c in compressor.on_session_start.call_args_list
            if c.kwargs.get("boundary_reason") == "compression"
        ]
        assert not comp_calls, (
            f"No compression hook should fire without session_db rotation, "
            f"got {comp_calls!r}"
        )

    def test_hook_failure_does_not_break_compression(self):
        """If the context engine raises from on_session_start, compression still completes."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)

            compressor = MagicMock()
            compressor.compress.return_value = [{"role": "user", "content": "summary"}]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False

            # Raise only on the compression-boundary call, not on earlier calls.
            def _raise_on_compression(*args, **kwargs):
                if kwargs.get("boundary_reason") == "compression":
                    raise RuntimeError("plugin exploded")
                return None
            compressor.on_session_start.side_effect = _raise_on_compression
            agent.context_compressor = compressor

            original_sid = agent.session_id

            # Must not raise
            compressed, _prompt = agent._compress_context(
                [{"role": "user", "content": "m"}], "sys", approx_tokens=100
            )
            assert compressed
            assert agent.session_id != original_sid


class TestMemoryProviderContextChain:
    """Regression tests for issue #23367.

    on_pre_compress() return value was silently discarded in
    conversation_compression.py — it was never forwarded to compress() as
    provider_context, so memory providers could never guide the compressor.
    """

    def _make_agent(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            return AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=None,
                session_id="test-session",
                skip_context_files=True,
                skip_memory=True,
            )

    def test_on_pre_compress_return_value_forwarded_to_compress(self):
        """provider_context from on_pre_compress must reach compress()."""
        agent = self._make_agent()

        # Inject a mock memory manager whose on_pre_compress returns a hint
        mock_mgr = MagicMock()
        mock_mgr.on_pre_compress.return_value = "User prefers PostgreSQL over MySQL"
        mock_mgr.build_system_prompt.return_value = ""
        agent._memory_manager = mock_mgr

        # Intercept compress() to capture what provider_context it receives
        received = {}
        compressor = MagicMock()

        def capture_compress(messages, **kwargs):
            received["provider_context"] = kwargs.get("provider_context", "")
            return [{"role": "user", "content": "summary"}]

        compressor.compress.side_effect = capture_compress
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        agent.context_compressor = compressor

        agent._compress_context(
            [{"role": "user", "content": "msg"}], "sys", approx_tokens=100
        )

        assert received.get("provider_context") == "User prefers PostgreSQL over MySQL", (
            "provider_context from on_pre_compress was not forwarded to compress(). "
            f"Got: {received.get('provider_context')!r}"
        )

    def test_empty_on_pre_compress_forwards_empty_string(self):
        """When on_pre_compress returns empty, compress() receives empty string."""
        agent = self._make_agent()

        mock_mgr = MagicMock()
        mock_mgr.on_pre_compress.return_value = ""
        mock_mgr.build_system_prompt.return_value = ""
        agent._memory_manager = mock_mgr

        received = {}
        compressor = MagicMock()

        def capture_compress(messages, **kwargs):
            received["provider_context"] = kwargs.get("provider_context", "MISSING")
            return [{"role": "user", "content": "summary"}]

        compressor.compress.side_effect = capture_compress
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        agent.context_compressor = compressor

        agent._compress_context(
            [{"role": "user", "content": "msg"}], "sys", approx_tokens=100
        )

        assert received.get("provider_context") == "", (
            f"Expected empty string, got: {received.get('provider_context')!r}"
        )

    def test_no_memory_manager_still_compresses(self):
        """Without a memory manager, compress() is still called normally."""
        agent = self._make_agent()
        agent._memory_manager = None

        received = {}
        compressor = MagicMock()

        def capture_compress(messages, **kwargs):
            received["called"] = True
            received["provider_context"] = kwargs.get("provider_context", "MISSING")
            return [{"role": "user", "content": "summary"}]

        compressor.compress.side_effect = capture_compress
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        agent.context_compressor = compressor

        agent._compress_context(
            [{"role": "user", "content": "msg"}], "sys", approx_tokens=100
        )

        assert received.get("called"), "compress() was never called"
        assert received.get("provider_context") == "", (
            f"Expected empty string without memory manager, got: {received.get('provider_context')!r}"
        )
