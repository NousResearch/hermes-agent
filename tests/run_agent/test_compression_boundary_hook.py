"""Test: the context engine is notified of a compression-boundary rollover.

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

from agent.conversation_compression import (
    finalize_context_engine_compression_notification,
)

class TestCompressionBoundaryHook:
    def _make_agent(self, session_db):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )
            # ROTATION fallback — pin in_place=False regardless of default (#38763).
            agent.compression_in_place = False
            return agent

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
            assert len(comp_calls) == 1

    def test_automatic_notification_follows_core_persistence(self):
        from hermes_state import SessionDB

        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.return_value = [
                {"role": "user", "content": "summary"}
            ]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            compressor.on_session_start.side_effect = (
                lambda *_args, **kwargs: events.append(
                    kwargs.get("boundary_reason")
                )
            )
            agent.context_compressor = compressor
            original_update = db.update_system_prompt

            def _record_update(*args, **kwargs):
                result = original_update(*args, **kwargs)
                events.append("persist")
                return result

            with patch.object(db, "update_system_prompt", side_effect=_record_update):
                agent._compress_context(
                    [{"role": "user", "content": "request"}],
                    "sys",
                    approx_tokens=100,
                )

            assert events == ["persist", "compression"]

    def test_failure_before_persistence_does_not_notify(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.side_effect = RuntimeError("synthetic compression failure")
            agent.context_compressor = compressor

            with pytest.raises(RuntimeError, match="synthetic compression failure"):
                agent._compress_context(
                    [{"role": "user", "content": "request"}],
                    "sys",
                    approx_tokens=100,
                )

            compressor.on_session_start.assert_not_called()

    def test_failure_during_persistence_does_not_notify(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.return_value = [
                {"role": "user", "content": "summary"}
            ]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor

            with patch.object(
                db,
                "update_system_prompt",
                side_effect=RuntimeError("synthetic commit failure"),
            ):
                agent._compress_context(
                    [{"role": "user", "content": "request"}],
                    "sys",
                    approx_tokens=100,
                )

            boundary_calls = [
                call
                for call in compressor.on_session_start.call_args_list
                if call.kwargs.get("boundary_reason") == "compression"
            ]
            assert boundary_calls == []

    def test_no_progress_does_not_notify(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.side_effect = lambda messages, **_kwargs: messages
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor
            messages = [{"role": "user", "content": "request"}]

            returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=100,
            )

            assert returned is messages
            compressor.on_session_start.assert_not_called()

    @pytest.mark.parametrize("committed", [True, False])
    def test_deferred_notification_finishes_exactly_once(self, committed):
        from hermes_state import SessionDB

        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.return_value = [
                {"role": "user", "content": "summary"}
            ]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            compressor.on_session_start.side_effect = (
                lambda *_args, **_kwargs: events.append("notify")
            )
            agent.context_compressor = compressor

            agent._compress_context(
                [{"role": "user", "content": "request"}],
                "sys",
                approx_tokens=100,
                force=True,
                defer_context_engine_notification=True,
            )

            assert events == []
            assert finalize_context_engine_compression_notification(
                agent, committed=committed
            ) is committed
            assert finalize_context_engine_compression_notification(
                agent, committed=True
            ) is False
            assert events == (["notify"] if committed else [])

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

    def test_plugin_noop_does_not_rotate_or_emit_boundary(self):
        """A plugin no-op must not create a compression child session.

        External context engines such as LCM can be above the host token
        threshold while having no eligible leaf backlog outside their protected
        fresh tail.  In that case they report a no-op and return the unchanged
        active context.  Treating that as a successful compression would split
        the session without creating summary/DAG state for the continuation.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)

            messages = [{"role": "user", "content": "tail-only pressure"}]
            compressor = MagicMock()
            # Return a *new* equal list, not the input object.  If we returned
            # the exact ``messages`` object, the identity no-op check
            # (``compressed is messages``) would short-circuit and the test
            # would pass without ever exercising the status-based no-op branch
            # this PR adds.  A distinct object forces the ``last_compression_status
            # == "noop"`` path to be the thing that prevents session rotation.
            compressor.compress.return_value = list(messages)
            compressor.compression_count = 0
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            compressor.last_compression_status = "noop"
            agent.context_compressor = compressor

            original_sid = agent.session_id
            compressed, _prompt = agent._compress_context(
                messages, "sys", approx_tokens=10_000
            )

            # Distinct object returned unchanged — proves the status branch
            # caught it rather than the identity check.
            assert compressed is not messages
            assert compressed == messages
            assert agent.session_id == original_sid
            comp_calls = [
                c for c in compressor.on_session_start.call_args_list
                if c.kwargs.get("boundary_reason") == "compression"
            ]
            assert not comp_calls

    def test_plugin_noop_private_field_fallback_does_not_rotate(self):
        """The legacy ``_last_compression_status`` private field is honored.

        Older context engines / compressors may not yet expose the public
        ``last_compression_status`` attribute.  The fix must fall back to the
        private ``_last_compression_status`` field so the no-op boundary skip
        still applies for those engines.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)

            messages = [{"role": "user", "content": "tail-only pressure"}]
            compressor = MagicMock()
            compressor.compress.return_value = list(messages)
            compressor.compression_count = 0
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            # Public attribute absent; only the legacy private field set.
            del compressor.last_compression_status
            compressor._last_compression_status = "noop"
            agent.context_compressor = compressor

            original_sid = agent.session_id
            compressed, _prompt = agent._compress_context(
                messages, "sys", approx_tokens=10_000
            )

            assert compressed is not messages
            assert compressed == messages
            assert agent.session_id == original_sid
            comp_calls = [
                c for c in compressor.on_session_start.call_args_list
                if c.kwargs.get("boundary_reason") == "compression"
            ]
            assert not comp_calls


class TestSessionCompressEvent:
    """The session:compress event_callback fires after a compression split."""

    def _make_agent(self, session_db, event_callback=None):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
                event_callback=event_callback,
            )
            # ROTATION fallback — pin in_place=False regardless of default (#38763).
            agent.compression_in_place = False
            return agent

    def _stub_compressor(self):
        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        return compressor

    def test_event_emitted_on_compression(self):
        from hermes_state import SessionDB

        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(
                db, event_callback=lambda et, ctx: events.append((et, ctx))
            )
            original_sid = agent.session_id
            agent.context_compressor = self._stub_compressor()

            agent._compress_context(
                [{"role": "user", "content": f"m{i}"} for i in range(10)],
                "sys",
                approx_tokens=10_000,
            )

            compress_events = [e for e in events if e[0] == "session:compress"]
            assert compress_events, f"session:compress not emitted, got {events!r}"
            _, ctx = compress_events[-1]
            assert ctx["session_id"] == agent.session_id
            assert ctx["old_session_id"] == original_sid
            assert ctx["compression_count"] == 1

    def test_no_callback_is_safe(self):
        """Compression must work when no event_callback is wired."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db, event_callback=None)
            agent.context_compressor = self._stub_compressor()
            compressed, _ = agent._compress_context(
                [{"role": "user", "content": "m"}], "sys", approx_tokens=100
            )
            assert compressed

    def test_callback_exception_does_not_break_compression(self):
        from hermes_state import SessionDB

        def _boom(event_type, ctx):
            raise RuntimeError("hook exploded")

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db, event_callback=_boom)
            original_sid = agent.session_id
            agent.context_compressor = self._stub_compressor()

            compressed, _ = agent._compress_context(
                [{"role": "user", "content": "m"}], "sys", approx_tokens=100
            )
            assert compressed
            assert agent.session_id != original_sid
