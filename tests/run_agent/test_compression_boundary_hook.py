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
from gateway.turn_lease import DurableTurnLease, TurnLeaseTimeoutError

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
            original_publish = db.publish_compression_child

            def _record_publish(*args, **kwargs):
                result = original_publish(*args, **kwargs)
                events.append("persist")
                return result

            with patch.object(
                db, "publish_compression_child", side_effect=_record_publish
            ):
                agent._compress_context(
                    [{"role": "user", "content": "request"}],
                    "sys",
                    approx_tokens=100,
                )

            assert events == ["persist", "compression"]

    def test_rotation_child_is_leased_before_publication_and_until_turn_release(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            owner_db = SessionDB(db_path=db_path)
            contender_db = SessionDB(db_path=db_path)
            lease = None
            successor = None
            agent = self._make_agent(owner_db)
            compressor = MagicMock()
            compressor.compress.return_value = [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": "tail question"},
            ]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor
            try:
                lease = DurableTurnLease.acquire(
                    owner_db,
                    agent.session_id,
                    owner_key="api:active-turn",
                    generation=1,
                    timeout=1,
                )
                assert lease is not None
                published_child = []
                original_publish = owner_db.publish_compression_child

                def _publish_only_after_child_lease(*args, **kwargs):
                    child_id = kwargs["child_session_id"]
                    assert owner_db.get_session_turn_lease_holder(child_id) == lease.holder
                    with pytest.raises(TurnLeaseTimeoutError):
                        DurableTurnLease.acquire(
                            contender_db,
                            child_id,
                            owner_key="api:contender",
                            generation=2,
                            timeout=0.05,
                        )
                    published_child.append(child_id)
                    return original_publish(*args, **kwargs)

                agent._durable_turn_lease_rebind = lease.rebind
                with patch.object(
                    owner_db,
                    "publish_compression_child",
                    side_effect=_publish_only_after_child_lease,
                ):
                    agent._compress_context(
                        [{"role": "user", "content": f"request {i}"} for i in range(10)],
                        "sys",
                        approx_tokens=10_000,
                    )

                assert published_child == [agent.session_id]
                with pytest.raises(TurnLeaseTimeoutError):
                    DurableTurnLease.acquire(
                        contender_db,
                        agent.session_id,
                        owner_key="api:still-running-contender",
                        generation=3,
                        timeout=0.05,
                    )
                lease.release()
                lease = None
                successor = DurableTurnLease.acquire(
                    contender_db,
                    agent.session_id,
                    owner_key="api:next-turn",
                    generation=4,
                    timeout=1,
                )
                assert successor is not None
            finally:
                if hasattr(agent, "_durable_turn_lease_rebind"):
                    delattr(agent, "_durable_turn_lease_rebind")
                if lease is not None:
                    lease.release()
                if successor is not None:
                    successor.release()
                owner_db.close()
                contender_db.close()

    def test_rotation_aborts_before_publication_when_child_lease_rebind_fails(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.return_value = [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": "tail question"},
            ]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor
            parent_id = agent.session_id
            agent._durable_turn_lease_rebind = lambda _child_id: False
            try:
                with patch.object(db, "publish_compression_child") as publish:
                    with pytest.raises(
                        RuntimeError,
                        match="active turn lease rejected compression child",
                    ):
                        agent._compress_context(
                            [
                                {"role": "user", "content": f"request {i}"}
                                for i in range(10)
                            ],
                            "sys",
                            approx_tokens=10_000,
                        )
                publish.assert_not_called()
                assert agent.session_id == parent_id
                assert db.find_live_compression_child(parent_id) is None
            finally:
                delattr(agent, "_durable_turn_lease_rebind")
                db.close()

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
