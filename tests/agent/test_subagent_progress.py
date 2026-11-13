"""
Tests for subagent progress relay (issue #169).

Verifies that:
- KawaiiSpinner.print_above() works with and without active spinner
- _build_child_progress_callback handles CLI/gateway/no-display paths
- Thinking events are relayed correctly
- Parallel callbacks don't share state
"""

import io
import sys
import pytest
from unittest.mock import MagicMock

from agent.display import KawaiiSpinner
from tools.delegate_tool import _build_child_progress_callback


# =========================================================================
# KawaiiSpinner.print_above tests
# =========================================================================

class TestPrintAbove:
    """Tests for KawaiiSpinner.print_above method."""


    def test_print_above_with_spinner_running(self):
        """print_above should clear spinner line and print text."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("test")
        spinner._out = buf
        spinner.running = True  # Pretend spinner is running (don't start thread)
        
        spinner.print_above("tool line")
        output = buf.getvalue()
        assert "tool line" in output
        assert "\r" in output  # Should start with carriage return to clear spinner line

    def test_print_above_uses_captured_stdout(self):
        """print_above should use self._out, not sys.stdout.
        This ensures it works inside redirect_stdout(devnull)."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("test")
        spinner._out = buf
        
        # Simulate redirect_stdout(devnull)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            spinner.print_above("should go to buf")
        finally:
            sys.stdout = old_stdout
        
        assert "should go to buf" in buf.getvalue()


# =========================================================================
# _build_child_progress_callback tests
# =========================================================================

class TestBuildChildProgressCallback:
    """Tests for child progress callback builder."""




    def test_gateway_batched_progress(self):
        """Gateway path: each tool.started relays a subagent.tool event, and a
        subagent.progress summary fires once BATCH_SIZE tools accumulate."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb

        cb = _build_child_progress_callback(0, "test goal", parent)

        # Each tool.started relays a subagent.tool event immediately (per-tool relay).
        for i in range(4):
            cb("tool.started", f"tool_{i}", f"arg_{i}", {})
        # 4 per-tool relays so far, no batch summary yet (BATCH_SIZE=5)
        events = [c.args[0] for c in parent_cb.call_args_list]
        assert events == ["subagent.tool"] * 4

        # 5th call triggers another per-tool relay PLUS the batch-size summary
        cb("tool.started", "tool_4", "arg_4", {})
        events = [c.args[0] for c in parent_cb.call_args_list]
        assert events == ["subagent.tool"] * 5 + ["subagent.progress"]
        summary_call = parent_cb.call_args_list[-1]
        summary_text = summary_call.kwargs.get("preview") or summary_call.args[2]
        assert "tool_0" in summary_text
        assert "tool_4" in summary_text


    def test_parallel_callbacks_independent(self):
        """Each child's callback batches tool names independently."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb

        cb0 = _build_child_progress_callback(0, "goal a", parent)
        cb1 = _build_child_progress_callback(1, "goal b", parent)

        # 3 tool.started per child = 6 per-tool relays; neither should hit
        # the batch-size summary (batch size = 5, counted per-child).
        for i in range(3):
            cb0("tool.started", f"tool_{i}", f"a_{i}", {})
            cb1("tool.started", f"other_{i}", f"b_{i}", {})

        events = [c.args[0] for c in parent_cb.call_args_list]
        assert events.count("subagent.tool") == 6
        assert "subagent.progress" not in events

    def test_task_index_prefix_in_batch_mode(self):
        """Batch mode (task_count > 1) should show 1-indexed prefix for all tasks."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("delegating")
        spinner._out = buf
        spinner.running = True
        
        parent = MagicMock()
        parent._delegate_spinner = spinner
        parent.tool_progress_callback = None
        
        # task_index=0 in a batch of 3 → prefix "[1/3]" (batch slot; a
        # delegation batch tag is prepended when the id is known)
        cb0 = _build_child_progress_callback(0, "test goal", parent, task_count=3)
        cb0("tool.started", "web_search", "test", {})
        output = buf.getvalue()
        assert "[1/3]" in output

        # task_index=2 in a batch of 3 → prefix "[3]"
        buf.truncate(0)
        buf.seek(0)
        cb2 = _build_child_progress_callback(2, "test goal", parent, task_count=3)
        cb2("tool.started", "web_search", "test", {})
        output = buf.getvalue()
        assert "[3/3]" in output



# =========================================================================
# Integration: thinking callback in run_agent.py
# =========================================================================







# =========================================================================
# Gateway batch flush tests
# =========================================================================

class TestBatchFlush:
    """Tests for gateway batch flush on subagent completion."""

    def test_flush_sends_remaining_batch(self):
        """_flush should send a final subagent.progress summary of any unsent
        tool names in the batch (less than BATCH_SIZE)."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb

        cb = _build_child_progress_callback(0, "test goal", parent)

        # Send 3 tools (below batch size of 5) — each relays subagent.tool
        cb("tool.started", "web_search", "query1", {})
        cb("tool.started", "read_file", "file.txt", {})
        cb("tool.started", "write_file", "out.txt", {})
        events = [c.args[0] for c in parent_cb.call_args_list]
        assert events == ["subagent.tool"] * 3  # per-tool relays so far
        assert "subagent.progress" not in events  # no batch-size summary yet

        # Flush should send the remaining 3 as a summary
        cb._flush()
        events = [c.args[0] for c in parent_cb.call_args_list]
        assert events[-1] == "subagent.progress"
        summary_call = parent_cb.call_args_list[-1]
        summary_text = summary_call.kwargs.get("preview") or summary_call.args[2]
        assert "web_search" in summary_text
        assert "write_file" in summary_text

    def test_flush_noop_when_batch_empty(self):
        """_flush should not send anything when batch is empty."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb

        cb = _build_child_progress_callback(0, "test goal", parent)
        cb._flush()
        parent_cb.assert_not_called()

    def test_flush_noop_when_no_parent_callback(self):
        """_flush should not crash when there's no parent callback."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("test")
        spinner._out = buf
        spinner.running = True

        parent = MagicMock()
        parent._delegate_spinner = spinner
        parent.tool_progress_callback = None

        cb = _build_child_progress_callback(0, "test goal", parent)
        cb("tool.started", "web_search", "test", {})
        cb._flush()  # Should not crash


# =========================================================================
# Route visibility and dynamic relay identity tests
# =========================================================================

class TestRouteVisibility:
    """Tests for route change notifications, relay dynamic identity, and controls sync."""

    def test_relay_on_route_change_updates_model_and_emits_progress(self):
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb
        session_ref = {"session_id": "child-123"}

        cb = _build_child_progress_callback(
            0, "test goal", parent, subagent_id="sub-1", model="model-initial", session_ref=session_ref,
        )
        assert cb.model == "model-initial"

        # Trigger route change
        cb.on_route_change("model-fallback-1", "openrouter")

        assert cb.model == "model-fallback-1"
        assert cb.provider == "openrouter"
        assert session_ref.get("model") == "model-fallback-1"
        assert session_ref.get("provider") == "openrouter"

        # subagent.progress emitted immediately
        assert parent_cb.called
        last_call = parent_cb.call_args_list[-1]
        assert last_call.args[0] == "subagent.progress"
        assert last_call.kwargs.get("model") == "model-fallback-1"
        preview_text = last_call.kwargs.get("preview") or last_call.args[2] or ""
        assert "model-fallback-1" in preview_text

    def test_consecutive_fallbacks_before_tokens_covered(self):
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb
        session_ref = {"session_id": "child-123"}

        cb = _build_child_progress_callback(
            0, "test goal", parent, subagent_id="sub-2", model="model-A", session_ref=session_ref,
        )

        # Fallback 1: Model A -> Model B
        cb.on_route_change("model-B", "prov-B")
        call1 = parent_cb.call_args_list[-1]
        assert call1.args[0] == "subagent.progress"
        assert call1.kwargs.get("model") == "model-B"

        # Fallback 2: Model B -> Model C
        cb.on_route_change("model-C", "prov-C")
        call2 = parent_cb.call_args_list[-1]
        assert call2.args[0] == "subagent.progress"
        assert call2.kwargs.get("model") == "model-C"

        # Now child starts running tools: subsequent relay calls inherit coherent dynamic model
        cb("tool.started", "read_file", "path.py", {})
        tool_call = parent_cb.call_args_list[-1]
        assert tool_call.args[0] == "subagent.tool"
        assert tool_call.kwargs.get("model") == "model-C"

    def test_relay_on_route_change_updates_active_subagents_registry(self):
        from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
        parent = MagicMock()
        parent._delegate_spinner = None
        parent.tool_progress_callback = MagicMock()
        sid = "test-subagent-reg-sync"

        with _active_subagents_lock:
            _active_subagents[sid] = {"subagent_id": sid, "model": "old-model", "status": "running"}

        try:
            cb = _build_child_progress_callback(
                0, "test goal", parent, subagent_id=sid, model="old-model",
            )
            cb.on_route_change("new-model", "new-prov")

            with _active_subagents_lock:
                rec = _active_subagents.get(sid)
                assert rec is not None
                assert rec["model"] == "new-model"
                assert rec["provider"] == "new-prov"
        finally:
            with _active_subagents_lock:
                _active_subagents.pop(sid, None)

    def test_bind_route_entry_notifies_route_change_callback(self):
        from agent.route_binding import bind_route_entry
        agent = MagicMock()
        agent.model = "initial-model"
        agent.provider = "initial-provider"
        agent.base_url = "https://initial"
        agent.api_mode = "chat_completions"
        agent.context_compressor = MagicMock()
        agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
        agent._ensure_lmstudio_runtime_loaded = MagicMock()

        notified = []
        agent.route_change_callback = lambda model, provider: notified.append((model, provider))

        from unittest.mock import patch
        fake_client = MagicMock()
        fake_client.base_url = "https://fallback"

        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(fake_client, "new-model")), \
             patch("agent.client_lifecycle._swap_fallback_clients"), \
             patch("agent.agent_runtime_helpers.sync_credential_pool_entry_id"), \
             patch("agent.chat_completion_helpers._update_fallback_context_compressor"), \
             patch("agent.chat_completion_helpers._reresolve_fallback_reasoning_config"), \
             patch("agent.chat_completion_helpers._rescope_fallback_extra_body"), \
             patch("agent.chat_completion_helpers._rebind_fallback_credential_pool"):
            entry = {"model": "fallback-model", "provider": "openrouter"}
            result = bind_route_entry(agent, entry, "openrouter", "fallback-model")
            assert result is not None
            assert len(notified) == 1
            assert notified[0][0] == "fallback-model"
            assert notified[0][1] == "openrouter"

    def test_bind_route_entry_fails_open_if_callback_raises(self, caplog):
        from agent.route_binding import bind_route_entry
        import logging
        agent = MagicMock()
        agent.model = "initial-model"
        agent.provider = "initial-provider"
        agent.base_url = "https://initial"
        agent.api_mode = "chat_completions"
        agent.context_compressor = MagicMock()
        agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
        agent._ensure_lmstudio_runtime_loaded = MagicMock()

        def raising_cb(model, provider):
            raise RuntimeError("simulated route callback explosion")

        agent.route_change_callback = raising_cb

        fake_client = MagicMock()
        fake_client.base_url = "https://fallback"

        from unittest.mock import patch
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(fake_client, "new-model")), \
             patch("agent.client_lifecycle._swap_fallback_clients"), \
             patch("agent.agent_runtime_helpers.sync_credential_pool_entry_id"), \
             patch("agent.chat_completion_helpers._update_fallback_context_compressor"), \
             patch("agent.chat_completion_helpers._reresolve_fallback_reasoning_config"), \
             patch("agent.chat_completion_helpers._rescope_fallback_extra_body"), \
             patch("agent.chat_completion_helpers._rebind_fallback_credential_pool"), \
             caplog.at_level(logging.WARNING):
            entry = {"model": "fallback-model", "provider": "openrouter"}
            result = bind_route_entry(agent, entry, "openrouter", "fallback-model")
            assert result is not None
            assert any("Route change callback failed" in r.message for r in caplog.records)

    def test_bind_route_entry_observer_raising_typeerror_called_once(self, caplog):
        from agent.route_binding import bind_route_entry
        import logging
        agent = MagicMock()
        agent.model = "initial-model"
        agent.provider = "initial-provider"
        agent.base_url = "https://initial"
        agent.api_mode = "chat_completions"
        agent.context_compressor = MagicMock()
        agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
        agent._ensure_lmstudio_runtime_loaded = MagicMock()

        calls = []
        def raising_type_error(model, provider):
            calls.append((model, provider))
            raise TypeError("observer signature error or internal bug")

        agent.route_change_callback = raising_type_error

        fake_client = MagicMock()
        fake_client.base_url = "https://fallback"

        from unittest.mock import patch
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(fake_client, "new-model")), \
             patch("agent.client_lifecycle._swap_fallback_clients"), \
             patch("agent.agent_runtime_helpers.sync_credential_pool_entry_id"), \
             patch("agent.chat_completion_helpers._update_fallback_context_compressor"), \
             patch("agent.chat_completion_helpers._reresolve_fallback_reasoning_config"), \
             patch("agent.chat_completion_helpers._rescope_fallback_extra_body"), \
             patch("agent.chat_completion_helpers._rebind_fallback_credential_pool"), \
             caplog.at_level(logging.WARNING):
            entry = {"model": "fallback-model", "provider": "openrouter"}
            result = bind_route_entry(agent, entry, "openrouter", "fallback-model")
            assert result is not None
            # Must be called exactly ONCE without duplicate retry
            assert len(calls) == 1
            # Must log constant warning without leaking exception content
            assert any(r.message == "Route change callback failed" for r in caplog.records)

    def test_child_route_callback_wiring_through_bind_route_entry(self):
        from agent.route_binding import bind_route_entry
        from tools.delegate_tool import _wire_child_route_callback
        from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock

        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb

        child_session_ref = {"session_id": "child-wire-1"}
        sid = "sub-wire-1"
        with _active_subagents_lock:
            _active_subagents[sid] = {"subagent_id": sid, "model": "initial-model", "provider": "initial-provider"}

        try:
            progress_cb = _build_child_progress_callback(
                0, "test goal", parent, subagent_id=sid, model="initial-model", session_ref=child_session_ref,
                provider="initial-provider",
            )
            child = MagicMock()
            child.model = "initial-model"
            child.provider = "initial-provider"
            child.base_url = "https://initial"
            child.api_mode = "chat_completions"
            child.context_compressor = MagicMock()
            child._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
            child._ensure_lmstudio_runtime_loaded = MagicMock()

            _wire_child_route_callback(child, progress_cb, child_session_ref, sid)

            fake_client = MagicMock()
            fake_client.base_url = "https://fallback"

            from unittest.mock import patch
            with patch("agent.auxiliary_client.resolve_provider_client", return_value=(fake_client, "fallback-model")), \
                 patch("agent.client_lifecycle._swap_fallback_clients"), \
                 patch("agent.agent_runtime_helpers.sync_credential_pool_entry_id"), \
                 patch("agent.chat_completion_helpers._update_fallback_context_compressor"), \
                 patch("agent.chat_completion_helpers._reresolve_fallback_reasoning_config"), \
                 patch("agent.chat_completion_helpers._rescope_fallback_extra_body"), \
                 patch("agent.chat_completion_helpers._rebind_fallback_credential_pool"):
                entry = {"model": "fallback-model", "provider": "openrouter"}
                result = bind_route_entry(child, entry, "openrouter", "fallback-model")
                assert result is not None

            # Verify wiring updated progress callback, session_ref, and active_subagents
            assert progress_cb.model == "fallback-model"
            assert progress_cb.provider == "openrouter"
            assert child_session_ref["model"] == "fallback-model"
            assert child_session_ref["provider"] == "openrouter"
            with _active_subagents_lock:
                assert _active_subagents[sid]["model"] == "fallback-model"
                assert _active_subagents[sid]["provider"] == "openrouter"

            # Verify subagent.progress emitted with updated identity
            last_call = parent_cb.call_args_list[-1]
            assert last_call.args[0] == "subagent.progress"
            assert last_call.kwargs.get("model") == "fallback-model"
            preview_text = last_call.kwargs.get("preview") or last_call.args[2] or ""
            assert "fallback-model" in preview_text
            assert "openrouter" in preview_text
        finally:
            with _active_subagents_lock:
                _active_subagents.pop(sid, None)

    def test_route_change_provider_only(self):
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb
        session_ref = {"session_id": "child-prov-only", "model": "same-model", "provider": "initial-provider"}

        cb = _build_child_progress_callback(
            0, "test goal", parent, subagent_id="sub-prov", model="same-model", session_ref=session_ref,
            provider="initial-provider",
        )
        assert cb.model == "same-model"
        assert cb.provider == "initial-provider"

        # Provider-only change: model remains same-model, provider becomes new-provider
        cb.on_route_change(model=None, provider="new-provider")
        assert cb.model == "same-model"
        assert cb.provider == "new-provider"
        assert session_ref["model"] == "same-model"
        assert session_ref["provider"] == "new-provider"

        last_call = parent_cb.call_args_list[-1]
        assert last_call.args[0] == "subagent.progress"
        assert last_call.kwargs.get("model") == "same-model"
        preview = last_call.kwargs.get("preview") or last_call.args[2] or ""
        assert "same-model" in preview
        assert "new-provider" in preview




if __name__ == "__main__":
    pytest.main([__file__, "-v"])

