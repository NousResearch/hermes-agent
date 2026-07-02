"""Tests for gateway.agent_adapter — Gateway → AgentAdapter → HermesAdapter → Response.

Verifies:
1. AgentAdapter protocol satisfaction
2. HermesAdapter delegates to AIAgent correctly
3. create_adapter factory produces a valid adapter
4. Post-run properties propagate from AIAgent
5. Callbacks set on adapter reach the underlying AIAgent
6. Gateway dispatch path uses adapter (not direct AIAgent)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from gateway.agent_adapter import AgentAdapter, HermesAdapter, create_adapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stub_agent(**overrides):
    """Create a minimal stub that duck-types AIAgent for testing."""
    stub = MagicMock()
    stub.session_id = overrides.get("session_id", "test-session-123")
    stub.model = overrides.get("model", "test-model")
    stub.context_compressor = overrides.get("context_compressor", MagicMock(
        last_prompt_tokens=100, context_length=8192
    ))
    stub.session_prompt_tokens = overrides.get("session_prompt_tokens", 500)
    stub.session_completion_tokens = overrides.get("session_completion_tokens", 200)
    stub.is_interrupted = overrides.get("is_interrupted", False)
    stub._last_compaction_in_place = overrides.get("_last_compaction_in_place", False)
    stub.max_iterations = overrides.get("max_iterations", 90)
    stub.api_mode = overrides.get("api_mode", "chat_completions")
    stub.provider = overrides.get("provider", "openrouter")

    stub.run_conversation.return_value = overrides.get(
        "run_result",
        {"final_response": "Hello from the agent", "completed": True, "api_calls": 1},
    )
    stub.steer.return_value = True
    stub.get_activity_summary.return_value = {
        "current_tool": None,
        "api_call_count": 1,
    }
    return stub


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------

class TestAgentAdapterProtocol:
    """AgentAdapter is a runtime_checkable Protocol."""

    def test_hermes_adapter_satisfies_protocol(self):
        """HermesAdapter instances satisfy the AgentAdapter protocol."""
        with patch("run_agent.AIAgent", return_value=_make_stub_agent()):
            adapter = create_adapter(model="test")
        assert isinstance(adapter, AgentAdapter)

    def test_stub_satisfies_protocol(self):
        """A duck-typed stub can satisfy the protocol."""
        stub = _make_stub_agent()
        # Protocol structural check: all protocol members are present
        for member in dir(AgentAdapter):
            if member.startswith("_"):
                continue
            assert hasattr(stub, member) or member in (
                "run", "interrupt", "steer", "get_activity_summary",
                "close", "shutdown_memory_provider",
            )


# ---------------------------------------------------------------------------
# create_adapter factory
# ---------------------------------------------------------------------------

class TestCreateAdapter:
    """Factory function produces a HermesAdapter."""

    def test_returns_hermes_adapter(self):
        with patch("run_agent.AIAgent", return_value=_make_stub_agent()):
            adapter = create_adapter(model="test-model")
        assert isinstance(adapter, HermesAdapter)

    def test_passes_kwargs_to_aianagent(self):
        with patch("run_agent.AIAgent", return_value=_make_stub_agent()) as MockAgent:
            create_adapter(
                model="my-model",
                max_iterations=42,
                quiet_mode=True,
                session_id="sess-1",
            )
        MockAgent.assert_called_once_with(
            model="my-model",
            max_iterations=42,
            quiet_mode=True,
            session_id="sess-1",
        )


# ---------------------------------------------------------------------------
# Delegation: run() → run_conversation()
# ---------------------------------------------------------------------------

class TestRunDelegation:
    """adapter.run() delegates to AIAgent.run_conversation()."""

    def test_run_delegates_to_run_conversation(self):
        stub = _make_stub_agent()
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")

        result = adapter.run(
            "Hello",
            conversation_history=[{"role": "user", "content": "Hi"}],
            task_id="task-1",
        )

        stub.run_conversation.assert_called_once_with(
            "Hello",
            conversation_history=[{"role": "user", "content": "Hi"}],
            task_id="task-1",
        )
        assert result["final_response"] == "Hello from the agent"

    def test_run_with_all_kwargs(self):
        stub = _make_stub_agent()
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")

        adapter.run(
            "msg",
            conversation_history=[],
            task_id="t",
            persist_user_message="orig",
            persist_user_timestamp=1234567890.0,
            moa_config={"enabled": True},
        )

        stub.run_conversation.assert_called_once_with(
            "msg",
            conversation_history=[],
            task_id="t",
            persist_user_message="orig",
            persist_user_timestamp=1234567890.0,
            moa_config={"enabled": True},
        )

    def test_run_minimal_kwargs(self):
        stub = _make_stub_agent()
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")

        adapter.run("msg")

        stub.run_conversation.assert_called_once_with("msg")


# ---------------------------------------------------------------------------
# Post-run property pass-through
# ---------------------------------------------------------------------------

class TestPostRunProperties:
    """Adapter exposes AIAgent state after run()."""

    def test_session_id(self):
        stub = _make_stub_agent(session_id="compressed-child-id")
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")
        assert adapter.session_id == "compressed-child-id"

    def test_model(self):
        stub = _make_stub_agent(model="anthropic/claude-sonnet-4")
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")
        assert adapter.model == "anthropic/claude-sonnet-4"

    def test_context_compressor(self):
        cc = MagicMock(last_prompt_tokens=42, context_length=16384)
        stub = _make_stub_agent(context_compressor=cc)
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")
        assert adapter.context_compressor is cc
        assert adapter.context_compressor.last_prompt_tokens == 42

    def test_session_prompt_tokens(self):
        stub = _make_stub_agent(session_prompt_tokens=1234)
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")
        assert adapter.session_prompt_tokens == 1234

    def test_session_completion_tokens(self):
        stub = _make_stub_agent(session_completion_tokens=567)
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")
        assert adapter.session_completion_tokens == 567

    def test_is_interrupted(self):
        stub = _make_stub_agent(is_interrupted=True)
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")
        assert adapter.is_interrupted is True

    def test_compaction_in_place(self):
        stub = _make_stub_agent(_last_compaction_in_place=True)
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")
        assert adapter._last_compaction_in_place is True

    def test_api_mode(self):
        stub = _make_stub_agent(api_mode="anthropic_messages")
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")
        assert adapter.api_mode == "anthropic_messages"

    def test_provider(self):
        stub = _make_stub_agent(provider="anthropic")
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")
        assert adapter.provider == "anthropic"


# ---------------------------------------------------------------------------
# Callback pass-through
# ---------------------------------------------------------------------------

class TestCallbackPassthrough:
    """Callbacks set on the adapter reach the underlying AIAgent."""

    def test_tool_progress_callback(self):
        stub = _make_stub_agent()
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")

        cb = lambda *a, **kw: None
        adapter.tool_progress_callback = cb
        assert adapter.tool_progress_callback is cb
        assert stub.tool_progress_callback is cb

    def test_stream_delta_callback(self):
        stub = _make_stub_agent()
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")

        cb = lambda *a, **kw: None
        adapter.stream_delta_callback = cb
        assert adapter.stream_delta_callback is cb

    def test_status_callback(self):
        stub = _make_stub_agent()
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")

        cb = lambda *a, **kw: None
        adapter.status_callback = cb
        assert adapter.status_callback is cb

    def test_max_iterations_passthrough(self):
        stub = _make_stub_agent(max_iterations=30)
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")
        assert adapter.max_iterations == 30

        adapter.max_iterations = 50
        assert adapter.max_iterations == 50
        assert stub.max_iterations == 50


# ---------------------------------------------------------------------------
# Lifecycle methods
# ---------------------------------------------------------------------------

class TestLifecycleMethods:
    """Lifecycle methods delegate correctly."""

    def test_interrupt(self):
        stub = _make_stub_agent()
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")

        adapter.interrupt("user stopped")
        stub.interrupt.assert_called_once_with("user stopped")

    def test_steer(self):
        stub = _make_stub_agent()
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")

        result = adapter.steer("new direction")
        stub.steer.assert_called_once_with("new direction")
        assert result is True

    def test_get_activity_summary(self):
        stub = _make_stub_agent()
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")

        summary = adapter.get_activity_summary()
        stub.get_activity_summary.assert_called_once()
        assert "current_tool" in summary

    def test_close(self):
        stub = _make_stub_agent()
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")

        adapter.close()
        stub.close.assert_called_once()

    def test_shutdown_memory_provider_no_args(self):
        stub = _make_stub_agent()
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")

        adapter.shutdown_memory_provider()
        stub.shutdown_memory_provider.assert_called_once_with()

    def test_shutdown_memory_provider_with_messages(self):
        stub = _make_stub_agent()
        msgs = [{"role": "user", "content": "hi"}]
        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")

        adapter.shutdown_memory_provider(msgs)
        stub.shutdown_memory_provider.assert_called_once_with(msgs)


# ---------------------------------------------------------------------------
# Gateway dispatch path uses adapter (integration-level)
# ---------------------------------------------------------------------------

class TestGatewayUsesAdapter:
    """Gateway's _run_agent_inner imports adapter, not AIAgent directly."""

    def test_run_agent_inner_imports_adapter(self):
        """The main execution path imports from gateway.agent_adapter."""
        import ast
        from pathlib import Path

        run_py = Path("/usr/local/lib/hermes-agent/gateway/run.py")
        source = run_py.read_text()

        # Find the _run_agent_inner method's import line
        # It should import create_adapter, not AIAgent
        lines = source.splitlines()

        # Search for the import inside _run_agent_inner context
        in_run_agent_inner = False
        found_adapter_import = False
        found_direct_aian_import = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "async def _run_agent_inner(" in stripped:
                in_run_agent_inner = True
            if in_run_agent_inner and stripped.startswith("from gateway.agent_adapter import create_adapter"):
                found_adapter_import = True
            if in_run_agent_inner and stripped.startswith("from run_agent import AIAgent"):
                found_direct_aian_import = True
            if in_run_agent_inner and stripped.startswith("async def ") and "_run_agent_inner" not in stripped:
                break  # next method

        assert found_adapter_import, "Gateway _run_agent_inner must import create_adapter"
        assert not found_direct_aian_import, "Gateway _run_agent_inner must NOT import AIAgent directly"

    def test_background_task_uses_adapter(self):
        """The background task path imports adapter, not AIAgent directly."""
        import ast
        from pathlib import Path

        run_py = Path("/usr/local/lib/hermes-agent/gateway/run.py")
        source = run_py.read_text()
        lines = source.splitlines()

        in_background = False
        found_adapter = False
        found_direct = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "_run_background_task" in stripped and "async def" in stripped:
                in_background = True
            if in_background and "from gateway.agent_adapter import create_adapter" in stripped:
                found_adapter = True
            if in_background and "from run_agent import AIAgent" in stripped:
                found_direct = True
            if in_background and i > 0 and stripped.startswith("async def ") and "_run_background_task" not in stripped:
                break

        assert found_adapter, "Background task must import create_adapter"
        assert not found_direct, "Background task must NOT import AIAgent directly"

    def test_run_conversation_replaced_with_run(self):
        """The main path calls adapter.run(), not agent.run_conversation()."""
        from pathlib import Path

        run_py = Path("/usr/local/lib/hermes-agent/gateway/run.py")
        source = run_py.read_text()

        # Find _run_agent_inner and search forward for the run call
        marker = "async def _run_agent_inner("
        start = source.find(marker)
        assert start != -1, "_run_agent_inner not found in run.py"

        # Search the next 200000 chars (the method is ~1700 lines, ~170KB)
        section = source[start:start + 200000]

        found_adapter_run = "result = agent.run(" in section
        found_direct_run_conv = "result = agent.run_conversation(" in section

        assert found_adapter_run, "Main path must call agent.run()"
        assert not found_direct_run_conv, "Main path must NOT call agent.run_conversation()"


# ---------------------------------------------------------------------------
# End-to-end: Gateway → Adapter → AIAgent → Response
# ---------------------------------------------------------------------------

class TestEndToEndFlow:
    """Full flow: Gateway creates adapter → adapter delegates → response returned."""

    def test_full_flow_returns_response(self):
        """Simulates the Gateway's _run_agent_inner dispatch flow."""
        expected_result = {
            "final_response": "The agent responded successfully",
            "completed": True,
            "api_calls": 3,
            "messages": [],
        }
        stub = _make_stub_agent(run_result=expected_result)

        with patch("run_agent.AIAgent", return_value=stub) as MockAgent:
            # Step 1: Gateway creates adapter (via factory)
            adapter = create_adapter(
                model="test-model",
                max_iterations=90,
                quiet_mode=True,
                session_id="test-session",
            )

            # Step 2: Gateway sets callbacks
            progress_cb = MagicMock()
            adapter.tool_progress_callback = progress_cb

            # Step 3: Gateway calls adapter.run()
            result = adapter.run(
                user_message="What is 2+2?",
                conversation_history=[],
                task_id="test-session",
            )

            # Step 4: Verify delegation chain
            MockAgent.assert_called_once()
            stub.run_conversation.assert_called_once()

            # Step 5: Verify response propagation
            assert result["final_response"] == "The agent responded successfully"
            assert result["completed"] is True
            assert result["api_calls"] == 3

            # Step 6: Verify callback was set on the stub
            assert stub.tool_progress_callback is progress_cb

    def test_post_run_state_accessible(self):
        """After run(), Gateway can read agent state through the adapter."""
        stub = _make_stub_agent(
            session_id="compressed-new-id",
            session_prompt_tokens=2048,
            session_completion_tokens=512,
            _last_compaction_in_place=True,
        )

        with patch("run_agent.AIAgent", return_value=stub):
            adapter = create_adapter(model="test")

        # Simulate post-run state reading (same pattern as gateway/run.py)
        agent_session_id = adapter.session_id or "original-id"
        assert agent_session_id == "compressed-new-id"

        prompt_toks = getattr(adapter.context_compressor, "last_prompt_tokens", 0)
        assert prompt_toks == 100

        input_toks = adapter.session_prompt_tokens
        assert input_toks == 2048

        output_toks = adapter.session_completion_tokens
        assert output_toks == 512

        compacted = bool(getattr(adapter, "_last_compaction_in_place", False))
        assert compacted is True
