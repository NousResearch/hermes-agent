"""Tests for Langfuse parent-child trace relationships."""
from __future__ import annotations
import json
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestParentTraceContext:
    """Test parent trace ID inheritance and propagation."""
    
    def test_parent_trace_id_storage(self):
        """AIAgent should store parent_trace_id and parent_observation_id."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            from run_agent import AIAgent
            
            agent = AIAgent(
                api_key="test-key",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                langfuse_enabled=True,
                parent_trace_id="parent-123",
                parent_observation_id="obs-456"
            )
            assert agent._parent_trace_id == "parent-123"
            assert agent._parent_observation_id == "obs-456"
        
    def test_current_trace_id_initialized_for_root(self):
        """Root agent should initialize a stable trace ID when Langfuse is enabled."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("run_agent.get_langfuse_client") as mock_get_client,
        ):
            from run_agent import AIAgent

            mock_langfuse = Mock()
            mock_langfuse.create_trace_id.return_value = "trace-init"
            mock_get_client.return_value = mock_langfuse
            
            agent = AIAgent(
                api_key="test-key",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                langfuse_enabled=True
            )
            assert agent._current_trace_id == "trace-init"
            assert agent._current_observation_id is None
        
    def test_capture_trace_context_from_response(self):
        """Should extract trace IDs from decorator context via Langfuse client."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("run_agent.get_langfuse_client") as mock_get_client,
        ):
            from run_agent import AIAgent
            
            # Mock Langfuse client
            mock_langfuse = Mock()
            mock_langfuse.create_trace_id.return_value = "trace-init"
            mock_langfuse.get_current_trace_id.return_value = "trace-abc"
            mock_langfuse.get_current_observation_id.return_value = "obs-xyz"
            mock_get_client.return_value = mock_langfuse
            
            agent = AIAgent(
                api_key="test-key",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                langfuse_enabled=True
            )
            
            # Response content doesn't matter with decorator approach
            mock_response = Mock()
            
            agent._capture_trace_context(mock_response)
            
            assert agent._current_trace_id == "trace-abc"
            assert agent._current_observation_id == "obs-xyz"
    
    def test_capture_trace_context_ignores_non_langfuse_response(self):
        """Should gracefully handle missing decorator context."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("run_agent.get_langfuse_client") as mock_get_client,
        ):
            from run_agent import AIAgent
            
            # Mock Langfuse client returning None (no active span)
            mock_langfuse = Mock()
            mock_langfuse.create_trace_id.return_value = "trace-init"
            mock_langfuse.get_current_trace_id.return_value = None
            mock_langfuse.get_current_observation_id.return_value = None
            mock_get_client.return_value = mock_langfuse
            
            agent = AIAgent(
                api_key="test-key",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                langfuse_enabled=True
            )
            
            mock_response = Mock()
            
            agent._capture_trace_context(mock_response)
            
            # Should remain unchanged when no decorator context
            assert agent._current_trace_id == "trace-init"
            assert agent._current_observation_id is None
    
    def test_capture_trace_context_when_langfuse_disabled(self):
        """Should not attempt to capture trace context when Langfuse disabled."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("run_agent.get_langfuse_client") as mock_get_client,
        ):
            from run_agent import AIAgent
            
            agent = AIAgent(
                api_key="test-key",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                langfuse_enabled=False
            )
            mock_response = Mock()
            
            agent._capture_trace_context(mock_response)
            
            # Should remain None since Langfuse is disabled
            assert agent._current_trace_id is None
            assert agent._current_observation_id is None
            mock_get_client.assert_not_called()

    def test_inject_observability_metadata_adds_trace_kwargs(self):
        """Should inject Langfuse trace kwargs and keep metadata."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                langfuse_enabled=True,
                parent_trace_id="parent-trace",
                parent_observation_id="parent-obs",
            )
            agent._langfuse_sampling = True
            agent._current_trace_id = "child-trace"

            result = agent._inject_observability_metadata({})

            # When parent context is provided, join the parent trace and chain
            # under the parent observation.
            assert result["trace_id"] == "parent-trace"
            assert result["parent_observation_id"] == "parent-obs"
            assert result["metadata"]["session_id"] == agent.session_id
            assert result["metadata"]["user_id"].startswith("hermes-agent/")
            assert "subagent" in result["metadata"]["tags"]
            assert result["metadata"]["is_subagent"] is True
            assert result["metadata"]["parent_trace_id"] == "parent-trace"
            assert result["metadata"]["parent_observation_id"] == "parent-obs"


class TestSubagentDelegation:
    """Test subagent delegation and trace inheritance."""
    
    def _create_parent_mock(self, *, langfuse_enabled=True, trace_id: str | None = "parent-trace-123",
                            observation_id: str | None = "parent-obs-456"):
        """Create a configured parent mock with common defaults."""
        parent = MagicMock()
        parent.base_url = "https://openrouter.ai/api/v1"
        parent.api_key = "parent-key"
        parent.provider = "openrouter"
        parent.api_mode = "chat_completions"
        parent.model = "anthropic/claude-sonnet-4"
        parent.platform = "cli"
        parent.providers_allowed = None
        parent.providers_ignored = None
        parent.providers_order = None
        parent.providers_sort = None
        parent._session_db = None
        parent._delegate_depth = 0
        parent._active_children = []
        parent._langfuse_enabled = langfuse_enabled
        parent._current_trace_id = trace_id
        parent._current_observation_id = observation_id
        parent.iteration_budget = MagicMock()
        parent.iteration_budget.remaining = 100
        return parent
    
    def test_delegate_envelope_contains_parent_trace_context(self):
        from tools.delegate_tool import delegate_task

        parent = self._create_parent_mock(
            langfuse_enabled=True,
            trace_id="parent-trace-123",
            observation_id="parent-obs-456",
        )

        captured = {}
        def _fake_spawn(envelope):
            captured["envelope"] = envelope
            return {"final_response": "done", "completed": True, "api_calls": 1, "messages": []}

        with (
            patch("tools.delegate_tool._spawn_subagent_process", side_effect=_fake_spawn),
            patch("tools.delegate_tool.get_langfuse_client") as mock_get_client,
        ):
            mock_langfuse = MagicMock()
            mock_span = MagicMock()
            mock_span.id = "delegate-span-123"
            mock_ctx = MagicMock()
            mock_ctx.__enter__.return_value = mock_span
            mock_ctx.__exit__.return_value = False
            mock_langfuse.start_as_current_span.return_value = mock_ctx
            mock_get_client.return_value = mock_langfuse

            json.loads(delegate_task(goal="Test task", parent_agent=parent))

        tc = captured["envelope"]["trace_context"]
        assert tc["parent_trace_id"] == "parent-trace-123"
        assert tc["parent_observation_id"] == "delegate-span-123"

    def test_delegate_envelope_no_parent_trace_context(self):
        from tools.delegate_tool import delegate_task

        parent = self._create_parent_mock(langfuse_enabled=True, trace_id=None, observation_id=None)

        captured = {}
        def _fake_spawn(envelope):
            captured["envelope"] = envelope
            return {"final_response": "done", "completed": True, "api_calls": 1, "messages": []}

        with patch("tools.delegate_tool._spawn_subagent_process", side_effect=_fake_spawn):
            json.loads(delegate_task(goal="Test task", parent_agent=parent))

        tc = captured["envelope"]["trace_context"]
        assert tc["parent_trace_id"] is None
        assert tc["parent_observation_id"] is None


class TestTraceIdPropagation:
    """Test trace ID propagation through conversation."""
    
    def test_trace_ids_extracted_after_api_call(self):
        """Trace IDs should be extracted from decorator context after call."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI") as mock_openai_class,
            patch("run_agent.get_langfuse_client") as mock_get_client,
        ):
            from run_agent import AIAgent
            
            # Mock Langfuse client
            mock_langfuse = MagicMock()
            mock_langfuse.create_trace_id.return_value = "trace-init"
            mock_langfuse.get_current_trace_id.return_value = "trace-from-api"
            mock_langfuse.get_current_observation_id.return_value = "obs-from-api"
            mock_get_client.return_value = mock_langfuse
            
            # Create mock client
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Hello"
            mock_response.choices[0].message.tool_calls = None
            mock_response.choices[0].finish_reason = "stop"
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            agent = AIAgent(
                api_key="test-key",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                langfuse_enabled=True,
            )
            agent.client = mock_client

            # Pretend we injected this trace_id for the call.
            agent._last_injected_trace_id = "trace-from-api"
            
            # Simulate _capture_trace_context
            agent._capture_trace_context(mock_response)
            
            # Verify trace IDs were captured from decorator context
            assert agent._current_trace_id == "trace-from-api"
            assert agent._current_observation_id == "obs-from-api"
    
    def test_child_generates_own_trace_id(self):
        """Child agent should generate its own trace IDs after instantiation."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("run_agent.get_langfuse_client") as mock_get_client,
        ):
            from run_agent import AIAgent
            
            # Mock Langfuse client
            mock_langfuse = MagicMock()
            mock_langfuse.create_trace_id.return_value = "trace-init"
            mock_langfuse.get_current_trace_id.return_value = "child-trace-123"
            mock_langfuse.get_current_observation_id.return_value = "child-obs-456"
            mock_get_client.return_value = mock_langfuse
            
            # Create child agent with parent context
            child = AIAgent(
                api_key="test-key",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                langfuse_enabled=True,
                parent_trace_id="parent-trace",
                parent_observation_id="parent-obs",
            )
            
            # Child starts with None current IDs
            assert child._current_trace_id is None
            assert child._current_observation_id is None
            
            # Simulate API call - response content doesn't matter with decorators
            mock_response = MagicMock()

            # Pretend we injected this trace_id for the call.
            child._last_injected_trace_id = "child-trace-123"
            
            child._capture_trace_context(mock_response)
            
            # Child now has its own trace IDs from decorator context
            assert child._current_trace_id == "child-trace-123"
            assert child._current_observation_id == "child-obs-456"
            
            # Parent context is preserved
            assert child._parent_trace_id == "parent-trace"
            assert child._parent_observation_id == "parent-obs"

    def test_capture_ignores_mismatched_trace_id(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("run_agent.get_langfuse_client") as mock_get_client,
        ):
            from run_agent import AIAgent

            mock_langfuse = MagicMock()
            mock_langfuse.get_current_trace_id.return_value = "trace-other"
            mock_langfuse.get_current_observation_id.return_value = "obs-other"
            mock_get_client.return_value = mock_langfuse

            agent = AIAgent(
                api_key="test-key",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                langfuse_enabled=True,
            )
            agent._current_trace_id = "trace-expected"
            agent._current_observation_id = "obs-expected"
            agent._last_injected_trace_id = "trace-expected"

            agent._capture_trace_context(MagicMock())

            assert agent._current_trace_id == "trace-expected"
            assert agent._current_observation_id == "obs-expected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
