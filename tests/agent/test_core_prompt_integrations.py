"""Tests for decoupled memory manager prompt extension hooks in system_prompt.py and conversation_loop.py."""

from unittest.mock import MagicMock, patch
import pytest

from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager
from agent.system_prompt import build_system_prompt_parts
from run_agent import AIAgent


class MockIntegrationProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "integration_mock"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def get_soul_extension_prompt(self):
        return "SOUL_EXTENSION_DATA"

    def get_user_profile_extension_prompt(self):
        return "USER_PROFILE_EXTENSION_DATA"

    def get_per_turn_context(self):
        return "<agent-context>\nPER_TURN_AGENT_CONTEXT\n</agent-context>\n\n<user-context>\nPER_TURN_USER_CONTEXT\n</user-context>"


def test_system_prompt_hook_injection():
    """Verify that system_prompt.py successfully calls MemoryManager hooks and injects their return values."""
    mgr = MemoryManager()
    mgr.add_provider(MockIntegrationProvider())

    agent = MagicMock()
    agent.load_soul_identity = False
    agent.skip_context_files = True
    agent._memory_manager = mgr
    agent._user_profile_enabled = True
    agent.valid_tool_names = []
    agent._kanban_worker_guidance = None
    agent._task_completion_guidance = True
    agent._memory_enabled = False

    # Mock agent._memory_store
    mock_store = MagicMock()
    mock_store.format_for_system_prompt.return_value = "MOCK_USER_PROFILE_BLOCK"
    agent._memory_store = mock_store

    parts = build_system_prompt_parts(agent)
    assert "SOUL_EXTENSION_DATA" in parts["stable"]
    assert "USER_PROFILE_EXTENSION_DATA" in parts["volatile"]


@patch("run_agent.AIAgent._build_system_prompt")
@patch("run_agent.AIAgent._interruptible_streaming_api_call")
@patch("run_agent.AIAgent._interruptible_api_call")
def test_conversation_loop_per_turn_context_injection(mock_api, mock_stream, mock_sys):
    """Verify that conversation_loop.py queries MemoryManager per-turn hook and appends it to user message."""
    mock_sys.return_value = "system prompt"

    mock_choice = MagicMock()
    mock_choice.message.content = "response"
    mock_choice.message.tool_calls = None
    mock_choice.message.refusal = None
    mock_choice.message.reasoning = None
    mock_choice.message.reasoning_content = None
    mock_choice.message.reasoning_details = None
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_response.model = "test-model"
    mock_response.id = "test-id"

    mock_stream.return_value = mock_response
    mock_api.return_value = mock_response

    # Initialize AIAgent
    agent = AIAgent(
        model="test/model",
        api_key="test-key",
        base_url="http://localhost:1234/v1",
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
    )
    agent.client = MagicMock()

    # Inject MemoryManager with MockIntegrationProvider
    mgr = MemoryManager()
    mgr.add_provider(MockIntegrationProvider())
    agent._memory_manager = mgr

    result = agent.run_conversation(
        user_message="Hello!",
        conversation_history=[],
    )

    # Verify that the message sent to the API contains the per-turn context in correct format
    mock_call_args = mock_stream.call_args or mock_api.call_args
    assert mock_call_args is not None

    called_api_kwargs = mock_call_args.args[0] if mock_call_args.args else mock_call_args.kwargs.get("api_kwargs")
    assert called_api_kwargs is not None
    
    called_messages = called_api_kwargs.get("messages")
    assert called_messages is not None
    
    user_msg_content = called_messages[-1]["content"]

    assert "Hello!" in user_msg_content
    assert "<agent-context>\nPER_TURN_AGENT_CONTEXT\n</agent-context>" in user_msg_content
    assert "<user-context>\nPER_TURN_USER_CONTEXT\n</user-context>" in user_msg_content
