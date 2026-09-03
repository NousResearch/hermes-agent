from unittest.mock import MagicMock, patch
import pytest

from agent.chat_completion_helpers import build_api_kwargs
from tools.vision_tools import _supports_media_in_tool_results

def test_supports_media_in_tool_results_for_custom_providers():
    """Verify that _supports_media_in_tool_results returns True for custom and ollama-launch."""
    assert _supports_media_in_tool_results("custom", "gemma4:latest") is True
    assert _supports_media_in_tool_results("ollama-launch", "gemma4:latest") is True
    assert _supports_media_in_tool_results("openai", "gpt-4o") is True
    assert _supports_media_in_tool_results("unknown-provider", "some-model") is False

def test_build_api_kwargs_suppresses_vision_analyze():
    """Verify that build_api_kwargs filters out the vision_analyze tool when native images are present."""
    # Create mock agent
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent.provider = "custom"
    agent.model = "gemma4:latest"
    agent.base_url = "http://localhost:11434/v1"
    agent.max_tokens = 2048
    agent.reasoning_config = None
    agent.request_overrides = {}
    agent.providers_allowed = None
    agent.providers_ignored = None
    agent.providers_order = None
    agent.provider_sort = None
    agent.provider_require_parameters = False
    agent.provider_data_collection = None
    agent._ephemeral_max_output_tokens = None
    agent._ollama_num_ctx = None
    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": "vision_analyze",
                "description": "Load an image",
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run shell command",
            }
        }
    ]
    
    # 1. Test when no native images are present
    messages_no_images = [
        {"role": "user", "content": "hello world"}
    ]
    
    # Mock transport builder
    mock_transport = MagicMock()
    agent._get_transport.return_value = mock_transport
    
    # Mock the return value of _prepare_messages_for_non_vision_model
    agent._prepare_messages_for_non_vision_model = lambda x: x
    agent._is_qwen_portal.return_value = False
    agent._is_openrouter_url.return_value = False
    agent._is_azure_openai_url.return_value = False
    agent._is_direct_openai_url.return_value = False
    agent._supports_reasoning_extra_body.return_value = False
    agent._resolved_api_call_timeout.return_value = 1800
    agent._max_tokens_param = lambda x: {}
    
    # We patch the provider profile resolver to return None (so it hits the legacy flag path in build_api_kwargs)
    with patch("providers.get_provider_profile", return_value=None):
        build_api_kwargs(agent, messages_no_images)
        
        # Verify that all tools (including vision_analyze) are passed to transport
        called_tools = mock_transport.build_kwargs.call_args[1].get("tools")
        assert len(called_tools) == 2
        assert any(t["function"]["name"] == "vision_analyze" for t in called_tools)
        
        # Reset mock
        mock_transport.reset_mock()
        
        # 2. Test when a native image is attached
        messages_with_images = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                ]
            }
        ]
        
        build_api_kwargs(agent, messages_with_images)
        
        # Verify that vision_analyze is filtered out of the tools list passed to transport
        called_tools = mock_transport.build_kwargs.call_args[1].get("tools")
        assert len(called_tools) == 1
        assert called_tools[0]["function"]["name"] == "run_command"
        assert not any(t["function"]["name"] == "vision_analyze" for t in called_tools)
