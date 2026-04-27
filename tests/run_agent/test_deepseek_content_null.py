import pytest
from unittest.mock import MagicMock

# We'll create a minimal mock of AIAgent to test the message preparation logic
class MockAIAgent:
    def __init__(self, provider="deepseek", model="deepseek-reasoner"):
        self.provider = provider
        self.model = model
        
    def _needs_deepseek_tool_reasoning(self):
        provider = (self.provider or "").lower()
        model = (self.model or "").lower()
        return (
            provider == "deepseek"
            or provider == "liaobots"
            or model.startswith("deepseek-reasoner")
        )

    def _prepare_api_messages(self, messages):
        # Extracted minimal logic from run_agent.py lines 9060-9079 and 9718-9734
        api_messages = []
        for msg in messages:
            api_msg = msg.copy()

            if (
                api_msg.get("role") == "assistant"
                and api_msg.get("content") == ""
                and api_msg.get("tool_calls")
                and self._needs_deepseek_tool_reasoning()
            ):
                api_msg["content"] = None

            api_messages.append(api_msg)
        return api_messages


def test_content_empty_string_converted_to_none():
    agent = MockAIAgent(provider="deepseek")
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]}
    ]
    
    api_messages = agent._prepare_api_messages(messages)
    
    assert len(api_messages) == 1
    assert api_messages[0]["content"] is None
    assert api_messages[0]["tool_calls"] is not None


def test_content_non_empty_unchanged():
    agent = MockAIAgent(provider="deepseek")
    messages = [
        {"role": "assistant", "content": "hello", "tool_calls": [{"id": "call_1"}]}
    ]
    
    api_messages = agent._prepare_api_messages(messages)
    
    assert len(api_messages) == 1
    assert api_messages[0]["content"] == "hello"


def test_no_tool_calls_unchanged():
    agent = MockAIAgent(provider="deepseek")
    messages = [
        {"role": "assistant", "content": ""}
    ]
    
    api_messages = agent._prepare_api_messages(messages)
    
    assert len(api_messages) == 1
    assert api_messages[0]["content"] == ""


def test_non_deepseek_unchanged():
    agent = MockAIAgent(provider="openai", model="gpt-4")
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]}
    ]
    
    api_messages = agent._prepare_api_messages(messages)
    
    assert len(api_messages) == 1
    assert api_messages[0]["content"] == ""


def test_user_message_unchanged():
    agent = MockAIAgent(provider="deepseek")
    messages = [
        {"role": "user", "content": "", "tool_calls": [{"id": "call_1"}]}
    ]
    
    api_messages = agent._prepare_api_messages(messages)
    
    assert len(api_messages) == 1
    assert api_messages[0]["content"] == ""


def test_tool_message_unchanged():
    agent = MockAIAgent(provider="deepseek")
    messages = [
        {"role": "tool", "content": "", "tool_calls": [{"id": "call_1"}]}
    ]
    
    api_messages = agent._prepare_api_messages(messages)
    
    assert len(api_messages) == 1
    assert api_messages[0]["content"] == ""
