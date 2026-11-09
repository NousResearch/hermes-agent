"""Tests for MemoryProvider.on_turn_complete hook and MemoryManager.on_turn_complete_all."""

import pytest
from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager
from tests.waiting import wait_until


class MockMemoryProvider(MemoryProvider):
    """Mock provider for testing on_turn_complete."""
    
    def __init__(self, name="mock"):
        self._name = name
        self.on_turn_complete_calls = []
        self.sync_turn_calls = []
    
    @property
    def name(self) -> str:
        return self._name
    
    def is_available(self) -> bool:
        return True
    
    def initialize(self, session_id: str, **kwargs) -> None:
        pass
    
    def sync_turn(self, user_content: str, assistant_content: str, **kwargs) -> None:
        self.sync_turn_calls.append((user_content, assistant_content, kwargs))
    
    def on_turn_complete(self, user_content: str, assistant_content: str, **kwargs) -> None:
        self.on_turn_complete_calls.append((user_content, assistant_content, kwargs))
    
    def get_tool_schemas(self) -> list:
        return []


def test_on_turn_complete_hook_called_after_sync():
    """Test that on_turn_complete is called separately from sync_turn."""
    provider = MockMemoryProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    
    # Simulate a turn
    user_text = "Hello, remember this important fact"
    assistant_text = "I'll remember that"
    session_id = "test-session"
    
    # Call sync_all (mimics the existing behavior)
    manager.sync_all(user_text, assistant_text, session_id=session_id)
    
    # Call on_turn_complete_all (the new hook)
    manager.on_turn_complete_all(user_text, assistant_text, session_id=session_id, should_store=True)
    
    wait_until(lambda: provider.on_turn_complete_calls, what="on_turn_complete to fire")
    
    # Both hooks should have been called
    assert len(provider.sync_turn_calls) > 0, "sync_turn should be called"
    assert len(provider.on_turn_complete_calls) > 0, "on_turn_complete should be called"
    
    # Verify on_turn_complete received should_store hint
    _, _, kwargs = provider.on_turn_complete_calls[0]
    assert kwargs.get("should_store") is True
    assert kwargs.get("session_id") == session_id


def test_storage_intent_detection():
    """Test _detect_storage_intent recognizes explicit storage keywords."""
    from run_agent import AIAgent
    
    # Positive cases - should detect storage intent
    storage_phrases = [
        "remember this: my API key is xyz",
        "Don't forget to call John tomorrow",
        "Keep in mind that the deadline is Friday",
        "Save this for later reference",
        "Store this important information",
        "Make a note that the server IP is 192.168.1.1",
        "Take note of this configuration",
        "This is important to remember",
        "Keep track of these requirements",
        "Write down this password",
    ]
    
    for phrase in storage_phrases:
        assert AIAgent._detect_storage_intent(phrase), f"Should detect storage intent in: {phrase}"
    
    # Negative cases - should NOT detect storage intent
    normal_phrases = [
        "What's the weather today?",
        "Can you help me with this code?",
        "Explain how recursion works",
        "Generate a report",
    ]
    
    for phrase in normal_phrases:
        assert not AIAgent._detect_storage_intent(phrase), f"Should NOT detect storage intent in: {phrase}"
    
    # Edge cases
    assert not AIAgent._detect_storage_intent(""), "Empty string should not trigger storage"
    assert not AIAgent._detect_storage_intent("   "), "Whitespace should not trigger storage"


def test_on_turn_complete_respects_suppression():
    """Test that on_turn_complete respects _fork_turn_hooks_suppressed."""
    provider = MockMemoryProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    
    # Simulate suppression (e.g., in a subagent)
    manager._fork_turn_hooks_suppressed = lambda: True
    
    manager.on_turn_complete_all("test user", "test assistant", session_id="test", should_store=True)
    
    # Suppression returns before anything is queued: no background work is pending.
    assert manager._background_futures == {}, "suppressed hook must not queue work"
    assert len(provider.on_turn_complete_calls) == 0, "on_turn_complete should be suppressed"


def test_on_turn_complete_with_trivial_content():
    """Test that on_turn_complete is not called for empty or trivial content."""
    provider = MockMemoryProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    
    # Empty user content
    manager.on_turn_complete_all("", "response", session_id="test", should_store=False)
    
    # Empty content returns before any work is queued.
    assert manager._background_futures == {}, "empty content must not queue work"
    assert len(provider.on_turn_complete_calls) == 0, "Should not call on_turn_complete with empty content"


def test_on_turn_complete_default_implementation():
    """Test that the default on_turn_complete implementation is a no-op."""
    
    class MinimalProvider(MemoryProvider):
        @property
        def name(self) -> str:
            return "minimal"
        
        def is_available(self) -> bool:
            return True
        
        def initialize(self, session_id: str, **kwargs) -> None:
            pass
        
        def get_tool_schemas(self) -> list:
            return []
    
    provider = MinimalProvider()
    
    # Should not raise - default implementation is a no-op
    provider.on_turn_complete("user", "assistant", session_id="test", should_store=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
