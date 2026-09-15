# Test Template: Gateway Feature

Template for testing gateway surface features.

## Basic Message Handler Test

```python
import pytest
import asyncio
from gateway.run import GatewayRunner
from gateway.platforms.base import MessageEvent


class MockMessageEvent:
    """Mock message event for testing."""
    def __init__(self, text, sender_id="test-user", channel_id="test-channel"):
        self.text = text
        self.sender_id = sender_id
        self.channel_id = channel_id
        self.platform = "test"
        self.message_id = "msg-123"
        self.timestamp = "2026-09-09T23:00:00Z"


@pytest.mark.asyncio
async def test_slash_command_gateway(isolated_hermes_home):
    """Test slash command via gateway."""
    config = {
        "gateway": {"enabled": True},
        "platforms": {"test": {"enabled": True}}
    }
    
    gateway = GatewayRunner(config=config)
    event = MockMessageEvent("/testcommand arg")
    
    result = await gateway._handle_command(event)
    
    assert result is not None
    assert "expected" in str(result).lower()


@pytest.mark.asyncio
async def test_message_queueing(isolated_hermes_home):
    """Test message queue during agent execution."""
    gateway = GatewayRunner(config={})
    
    # Simulate agent running
    gateway._active_sessions.add("test-session")
    
    # Send message (should queue)
    event = MockMessageEvent("queued message")
    await gateway._process_message_background(event)
    
    # Verify queued
    assert len(gateway._pending_messages.get("test-session", [])) > 0
```

## Platform Adapter Test

```python
@pytest.mark.asyncio
async def test_platform_adapter_send():
    """Test platform adapter sending messages."""
    from gateway.platforms.test_adapter import TestAdapter
    
    adapter = TestAdapter(config={
        "test_token": "test-token"
    })
    
    await adapter.start()
    
    result = await adapter.send_message(
        text="Test message",
        channel_id="ch-123"
    )
    
    assert result.message_id is not None
    assert result.success


@pytest.mark.asyncio
async def test_platform_adapter_receive():
    """Test platform adapter receiving messages."""
    adapter = TestAdapter(config={})
    await adapter.start()
    
    # Mock incoming message
    received = []
    async def handler(event):
        received.append(event)
    
    adapter.set_message_handler(handler)
    
    # Simulate platform webhook/event
    await adapter._simulate_incoming("Test message", "user-123")
    
    assert len(received) == 1
    assert received[0].text == "Test message"
```

## Multiplex Profile Test

```python
@pytest.mark.asyncio
async def test_multiplex_profile_isolation(tmp_path, monkeypatch):
    """Test profile isolation in gateway."""
    # Setup two profiles
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    (default_home / ".env").write_text("TEST_VAR=default")
    
    profile_home = tmp_path / ".hermes" / "profiles" / "profile1"
    profile_home.mkdir(parents=True)
    (profile_home / ".env").write_text("TEST_VAR=profile1")
    
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    
    gateway = GatewayRunner(config={
        "gateway": {"multiplex_profiles": True}
    })
    
    # Test default profile sees default env
    with gateway._profile_runtime_scope(None):
        from agent.secret_scope import get_scoped_secret
        assert get_scoped_secret("TEST_VAR") == "default"
    
    # Test profile1 sees its own env
    with gateway._profile_runtime_scope("profile1"):
        assert get_scoped_secret("TEST_VAR") == "profile1"
```

## Tool Availability Test

```python
def test_gateway_toolset_loading():
    """Test gateway loads correct toolsets."""
    from gateway.run import GatewayRunner
    
    gateway = GatewayRunner(config={})
    tools = gateway._load_enabled_toolsets(platform="telegram")
    
    tool_names = [t["name"] for t in tools]
    
    # Core tools present
    assert "terminal" in tool_names
    assert "read_file" in tool_names
    
    # Messaging tools present
    assert "send_message" in tool_names
    
    # CLI-only tools absent
    assert "clarify" not in tool_names
```

## Authorization Test

```python
@pytest.mark.asyncio
async def test_gateway_authorization(isolated_hermes_home):
    """Test user authorization."""
    config = {
        "gateway": {
            "allow_all_users": False,
            "allowed_users": ["user-123"]
        }
    }
    
    gateway = GatewayRunner(config=config)
    
    # Allowed user
    event_allowed = MockMessageEvent("test", sender_id="user-123")
    assert await gateway._is_authorized(event_allowed)
    
    # Disallowed user
    event_denied = MockMessageEvent("test", sender_id="user-999")
    assert not await gateway._is_authorized(event_denied)
```

## Streaming Response Test

```python
@pytest.mark.asyncio
async def test_streaming_response():
    """Test streaming message delivery."""
    adapter = TestAdapter(config={})
    await adapter.start()
    
    # Start stream
    stream_id = await adapter.start_stream(channel_id="ch-123")
    
    # Send chunks
    await adapter.send_stream_chunk(stream_id, "Hello ")
    await adapter.send_stream_chunk(stream_id, "world!")
    
    # End stream
    await adapter.finish_stream(stream_id)
    
    # Verify final message
    messages = adapter.get_sent_messages()
    assert len(messages) == 1
    assert messages[0].text == "Hello world!"
```

## Fixtures

```python
@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME for gateway tests."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("""
gateway:
  enabled: true
""")
    (home / "logs").mkdir()
    (home / "gateway_state").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def mock_platform_config():
    """Mock platform configuration."""
    return {
        "enabled": True,
        "token": "test-token",
        "allowed_channels": ["ch-123"]
    }
```

## Notes

- Gateway tests are async - use `@pytest.mark.asyncio`
- Use `MockMessageEvent` for consistent test events
- Test both authorized and unauthorized users
- Verify message queueing when agent is busy
- Test streaming if platform supports it
- Always cleanup adapters with `await adapter.stop()`
