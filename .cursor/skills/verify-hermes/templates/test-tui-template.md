# Test Template: TUI Feature

Template for testing TUI surface features (JSON-RPC gateway methods).

## Basic JSON-RPC Method Test

```python
import pytest
import json
from tui_gateway.server import TUIGateway


def test_prompt_submit(isolated_hermes_home):
    """Test prompt.submit RPC method."""
    gateway = TUIGateway()
    
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "prompt.submit",
        "params": {"text": "test prompt"}
    }
    
    response = gateway.handle_request(request)
    
    assert "result" in response or "error" in response
    # If streaming, expect message.delta events
    # If complete, expect message.complete event


def test_slash_exec(isolated_hermes_home):
    """Test slash.exec RPC method."""
    gateway = TUIGateway()
    
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "slash.exec",
        "params": {"command": "/help"}
    }
    
    response = gateway.handle_request(request)
    
    assert response.get("result") is not None
    result = response["result"]
    assert result.get("type") in ["output", "skill", "error"]
```

## Session Management Test

```python
def test_session_list(isolated_hermes_home):
    """Test session.list RPC method."""
    gateway = TUIGateway()
    
    # Create some test sessions first
    gateway.handle_request({
        "method": "prompt.submit",
        "params": {"text": "test 1"}
    })
    
    # List sessions
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "session.list",
        "params": {}
    }
    
    response = gateway.handle_request(request)
    
    assert "result" in response
    sessions = response["result"]
    assert isinstance(sessions, list)
    assert len(sessions) > 0


def test_session_resume(isolated_hermes_home):
    """Test session.resume RPC method."""
    gateway = TUIGateway()
    
    request = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "session.resume",
        "params": {"session_id": "test-session-123"}
    }
    
    response = gateway.handle_request(request)
    
    assert "result" in response or "error" in response
```

## Completion Test

```python
def test_slash_completion(isolated_hermes_home):
    """Test complete.slash RPC method."""
    gateway = TUIGateway()
    
    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "complete.slash",
        "params": {"prefix": "/hel"}
    }
    
    response = gateway.handle_request(request)
    
    assert "result" in response
    completions = response["result"]
    assert isinstance(completions, list)
    assert any("/help" in c for c in completions)


def test_path_completion(isolated_hermes_home):
    """Test complete.path RPC method."""
    gateway = TUIGateway()
    
    request = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "complete.path",
        "params": {"prefix": "./scr"}
    }
    
    response = gateway.handle_request(request)
    
    assert "result" in response
    completions = response["result"]
    assert isinstance(completions, list)
```

## Tool Activity Test

```python
def test_tool_events(isolated_hermes_home):
    """Test tool activity events."""
    gateway = TUIGateway()
    events = []
    
    def event_handler(event):
        events.append(event)
    
    gateway.set_event_handler(event_handler)
    
    # Trigger action that uses tools
    gateway.handle_request({
        "method": "prompt.submit",
        "params": {"text": "read test.txt"}
    })
    
    # Should see tool.start and tool.complete events
    tool_events = [e for e in events if e.get("method", "").startswith("tool.")]
    assert len(tool_events) > 0
```

## Streaming Test

```python
def test_message_streaming(isolated_hermes_home):
    """Test message streaming events."""
    gateway = TUIGateway()
    events = []
    
    def event_handler(event):
        events.append(event)
    
    gateway.set_event_handler(event_handler)
    
    # Submit prompt
    gateway.handle_request({
        "method": "prompt.submit",
        "params": {"text": "hello"}
    })
    
    # Should see message.delta events
    delta_events = [e for e in events if e.get("method") == "message.delta"]
    assert len(delta_events) > 0
    
    # Should see message.complete event
    complete_events = [e for e in events if e.get("method") == "message.complete"]
    assert len(complete_events) == 1
```

## Config Test

```python
def test_gateway_ready(isolated_hermes_home):
    """Test gateway.ready event includes config."""
    gateway = TUIGateway()
    
    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "gateway.init",
        "params": {}
    }
    
    response = gateway.handle_request(request)
    
    assert "result" in response
    ready_data = response["result"]
    assert "skin" in ready_data  # Skin config
    assert "version" in ready_data  # Hermes version
```

## Error Handling Test

```python
def test_invalid_method(isolated_hermes_home):
    """Test error handling for invalid methods."""
    gateway = TUIGateway()
    
    request = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "invalid.method",
        "params": {}
    }
    
    response = gateway.handle_request(request)
    
    assert "error" in response
    error = response["error"]
    assert error["code"] == -32601  # Method not found


def test_invalid_params(isolated_hermes_home):
    """Test error handling for invalid params."""
    gateway = TUIGateway()
    
    request = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "prompt.submit",
        "params": {}  # Missing 'text'
    }
    
    response = gateway.handle_request(request)
    
    assert "error" in response
```

## Integration Test (Full Flow)

```python
@pytest.mark.integration
def test_tui_full_conversation_flow(isolated_hermes_home):
    """Test complete conversation flow."""
    gateway = TUIGateway()
    events = []
    
    def event_handler(event):
        events.append(event)
    
    gateway.set_event_handler(event_handler)
    
    # 1. Initialize
    init_response = gateway.handle_request({
        "method": "gateway.init",
        "params": {}
    })
    assert "result" in init_response
    
    # 2. Submit prompt
    prompt_response = gateway.handle_request({
        "method": "prompt.submit",
        "params": {"text": "hello"}
    })
    
    # 3. Verify events
    assert any(e.get("method") == "message.delta" for e in events)
    assert any(e.get("method") == "message.complete" for e in events)
    
    # 4. List sessions
    sessions_response = gateway.handle_request({
        "method": "session.list",
        "params": {}
    })
    assert len(sessions_response["result"]) > 0
```

## Fixtures

```python
@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME for TUI tests."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("""
display:
  skin: default
""")
    (home / "logs").mkdir()
    (home / "state.db").touch()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_TUI", "1")
    return home


@pytest.fixture
def mock_tui_request():
    """Factory for creating TUI RPC requests."""
    counter = {"id": 0}
    
    def make_request(method, params=None):
        counter["id"] += 1
        return {
            "jsonrpc": "2.0",
            "id": counter["id"],
            "method": method,
            "params": params or {}
        }
    
    return make_request
```

## React Component Test (Vitest)

For testing React components in `ui-tui/src/`:

```typescript
// ui-tui/src/app/components/messageLine.test.tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MessageLine } from './messageLine'

describe('MessageLine', () => {
  it('renders user message', () => {
    render(<MessageLine role="user" content="test message" />)
    expect(screen.getByText('test message')).toBeInTheDocument()
  })

  it('renders assistant message with markdown', () => {
    render(<MessageLine role="assistant" content="**bold** text" />)
    expect(screen.getByText(/bold/)).toBeInTheDocument()
  })
})
```

## Notes

- TUI gateway tests use JSON-RPC protocol over stdio
- Event handler captures async events (tool progress, streaming)
- Use `HERMES_TUI=1` env var to signal TUI mode
- React component tests use Vitest + React Testing Library
- For full integration tests, verify complete request/event flow
- Test both success and error paths
- Verify JSON-RPC spec compliance (id, result, error fields)
