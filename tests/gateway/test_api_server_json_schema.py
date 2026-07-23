"""
Test API server handling of JSON schema structured output requests.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.fixture
def api_server():
    """Create APIServerAdapter instance for testing."""
    config = PlatformConfig(enabled=True, extra={"key": "test-key"})
    adapter = APIServerAdapter(config)
    return adapter


@pytest.mark.asyncio
async def test_responses_endpoint_rejects_json_schema(api_server):
    """Test that /v1/responses endpoint properly rejects json_schema format requests."""
    request = AsyncMock()
    request.headers = {"Authorization": "Bearer test-key"}
    request.json = AsyncMock(return_value={
        "input": "Reply with PONG",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "Probe",
                "schema": {
                    "type": "object",
                    "properties": {"word": {"type": "string"}},
                    "required": ["word"],
                    "additionalProperties": False
                }
            }
        }
    })
    
    response = await api_server._handle_responses(request)
    
    assert response.status == 400
    
    response_data = json.loads(response.text)
    assert "error" in response_data
    assert "json_schema" in response_data["error"]["message"]
    assert "not yet supported" in response_data["error"]["message"]


@pytest.mark.asyncio
async def test_responses_endpoint_accepts_non_json_schema(api_server):
    """Test that /v1/responses endpoint works normally without json_schema format."""
    request = AsyncMock()
    request.headers = {"Authorization": "Bearer test-key"}
    request.json = AsyncMock(return_value={
        "input": "Hello",
        "instructions": "Be helpful"
    })
    
    with patch.object(api_server, '_run_agent') as mock_run_agent:
        mock_run_agent.return_value = ({"response": "Hi there!"}, {"total_tokens": 10})
        
        response = await api_server._handle_responses(request)
        
        assert response.status == 200
        mock_run_agent.assert_called_once()


@pytest.mark.asyncio
async def test_responses_endpoint_ignores_empty_text_field(api_server):
    """Test that empty text field doesn't trigger json_schema check."""
    request = AsyncMock()
    request.headers = {"Authorization": "Bearer test-key"}
    request.json = AsyncMock(return_value={
        "input": "Hello",
        "text": {}
    })
    
    with patch.object(api_server, '_run_agent') as mock_run_agent:
        mock_run_agent.return_value = ({"response": "Hi!"}, {"total_tokens": 5})
        
        response = await api_server._handle_responses(request)
        
        assert response.status == 200
        mock_run_agent.assert_called_once()


@pytest.mark.asyncio
async def test_responses_endpoint_rejects_json_schema_in_tools(api_server):
    """Test that json_schema in tool definitions is also rejected."""
    request = AsyncMock()
    request.headers = {"Authorization": "Bearer test-key"}
    request.json = AsyncMock(return_value={
        "input": "Use the tool",
        "tools": [
            {
                "name": "get_data",
                "response_format": {
                    "type": "json_schema",
                    "schema": {"type": "object"}
                }
            }
        ]
    })
    
    response = await api_server._handle_responses(request)
    
    assert response.status == 400
    response_data = json.loads(response.text)
    assert "error" in response_data
    assert "tool definitions" in response_data["error"]["message"]