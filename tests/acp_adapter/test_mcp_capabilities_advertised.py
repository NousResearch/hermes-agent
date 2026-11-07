import pytest

from acp_adapter.server import HermesACPAgent


@pytest.mark.asyncio
async def test_initialize_advertises_http_mcp_capability():
    """External runtimes refuse to attach HTTP MCP servers unless initialize() says so (#137500)."""
    response = await HermesACPAgent().initialize()

    assert response.agent_capabilities is not None
    assert response.agent_capabilities.mcp_capabilities is not None
    assert response.agent_capabilities.mcp_capabilities.http is True


@pytest.mark.asyncio
async def test_initialize_serializes_mcp_capabilities_with_wire_alias():
    serialized = (
        await HermesACPAgent().initialize()
    ).agent_capabilities.mcp_capabilities.model_dump(by_alias=True, exclude_none=True)

    assert serialized == {"http": True, "sse": False}
