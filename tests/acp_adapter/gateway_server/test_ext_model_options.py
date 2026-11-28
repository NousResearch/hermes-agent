"""The model picker's inventory over ACP: ``_model/options`` (#160752).

ACP's session replies carry only ``availableModels``; the picker that offers the shared inventory
(pricing, capabilities, planned usage) is the TUI's ``model.options``. An editor names the session
its picker belongs to, and the payload centers that session's model on the inventory.

The inventory itself is stubbed: these tests are about the wire contract, the flags and the session
the payload is centered on, not about provider discovery.
"""
import pytest
from acp.exceptions import RequestError

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli import inventory
from hermes_cli.inventory import ConfigContext


@pytest.fixture
def inventory_stub(monkeypatch):
    """A disk config of this test's own, and the payload build's context and flags recorded."""
    seen = {}

    monkeypatch.setattr(inventory, "load_picker_context", lambda: ConfigContext(
        current_provider="openrouter", current_model="disk-model", current_base_url="",
        user_providers={}, custom_providers=[]))

    def build_models_payload(ctx, **kwargs):
        seen["ctx"], seen["flags"] = ctx, kwargs
        return {"providers": [], "model": ctx.current_model, "provider": ctx.current_provider}

    monkeypatch.setattr(inventory, "build_models_payload", build_models_payload)
    return seen


def _agent_with_session(info: dict) -> GatewayACPAgent:
    agent = GatewayACPAgent()
    agent._snapshots["logical"] = {"info": info}
    return agent


@pytest.mark.asyncio
async def test_payload_is_the_picker_inventory_centered_on_the_session(inventory_stub):
    agent = _agent_with_session({"model": "session-model", "provider": "openrouter"})
    payload = await agent.ext_method("model/options", {
        "sessionId": "logical", "explicitOnly": True, "includeUnconfigured": True, "refresh": True})

    assert set(payload) == {"providers", "model", "provider"}
    assert payload["model"] == "session-model"
    flags = inventory_stub["flags"]
    # The picker payload, not the bare model list, and the three flags as the TUI names them.
    assert flags["for_picker"] is True and flags["picker_hints"] is True
    assert (flags["explicit_only"], flags["include_unconfigured"], flags["refresh"]) == (True, True, True)


@pytest.mark.asyncio
async def test_without_a_session_the_picker_reads_the_profile_config(inventory_stub):
    payload = await GatewayACPAgent().ext_method("model/options", {})

    assert payload["model"] == "disk-model"
    flags = inventory_stub["flags"]
    assert (flags["explicit_only"], flags["include_unconfigured"], flags["refresh"]) == (False, False, False)


@pytest.mark.asyncio
async def test_the_editors_alias_reads_the_logical_sessions_model(inventory_stub):
    agent = _agent_with_session({"model": "logical-model"})
    agent._aliases["held-id"] = "logical"

    assert (await agent.ext_method("model/options", {"sessionId": "held-id"}))["model"] == "logical-model"


@pytest.mark.asyncio
async def test_an_unknown_session_is_refused_with_its_id(inventory_stub):
    with pytest.raises(RequestError) as refused:
        await GatewayACPAgent().ext_method("model/options", {"sessionId": "gone"})
    assert refused.value.code == -32602
    assert refused.value.data == {"reason": "unknown session gone"}


@pytest.mark.asyncio
async def test_the_wire_reaches_the_handler(inventory_stub):
    """``HermesACPAgent()`` answers as a ``GatewayACPAgent``, so the handler must live on it."""
    from acp.agent.router import build_agent_router
    payload = await build_agent_router(GatewayACPAgent())("_model/options", {}, False)
    assert set(payload) == {"providers", "model", "provider"}
