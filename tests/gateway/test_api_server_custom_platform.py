"""Custom API transports retain their own policy-enforcing execution factory."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.mark.asyncio
async def test_custom_platform_factory_policy_remains_authoritative(monkeypatch):
    class PolicyAdapter(APIServerAdapter):
        def _create_agent(self, **kwargs):
            raise PermissionError("custom transport policy refused")

    adapter = PolicyAdapter(PlatformConfig(enabled=True))
    adapter.platform = "custom_transport"
    authority = object()
    runner = SimpleNamespace(session_authority=authority)
    adapter.gateway_runner = runner
    canonical = AsyncMock(return_value=({"final_response": "bypassed"}, {}))
    monkeypatch.setattr("gateway.session_api_turn.run_api_turn", canonical)

    with pytest.raises(PermissionError, match="custom transport policy refused"):
        await adapter._run_agent("hello", [], session_id="custom-session")

    canonical.assert_not_awaited()
    assert adapter.gateway_runner is runner
    assert runner.session_authority is authority
    assert adapter._inflight_agent_runs == 0


@pytest.mark.asyncio
async def test_builtin_api_retains_canonical_authority(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    assert adapter.platform == Platform.API_SERVER
    adapter.gateway_runner = SimpleNamespace(session_authority=object())
    result = ({"final_response": "canonical"}, {"total_tokens": 1})
    canonical = AsyncMock(return_value=result)
    monkeypatch.setattr("gateway.session_api_turn.run_api_turn", canonical)
    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: pytest.fail("legacy factory used"))

    assert await adapter._run_agent("hello", [], session_id="api-session") == result
    canonical.assert_awaited_once()
    assert canonical.call_args.args == (adapter,)
    assert canonical.call_args.kwargs["session_id"] == "api-session"
