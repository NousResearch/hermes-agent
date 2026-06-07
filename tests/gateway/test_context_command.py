from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_gateway_context_1m_saves_config_and_evicts_agent(tmp_path):
    from gateway.run import GatewayRunner
    from hermes_cli.config import read_raw_config
    from hermes_cli.context_cmd import ONE_M_CONTEXT_LENGTH

    runner = SimpleNamespace(
        _session_key_for_source=lambda source: "telegram:chat",
        _evict_cached_agent=MagicMock(),
    )
    event = SimpleNamespace(
        get_command_args=lambda: "1m",
        source=SimpleNamespace(platform="telegram", chat_id="chat"),
    )

    with patch.dict("os.environ", {"HERMES_HOME": str(tmp_path)}):
        result = await GatewayRunner._handle_context_command(runner, event)
        raw = read_raw_config()

    assert "1,000,000" in result
    assert raw["model"]["context_length"] == ONE_M_CONTEXT_LENGTH
    runner._evict_cached_agent.assert_called_once_with("telegram:chat")


@pytest.mark.asyncio
async def test_gateway_context_status_does_not_evict_agent(tmp_path):
    from gateway.run import GatewayRunner

    runner = SimpleNamespace(
        _session_key_for_source=lambda source: "telegram:chat",
        _evict_cached_agent=MagicMock(),
    )
    event = SimpleNamespace(
        get_command_args=lambda: "status",
        source=SimpleNamespace(platform="telegram", chat_id="chat"),
    )

    with patch.dict("os.environ", {"HERMES_HOME": str(tmp_path)}):
        result = await GatewayRunner._handle_context_command(runner, event)

    assert "auto-detect" in result
    runner._evict_cached_agent.assert_not_called()
