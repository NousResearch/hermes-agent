"""Gateway background work keeps the initiating conversation's notebook."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools.memory_tool import MemoryStore


@pytest.mark.asyncio
async def test_background_agent_inherits_the_real_gateway_memory_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "memory:\n  scope: conversation\n  memory_enabled: true\n  user_profile_enabled: true\n"
    )
    monkeypatch.setattr("tools.tirith_security.ensure_installed", lambda **kwargs: False)
    monkeypatch.setattr("run_agent.OpenAI", MagicMock())
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(
        send=AsyncMock(), extract_media=lambda text: ([], text),
        extract_images=lambda text: ([], text),
    )
    runner.adapters[Platform.WHATSAPP] = adapter
    source = SessionSource(
        platform=Platform.WHATSAPP, chat_type="dm",
        chat_id="100000001@s.whatsapp.net", user_id="100000001@s.whatsapp.net",
    )
    expected = MemoryStore(gateway_session_key=runner._session_key_for_source(source))
    expected.load_from_disk()
    assert expected.add("user", "Private AMBER-LANTERN project")["success"]
    runtime = {
        "api_key": "test-only", "provider": "openrouter", "api_mode": "chat_completions",
        "base_url": "http://test.invalid",
    }
    monkeypatch.setattr(runner, "_resolve_session_agent_runtime", lambda **kwargs: ("gpt-4.1", runtime))
    monkeypatch.setattr(runner, "_resolve_turn_agent_config", lambda *args: {"model": "gpt-4.1", "runtime": runtime})
    observed = []

    def infer(agent, **kwargs):
        observed.append(agent._build_system_prompt())
        return {"final_response": "Internal fixture response"}

    monkeypatch.setattr("run_agent.AIAgent.run_conversation", infer)
    try:
        await runner._run_background_task_inner("internal fixture", source, "fixture-task")
        assert len(observed) == 1
        assert "AMBER-LANTERN" in observed[0]
        adapter.send.assert_awaited_once()
    finally:
        runner._shutdown_executor()
        if runner._session_db:
            runner._session_db._db.close()
