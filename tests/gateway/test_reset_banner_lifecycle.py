"""Exercise real store rotation and banner construction in a disposable home."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionStore, SessionSource
from gateway.platforms.event import MessageEvent


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("command", ["/new", "/reset"])
async def test_reset_banner_is_new_session(tmp_path, monkeypatch, existing, command):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("model:\n  default: test-model\n  context_length: 2000\nagent:\n  reasoning_effort: low\n  service_tier: fast\n", encoding="utf-8")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    source = SessionSource(platform=Platform.MATRIX, chat_id="room", user_id="user", chat_type="dm")
    key = runner._session_key_for_source(source)
    old = runner.session_store.get_or_create_session(source) if existing else None
    runner._set_session_reasoning_override(key, {"enabled": True, "effort": "high"})
    runner._set_session_service_tier_override(key, None)
    runner._session_model_overrides[key] = {"model": "old-model"}
    runner._agent_cache_lock = None
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._background_tasks = set()
    with patch.object(runner, "_fire_session_reset_hooks", AsyncMock()), \
         patch.object(runner, "_telegram_topic_new_header", return_value=None), \
         patch.object(runner, "_is_telegram_topic_lane", return_value=False), \
         patch("gateway.run._resolve_runtime_agent_kwargs", return_value={}), \
         patch("hermes_cli.lifecycle.invoke_hook"), \
         patch("tools.async_delegation.interrupt_for_session"), \
         patch("gateway.slash_commands_session._reset_process_scoped_tool_state"), \
         patch("hermes_cli.tips.get_random_tip", return_value="Synthetic tip"):
        reply = await runner._handle_reset_command(MessageEvent(text=command, source=source, message_id="m"))
    text = str(reply)
    fresh = runner.session_store._entries[key]
    assert fresh.session_id in text
    assert "◆ Session:" in text
    if old:
        assert fresh.session_id != old.session_id
        assert old.session_id not in text
    assert "Main reasoning: low" in text
    assert "Model: `test-model`" in text
    assert runner._resolve_session_reasoning_config(session_key=key) == {"enabled": True, "effort": "low"}
    assert runner._resolve_session_service_tier(session_key=key) == "priority"
    assert "Service tier (requested): priority" in text
    assert "old-model" not in text
    assert "Synthetic tip" in text
