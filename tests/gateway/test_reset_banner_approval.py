"""Reset notices agree with the real approval bypass state after rotation."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from tools import approval


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/new", "/reset"])
@pytest.mark.parametrize("mode", ["manual", "smart", "off"])
@pytest.mark.parametrize("process_override", [False, True])
@pytest.mark.parametrize("scoped_bypass", [False, True])
async def test_reset_banner_effective_approval(
    tmp_path, monkeypatch, command, mode, process_override, scoped_bypass,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", process_override)
    # Isolate only the in-memory registry, not the real toggle or boundary cleanup.
    monkeypatch.setattr(approval, "_session_yolo", set())
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "test-model", "context_length": 2000},
        "approvals": {"mode": mode},
    }), encoding="utf-8")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner._agent_cache_lock = None
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._background_tasks = set()
    source = SessionSource(platform=Platform.MATRIX, chat_id="room", user_id="user", chat_type="dm")
    other = SessionSource(platform=Platform.MATRIX, chat_id="other-room", user_id="user", chat_type="dm")
    key = runner._session_key_for_source(source)
    other_key = runner._session_key_for_source(other)
    old = runner.session_store.get_or_create_session(source)
    if scoped_bypass:
        await runner._handle_yolo_command(MessageEvent(text="/yolo", source=source))
    await runner._handle_yolo_command(MessageEvent(text="/yolo", source=other))
    assert approval.is_session_yolo_enabled(key) is scoped_bypass
    assert approval.is_approval_bypass_active_for_session(key) is (
        process_override or scoped_bypass or mode == "off"
    )

    with patch.object(runner, "_fire_session_reset_hooks", AsyncMock()), \
         patch.object(runner, "_telegram_topic_new_header", return_value=None), \
         patch.object(runner, "_is_telegram_topic_lane", return_value=False), \
         patch("gateway.run._resolve_runtime_agent_kwargs", return_value={}), \
         patch("hermes_cli.lifecycle.invoke_hook"), \
         patch("tools.async_delegation.interrupt_for_session"), \
         patch("gateway.slash_commands_session._reset_process_scoped_tool_state"), \
         patch("hermes_cli.tips.get_random_tip", return_value="Synthetic tip"):
        reply = await runner._handle_reset_command(MessageEvent(text=command, source=source))

    fresh = runner.session_store._entries[key]
    assert fresh.session_id != old.session_id
    assert fresh.session_id in str(reply)
    assert old.session_id not in str(reply)
    assert not approval.is_session_yolo_enabled(key)
    assert approval.is_session_yolo_enabled(other_key)
    assert approval._YOLO_MODE_FROZEN is process_override
    assert approval.is_approval_bypass_active_for_session(key) is (process_override or mode == "off")
    expected = "off (runtime override)" if process_override else mode
    approval_lines = [line for line in str(reply).splitlines() if line.startswith("◆ Tool approval:")]
    assert approval_lines == [f"◆ Tool approval: {expected}"]
