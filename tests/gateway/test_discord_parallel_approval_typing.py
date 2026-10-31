"""Real concurrent protected-file tools must not resume typing between approvals."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.run_turn_runner import TurnRunner
from plugins.platforms.discord.adapter import DiscordAdapter, discord
from gateway.config import PlatformConfig
from run_agent import AIAgent
from tools import approval
from tests.gateway.test_discord_approval_typing import interaction


def file_agent():
    from tools.file_tools import WRITE_FILE_SCHEMA
    with (
        patch("model_tools.get_tool_definitions", return_value=[
            {"type": "function", "function": WRITE_FILE_SCHEMA}]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, platform="discord")
    agent._flush_messages_to_session_db = lambda *args, **kwargs: True
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("separate_sessions", [False, True], ids=["parallel-tools", "two-users"])
@pytest.mark.parametrize("control", ["button", "approve", "deny"])
async def test_real_parallel_file_approvals_keep_typing_paused(tmp_path, monkeypatch, separate_sessions, control):
    # Import gateway at collection time (as production does before the worker starts).
    # The actual dispatch planner, file-write guard, queue and card renderer stay real.
    monkeypatch.setattr(discord.ui.View, "remove_item", lambda self, item: None, raising=False)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *args, **kwargs: None)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    from gateway.session import SessionSource, build_session_key
    from gateway.config import Platform
    source = SessionSource(platform=Platform.DISCORD, chat_id="77", chat_type="group", user_id="user")
    key = build_session_key(source)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._allowed_user_ids = {"user"}
    cards = asyncio.Queue()

    async def send(**kwargs):
        cards.put_nowait(kwargs["view"])
        return SimpleNamespace(id=cards.qsize())

    adapter._client = SimpleNamespace(
        get_channel=lambda channel_id: SimpleNamespace(send=send),
        http=SimpleNamespace(request=AsyncMock()),
    )
    runner = object.__new__(TurnRunner)
    runner._runner = SimpleNamespace(_consume_pending_native_image_paths=lambda key: [])
    runner._ctx = SimpleNamespace(
        session_key=key, _status_adapter=adapter, _status_chat_id="77",
        _status_thread_metadata=None, _loop_for_step=asyncio.get_running_loop(),
        stream_consumer_holder=[None], _run_still_current=lambda: True,
        message="update project instructions", source=source, session_id=key,
        title_user_message=None, persist_user_display_kind=None, persist_user_display_metadata=None,
        moa_config=None, inbound_message_id=None, mute_notification_reply=False, reply_expected=True,
        native_images=[],
    )
    agent = file_agent()
    paths = [tmp_path / "a" / "AGENTS.md", tmp_path / "b" / "CLAUDE.md"]
    for path in paths:
        path.parent.mkdir()
        path.write_text("original")
    calls = [SimpleNamespace(id=str(i), type="function", function=SimpleNamespace(
        name="write_file", arguments=json.dumps({"path": str(path), "content": "new"})))
        for i, path in enumerate(paths)]
    messages = []

    def launch(turn_runner, turn_agent, turn_calls, rows):
        def conversation(message, **kwargs):
            turn_agent._execute_tool_calls(SimpleNamespace(tool_calls=turn_calls), rows, turn_runner._ctx.session_key)
            return {"final_response": "done"}
        monkeypatch.setattr(turn_agent, "run_conversation", conversation)
        return asyncio.create_task(asyncio.to_thread(
            turn_runner._run_conversation_with_approval, turn_agent, [], None, None, None))

    runners = {key: runner}
    workers = []
    if separate_sessions:
        other_source = SessionSource(platform=Platform.DISCORD, chat_id="77", chat_type="group", user_id="other")
        other_key = build_session_key(other_source)
        assert other_key != key
        other = object.__new__(TurnRunner)
        other._runner = runner._runner
        other._ctx = SimpleNamespace(**{**vars(runner._ctx), "session_key": other_key,
                                       "session_id": other_key, "source": other_source})
        runners[other_key] = other
        workers.append(launch(runner, agent, calls[:1], messages))
        workers.append(launch(other, file_agent(), calls[1:], []))
    else:
        workers.append(launch(runner, agent, calls, messages))

    async def decide(view, first=False):
        if control == "button":
            await view._resolve(interaction(), "deny", None, "gateway.exec_approval.action_deny")
        else:
            from gateway.run import GatewayRunner
            from gateway.platforms.event import MessageEvent
            command_runner = object.__new__(GatewayRunner)
            command_runner._delivery_adapter_for = lambda source: adapter
            command_runner._session_key_for_source = build_session_key
            command_runner._pending_approvals = {}
            event = MessageEvent(text=f"/{control if first else 'deny'}", source=runners[view.session_key]._ctx.source)
            handler = command_runner._handle_approve_command if first and control == "approve" else command_runner._handle_deny_command
            await handler(event)

    try:
        first = await asyncio.wait_for(cards.get(), 10)
        second = await asyncio.wait_for(cards.get(), 10)
        pending = [entry for session in runners for entry in approval.list_gateway_approvals(session)]
        assert len(pending) == 2
        assert {entry["command"] for entry in pending} == {
            "<write to AGENTS.md>", "<write to CLAUDE.md>"}
        assert all(not worker.done() for worker in workers)
        assert "77" in adapter._typing_paused
        if control == "button":
            # A card selects its own request, not the oldest wait in the chat.
            first, second = second, first
        await decide(first, first=True)
        remaining = [entry for session in runners for entry in approval.list_gateway_approvals(session)]
        assert len(remaining) == 1
        if control == "button":
            assert remaining[0]["request_id"] == second.request_id
        assert "77" in adapter._typing_paused, "first decision resumed a chat with another real tool awaiting approval"
        await adapter.send_typing("77")
        await asyncio.sleep(0)
        adapter._client.http.request.assert_not_awaited()
        await decide(second)
        await asyncio.wait_for(asyncio.gather(*workers), 10)
        assert "77" not in adapter._typing_paused
        assert len(messages) == (1 if separate_sessions else 2)
        if control != "approve":
            assert all(path.read_text() == "original" for path in paths)
    finally:
        for session in runners:
            approval.resolve_gateway_approval(session, "deny", resolve_all=True)
        await asyncio.wait_for(asyncio.gather(*workers), 10)
        await adapter.stop_typing("77")
