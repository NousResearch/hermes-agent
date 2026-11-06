"""Terminal and execute_code share fenced gateway progress, including #59273."""

import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.run_turn_runner import TurnRunner
from gateway.config import PlatformConfig
from gateway.turn_context import TurnContext
from tests.gateway.test_run_progress_topics import CodeBlockProgressAdapter, _run_with_agent
from tests.waiting import wait_until


def _turn(mode, capable=True):
    adapter = SimpleNamespace(supports_code_blocks=capable, format_tool_preview=lambda p: p.text)
    ctx = TurnContext(progress_mode=mode, tool_progress_enabled=True,
                      progress_queue=queue.Queue(), _run_still_current=lambda: True)
    return TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda _: adapter), ctx), ctx


@pytest.mark.parametrize("mode", ["all", "new", "verbose"])
@pytest.mark.parametrize("tool,key", [("terminal", "command"), ("execute_code", "code")])
@pytest.mark.parametrize("capable", [True, False])
def test_code_progress_respects_mode_and_platform(monkeypatch, mode, tool, key, capable):
    monkeypatch.setattr("agent.display.get_tool_preview_max_len", lambda: 40)
    turn, ctx = _turn(mode, capable)
    code = "print('first')\nprint('second')"
    turn.progress_callback("tool.started", tool, code, {key: code})
    text = ctx.progress_queue.get_nowait()
    assert ("```" in text) is capable
    if capable:
        expected = code if mode == "verbose" else "print('first') ..."
        assert text.endswith(f"```\n{expected}\n```")
    turn.progress_callback("tool.started", tool, "next", {key: "next"})
    assert ctx.progress_queue.empty() is (mode == "new")


@pytest.mark.parametrize("mode", ["all", "verbose"])
def test_only_adjacent_blocks_of_the_same_tool_omit_the_header(mode):
    turn, ctx = _turn(mode)
    sequence = [("execute_code", {"code": "first"}), ("execute_code", {"code": "second"}),
                ("terminal", {"command": "pwd"}), ("execute_code", {"code": "third"}),
                ("search_files", {}), ("execute_code", {"code": "fourth"})]
    for index, (tool, args) in enumerate(sequence):
        turn.progress_callback("tool.started", tool, None, args)
        text = ctx.progress_queue.get_nowait()
        if tool in ("terminal", "execute_code"):
            assert text.startswith("```\n") is (index == 1)
            if index != 1:
                assert tool in text.split("```", 1)[0]


@pytest.mark.asyncio
async def test_execute_code_progress_reaches_the_gateway_adapter(monkeypatch, tmp_path):
    class Capture(CodeBlockProgressAdapter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            Agent.adapter = self

    class Agent:
        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            self.tool_progress_callback("tool.started", "execute_code", "print('first')",
                                        {"code": "print('first')\nprint('second')"})
            wait_until(lambda: self.adapter.sent, what="execute_code progress was sent")
            return {"final_response": "done", "messages": [], "api_calls": 1}

    adapter, result = await _run_with_agent(monkeypatch, tmp_path, Agent, session_id="code-progress",
                                           config_data={"display": {"tool_progress": "all"}}, adapter_cls=Capture)
    assert result["final_response"] == "done"
    text = " ".join(call["content"] for call in adapter.sent + adapter.edits)
    assert "```\nprint('first') ...\n```" in text
    assert "print('second')" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["all", "verbose"])
@pytest.mark.parametrize("tool,key", [("terminal", "command"), ("execute_code", "code")])
async def test_slack_send_and_edit_preserve_the_bare_code_block(mode, tool, key):
    from tests.gateway.test_slack import SlackAdapter

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._app = SimpleNamespace(client=AsyncMock())
    adapter._running = True
    ctx = TurnContext(progress_mode=mode, progress_queue=queue.Queue())
    turn = TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda _: adapter), ctx)
    code = "first_line\nsecond_line"
    text = turn._progress_build_message(tool, code, {key: code})
    text = ctx.progress_queue.get_nowait() if text is None else text
    client = adapter._app.client
    client.chat_postMessage.return_value = {"ok": True, "ts": "123.456"}
    client.chat_update.return_value = {"ok": True, "ts": "123.456"}

    assert (await adapter.send("C123", text)).success
    assert (await adapter.edit_message("C123", "123.456", text)).success
    body = code if mode == "verbose" else "first_line ..."
    for method in (client.chat_postMessage, client.chat_update):
        assert method.call_args.kwargs["text"].endswith(f"```\n{body}\n```")
