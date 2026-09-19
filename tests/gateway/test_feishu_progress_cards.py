"""Quiet per-turn progress contracts; no network or live config required."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest



def test_progress_ledger_keeps_actual_previews_and_bounds_without_reasoning():
    from gateway.progress_cards import TurnProgress
    import tools.terminal_tool  # register the real tool's display emoji
    progress = TurnProgress(None, "oc-chat", session_id="session-a")
    progress.record({"type": "tool.started", "tool_call_id": "a", "tool_name": "terminal", "preview": "printf hello"})
    progress.record({"type": "tool.started", "tool_call_id": "b", "tool_name": "terminal", "preview": "pwd"})
    progress.record({"type": "tool.completed", "tool_call_id": "b", "tool_name": "terminal", "is_error": True})
    progress.record({"type": "commentary", "text": "Checking the source first."})
    progress.record({"type": "reasoning.available", "text": "PRIVATE THOUGHT"})
    progress.record({"type": "_thinking", "text": "PRIVATE THOUGHT"})
    progress.record(("__reset__",))
    text = "\n".join(progress.snapshot()["details"])
    assert "💻 terminal" in text and "printf hello" in text and "pwd" in text
    assert "running" in text and "failed" in text
    assert "Checking the source first." in text
    assert "PRIVATE THOUGHT" not in text
    for n in range(150):
        progress.record({"type": "commentary", "text": f"{n} API_KEY=secret-value-123456 https://example.com/?ticket=sensitive-ticket " + "x" * 2000})
    snapshot = progress.snapshot()
    assert snapshot["omitted"] > 0
    assert len(snapshot["details"]) <= 24
    assert len(json.dumps(snapshot).encode()) < 24000
    assert "secret-value-123456" not in json.dumps(snapshot)
    assert "sensitive-ticket" not in json.dumps(snapshot)
    assert all(len(line) <= 600 for line in snapshot["details"])

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


class WiredFeishuAdapter(FeishuAdapter):
    def __init__(self, platform=None):
        super().__init__(PlatformConfig(enabled=True, extra={"progress_cards": True}))
        self.cards = []
        self.sent = []
        response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om-card"))
        self._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(patch=Mock()))))
        async def create(**kwargs):
            self.cards.append((None, json.loads(kwargs["payload"]), kwargs))
            return response
        async def patch(method, request):
            self.cards.append((request.message_id, json.loads(request.request_body.content), {}))
            return response
        self._send_raw_message = create
        self._run_blocking = patch

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        from gateway.platforms.base import SendResult
        self.sent.append(content)
        return SendResult(success=True, message_id="normal-message")


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("outcome", ["completed", "failed", "interrupted", "exception", "cancelled_cleanup"])
async def test_full_gateway_wires_one_card_and_preserves_final(monkeypatch, tmp_path, streaming, outcome):
    import time
    from gateway.config import Platform
    from tests.gateway.test_run_progress_topics import _run_with_agent
    release_calls = []
    if outcome == "cancelled_cleanup":
        import asyncio
        import gateway.progress_cards as progress_cards
        import tests.gateway.test_run_progress_topics as helpers
        original_finish, original_make = progress_cards.finish_progress_card, helpers._make_runner
        async def cancelling_finish(*args, **kwargs):
            await original_finish(*args, **kwargs)
            raise asyncio.CancelledError()
        def make_runner(adapter):
            runner = original_make(adapter)
            original_release = runner._release_running_agent_state
            def release(*args, **kwargs):
                release_calls.append(True)
                return original_release(*args, **kwargs)
            runner._release_running_agent_state = release
            return runner
        monkeypatch.setattr(progress_cards, "finish_progress_card", cancelling_finish)
        monkeypatch.setattr(helpers, "_make_runner", make_runner)
    class Agent:
        def __init__(self, **kwargs):
            self.tools = []
            self.is_interrupted = False
        def run_conversation(self, message, **kwargs):
            assert self.tool_start_callback is not None
            assert self.tool_complete_callback is not None
            assert self.interim_assistant_callback is not None
            # Quiet cards buffer text deltas: no speculative commentary bubbles,
            # and the final is still returned to normal gateway delivery.
            assert self.stream_delta_callback is None
            self.tool_start_callback("a", "skill_view", {"name": "codex"})
            self.tool_progress_callback("tool.started", "skill_view", "codex", {"name": "codex"})
            self.interim_assistant_callback("I will inspect the source first.")
            self.tool_progress_callback("_thinking", "HIDDEN SCRATCH")
            time.sleep(0.2)
            self.tool_complete_callback("a", "skill_view", {}, '{"success": true}')
            self.tool_start_callback("b", "terminal", {"command": "pwd"})
            self.tool_start_callback("c", "read_file", {"path": "AGENTS.md"})
            self.tool_start_callback("d", "search_files", {"pattern": "progress"})
            self.tool_complete_callback("b", "terminal", {}, '{"success": true}')
            self.tool_complete_callback("c", "read_file", {}, '{"success": true}')
            self.tool_complete_callback("d", "search_files", {}, '{"success": true}')
            if outcome == "exception":
                raise RuntimeError("test failure")
            self.is_interrupted = outcome == "interrupted"
            return {"final_response": "Final answer remains visible", "messages": [], "api_calls": 1,
                    "failed": outcome == "failed", "interrupted": outcome == "interrupted"}
    instances = []
    def adapter_factory(platform):
        adapter = WiredFeishuAdapter(platform)
        instances.append(adapter)
        return adapter
    async def invoke():
        return await _run_with_agent(
        monkeypatch, tmp_path, Agent, session_id="feishu-progress-test", platform=Platform.FEISHU,
        adapter_cls=adapter_factory,
        config_data={"display": {"tool_progress": "off", "thinking_progress": True,
                                 "interim_assistant_messages": False, "cleanup_progress": True},
                     "streaming": {"enabled": streaming}},
    )
    if outcome == "cancelled_cleanup":
        with pytest.raises(asyncio.CancelledError):
            await invoke()
        assert release_calls, "Card cleanup cancellation must not skip runner teardown"
        return
    if outcome == "exception":
        with pytest.raises(RuntimeError, match="test failure"):
            await invoke()
        adapter, result = instances[0], {}
    else:
        adapter, result = await invoke()
    assert adapter.cards
    assert sum(mid is None for mid, _, _ in adapter.cards) == 1
    assert all(mid == "om-card" for mid, _, _ in adapter.cards[1:])
    assert not adapter.sent
    final_card = adapter.cards[-1][1]
    text = json.dumps(final_card, ensure_ascii=False)
    expected_colour, expected_label = {
        "completed": ("green", "✓ 本轮已结束"),
        "failed": ("red", "✕ 执行失败"),
        "exception": ("red", "✕ 执行失败"),
        "interrupted": ("grey", "■ 已停止"),
    }[outcome]
    assert final_card["header"]["template"] == expected_colour
    assert final_card["header"]["title"]["content"] == f"Hermes · {expected_label}"
    assert "Reading skill codex" in text
    assert "terminal" in text and "pwd" in text
    assert "Reading AGENTS.md" in text and "Searching files" in text
    assert "I will inspect the source first." in text
    assert "HIDDEN SCRATCH" not in text
    if outcome != "exception":
        assert result["final_response"] == "Final answer remains visible"
        assert not result.get("already_sent")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "failed", "interrupted"])
async def test_turn_card_throttles_but_flushes_terminal_status(monkeypatch, outcome):
    from gateway.progress_cards import TurnProgress
    from gateway.platforms.base import SendResult
    clock = [10.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    adapter = SimpleNamespace(send_progress_card=AsyncMock(return_value=SendResult(success=True, message_id="card-1")), send=AsyncMock())
    progress = TurnProgress(adapter, "chat", session_id="sess", reply_to="anchor", metadata={"thread_id": "thread"})
    progress.record({"type": "tool.started", "tool_call_id": "a", "tool_name": "terminal", "preview": "pwd"})
    await progress.publish()
    progress.record({"type": "commentary", "text": "Checking."})
    await progress.publish()
    assert adapter.send_progress_card.await_count == 1
    clock[0] += 2
    await progress.publish()
    assert adapter.send_progress_card.await_count == 2
    await progress.finish(outcome)
    calls = adapter.send_progress_card.call_args_list
    assert len(calls) == 3
    assert calls[0].kwargs["message_id"] is None
    assert all(c.kwargs["message_id"] == "card-1" for c in calls[1:])
    assert calls[-1].args[1]["status"] == outcome
    assert not any(" — running" in d for d in calls[-1].args[1]["details"])
    await progress.finish(outcome)
    assert adapter.send_progress_card.await_count == 3
    adapter.send.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["create", "patch", "exception"])
async def test_card_failure_attempts_only_one_compact_fallback(stage):
    from gateway.progress_cards import TurnProgress
    from gateway.platforms.base import SendResult
    adapter = SimpleNamespace(send_progress_card=AsyncMock(), send=AsyncMock(side_effect=RuntimeError("also unavailable")))
    ok = SendResult(success=True, message_id="card-1")
    failure = SendResult(success=False, error="secret payload must not appear")
    adapter.send_progress_card.side_effect = [ok, failure] if stage == "patch" else [RuntimeError("private") if stage == "exception" else failure]
    progress = TurnProgress(adapter, "chat", session_id="session-inspect", reply_to="anchor", metadata={"thread_id": "thread"})
    progress.record({"type": "commentary", "text": "First"})
    await progress.publish()
    await progress.finish("failed")
    for _ in range(30):
        await progress.publish(force=True)
    assert adapter.send.await_count == 1
    fallback = adapter.send.call_args
    assert "hermes sessions browse" in fallback.kwargs["content"]
    assert "session-inspect" in fallback.kwargs["content"]
    assert "private" not in fallback.kwargs["content"]
    assert fallback.kwargs["reply_to"] == "anchor"
    assert fallback.kwargs["metadata"]["thread_id"] == "thread"
    assert adapter.send_progress_card.await_count == (3 if stage == "patch" else 1)


@pytest.mark.asyncio
async def test_native_card_creates_once_then_patches_same_message():
    adapter = FeishuAdapter(PlatformConfig(enabled=True, extra={"progress_cards": True}))
    response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om-card"))
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(patch=Mock()))))
    adapter._send_raw_message = AsyncMock(return_value=response)
    adapter._run_blocking = AsyncMock(return_value=response)
    snapshot = {"status": "running", "details": ["web_search — running"], "omitted": 0, "session_id": "test-session"}

    assert adapter.progress_cards_enabled() is True
    first = await adapter.send_progress_card("oc-chat", snapshot, reply_to="om-anchor", metadata={"thread_id": "omt-thread"})
    assert first.success and first.message_id == "om-card"
    sent = adapter._send_raw_message.call_args.kwargs
    assert sent["msg_type"] == "interactive"
    assert sent["reply_to"] == "om-anchor"
    assert sent["metadata"]["thread_id"] == "omt-thread"
    card = json.loads(sent["payload"])
    assert card["schema"] == "2.0"
    panel = next(e for e in card["body"]["elements"] if e["tag"] == "collapsible_panel")
    assert panel["expanded"] is False
    assert "web_search" in json.dumps(panel)

    snapshot["status"] = "completed"
    patched = await adapter.send_progress_card("oc-chat", snapshot, message_id=first.message_id)
    assert patched.success and patched.message_id == first.message_id
    assert adapter._send_raw_message.await_count == 1
    method, request = adapter._run_blocking.call_args.args
    assert method is adapter._client.im.v1.message.patch
    assert request.message_id == "om-card"
    updated = json.loads(request.request_body.content)
    assert updated["header"]["template"] == "green"
    assert updated["header"]["title"]["content"] == "Hermes · ✓ 本轮已结束"


@pytest.mark.asyncio
async def test_chinese_card_respects_serialized_transport_budget():
    adapter = WiredFeishuAdapter()
    snapshot = {"status": "running", "details": [f"{i} 中文内容\\\"&" * 100 for i in range(24)], "omitted": 0, "session_id": "byte-test"}
    await adapter.send_progress_card("oc-chat", snapshot)
    payload = adapter.cards[0][2]["payload"]
    assert len(json.dumps({"content": payload}).encode("utf-8")) < 28000
    assert "23 中文" in payload


def test_late_completion_does_not_resurrect_evicted_tool():
    from gateway.progress_cards import TurnProgress
    p = TurnProgress(None, "chat", session_id="evicted")
    for i in range(30):
        p.record({"type": "tool.started", "tool_call_id": str(i), "tool_name": "read_file", "preview": f"file-{i}"})
    before = p.snapshot()
    p.record({"type": "tool.completed", "tool_call_id": "0", "tool_name": "read_file"})
    assert p.snapshot() == before


@pytest.mark.parametrize("redacted", ['{"command": "broken}', '[]', 'error'])
def test_tool_start_survives_invalid_redacted_json_without_raw_preview(monkeypatch, redacted):
    import queue
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    import agent.redact
    def bad_redactor(value, **kwargs):
        if redacted == 'error':
            raise ValueError('unavailable')
        return redacted
    monkeypatch.setattr(agent.redact, 'redact_sensitive_text', bad_redactor)
    ctx = TurnContext(progress_queue=queue.Queue(), _run_still_current=lambda: True, _progress_cards=True)
    TurnRunner(None, ctx).native_tool_start_callback('call-1', 'terminal', {'command': 'echo PRIVATE_SENTINEL'})
    event = ctx.progress_queue.get_nowait()
    assert event['tool_call_id'] == 'call-1'
    assert 'terminal' in event['label']
    assert 'PRIVATE_SENTINEL' not in json.dumps(event)


def test_progress_cards_are_explicit_yaml_opt_in():
    for value in (None, False, "false", "true", 1):
        assert FeishuAdapter(PlatformConfig(extra={"progress_cards": value})).progress_cards_enabled() is False
