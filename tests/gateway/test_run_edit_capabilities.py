"""Adapter edit capabilities on the gateway turn paths: the SUPPORTS_STREAMING_EDITS and
SUPPORTS_PROGRESS_EDITS gates, Signal-style timestamp chains (EDIT_RESULT_ID_IS_NEXT_TARGET) on the
progress bubble and the heartbeat, and progress sends that return no edit handle."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult, utf16_len
from plugins.platforms.signal.adapter import SignalAdapter
from tests.gateway.test_run_progress_topics import _ACTIVE_ADAPTER, FakeAgent, ProgressCaptureAdapter, _run_with_agent
from tests.waiting import wait_until

_SIGNAL_PROGRESS_ON = {"display": {"platforms": {"signal": {"tool_progress": "all"}}, "interim_assistant_messages": False}}


class _MintingProgressAdapter(ProgressCaptureAdapter):
    """Every send/edit answers with a fresh id, as Signal's timestamps and Matrix's replacement events do."""

    SUPPORTS_MESSAGE_EDITING = True
    SUPPORTS_STREAMING_EDITS = False
    SUPPORTS_PROGRESS_EDITS = True

    def __init__(self, platform=Platform.SIGNAL):
        super().__init__(platform=platform)
        self._minted = 0

    def _mint(self):
        self._minted += 1
        return f"progress-{self._minted}"

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)
        return SendResult(success=True, message_id=self._mint())

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        await super().edit_message(chat_id, message_id, content)
        return SendResult(success=True, message_id=self._mint())


class _TimestampChainProgressAdapter(_MintingProgressAdapter):
    EDIT_RESULT_ID_IS_NEXT_TARGET = True


class _NoProgressEditAdapter(_MintingProgressAdapter):
    """Explicit edits allowed, automatic progress edits declined."""

    SUPPORTS_PROGRESS_EDITS = False


class _MissingIdProgressAdapter(_MintingProgressAdapter):
    """Delivers progress but never returns an edit handle."""

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        await ProgressCaptureAdapter.send(self, chat_id, content, reply_to=reply_to, metadata=metadata)
        return SendResult(success=True, message_id=None)

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        raise AssertionError("progress without an edit handle must not attempt edits")


class _PacedToolAgent:
    """One tool event, then two more once the bubble exists. The progress drain throttles edits to
    one per 1.5 s, so the pair lands as ONE update, and the turn-end flush edits the bubble once more:
    two successive edits of one bubble without the test sleeping through the throttle."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        adapter = _ACTIVE_ADAPTER["adapter"]
        self.tool_progress_callback("tool.started", "terminal", "first", {})
        wait_until(lambda: adapter.sent, what="the progress bubble was sent")
        self.tool_progress_callback("tool.started", "browser_navigate", "https://example.com/second", {})
        self.tool_progress_callback("tool.started", "terminal", "third", {})
        wait_until(lambda: len(adapter.sent) + len(adapter.edits) >= 2, what="the throttled update was delivered")
        return {"final_response": "done", "messages": [], "api_calls": 1}


class _StreamingAgent:
    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        if self.stream_delta_callback:
            self.stream_delta_callback("Hello")
            self.stream_delta_callback(" world")
        return {"final_response": "Hello world", "messages": [], "api_calls": 1}


class _RecordingSignalAdapter(SignalAdapter):
    """The real Signal adapter with its JSON-RPC stubbed; every send/edit params dict is kept."""

    def __init__(self, platform=Platform.SIGNAL):
        super().__init__(PlatformConfig(enabled=True, extra={"http_url": "http://localhost:8080",
                                                             "account": "+155****0000"}))
        self.rpc_sends = []

    async def _rpc(self, method, params, rpc_id=None, **kwargs):
        if method == "send":
            self.rpc_sends.append(dict(params))
        return {"timestamp": 1712345678000 + len(self.rpc_sends)}


class _EmojiBurstAgent:
    """One progress line, then a burst of emoji-heavy lines (two UTF-16 units per rocket) that the
    turn-end drain rolls into fresh bubbles."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        adapter = _ACTIVE_ADAPTER["adapter"]
        self.tool_progress_callback("tool.started", "emoji_probe", "000 " + "\U0001F680" * 36, {})
        wait_until(lambda: adapter.rpc_sends, what="the progress bubble was sent")
        for idx in range(1, 200):
            self.tool_progress_callback("tool.started", "emoji_probe", f"{idx:03d} " + "\U0001F680" * 36, {})
        return {"final_response": "done", "messages": [], "api_calls": 1}


async def _run_signal(monkeypatch, tmp_path, agent_cls, adapter_cls, session_id, config_data=None):
    return await _run_with_agent(
        monkeypatch, tmp_path, agent_cls, session_id=session_id, config_data=config_data or _SIGNAL_PROGRESS_ON,
        platform=Platform.SIGNAL, chat_id="+155****4567", chat_type="dm", thread_id=None, adapter_cls=adapter_cls,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("opted_out", "interim"), [
    (True, False),  # no consumer at all: the config builder refuses streaming
    (True, True),  # an interim-commentary consumer exists, but is never fed token deltas
    (False, False),  # control: the same setup streams when streaming edits are allowed
])
async def test_token_streaming_honours_the_streaming_edit_opt_out(monkeypatch, tmp_path, opted_out, interim):
    """An editable adapter that declines streaming edits (Signal) gets no streamed preview even with
    streaming switched on for it."""
    config = {
        "display": {"platforms": {"signal": {"streaming": True}}, "tool_progress": "off",
                    "interim_assistant_messages": interim},
        "streaming": {"enabled": True, "edit_interval": 0.01, "buffer_threshold": 1},
    }
    adapter_cls = _MintingProgressAdapter if opted_out else type(
        "_StreamingEditsAdapter", (_MintingProgressAdapter,), {"SUPPORTS_STREAMING_EDITS": True})
    adapter, result = await _run_signal(monkeypatch, tmp_path, _StreamingAgent, adapter_cls, "sess-stream-gate", config)

    assert result["final_response"] == "Hello world"
    if opted_out:
        assert result.get("already_sent") is not True
        assert adapter.sent == []
        assert adapter.edits == []
    else:
        assert adapter.sent, "control: a streaming-capable adapter must receive the streamed preview"


@pytest.mark.parametrize("opted_out", [True, False])
def test_stream_consumer_config_treats_a_streaming_opt_out_as_non_editing(opted_out):
    """The shared config builder: no consumer when the caller asks to skip non-editable platforms,
    and no edit cursor on the cursorless fallback."""
    from gateway.config import StreamingConfig
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    runner = object.__new__(GatewayRunner)
    source = SessionSource(platform=Platform.SIGNAL, chat_id="+155****4567", chat_type="dm")
    adapter = _MintingProgressAdapter()
    if not opted_out:
        adapter.SUPPORTS_STREAMING_EDITS = True
    scfg = StreamingConfig()

    fallback_cfg, _ = runner._build_stream_consumer_config(source, scfg, adapter, on_missing_cursor="fallback")

    assert fallback_cfg.cursor == ("" if opted_out else scfg.cursor)
    if opted_out:
        with pytest.raises(RuntimeError):
            runner._build_stream_consumer_config(source, scfg, adapter, on_missing_cursor="raise")
    else:
        runner._build_stream_consumer_config(source, scfg, adapter, on_missing_cursor="raise")


@pytest.mark.asyncio
async def test_progress_respects_explicit_progress_edit_opt_out(monkeypatch, tmp_path):
    """An adapter may allow explicit edits while declining the automatic progress bubble."""
    adapter, result = await _run_signal(monkeypatch, tmp_path, FakeAgent, _NoProgressEditAdapter, "sess-no-progress")

    assert result["final_response"] == "done"
    assert adapter.sent == []
    assert adapter.edits == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("adapter_cls", "targets"), [
    (_TimestampChainProgressAdapter, ["progress-1", "progress-2"]),  # Signal: each edit mints the next
    (_MintingProgressAdapter, ["progress-1", "progress-1"]),  # Matrix: keep editing the original event
])
async def test_progress_edits_follow_the_adapter_edit_target_contract(monkeypatch, tmp_path, adapter_cls, targets):
    adapter, result = await _run_signal(monkeypatch, tmp_path, _PacedToolAgent, adapter_cls, "sess-progress-chain")

    assert result["final_response"] == "done"
    assert len(adapter.sent) == 1
    assert [call["message_id"] for call in adapter.edits] == targets
    assert [len(call["content"].splitlines()) for call in adapter.edits] == [3, 3]


@pytest.mark.asyncio
async def test_progress_without_an_edit_handle_degrades_to_one_line_per_update(monkeypatch, tmp_path):
    """A delivered but unaddressable bubble must not replay the accumulated transcript each tick."""
    adapter, result = await _run_signal(monkeypatch, tmp_path, _PacedToolAgent, _MissingIdProgressAdapter, "sess-no-id")

    assert result["final_response"] == "done"
    assert [len(call["content"].splitlines()) for call in adapter.sent] == [1, 1]
    assert adapter.edits == []


@pytest.mark.asyncio
async def test_signal_progress_bubbles_roll_over_within_the_utf16_limit(monkeypatch, tmp_path):
    """Signal's 8000 limit counts UTF-16 units: an emoji-heavy bubble must roll over before it, or
    send() splits it into "(1/2)" parts and later edits push the whole oversized text at one part."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    adapter, result = await _run_signal(monkeypatch, tmp_path, _EmojiBurstAgent, _RecordingSignalAdapter, "sess-utf16")

    messages = [params["message"] for params in adapter.rpc_sends]
    assert result["final_response"] == "done"
    assert sum("editTimestamp" not in params for params in adapter.rpc_sends) >= 2  # rolled into a new bubble
    assert any("editTimestamp" in params for params in adapter.rpc_sends)
    assert max(utf16_len(text) for text in messages) <= adapter.MAX_MESSAGE_LENGTH
    assert not any(text.endswith("(1/2)") for text in messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(("chains", "targets"), [(True, ["hb-1", "hb-2"]), (False, ["hb-1", "hb-1"])])
async def test_long_running_heartbeat_follows_the_adapter_edit_target_contract(monkeypatch, chains, targets):
    """The in-place "still working" heartbeat targets each fresh timestamp only on chained adapters."""
    from gateway.run import GatewayRunner
    from gateway.turn_context import TurnContext

    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.01")
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._draining = runner._restart_requested = False
    adapter = MagicMock()
    adapter.EDIT_RESULT_ID_IS_NEXT_TARGET = chains
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="hb-1"))
    minted = iter(["hb-2", "hb-3"])

    async def _edit(chat_id, message_id, content):
        result = SimpleNamespace(success=True, message_id=next(minted))
        if result.message_id == "hb-3":
            runner._restart_requested = True  # stop the loop after the second edit
        return result

    adapter.edit_message = AsyncMock(side_effect=_edit)
    runner._delivery_adapter_for = lambda source: adapter
    runner._agent_activity_summary = staticmethod(lambda agent: None)
    agent = MagicMock()
    runner._running_agents["sess"] = agent
    disp = MagicMock()
    disp._display_surface_mode.return_value = "on"
    disp.resolve_display_setting.return_value = False
    ctx = TurnContext(source=SimpleNamespace(chat_id="c", platform=Platform.SIGNAL), session_key="sess")
    ctx.agent_holder[0] = agent

    await asyncio.wait_for(runner._run_agent_notify_long_running(disp, ctx, [None]), 5)

    assert adapter.send.await_count == 1
    assert [call.args[1] for call in adapter.edit_message.await_args_list] == targets
