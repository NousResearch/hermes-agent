"""Transport-neutral live HUD state and publisher contracts."""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.live_hud import LiveHUDProjector, LiveHUDPublisher
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from plugins.platforms.telegram.adapter import TelegramAdapter


class CaptureTransport:
    def __init__(
        self,
        *,
        fail_send: bool = False,
        fail_edits: bool = False,
        send_message_id: str | None = "hud-1",
    ) -> None:
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self.sent_event = asyncio.Event()
        self.fail_send = fail_send
        self.fail_edits = fail_edits
        self.send_message_id = send_message_id

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        self.sent_event.set()
        return SendResult(success=not self.fail_send, message_id=self.send_message_id)

    async def edit_message(
        self, chat_id, message_id, content, *, finalize=False, metadata=None
    ):
        self.edits.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "content": content,
            "finalize": finalize,
            "metadata": metadata,
        })
        if self.fail_edits:
            return SendResult(success=False, error="edit failed")
        return SendResult(success=True, message_id=message_id)


def test_projector_renders_only_closed_vocabulary_and_numeric_counters():
    private_values = (
        "private-user-marker",
        "private-path-marker",
        "private-url-marker",
        "private-id-marker",
    )
    hud = LiveHUDProjector(started_at=100.0, clock=lambda: 207.0)
    hud.observe(
        "tool.started", tool_name="terminal", args={"command": private_values[1]}
    )
    hud.observe(
        "tool.completed", tool_name="terminal", result={"output": private_values[0]}
    )
    hud.observe("subagent.start", preview=private_values[2], goal=private_values[3])
    hud.observe("status", preview=" ".join(private_values))

    active = hud.render()
    assert "⚡ HERMES · ACTIVE" in active
    assert "Task: Current turn" in active
    assert "● shell" in active
    assert "Subagents: 1 active" in active
    assert "Tools: 1 · Runtime: 01:47" in active
    assert not any(value in active for value in private_values)

    hud.observe("subagent.complete", status="completed", goal=private_values[3])
    hud.complete({"api_calls": 3, "usage": {"input_tokens": 1200, "output_tokens": 80}})
    complete = hud.render()
    assert "✓ HERMES · COMPLETE" in complete
    assert "Tools: 1 · Runtime: 01:47" in complete
    assert "Usage: 3 calls · 1.2k in · 80 out" in complete
    assert "◉ Working" not in complete


@pytest.mark.parametrize(
    ("result", "header"),
    [
        ({"completed": True}, "✓ HERMES · COMPLETE"),
        ({"failed": True}, "✗ HERMES · FAILED"),
        ({"interrupted": True, "completed": False}, "■ HERMES · INTERRUPTED"),
        ({"completed": False}, "⚠ HERMES · INCOMPLETE"),
    ],
)
def test_projector_renders_terminal_turn_state(result, header):
    hud = LiveHUDProjector(started_at=0.0, clock=lambda: 1.0)
    hud.complete(result)
    assert header in hud.render()


@pytest.mark.parametrize("status", ["failed", "error", "timeout"])
def test_projector_preserves_failed_subagent_count_without_rendering_payload(status):
    marker = "private-subagent-payload"
    hud = LiveHUDProjector(started_at=0.0, clock=lambda: 1.0)
    hud.observe("subagent.start", goal=marker)
    hud.observe("subagent.complete", status=status, goal=marker, summary=marker)
    assert "1 failed" in hud.render()
    assert marker not in hud.render()


def test_projector_handles_parallel_same_tool_without_rendering_ambiguous_payloads():
    marker = "private-parallel-result"
    hud = LiveHUDProjector(started_at=0.0, clock=lambda: 1.0)
    hud.observe("tool.started", tool_name="terminal", args={"command": marker})
    hud.observe("tool.started", tool_name="terminal", args={"command": marker})
    hud.observe("tool.completed", tool_name="terminal", result=marker)
    assert hud._pending_tools == {"terminal": 1}
    assert marker not in hud.render()
    hud.observe("tool.completed", tool_name="terminal", result=marker)
    assert hud._pending_tools == {}
    assert "Tools: 2" in hud.render()


def test_projector_maps_unknown_tool_names_to_generic_label():
    marker = "private-custom-tool-name"
    hud = LiveHUDProjector(started_at=0.0, clock=lambda: 1.0)
    hud.observe("tool.started", tool_name=marker)
    assert "● tool" in hud.render()
    assert marker not in hud.render()


@pytest.mark.asyncio
async def test_publisher_sends_once_then_edits_same_message_to_completion():
    transport = CaptureTransport()
    publisher = LiveHUDPublisher(
        transport=transport,
        chat_id="42",
        metadata={"message_thread_id": "7"},
        min_edit_interval=0.0,
        refresh_interval=0.01,
    )
    task = asyncio.create_task(publisher.run())
    await asyncio.wait_for(transport.sent_event.wait(), timeout=2)
    publisher.observe(
        "tool.started", tool_name="terminal", args={"command": "pytest tests/"}
    )
    publisher.observe(
        "tool.completed",
        tool_name="terminal",
        result='{"output":"2 passed","exit_code":0}',
    )
    publisher.complete({"api_calls": 1, "completed": True})
    await asyncio.wait_for(task, timeout=2)

    assert len(transport.sent) == 1
    assert transport.sent[0]["metadata"] == {
        "message_thread_id": "7",
        "_interim_send": True,
    }
    assert transport.edits
    assert {edit["message_id"] for edit in transport.edits} == {"hud-1"}
    assert transport.edits[-1]["finalize"] is True
    assert "✓ HERMES · COMPLETE" in transport.edits[-1]["content"]
    assert "pytest tests/" not in transport.edits[-1]["content"]
    assert "2 passed" not in transport.edits[-1]["content"]


@pytest.mark.asyncio
async def test_publisher_edit_failure_is_fail_open_and_never_sends_replacement():
    transport = CaptureTransport(fail_edits=True)
    publisher = LiveHUDPublisher(
        transport=transport,
        chat_id="42",
        min_edit_interval=0.0,
        refresh_interval=0.01,
    )
    task = asyncio.create_task(publisher.run())
    await asyncio.wait_for(transport.sent_event.wait(), timeout=2)
    publisher.observe("tool.started", tool_name="read_file", args={"path": "README.md"})
    publisher.complete({"failed": True, "error": "provider failed"})
    await asyncio.wait_for(task, timeout=2)

    assert len(transport.sent) == 1
    assert len(transport.edits) == 1
    assert publisher.failed is True


@pytest.mark.asyncio
async def test_publisher_transport_exception_never_logs_exception_text(caplog):
    marker = "private-exception-marker"

    class RaisingTransport(CaptureTransport):
        async def send(self, *_args, **_kwargs):
            raise RuntimeError(marker)

    publisher = LiveHUDPublisher(
        transport=RaisingTransport(),
        chat_id="42",
        min_edit_interval=0.0,
        refresh_interval=0.01,
    )
    await publisher.run()

    assert publisher.failed is True
    assert marker not in caplog.text


@pytest.mark.asyncio
async def test_telegram_adapter_shares_edit_budget_across_callers(monkeypatch):
    telegram_module = importlib.import_module("plugins.platforms.telegram.adapter")
    now = [10.0]
    sleeps: list[float] = []

    async def advance(delay):
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr(telegram_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(telegram_module.asyncio, "sleep", advance)
    monkeypatch.setattr(telegram_module, "_MESSAGE_EDIT_LANE_CAP", 2)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    bot = SimpleNamespace(edit_message_text=AsyncMock())
    adapter._bot = bot

    await adapter._edit_text("shared-chat", "1", "first")
    await adapter._edit_text("shared-chat", "1", "second")
    await adapter._edit_text("other-chat", "1", "third")
    await adapter._edit_text("third-chat", "1", "fourth")
    await adapter._edit_text("fourth-chat", "1", "fifth")

    # Once the per-chat map is full, excess chats share one conservative overflow
    # lane instead of evicting a still-rate-limited destination.
    assert sleeps == [pytest.approx(0.8), pytest.approx(0.8)]
    assert bot.edit_message_text.await_count == 5
    assert len(adapter._message_edit_locks) == 2
    assert len(adapter._message_edit_last_at) == 2


@pytest.mark.asyncio
async def test_telegram_adapter_holds_chat_lane_until_edit_request_finishes(
    monkeypatch,
):
    telegram_module = importlib.import_module("plugins.platforms.telegram.adapter")
    monkeypatch.setattr(telegram_module, "_MIN_MESSAGE_EDIT_INTERVAL_SECS", 0.0)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_edit(**_kwargs):
        started.set()
        await release.wait()

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    bot = SimpleNamespace(edit_message_text=AsyncMock(side_effect=blocked_edit))
    adapter._bot = bot
    first = asyncio.create_task(adapter._edit_text("shared-chat", "1", "first"))
    await asyncio.wait_for(started.wait(), timeout=2)
    second = asyncio.create_task(adapter._edit_text("shared-chat", "1", "second"))
    await asyncio.sleep(0)
    assert bot.edit_message_text.await_count == 1
    release.set()
    await asyncio.gather(first, second)
    assert bot.edit_message_text.await_count == 2


@pytest.mark.asyncio
async def test_telegram_rich_edit_uses_shared_lane_and_retries_short_flood(monkeypatch):
    telegram_module = importlib.import_module("plugins.platforms.telegram.adapter")
    monkeypatch.setattr(telegram_module, "_MIN_MESSAGE_EDIT_INTERVAL_SECS", 0.0)

    class ShortFlood(Exception):
        retry_after = 0.01

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    bot = SimpleNamespace(do_api_request=AsyncMock(side_effect=[ShortFlood(), {}]))
    adapter._bot = bot
    result = await adapter._try_edit_rich("shared-chat", "1", "**status**")

    assert result is not None and result.success is True
    assert bot.do_api_request.await_count == 2
    assert "shared-chat" in adapter._message_edit_locks


@pytest.mark.asyncio
async def test_telegram_final_edit_does_not_plain_retry_long_flood():
    class LongFlood(Exception):
        retry_after = 30.0

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    bot = SimpleNamespace(edit_message_text=AsyncMock(side_effect=LongFlood()))
    adapter._bot = bot
    result = await adapter.edit_message("shared-chat", "1", "status", finalize=True)

    assert result.success is False
    assert result.retry_after == 30.0
    assert bot.edit_message_text.await_count == 1


@pytest.mark.asyncio
async def test_telegram_rich_and_legacy_edits_share_in_flight_lock(monkeypatch):
    telegram_module = importlib.import_module("plugins.platforms.telegram.adapter")
    monkeypatch.setattr(telegram_module, "_MIN_MESSAGE_EDIT_INTERVAL_SECS", 0.0)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_rich(*_args, **_kwargs):
        started.set()
        await release.wait()
        return {}

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    bot = SimpleNamespace(
        do_api_request=AsyncMock(side_effect=blocked_rich),
        edit_message_text=AsyncMock(),
    )
    adapter._bot = bot
    rich = asyncio.create_task(adapter._try_edit_rich("shared-chat", "1", "**status**"))
    await asyncio.wait_for(started.wait(), timeout=2)
    legacy = asyncio.create_task(adapter._edit_text("shared-chat", "1", "status"))
    await asyncio.sleep(0)
    assert bot.edit_message_text.await_count == 0
    release.set()
    await asyncio.gather(rich, legacy)
    assert bot.edit_message_text.await_count == 1


@pytest.mark.parametrize(
    "transport",
    [CaptureTransport(fail_send=True), CaptureTransport(send_message_id=None)],
)
@pytest.mark.asyncio
async def test_publisher_initial_send_failure_never_sends_replacement(transport):
    publisher = LiveHUDPublisher(
        transport=transport,
        chat_id="42",
        min_edit_interval=0.0,
        refresh_interval=0.01,
    )
    await asyncio.wait_for(publisher.run(), timeout=2)
    assert len(transport.sent) == 1
    assert transport.edits == []
    assert publisher.failed is True


@pytest.mark.asyncio
async def test_publisher_skips_unchanged_nonfinal_edit():
    transport = CaptureTransport()
    publisher = LiveHUDPublisher(
        transport=transport,
        chat_id="42",
        min_edit_interval=0.0,
        refresh_interval=0.01,
    )
    publisher.message_id = "hud-1"
    publisher._last_content = publisher.projector.render()

    assert await publisher._publish(final=False) is True
    assert transport.edits == []


def test_publisher_completion_is_nonblocking_when_event_queue_is_full():
    publisher = LiveHUDPublisher(
        transport=CaptureTransport(),
        chat_id="42",
        min_edit_interval=0.0,
        refresh_interval=0.01,
    )
    for index in range(publisher.events.maxsize):
        publisher.events.put_nowait(("status", None, f"event {index}", None, {}))

    publisher.complete({"completed": True, "api_calls": 1})
    changed, final = publisher._drain()

    assert changed is True
    assert final is True
    assert "✓ HERMES · COMPLETE" in publisher.projector.render()


def test_live_hud_config_is_opt_in_and_platform_scoped():
    from gateway.display_config import resolve_display_setting

    assert resolve_display_setting({}, "telegram", "live_hud") is False
    config = {"display": {"platforms": {"telegram": {"live_hud": True}}}}
    assert resolve_display_setting(config, "telegram", "live_hud") is True
    assert resolve_display_setting(config, "discord", "live_hud") is False


class CaptureHUD:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def observe(self, *args, **kwargs) -> None:
        self.events.append((args, kwargs))


def test_turn_runner_tees_structured_runtime_events_to_hud_when_tool_progress_is_off():
    hud = CaptureHUD()
    ctx = TurnContext(
        source=SimpleNamespace(
            chat_id="42", platform=SimpleNamespace(value="telegram")
        ),
        _run_still_current=lambda: True,
        progress_queue=None,
        tool_progress_enabled=False,
        live_hud_enabled=True,
        live_hud_publisher=hud,
        _status_adapter=None,
    )
    runner = TurnRunner(SimpleNamespace(), ctx)

    runner.progress_callback(
        "tool.started", "terminal", "pytest", {"command": "pytest"}
    )
    runner.progress_callback(
        "tool.completed",
        "terminal",
        None,
        None,
        duration=1.2,
        is_error=False,
        result='{"output":"4 passed"}',
    )
    runner.progress_callback("subagent.start", preview="review the bridge")
    runner.progress_callback(
        "subagent.complete",
        preview="review failed",
        status="error",
        goal="review the bridge",
        summary="review failed",
    )
    runner._status_callback_sync("compacting", "Compacting context")

    assert [event[0][0] for event in hud.events] == [
        "tool.started",
        "tool.completed",
        "subagent.start",
        "subagent.complete",
        "status",
    ]
    assert hud.events[1][1]["result"] == '{"output":"4 passed"}'
    assert hud.events[3][1]["status"] == "error"
    assert hud.events[4][0][2] == "Compacting context"


class HUDIntegrationAdapter(BasePlatformAdapter):
    def __init__(self) -> None:
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent: list[dict] = []
        self.edits: list[dict] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id="live-hud-1")

    async def edit_message(
        self, chat_id, message_id, content, *, finalize=False, metadata=None
    ):
        self.edits.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "content": content,
            "finalize": finalize,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def stop_typing(self, chat_id):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class HUDIntegrationAgent:
    def __init__(self, **kwargs) -> None:
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.status_callback = kwargs.get("status_callback")
        self.tools = []
        self.context_compressor = SimpleNamespace(
            last_prompt_tokens=1000, context_length=200000
        )
        self.session_prompt_tokens = 1000
        self.session_completion_tokens = 25

    def run_conversation(
        self, message, conversation_history=None, task_id=None, **kwargs
    ):
        self.status_callback("compacting", "Compacting context")
        self.tool_progress_callback(
            "tool.started", "terminal", "pytest tests/", {"command": "pytest tests/"}
        )
        time.sleep(0.05)
        self.tool_progress_callback(
            "tool.completed",
            "terminal",
            None,
            None,
            duration=0.05,
            is_error=False,
            result='{"output":"8 passed in 0.1s","exit_code":0}',
        )
        self.tool_progress_callback("subagent.start", preview="Review HUD")
        self.tool_progress_callback(
            "subagent.complete",
            preview="Looks good",
            status="completed",
            goal="Review HUD",
        )
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 2,
            "completed": True,
            "failed": False,
            "interrupted": False,
            "usage": {"input_tokens": 1000, "output_tokens": 25},
        }


def _gateway_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


@pytest.mark.parametrize(
    "scenario",
    [
        "normal",
        "post-turn-failure",
        "queued-followup",
        "queued-followup-timeout",
        "queued-followup-inherited-suppression",
    ],
)
@pytest.mark.asyncio
async def test_gateway_telegram_live_hud_uses_one_message_and_finalizes_it(
    monkeypatch,
    tmp_path,
    scenario,
):
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.dump({
            "display": {
                "platforms": {
                    "telegram": {
                        "live_hud": True,
                        "tool_progress": "all",
                        "interim_assistant_messages": False,
                    }
                }
            }
        }),
        encoding="utf-8",
    )
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = HUDIntegrationAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.01")

    adapter = HUDIntegrationAdapter()
    runner = _gateway_runner(adapter)
    if scenario == "post-turn-failure":
        runner._run_agent_evict_on_fallback = lambda _ctx: (_ for _ in ()).throw(
            RuntimeError("post-turn failure")
        )
    elif scenario in {
        "queued-followup",
        "queued-followup-timeout",
        "queued-followup-inherited-suppression",
    }:
        hud_release = asyncio.Event()
        if scenario == "queued-followup-timeout":
            run_turn = importlib.import_module("gateway.run_turn")
            monkeypatch.setattr(run_turn, "_LIVE_HUD_CLEANUP_TIMEOUT", 0.01)

            async def stubborn_hud(_self):
                while not hud_release.is_set():
                    try:
                        await hud_release.wait()
                    except asyncio.CancelledError:
                        continue

            monkeypatch.setattr(TurnRunner, "send_live_hud", stubborn_hud)

        async def drain_pending(*_args):
            return None, "queued"

        async def run_followup(*args, **kwargs):
            if scenario == "queued-followup-timeout":
                assert kwargs["suppress_live_hud"] is True
                hud_release.set()
                await asyncio.sleep(0)
            elif scenario == "queued-followup-inherited-suppression":
                assert kwargs["suppress_live_hud"] is True
            else:
                assert kwargs["suppress_live_hud"] is False
                assert adapter.edits[-1]["finalize"] is True
                assert "✓ HERMES · COMPLETE" in adapter.edits[-1]["content"]
            return args[5]

        runner._run_agent_drain_pending = drain_pending
        runner._run_agent_queued_followup = run_followup
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
        thread_id="7",
    )

    run = runner._run_agent(
        message="Implement Telegram Live HUD",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-hud",
        session_key="agent:main:telegram:dm:42:7",
        _suppress_live_hud=scenario == "queued-followup-inherited-suppression",
    )
    if scenario == "post-turn-failure":
        with pytest.raises(RuntimeError, match="post-turn failure"):
            await run
        assert len(adapter.sent) == 1
        assert "✗ HERMES · FAILED" in adapter.edits[-1]["content"]
        return
    result = await run

    if scenario in {
        "queued-followup-timeout",
        "queued-followup-inherited-suppression",
    }:
        assert result["final_response"] == "done"
        return

    assert result["final_response"] == "done"
    assert len(adapter.sent) == 1
    assert "⚡ HERMES · ACTIVE" in adapter.sent[0]["content"]
    assert adapter.edits
    assert {edit["message_id"] for edit in adapter.edits} == {"live-hud-1"}
    assert "✓ HERMES · COMPLETE" in adapter.edits[-1]["content"]
    assert "Usage: 2 calls · 1.0k in · 25 out" in adapter.edits[-1]["content"]
    all_hud_content = "\n".join([
        adapter.sent[0]["content"],
        *(edit["content"] for edit in adapter.edits),
    ])
    assert "Implement Telegram Live HUD" not in all_hud_content
    assert "pytest tests/" not in all_hud_content
    assert "8 passed in 0.1s" not in all_hud_content
    assert "Review HUD" not in all_hud_content
    assert "Compacting context" not in all_hud_content
    assert adapter.edits[-1]["finalize"] is True
    assert (adapter.sent[0]["metadata"] or {}).get("thread_id") == "7"
    assert (adapter.sent[0]["metadata"] or {}).get("_interim_send") is True
    assert result.get("already_sent") is None

    runner._should_send_voice_reply = lambda *args, **kwargs: False
    delivered = await runner._hmwa_deliver_turn_response(
        SimpleNamespace(),
        source,
        SimpleNamespace(session_id="sess-hud"),
        "agent:main:telegram:dm:42:7",
        1,
        result,
        [],
        result["final_response"],
        "",
        False,
    )
    assert delivered == "done"
    await adapter.send(
        source.chat_id,
        delivered,
        metadata={"thread_id": source.thread_id},
    )
    assert len(adapter.sent) == 2
    assert adapter.sent[-1]["content"] == "done"
    assert adapter.sent[-1]["metadata"] == {"thread_id": "7"}


@pytest.mark.asyncio
async def test_telegram_hud_send_preserves_topic_and_reply_anchor():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=11)),
    )
    adapter._bot = bot

    result = await adapter.send(
        "100",
        "HUD",
        reply_to="5",
        metadata={"thread_id": "7", "_interim_send": True},
    )

    assert result.success is True
    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["message_thread_id"] == 7
    assert kwargs["reply_to_message_id"] == 5


@pytest.mark.asyncio
async def test_cleanup_releases_session_and_children_when_cancelled_during_hud_finalize():
    adapter = HUDIntegrationAdapter()
    runner = _gateway_runner(adapter)
    released: list[tuple] = []
    runner._release_running_agent_state = lambda key, run_generation=None: (
        released.append((key, run_generation))
    )
    runner._draining = False
    turn_ctx = TurnContext(session_key="session-1", run_generation=4)

    hud_started = asyncio.Event()
    hud_release = asyncio.Event()
    tracking_started = asyncio.Event()

    async def wait_for_hud_release():
        hud_started.set()
        await hud_release.wait()

    async def track_forever():
        tracking_started.set()
        await asyncio.Event().wait()

    hud_task = asyncio.create_task(wait_for_hud_release())
    tracking_task = asyncio.create_task(track_forever())
    await asyncio.wait_for(hud_started.wait(), timeout=2)
    await asyncio.wait_for(tracking_started.wait(), timeout=2)
    cleanup = asyncio.create_task(
        runner._run_agent_cleanup_turn_tasks(
            turn_ctx,
            progress_task=None,
            log_task=None,
            hud_task=hud_task,
            interrupt_monitor=None,
            _notify_task=None,
            tracking_task=tracking_task,
            stream_task=None,
        )
    )
    await asyncio.sleep(0)  # let cleanup enter its shielded HUD wait
    cleanup.cancel()
    await asyncio.sleep(0)
    assert hud_task.cancelled() is False
    hud_release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cleanup, timeout=2)
    assert released == [("session-1", 4)]
    assert hud_task.done()
    assert tracking_task.cancelled()


@pytest.mark.asyncio
async def test_cleanup_does_not_reawait_hud_that_suppresses_cancellation(monkeypatch):
    run_turn = importlib.import_module("gateway.run_turn")
    monkeypatch.setattr(run_turn, "_LIVE_HUD_CLEANUP_TIMEOUT", 0.01)
    runner = _gateway_runner(HUDIntegrationAdapter())
    released: list[tuple] = []
    runner._release_running_agent_state = lambda key, run_generation=None: (
        released.append((key, run_generation))
    )
    runner._draining = False
    turn_ctx = TurnContext(session_key="session-1", run_generation=4)
    hud_release = asyncio.Event()

    async def stubborn_hud():
        while not hud_release.is_set():
            try:
                await hud_release.wait()
            except asyncio.CancelledError:
                continue

    async def track_forever():
        await asyncio.Event().wait()

    hud_task = asyncio.create_task(stubborn_hud())
    tracking_task = asyncio.create_task(track_forever())
    await asyncio.wait_for(
        runner._run_agent_cleanup_turn_tasks(
            turn_ctx,
            progress_task=None,
            log_task=None,
            hud_task=hud_task,
            interrupt_monitor=None,
            _notify_task=None,
            tracking_task=tracking_task,
            stream_task=None,
        ),
        timeout=1,
    )

    assert released == [("session-1", 4)]
    assert not hud_task.done()
    assert tracking_task.cancelled()
    hud_release.set()
    await asyncio.wait_for(hud_task, timeout=1)
