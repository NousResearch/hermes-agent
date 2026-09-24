"""End-to-end todo progress tests through the gateway and Telegram adapter."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from agent.tool_executor import _execute_tool_calls_sequential
from gateway.config import Platform, PlatformConfig
from gateway.display_config import resolve_display_setting, resolve_tool_progress
from gateway.run_turn import GatewayTurnMixin
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.live_todo import TodoSource
from hermes_cli.plugins import get_plugin_manager
from gateway.turn_context import TurnContext
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools.todo_tool import TodoStore


@pytest.fixture(autouse=True)
def loaded_plugin(tmp_path, monkeypatch):
    """Real installed entry point, real profile config and loader (no registry fake)."""
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [hermes-telegram-experience]\n  entries:\n"
        "    hermes-telegram-experience:\n      settings:\n        enabled: true\n"
        "        scope:\n"
        "          routes:\n"
        "            - {profile: default, platform: telegram, chat_id: '-100', thread_id: '7'}\n"
        "            - {profile: default, platform: telegram, chat_id: '-100123', thread_id: '11'}\n"
        "            - {profile: default, platform: telegram, chat_id: '-100123', thread_id: '22'}\n"
        "            - {profile: default, platform: telegram, chat_id: '-100789', thread_id: '44'}\n"
        "            - {profile: default, platform: telegram, chat_id: '-100456', thread_id: '33'}\n"
        "          task_resources: []\n"
    )
    manager = get_plugin_manager()
    manager.discover_and_load()
    assert manager._live_todo_registration.active
    yield manager
    manager.unload()


class StubTelegramBot:
    """Bot API surface used by TelegramAdapter.send/edit_message, with no network."""

    def __init__(self, *, block_sends: bool = False) -> None:
        self.sent: list[dict] = []
        self.edited: list[dict] = []
        self.send_started = asyncio.Event()
        self.release_sends = asyncio.Event()
        self.block_sends = block_sends

    async def send_message(self, **kwargs):
        self.sent.append(dict(kwargs))
        self.send_started.set()
        if self.block_sends:
            await self.release_sends.wait()
        return SimpleNamespace(message_id=len(self.sent))

    async def edit_message_text(self, **kwargs):
        self.edited.append(dict(kwargs))
        return SimpleNamespace(message_id=kwargs["message_id"])


class GatewayHarness(GatewayTurnMixin):
    def __init__(self, adapter: TelegramAdapter) -> None:
        self.adapter = adapter
        self._draining = False
        self.released: list[tuple[str, int | None]] = []

    def _delivery_adapter_for(self, source):
        return self.adapter

    def _release_running_agent_state(self, session_key, *, run_generation=None):
        self.released.append((session_key, run_generation))

    def _update_runtime_status(self, status):
        return None


class ToolAgent:
    """Small real-tool-executor host; its store and commit hooks are not fakes."""

    def __init__(self, store: TodoStore) -> None:
        self._todo_store = store
        self.session_id = "session-1"
        self._current_turn_id = "turn-1"
        self._current_api_request_id = "request-1"
        self._interrupt_requested = False
        self._incremental_persistence_failed = False
        self._last_persistence_error_cause = None
        self._current_tool = None
        self._trim_after_tool_batch = False
        self.quiet_mode = False
        self.tool_progress_mode = "all"
        self.verbose_logging = False
        self.log_prefix = ""
        self.log_prefix_chars = 40
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self.context_compressor = SimpleNamespace(context_length=None)
        self._context_engine_tool_names = set()
        self._memory_manager = None
        self._subdirectory_hints = SimpleNamespace(check_tool_call=lambda *_args: None)
        self._tool_guardrails = SimpleNamespace(
            before_call=lambda *_args: SimpleNamespace(allows_execution=True)
        )
        self._checkpoint_mgr = SimpleNamespace(enabled=False)
        self._tool_worker_threads_lock = threading.Lock()
        self._tool_worker_threads: set[int] = set()
        self.flushed: list[list[dict]] = []
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None

    def _append_guardrail_observation(self, _name, _args, result, **_kwargs):
        return result

    def _record_file_mutation_result(self, *_args, **_kwargs):
        return None

    def _tool_result_content_for_active_model(self, _name, result):
        return result

    def _flush_messages_to_session_db(self, messages):
        self.flushed.append(list(messages))
        return True

    def _touch_activity(self, _message):
        return None

    def _vprint(self, *_args, **_kwargs):
        return None

    def _safe_print(self, *_args, **_kwargs):
        return None

    def _should_emit_quiet_tool_messages(self):
        return False

    def _should_start_quiet_spinner(self):
        return False

    def _apply_pending_steer_to_tool_results(self, *_args):
        return None


def _telegram_adapter(bot: StubTelegramBot) -> TelegramAdapter:
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="unit-test-token", typing_indicator=False)
    )
    adapter._bot = bot
    return adapter


def _display(user_config: dict) -> SimpleNamespace:
    progress_mode, explicit = resolve_tool_progress(user_config, "telegram")
    return SimpleNamespace(
        user_config=user_config,
        platform_key="telegram",
        progress_mode=progress_mode,
        _tool_progress_explicit=explicit,
        resolve_display_setting=resolve_display_setting,
    )


def _context(agent: ToolAgent, *, chat_id: str, thread_id: str, current) -> TurnContext:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        thread_id=thread_id,
        profile="default",
    )
    return TurnContext(
        source=source,
        _run_still_current=current,
        progress_mode="all",
        tool_progress_enabled=True,
        progress_queue=queue.Queue(),
        session_id=f"session-{thread_id}",
        session_key=f"agent:default:telegram:{chat_id}:{thread_id}",
        run_generation=int(thread_id),
        agent_holder=[agent],
        _progress_metadata={"message_thread_id": thread_id},
        _progress_reply_to=None,
    )


def _owner(harness: GatewayHarness, agent: ToolAgent, *, chat_id: str, thread_id: str, current):
    ctx = _context(agent, chat_id=chat_id, thread_id=thread_id, current=current)
    owner = GatewayTurnMixin._run_agent_create_todo_progress_owner(
        harness, _display({"display": {"task_progress": True}}), ctx
    )
    assert isinstance(owner, TodoSource)
    ctx._todo_progress_owner = owner
    runner = TurnRunner(harness, ctx)
    agent.tool_progress_callback = runner.progress_callback
    return ctx, owner


def _todo_call(call_id: str, item: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="todo_list",
            arguments=json.dumps({"todos": [item]}, ensure_ascii=False),
        ),
    )


def _execute_todo(agent: ToolAgent, call_id: str, item: dict) -> list[dict]:
    messages: list[dict] = []
    _execute_tool_calls_sequential(
        agent,
        SimpleNamespace(tool_calls=[_todo_call(call_id, item)]),
        messages,
        "task-1",
        finalize=False,
    )
    return messages


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("timed out waiting for integration delivery")
        await asyncio.sleep(0.005)


@pytest.mark.parametrize(
    ("config", "expected_owner"),
    [
        ({}, True),
        ({"display": {"task_progress": False}}, True),
        ({"display": {"task_progress": True, "tool_progress": False}}, False),
        ({"display": {"task_progress": True}}, True),
    ],
)
@pytest.mark.asyncio
async def test_gateway_owner_respects_quiet_and_task_progress_config(config, expected_owner):
    bot = StubTelegramBot()
    harness = GatewayHarness(_telegram_adapter(bot))
    agent = ToolAgent(TodoStore())
    current = lambda: True
    ctx = _context(agent, chat_id="-100", thread_id="7", current=current)

    owner = GatewayTurnMixin._run_agent_create_todo_progress_owner(harness, _display(config), ctx)

    assert (owner is not None) is expected_owner


@pytest.mark.asyncio
async def test_committed_todo_results_reach_real_telegram_adapter_in_isolated_topics():
    bot = StubTelegramBot()
    adapter = _telegram_adapter(bot)
    harness = GatewayHarness(adapter)
    current = lambda: True

    agents_and_owners = []
    for topic, task_id in (("11", "topic11"), ("22", "topic22")):
        agent = ToolAgent(TodoStore())
        ctx, owner = _owner(
            harness, agent, chat_id="-100123", thread_id=topic, current=current
        )
        owner_task = asyncio.create_task(owner.run())
        agents_and_owners.append((agent, ctx, owner, owner_task, task_id))

        messages = _execute_todo(
            agent,
            f"call-{topic}-1",
            {"id": task_id, "content": f"work for {task_id}", "status": "in_progress"},
        )
        assert agent._todo_store.snapshot()["revision"] == 1
        assert agent.flushed and messages[-1]["role"] == "tool"

    await _wait_until(lambda: len(bot.sent) == 2)
    assert {
        (entry["chat_id"], entry["message_thread_id"])
        for entry in bot.sent
    } == {(-100123, 11), (-100123, 22)}
    assert "topic11" in next(entry["text"] for entry in bot.sent if entry["message_thread_id"] == 11)
    assert "topic22" in next(entry["text"] for entry in bot.sent if entry["message_thread_id"] == 22)

    for agent, _ctx, owner, _task, task_id in agents_and_owners:
        _execute_todo(
            agent,
            f"call-{task_id}-2",
            {"id": task_id, "content": f"done for {task_id}", "status": "completed"},
        )
        assert agent._todo_store.snapshot()["revision"] == 2

    await _wait_until(lambda: len(bot.edited) == 2)
    assert {entry["message_id"] for entry in bot.edited} == {
        int(owner.message_id)
        for _agent, _ctx, owner, _task, _task_id in agents_and_owners
    }
    for agent, _ctx, owner, owner_task, task_id in agents_and_owners:
        assert owner.message_id is not None
        edit = next(entry for entry in bot.edited if entry["message_id"] == int(owner.message_id))
        assert f"done for {task_id}" in edit["text"]
        owner.close()
        await asyncio.wait_for(owner_task, timeout=1)


@pytest.mark.asyncio
async def test_real_telegram_adapter_settles_stale_post_await_send_result():
    bot = StubTelegramBot(block_sends=True)
    adapter = _telegram_adapter(bot)
    harness = GatewayHarness(adapter)
    agent = ToolAgent(TodoStore())
    current = [True]
    _ctx, owner = _owner(
        harness, agent, chat_id="-100789", thread_id="44", current=lambda: current[0]
    )
    owner_task = asyncio.create_task(owner.run())

    _execute_todo(
        agent,
        "call-stale-send",
        {"id": "stale", "content": "accepted remotely", "status": "in_progress"},
    )
    await asyncio.wait_for(bot.send_started.wait(), timeout=1)
    current[0] = False
    bot.release_sends.set()
    await asyncio.wait_for(owner_task, timeout=1)

    assert owner.message_id == "1"  # Host settles exact evidence without reviving the writer.
    assert not owner.active
    assert len(bot.sent) == 1  # The accepted Bot API request cannot be unsent.


@pytest.mark.asyncio
async def test_gateway_cleanup_drains_committed_todo_snapshot_before_releasing_turn():
    bot = StubTelegramBot(block_sends=True)
    harness = GatewayHarness(_telegram_adapter(bot))
    agent = ToolAgent(TodoStore())
    ctx, owner = _owner(
        harness, agent, chat_id="-100456", thread_id="33", current=lambda: True
    )
    ctx.session_key = "agent:default:telegram:-100456:33"
    owner_task = asyncio.create_task(owner.run())
    ctx._todo_progress_task = owner_task

    _execute_todo(
        agent,
        "call-drain-1",
        {"id": "drain", "content": "must survive cleanup", "status": "in_progress"},
    )
    await asyncio.wait_for(bot.send_started.wait(), timeout=1)

    tracking_task = asyncio.create_task(asyncio.sleep(3600))
    interrupt_monitor = asyncio.create_task(asyncio.sleep(3600))
    notify_task = asyncio.create_task(asyncio.sleep(3600))
    cleanup_task = asyncio.create_task(
        harness._run_agent_cleanup_turn_tasks(
            ctx,
            progress_task=None,
            log_task=None,
            interrupt_monitor=interrupt_monitor,
            _notify_task=notify_task,
            tracking_task=tracking_task,
            stream_task=None,
        )
    )
    await asyncio.sleep(0)
    assert not cleanup_task.done()
    assert harness.released == []

    bot.release_sends.set()
    await asyncio.wait_for(cleanup_task, timeout=1)
    assert owner_task.done()
    assert ctx._todo_progress_task is None
    assert harness.released == [(ctx.session_key, ctx.run_generation)]
    assert len(bot.sent) == 1
    assert "must survive cleanup" in bot.sent[0]["text"]
    assert not owner.publish(
        {"todos": [{"id": "late", "status": "completed"}], "revision": 2},
        agent._todo_store.incarnation,
    )


@pytest.mark.asyncio
async def test_final_answer_is_separate_and_does_not_complete_todos():
    bot = StubTelegramBot()
    adapter = _telegram_adapter(bot)
    harness = GatewayHarness(adapter)
    agent = ToolAgent(TodoStore())
    _, owner = _owner(harness, agent, chat_id="-100", thread_id="7", current=lambda: True)
    task = asyncio.create_task(owner.run())
    _execute_todo(agent, "start", {"id": "a", "content": "still pending", "status": "pending"})
    await _wait_until(lambda: owner.message_id is not None)
    progress_id = owner.message_id
    final = await adapter.send("-100", "Separate final answer", metadata={"message_thread_id": "7"})
    assert final.success and final.message_id != progress_id
    _execute_todo(agent, "more", {"id": "a", "content": "still pending later", "status": "pending"})
    await _wait_until(lambda: len(bot.edited) == 1)
    assert str(bot.edited[0]["message_id"]) == progress_id
    assert agent._todo_store.snapshot()["todos"][0]["status"] == "pending"
    assert len(bot.sent) == 2
    await owner.finish()
    await task


@pytest.mark.asyncio
async def test_failed_canonical_flush_never_publishes_todo_snapshot():
    bot = StubTelegramBot()
    harness = GatewayHarness(_telegram_adapter(bot))
    agent = ToolAgent(TodoStore())
    _, owner = _owner(harness, agent, chat_id="-100", thread_id="7", current=lambda: True)
    # Real executor can mutate the TodoStore, but a failed canonical tool-result flush
    # must not produce a committed source event.
    agent._flush_messages_to_session_db = lambda messages: not messages or messages[-1]["role"] != "tool"
    _execute_todo(agent, "failed-flush", {"id": "a", "content": "not durable", "status": "pending"})
    assert agent._todo_store.snapshot()["revision"] == 1
    assert agent._incremental_persistence_failed
    assert owner.revision == -1 and bot.sent == []
    await owner.finish()


@pytest.mark.asyncio
@pytest.mark.parametrize("quiet", ["heartbeat", "muted", "other-platform"])
async def test_quiet_sources_and_other_platforms_have_no_todo_consumer(quiet):
    bot = StubTelegramBot()
    harness = GatewayHarness(_telegram_adapter(bot))
    agent = ToolAgent(TodoStore())
    ctx = _context(agent, chat_id="-100", thread_id="7", current=lambda: True)
    if quiet == "heartbeat":
        ctx.scheduled_heartbeat = True
    elif quiet == "muted":
        ctx.mute_notification_reply = True
    else:
        ctx.source.platform = Platform.SLACK
    assert harness._run_agent_create_todo_progress_owner(_display({}), ctx) is None
    assert bot.sent == []
