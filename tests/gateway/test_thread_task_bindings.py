import asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_context, build_session_context_prompt
from gateway.thread_tasks import ThreadTaskBindingStore, parse_todoist_task_id


def _discord_thread_source(chat_name="Hermes / #hermes / [T:8847339201] context recovery"):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-123",
        chat_name=chat_name,
        chat_type="thread",
        thread_id="thread-123",
        parent_chat_id="channel-456",
        guild_id="guild-789",
        user_id="user-1",
        user_name="alex",
    )


def test_parse_todoist_task_id_from_thread_titles_and_urls():
    assert parse_todoist_task_id("[T:8847339201] context recovery") == "8847339201"
    assert parse_todoist_task_id("T:8847339202 - context recovery") == "8847339202"
    assert parse_todoist_task_id("https://todoist.com/showTask?id=8847339203") == "8847339203"
    assert parse_todoist_task_id("no task here") is None


def test_thread_title_fallback_is_injected_into_session_context(tmp_path):
    store = ThreadTaskBindingStore(tmp_path / "bindings.json")
    source = _discord_thread_source()
    binding = store.resolve_for_source(source)
    assert binding is not None
    source.task_binding = binding.to_dict()

    config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake")},
    )
    prompt = build_session_context_prompt(build_session_context(source, config))

    assert "**Bound Todoist Task:**" in prompt
    assert "Task ID: `8847339201`" in prompt
    assert "https://todoist.com/showTask?id=8847339201" in prompt
    assert "Binding source: thread-title" in prompt


def test_manual_binding_round_trips_and_takes_precedence_over_title(tmp_path):
    store = ThreadTaskBindingStore(tmp_path / "bindings.json")
    source = _discord_thread_source()

    store.bind(source, "9999999999", task_title="Manual recovery task")
    binding = store.resolve_for_source(source)

    assert binding is not None
    assert binding.task_id == "9999999999"
    assert binding.task_title == "Manual recovery task"
    assert binding.source == "manual"


def test_bind_task_command_persists_binding(tmp_path):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.thread_task_bindings = ThreadTaskBindingStore(tmp_path / "bindings.json")
    source = _discord_thread_source(chat_name="Hermes / #hermes / context recovery")
    event = MessageEvent(text="/bind-task 8847339204 Router context recovery", source=source)

    reply = asyncio.run(runner._handle_bind_task_command(event))
    binding = runner.thread_task_bindings.resolve_for_source(source)

    assert "8847339204" in reply
    assert "Future turns will include" in reply
    assert "https://todoist.com/showTask?id=8847339204" in reply
    assert binding is not None
    assert binding.task_id == "8847339204"
    assert binding.task_title == "Router context recovery"


def test_source_with_thread_task_binding_attaches_prompt_metadata(tmp_path):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.thread_task_bindings = ThreadTaskBindingStore(tmp_path / "bindings.json")
    source = _discord_thread_source(chat_name="Hermes / #hermes / context recovery")
    runner.thread_task_bindings.bind(source, "8847339205", task_title="Injected task")

    enriched = runner._source_with_thread_task_binding(source)

    assert enriched is not source
    assert enriched.task_binding is not None
    assert enriched.task_binding["task_id"] == "8847339205"
