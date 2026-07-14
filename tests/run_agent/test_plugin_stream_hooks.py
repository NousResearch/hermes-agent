import threading
import time

from types import SimpleNamespace
from unittest.mock import patch


def _agent():
    from run_agent import AIAgent

    return AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def _make_stream_chunk(content=None, finish_reason=None):
    delta = SimpleNamespace(content=content, reasoning_content=None, reasoning=None, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model")


def test_stream_observer_hooks_are_valid_plugin_hooks():
    from hermes_cli.plugins import VALID_HOOKS

    assert {
        "on_stream_start",
        "on_stream_delta",
        "on_stream_end",
        "on_interim_message",
    }.issubset(VALID_HOOKS)


def test_stream_delta_plugin_hook_is_queued_off_token_path(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []

    def invoke_hook(hook_name, **kwargs):
        time.sleep(0.2)
        calls.append((hook_name, kwargs))
        return []

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "on_stream_delta")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)

    agent = _agent()

    started = time.monotonic()
    agent._fire_stream_delta("hello")
    elapsed = time.monotonic() - started

    assert elapsed < 0.05
    _wait_for(lambda: calls)
    shutdown_plugin_stream_hook_dispatcher()

    assert calls[0][0] == "on_stream_delta"
    assert calls[0][1]["delta"] == "hello"
    assert calls[0][1]["kind"] == "text"
    assert calls[0][1]["model"] == "test/model"
    assert calls[0][1]["provider"] == "openrouter"


def test_stream_delta_plugin_hook_error_does_not_break_streaming(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    ui_deltas = []

    def invoke_hook(_hook_name, **_kwargs):
        raise RuntimeError("plugin failed")

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "on_stream_delta")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)

    agent = _agent()
    agent.stream_delta_callback = ui_deltas.append

    agent._fire_stream_delta("still visible")
    shutdown_plugin_stream_hook_dispatcher()

    assert ui_deltas == ["still visible"]


def test_stream_hook_queue_drops_oldest_pending_event_when_full(monkeypatch):
    from agent.plugin_stream_hooks import enqueue_plugin_stream_hook, shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    monkeypatch.setenv("HERMES_PLUGIN_STREAM_HOOK_QUEUE_SIZE", "1")
    delivered = []
    first_delivered = threading.Event()
    release_worker = threading.Event()

    def invoke_hook(_hook_name, **kwargs):
        delivered.append(kwargs["delta"])
        first_delivered.set()
        release_worker.wait(timeout=1.0)
        return []

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "on_stream_delta")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)

    assert enqueue_plugin_stream_hook("on_stream_delta", delta="first") is True
    assert first_delivered.wait(timeout=1.0)
    assert enqueue_plugin_stream_hook("on_stream_delta", delta="second") is True
    assert enqueue_plugin_stream_hook("on_stream_delta", delta="third") is True

    release_worker.set()
    _wait_for(lambda: "third" in delivered)
    shutdown_plugin_stream_hook_dispatcher()

    assert delivered == ["first", "third"]


def test_reasoning_stream_delta_plugin_hook_is_opt_in(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "on_stream_delta")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda hook_name, **kwargs: calls.append((hook_name, kwargs)) or [])

    agent = _agent()
    agent._fire_reasoning_delta("private chain")
    shutdown_plugin_stream_hook_dispatcher()

    assert calls == []

    with patch("hermes_cli.config.cfg_get", return_value=True):
        agent._fire_reasoning_delta("visible reasoning")
        _wait_for(lambda: calls)
        shutdown_plugin_stream_hook_dispatcher()

    assert calls[0][0] == "on_stream_delta"
    assert calls[0][1]["kind"] == "reasoning"
    assert calls[0][1]["delta"] == "visible reasoning"


def test_interim_message_plugin_hook_is_queued(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "on_interim_message")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda hook_name, **kwargs: calls.append((hook_name, kwargs)) or [])

    agent = _agent()
    agent._emit_interim_assistant_message({"content": "I will inspect the files first."})
    _wait_for(lambda: calls)
    shutdown_plugin_stream_hook_dispatcher()

    assert calls[0][0] == "on_interim_message"
    assert calls[0][1]["text"] == "I will inspect the files first."
    assert calls[0][1]["already_streamed"] is False


def test_stream_plugin_hook_counts_as_stream_consumer(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "on_stream_delta")

    agent = _agent()

    assert agent._has_stream_consumers() is True


def test_stream_lifecycle_plugin_hooks_are_queued(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name in {"on_stream_start", "on_stream_end"})
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda hook_name, **kwargs: calls.append((hook_name, kwargs)) or [])

    agent = _agent()
    agent._emit_stream_start()
    agent._emit_stream_end(final_text="done", finished=True, error=None)
    _wait_for(lambda: len(calls) == 2)
    shutdown_plugin_stream_hook_dispatcher()

    assert [call[0] for call in calls] == ["on_stream_start", "on_stream_end"]
    assert calls[0][1]["model"] == "test/model"
    assert calls[1][1]["final_text"] == "done"
    assert calls[1][1]["finished"] is True
    assert calls[1][1]["error"] is None


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_chat_completion_stream_emits_lifecycle_hooks(_mock_close, mock_create, monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []
    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook",
        lambda name: name in {"on_stream_start", "on_stream_delta", "on_stream_end"},
    )
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda hook_name, **kwargs: calls.append((hook_name, kwargs)) or [])

    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: iter([
                    _make_stream_chunk(content="hello "),
                    _make_stream_chunk(content="world"),
                    _make_stream_chunk(finish_reason="stop"),
                ])
            )
        )
    )
    mock_create.return_value = mock_client

    agent = _agent()
    agent.api_mode = "chat_completions"
    response = agent._interruptible_streaming_api_call({})

    _wait_for(lambda: [call[0] for call in calls].count("on_stream_end") == 1)
    shutdown_plugin_stream_hook_dispatcher()

    assert response.choices[0].message.content == "hello world"
    assert [call[0] for call in calls] == [
        "on_stream_start",
        "on_stream_delta",
        "on_stream_delta",
        "on_stream_end",
    ]
    assert calls[-1][1]["final_text"] == "hello world"
    assert calls[-1][1]["finished"] is True
