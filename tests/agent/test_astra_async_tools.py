"""Focused deterministic coverage for direct GPT-6 Astra async application tools."""

from types import SimpleNamespace
import threading
import time

import pytest

from agent.astra_async_tools import AstraAsyncExecutor, is_direct_astra, provider_async_marker
from agent.codex_responses_adapter import _response_tool_call, _responses_tools
from agent.codex_runtime import _consume_codex_event_stream


def _tool(name, call_id):
    call = SimpleNamespace(
        type="function_call", id=f"fc_{call_id}", call_id=f"call_{call_id}", name=name,
        arguments="{}", async_=True,
    )
    return call


def test_astra_schema_marker_is_explicit_and_non_astra_shape_is_unchanged():
    source = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
    plain = _responses_tools(source)
    async_tools = _responses_tools(source, async_tools=True)
    assert "async" not in plain[0]
    assert async_tools[0]["async"] is True


def test_stream_preserves_async_marker_and_admits_only_completed_call():
    admitted = []
    events = [
        SimpleNamespace(type="response.output_item.added", output_index=0, item=SimpleNamespace(
            type="function_call", id="fc_1", call_id="call_1", name="read_file", arguments="", **{"async": True},
        )),
        SimpleNamespace(type="response.function_call_arguments.delta", item_id="fc_1", delta="{}"),
        SimpleNamespace(type="response.function_call_arguments.done", item_id="fc_1", arguments="{}"),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(id="resp_1", status="completed")),
    ]
    final = _consume_codex_event_stream(events, model="gpt-6-astra", on_async_tool_call=admitted.append)
    assert len(admitted) == 1
    assert admitted[0].call_id == "call_1"
    assert getattr(admitted[0], "async", False) is True
    assert final.output[0].call_id == "call_1"


@pytest.mark.parametrize("marker", ["async", "async_"])
def test_async_marker_spellings_survive_raw_and_normalized_paths(marker):
    emitted = []
    item = {"type": "function_call", "id": "fc_alias", "call_id": "call_alias", "name": "read_file",
            "arguments": "{}", marker: True}
    events = [
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {"type": "response.function_call_arguments.done", "item_id": "fc_alias", "arguments": "{}"},
        {"type": "response.completed", "response": {"id": "resp_alias", "status": "completed"}},
    ]
    _consume_codex_event_stream(events, model="gpt-6-astra", on_async_tool_call=emitted.append)
    assert len(emitted) == 1
    assert provider_async_marker(emitted[0])

    normalized = _response_tool_call(
        SimpleNamespace(type="function_call", id="fc_alias", call_id="call_alias", name="read_file",
                        arguments="{}", **{marker: True}),
        "function_call", 0,
    )
    assert provider_async_marker(normalized)


@pytest.mark.parametrize(("base_url", "expected"), [
    ("https://api.openai.com/v1", True),
    ("https://evil.api.openai.com/v1", False),
    ("https://api.openai.com.evil.test/v1", False),
])
def test_direct_astra_requires_exact_official_hostname(base_url, expected):
    agent = SimpleNamespace(api_mode="codex_responses", model="gpt-6-astra", base_url=base_url)
    assert is_direct_astra(agent) is expected


class _FakeAgent:
    api_mode = "codex_responses"
    model = "gpt-6-astra"
    base_url = "https://api.openai.com/v1"
    session_id = "session-test"
    session_start = None
    _interrupt_requested = False

    def __init__(self, persisted):
        self.persisted = persisted
        self._session_messages = []
        self.verbose_logging = False
        self.reasoning_callback = None
        self.stream_delta_callback = None
        self._stream_callback = None

    def _extract_reasoning(self, message):
        return None

    def _strip_think_blocks(self, content):
        return content

    def _split_responses_tool_id(self, value):
        return (value, value)

    def _derive_responses_function_call_id(self, call_id, response_item_id=None):
        return response_item_id or f"fc_{call_id.removeprefix('call_')}"

    def _deterministic_call_id(self, name, arguments, index):
        return f"call_deterministic_{index}"

    def _flush_messages_to_session_db(self, messages):
        self.persisted.append([m.copy() for m in messages])
        return True


@pytest.fixture
def patched_executor(monkeypatch):
    import agent.astra_async_tools as module
    import agent.tool_executor as tool_executor

    class Parsed:
        def __init__(self, call):
            self.call = call
            self.name = call.name
            self.args = {}
            self.middleware_trace = []
            self.parse_error = "invalid arguments" if call.name == "malformed" else None
            self.scope_block = None

        def ref(self, task_id):
            return SimpleNamespace(name=self.name, args={}, task_id=task_id, call_id=self.call.call_id, trace=[])

    monkeypatch.setattr(tool_executor, "_parse_tool_call", lambda agent, call: Parsed(call))
    monkeypatch.setattr(tool_executor, "_resolve_sequential_dispatch", lambda *args: SimpleNamespace())

    running = set()
    overlap = threading.Event()
    starts = []

    def run_call(agent, dispatch, ref, **kwargs):
        starts.append(ref.call_id)
        running.add(ref.call_id)
        if len(running) >= 2:
            overlap.set()
        time.sleep(0.03)
        running.remove(ref.call_id)
        return SimpleNamespace(result=f"result:{ref.call_id}", args={}, middleware_trace=[], blocked=False, dispatched=True), 0.03

    monkeypatch.setattr(tool_executor, "_run_sequential_call", run_call)
    committed = []

    def publish(agent, messages, ref, managed, **kwargs):
        committed.append(ref.call_id)
        messages.append({"role": "tool", "tool_call_id": ref.call_id, "content": managed.result})
        return True

    monkeypatch.setattr(tool_executor, "_publish_sequential_result", publish)
    monkeypatch.setattr(tool_executor, "_budget_for_agent", lambda agent: object())
    monkeypatch.setattr(tool_executor, "_finalize_tool_batch", lambda *args, **kwargs: None)

    def plan(calls, **kwargs):
        segments = []
        current = []
        for call in calls:
            if call.name == "unsafe":
                if current:
                    segments.append(("parallel", current) if len(current) > 1 else ("sequential", current))
                    current = []
                segments.append(("sequential", [call]))
            else:
                current.append(call)
        if current:
            segments.append(("parallel", current) if len(current) > 1 else ("sequential", current))
        return segments

    monkeypatch.setattr(module, "_plan_tool_batch_segments", plan)
    return overlap, starts, committed


def test_admit_persists_before_handler_and_barriers_keep_result_order(patched_executor):
    overlap, starts, committed = patched_executor
    persisted = []
    agent = _FakeAgent(persisted)
    executor = AstraAsyncExecutor(agent, [], "task")

    first = _tool("safe_a", "a")
    assert executor.admit(first) is True
    # The first worker may start immediately, but its assistant fragment was flushed first.
    assert persisted and persisted[0][0]["role"] == "assistant"

    second = _tool("safe_b", "b")
    assert executor.admit(second) is True
    assert executor.admit(_tool("unsafe", "c")) is True
    assert executor.admit(_tool("safe_d", "d")) is True
    assert executor.finish_stream() is True
    assert overlap.is_set()
    assert committed == ["call_a", "call_b", "call_c", "call_d"]
    assert starts.count("call_c") == 1


def test_empty_executor_is_retired_before_later_turn_state_can_be_reused():
    agent = _FakeAgent([])
    stale_messages = [{"role": "assistant", "content": "stale"}]
    stale = AstraAsyncExecutor(agent, stale_messages, "task")
    assert stale.retire_empty() is True
    assert stale.closed is True

    fresh_messages = [{"role": "user", "content": "new"}]
    fresh = AstraAsyncExecutor(agent, fresh_messages, "task")
    assert fresh.messages is fresh_messages
    assert fresh.messages is not stale.messages
    fresh.retire_empty()


def test_mixed_assistant_text_is_persisted_before_async_results(patched_executor):
    persisted = []
    agent = _FakeAgent(persisted)
    executor = AstraAsyncExecutor(agent, [], "task")
    assert executor.admit(_tool("safe", "text")) is True
    assert executor.finish_stream(assistant_content="Visible assistant text") is True
    assistant_rows = [row for batch in persisted for row in batch if row.get("role") == "assistant"]
    assert any(row.get("content") == "Visible assistant text" for row in assistant_rows)


def test_interrupt_retirement_is_idempotent_and_does_not_re_admit(patched_executor):
    persisted = []
    agent = _FakeAgent(persisted)
    executor = AstraAsyncExecutor(agent, [], "task")
    call = _tool("safe", "once")
    assert executor.admit(call) is True
    executor.abort_stream()
    assert executor.admit(call) is False


def test_parse_error_is_reported_without_handler_dispatch(patched_executor, monkeypatch):
    _, starts, _ = patched_executor
    import agent.tool_executor as tool_executor

    invalid = []

    def append_invalid(agent, messages, ref, parse_error):
        invalid.append(ref.call_id)
        messages.append({"role": "tool", "tool_call_id": ref.call_id, "content": parse_error})
        return True

    monkeypatch.setattr(tool_executor, "_append_invalid_arguments_result", append_invalid)
    agent = _FakeAgent([])
    executor = AstraAsyncExecutor(agent, [], "task")
    assert executor.admit(_tool("malformed", "bad")) is True
    assert executor.finish_stream() is True
    assert starts == []
    assert invalid == ["call_bad"]
    assert agent._session_messages[-1]["content"] == "invalid arguments"


def test_announced_order_blocks_later_completion_until_earlier_call_is_admitted(patched_executor):
    _, starts, committed = patched_executor
    agent = _FakeAgent([])
    executor = AstraAsyncExecutor(agent, [], "task")
    first, second = _tool("safe_a", "a"), _tool("safe_b", "b")
    assert executor.reserve(first) is True
    assert executor.reserve(second) is True
    assert executor.admit(second) is True
    assert starts == []
    assert executor.admit(first) is True
    assert executor.finish_stream() is True
    assert committed == ["call_a", "call_b"]


def test_terminal_settlement_admits_pending_announced_call_before_retirement(patched_executor):
    _, starts, committed = patched_executor
    agent = _FakeAgent([])
    executor = AstraAsyncExecutor(agent, [], "task")
    first, second = _tool("safe_a", "a"), _tool("safe_b", "b")
    assert executor.admit(first) is True
    final = _consume_codex_event_stream([
        SimpleNamespace(type="response.output_item.added", output_index=1, item=second),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(id="resp", status="completed")),
    ], model="gpt-6-astra", on_async_tool_announcement=executor.reserve)
    assert executor.has_pending
    assert executor.finish_stream(settled_calls=final.output) is True
    assert starts == ["call_a", "call_b"]
    assert committed == ["call_a", "call_b"]


def test_publish_failure_recovers_once_without_reexecuting_handler(patched_executor, monkeypatch):
    _, starts, _ = patched_executor
    import agent.tool_executor as tool_executor

    publish_calls = []

    def fail_publish(agent, messages, ref, managed, **kwargs):
        publish_calls.append(ref.call_id)
        messages.append({"role": "tool", "tool_call_id": ref.call_id, "content": managed.result})
        return False

    monkeypatch.setattr(tool_executor, "_publish_sequential_result", fail_publish)
    flushes = []
    agent = _FakeAgent([])

    def flush(messages):
        flushes.append([row.copy() for row in messages])
        return True

    agent._flush_messages_to_session_db = flush
    executor = AstraAsyncExecutor(agent, [], "task")
    assert executor.admit(_tool("safe", "fail")) is True
    assert executor.finish_stream() is True
    assert starts == ["call_fail"]
    assert publish_calls == ["call_fail"]
    assert len(flushes) == 2
    assert "[Orphan recovery:" in flushes[-1][-1]["content"]


def test_settle_required_persists_ordered_prefix_once_and_finish_stream_is_idempotent(patched_executor, monkeypatch):
    _, starts, committed = patched_executor
    import agent.tool_executor as tool_executor

    persisted = []
    agent = _FakeAgent(persisted)

    def publish(agent, messages, ref, managed, **kwargs):
        committed.append(ref.call_id)
        messages.append({"role": "tool", "tool_call_id": ref.call_id, "content": managed.result})
        assert agent._flush_messages_to_session_db(messages) is True
        return True

    monkeypatch.setattr(tool_executor, "_publish_sequential_result", publish)
    executor = AstraAsyncExecutor(agent, [], "task")
    executor.admit(_tool("safe_a", "a"))
    executor.admit(_tool("unsafe", "b"))
    last = _tool("safe_c", "c")
    executor.admit(last)

    assert executor.settle_required([last.call_id]) is True
    assert committed == ["call_a", "call_b", "call_c"]
    assert [row["tool_call_id"] for row in persisted[-1] if row.get("role") == "tool"] == committed
    assert executor.settle_required([last.call_id]) is True
    assert committed == ["call_a", "call_b", "call_c"]

    assert executor.finish_stream() is True
    assert executor.finish_stream() is True
    assert committed == ["call_a", "call_b", "call_c"]
    assert [row["tool_call_id"] for row in agent._session_messages if row.get("role") == "tool"] == committed
    assert starts.count("call_c") == 1


def test_settle_required_persistence_failure_never_reexecutes_or_exposes_result(patched_executor, monkeypatch):
    _, starts, _ = patched_executor
    import agent.tool_executor as tool_executor

    persisted = []
    agent = _FakeAgent(persisted)

    def flush(messages):
        if any(row.get("role") == "tool" for row in messages):
            return False
        persisted.append([row.copy() for row in messages])
        return True

    agent._flush_messages_to_session_db = flush

    def publish(agent, messages, ref, managed, **kwargs):
        messages.append({"role": "tool", "tool_call_id": ref.call_id, "content": managed.result})
        return agent._flush_messages_to_session_db(messages)

    monkeypatch.setattr(tool_executor, "_publish_sequential_result", publish)
    executor = AstraAsyncExecutor(agent, [], "task")
    call = _tool("safe", "no-durable-result")
    assert executor.admit(call) is True

    assert executor.settle_required([call.call_id]) is False
    assert starts == [call.call_id]
    assert all(not any(row.get("role") == "tool" for row in batch) for batch in persisted)
