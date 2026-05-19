import json
from types import SimpleNamespace

from agent.agent_runtime_helpers import invoke_tool
from agent.tool_executor import (
    _record_harness_tool_result,
    _record_harness_tool_start,
    execute_tool_calls_sequential,
)
from hermes_constants import get_hermes_home


class _Agent:
    session_id = "sess-tool"


class _Store:
    def __init__(self):
        self.added = []

    def add(self, target, content):
        self.added.append((target, content))
        return {"success": True}


class _MemoryManager:
    def __init__(self):
        self.writes = []

    def on_memory_write(self, action, target, content, metadata=None):
        self.writes.append((action, target, content, metadata or {}))

    def has_tool(self, _name):
        return False


class _Guardrails:
    def before_call(self, _name, _args):
        return SimpleNamespace(allows_execution=True)


class _SequentialAgent(_Agent):
    def __init__(self):
        self._interrupt_requested = False
        self.log_prefix = ""
        self._tool_guardrails = _Guardrails()
        self.quiet_mode = True
        self.verbose_logging = False
        self.log_prefix_chars = 120
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self._checkpoint_mgr = SimpleNamespace(enabled=False)
        self._memory_store = _Store()
        self._memory_manager = _MemoryManager()
        self._context_engine_tool_names = set()
        self.valid_tool_names = set()
        self._subdirectory_hints = SimpleNamespace(check_tool_call=lambda *_args, **_kwargs: "")
        self.tool_delay = 0

    def _vprint(self, *_args, **_kwargs):
        pass

    def _touch_activity(self, *_args, **_kwargs):
        pass

    def _should_emit_quiet_tool_messages(self):
        return False

    def _should_start_quiet_spinner(self):
        return False

    def _build_memory_write_metadata(self, **kwargs):
        return kwargs

    def _append_guardrail_observation(self, _name, _args, result, failed=False):
        return result

    def _record_file_mutation_result(self, *_args, **_kwargs):
        pass

    def _tool_result_content_for_active_model(self, _name, result):
        return result

    def _apply_pending_steer_to_tool_results(self, *_args, **_kwargs):
        pass


def _memory_call(content: str):
    return SimpleNamespace(
        id="call-memory",
        function=SimpleNamespace(
            name="memory",
            arguments=json.dumps({"action": "add", "target": "memory", "content": content}),
        ),
    )


def test_tool_start_records_argument_shape_without_raw_content(_isolate_hermes_home):
    args = {
        "content": "remember this private sentence",
        "command": "printf sk-abcdefghijklmnopqrstuvwxyz123456",
        "question": "what is the private deployment password?",
        "nested": {"api_key": "sk-abcdefghijklmnopqrstuvwxyz123456"},
        "count": 3,
    }

    _record_harness_tool_start(_Agent(), "memory", args)

    event_file = get_hermes_home() / "harness" / "harness-events.jsonl"
    raw = event_file.read_text()
    row = json.loads(raw.splitlines()[0])
    payload = row["payload"]

    assert payload["tool_name"] == "memory"
    assert "args" not in payload
    assert "arg_summary" in payload
    assert payload["arg_summary"]["arg_keys"] == [
        "content",
        "command",
        "question",
        "nested",
        "count",
    ]
    assert payload["arg_summary"]["arg_shapes"]["content"] == {
        "type": "str",
        "length": len(args["content"]),
    }
    assert payload["arg_summary"]["arg_shapes"]["command"] == {
        "type": "str",
        "length": len(args["command"]),
    }
    assert payload["arg_summary"]["arg_shapes"]["nested"] == {
        "type": "dict",
        "key_count": 1,
        "keys": ["api_key"],
    }
    assert payload["arg_summary"]["arg_shapes"]["count"] == {"type": "number"}

    assert "remember this private sentence" not in raw
    assert "printf sk-" not in raw
    assert "private deployment password" not in raw
    assert "sk-abc...3456" not in raw


def test_tool_result_records_error_metadata_without_raw_content(_isolate_hermes_home):
    raw_error = "private result error: customer database password failed"

    _record_harness_tool_result(
        _Agent(),
        "terminal",
        {},
        {"success": False, "error": raw_error},
        duration=0.25,
        is_error=True,
    )

    event_file = get_hermes_home() / "harness" / "harness-events.jsonl"
    raw = event_file.read_text()
    row = json.loads(raw.splitlines()[0])
    payload = row["payload"]

    assert row["event_type"] == "tool.error"
    assert payload["result_error_present"] is True
    assert payload["result_error_chars"] == len(raw_error)
    assert payload["result_error_sha256"]
    assert "result_error" not in payload
    assert raw_error not in raw
    assert "customer database password" not in raw


def test_sequential_memory_rejection_does_not_bridge_to_external_memory(_isolate_hermes_home):
    agent = _SequentialAgent()
    content = "Fixed issue #1234 and committed sha abcdef123456."
    messages = []
    assistant_message = SimpleNamespace(tool_calls=[_memory_call(content)])

    execute_tool_calls_sequential(agent, assistant_message, messages, effective_task_id="task-1")

    assert agent._memory_store.added == []
    assert agent._memory_manager.writes == []
    assert messages
    result = json.loads(messages[0]["content"])
    assert result["success"] is False
    assert result["admission"]["decision"] == "reject"


def test_invoke_tool_memory_rejection_does_not_bridge_to_external_memory(_isolate_hermes_home):
    agent = _SequentialAgent()
    content = "Fixed issue #1234 and committed sha abcdef123456."

    raw = invoke_tool(
        agent,
        "memory",
        {"action": "add", "target": "memory", "content": content},
        effective_task_id="task-1",
        tool_call_id="call-memory",
        pre_tool_block_checked=True,
    )

    assert agent._memory_store.added == []
    assert agent._memory_manager.writes == []
    result = json.loads(raw)
    assert result["success"] is False
    assert result["admission"]["decision"] == "reject"


def test_sequential_successful_memory_write_still_bridges_to_external_memory(_isolate_hermes_home):
    agent = _SequentialAgent()
    content = "User prefers concise replies."
    messages = []
    assistant_message = SimpleNamespace(tool_calls=[_memory_call(content)])

    execute_tool_calls_sequential(agent, assistant_message, messages, effective_task_id="task-1")

    assert agent._memory_store.added == [("memory", content)]
    assert agent._memory_manager.writes == [
        ("add", "memory", content, {"task_id": "task-1", "tool_call_id": "call-memory"})
    ]


def test_invoke_tool_successful_memory_write_still_bridges_to_external_memory(_isolate_hermes_home):
    agent = _SequentialAgent()
    content = "User prefers concise replies."

    raw = invoke_tool(
        agent,
        "memory",
        {"action": "add", "target": "memory", "content": content},
        effective_task_id="task-1",
        tool_call_id="call-memory",
        pre_tool_block_checked=True,
    )

    result = json.loads(raw)
    assert result["success"] is True
    assert agent._memory_store.added == [("memory", content)]
    assert agent._memory_manager.writes == [
        ("add", "memory", content, {"task_id": "task-1", "tool_call_id": "call-memory"})
    ]
