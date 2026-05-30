import json
from types import SimpleNamespace

from agent.tool_executor import execute_tool_calls_sequential


class _AllowAllGuardrails:
    def before_call(self, _name, _args):
        return SimpleNamespace(allows_execution=True)


class _NoSubdirHints:
    def check_tool_call(self, _name, _args):
        return None


class _FakeAgent:
    def __init__(self):
        self._interrupt_requested = False
        self.quiet_mode = True
        self.verbose_logging = False
        self.log_prefix = ""
        self.log_prefix_chars = 120
        self._checkpoint_mgr = SimpleNamespace(enabled=False)
        self._tool_guardrails = _AllowAllGuardrails()
        self._context_engine_tool_names = set()
        self._memory_manager = None
        self._subdirectory_hints = _NoSubdirHints()
        self.valid_tool_names = set()
        self.session_id = "session-1"
        self.tool_delay = 0
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self.subagent_events = []

    def _should_emit_quiet_tool_messages(self):
        return False

    def _should_start_quiet_spinner(self):
        return False

    def _vprint(self, *_args, **_kwargs):
        return None

    def _touch_activity(self, *_args, **_kwargs):
        return None

    def _record_file_mutation_result(self, *_args, **_kwargs):
        return None

    def _append_guardrail_observation(self, _name, _args, result, failed=False):
        return result

    def _tool_result_content_for_active_model(self, _name, result):
        return result

    def _apply_pending_steer_to_tool_results(self, *_args, **_kwargs):
        return None

    def _invoke_tool(self, function_name, function_args, *_args, **_kwargs):
        assert function_name in {"ao_delegate_task", "ao_delegate_batch"}
        if self.tool_progress_callback:
            self.tool_progress_callback(
                "subagent.start",
                tool_name=function_name,
                preview="AO spawned",
                subagent_id="ao:oryn-workspace-1",
                runtime="ao",
                status="running",
            )
        return json.dumps({"ok": True, "status": "completed"})


def _tool_call(name, arguments):
    return SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def test_sequential_ao_delegate_task_routes_through_agent_invoke_tool():
    agent = _FakeAgent()
    agent.tool_progress_callback = lambda event, **payload: agent.subagent_events.append((event, payload))
    messages = []
    assistant = SimpleNamespace(
        tool_calls=[
            _tool_call(
                "ao_delegate_task",
                {
                    "prompt": "Read-only inspection",
                    "project_id": "OrynWorkspace",
                },
            )
        ]
    )

    execute_tool_calls_sequential(agent, assistant, messages, effective_task_id="task-1")

    assert agent.subagent_events
    assert agent.subagent_events[0][0] == "subagent.start"
    assert agent.subagent_events[0][1]["subagent_id"] == "ao:oryn-workspace-1"
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["name"] == "ao_delegate_task"


def test_sequential_ao_delegate_batch_routes_through_agent_invoke_tool():
    agent = _FakeAgent()
    agent.tool_progress_callback = lambda event, **payload: agent.subagent_events.append((event, payload))
    messages = []
    assistant = SimpleNamespace(
        tool_calls=[
            _tool_call(
                "ao_delegate_batch",
                {
                    "tasks": [
                        {
                            "goal": "Read-only inspection",
                            "project_id": "OrynWorkspace",
                            "prompt": "Inspect the board.",
                        }
                    ],
                },
            )
        ]
    )

    execute_tool_calls_sequential(agent, assistant, messages, effective_task_id="task-1")

    assert agent.subagent_events
    assert agent.subagent_events[0][0] == "subagent.start"
    assert agent.subagent_events[0][1]["tool_name"] == "ao_delegate_batch"
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["name"] == "ao_delegate_batch"
