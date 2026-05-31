import inspect

from agent.workflow_orchestrator import WorkflowPlan, WorkflowResult, WorkflowSubtask
from cli import HermesCLI
from hermes_cli.commands import resolve_command


def test_workflow_command_registered():
    resolved = resolve_command("workflow")

    assert resolved is not None
    assert resolved.name == "workflow"


def test_ultracode_chat_uses_workflow_orchestrator_not_directive():
    src = inspect.getsource(HermesCLI.chat)

    assert "_ULTRACODE_DIRECTIVE" not in src
    assert "delegate_task (batch mode)" not in src
    assert "if _ultracode_active:" in src
    assert "self._workflow_result_dict(str(agent_message))" in src


def test_workflow_result_dict_preserves_turn_history(monkeypatch):
    class FakeOrchestrator:
        def __init__(self, agent):
            self.agent = agent

        def run(self, task):
            return WorkflowResult(
                task=task,
                plan=WorkflowPlan(
                    mode="parallel",
                    subtasks=[WorkflowSubtask(goal="A"), WorkflowSubtask(goal="B")],
                ),
                child_results=[{"task_index": 0, "status": "completed", "summary": "A done"}],
                final_response="merged answer",
                delegated=True,
            )

    class Agent:
        def __init__(self):
            self.persisted = None

        def _persist_session(self, messages, conversation_history=None):
            self.persisted = (messages, conversation_history)

    monkeypatch.setattr("agent.workflow_orchestrator.WorkflowOrchestrator", FakeOrchestrator)
    cli = object.__new__(HermesCLI)
    cli.agent = Agent()
    cli.conversation_history = [{"role": "user", "content": "do workflow"}]

    result = cli._workflow_result_dict("do workflow")

    assert result is not None
    assert result["final_response"] == "merged answer"
    expected_messages = [
        {"role": "user", "content": "do workflow"},
        {"role": "assistant", "content": "merged answer"},
    ]
    assert result["messages"] == expected_messages
    assert cli.agent.persisted == (expected_messages, [{"role": "user", "content": "do workflow"}])
    assert result["workflow_plan"]["mode"] == "parallel"
    assert result["workflow_plan"]["subtasks"] == [
        {"goal": "A", "context": ""},
        {"goal": "B", "context": ""},
    ]


def test_handle_workflow_command_updates_history_and_prints_response():
    cli = object.__new__(HermesCLI)
    cli._init_agent = lambda **kwargs: True
    cli.conversation_history = []
    printed = []

    expected_messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "answer"},
    ]
    cli._workflow_result_dict = lambda task: {
        "final_response": "answer",
        "messages": expected_messages,
    }
    cli._console_print = printed.append

    assert cli.process_command("/workflow task") is True

    assert cli.conversation_history == expected_messages
    assert printed == ["answer"]


def test_workflow_result_dict_returns_none_for_single_subtask(monkeypatch):
    class FakeOrchestrator:
        def __init__(self, agent):
            self.agent = agent

        def run(self, task):
            return WorkflowResult(
                task=task,
                plan=WorkflowPlan(subtasks=[WorkflowSubtask(goal="inline")]),
                child_results=[],
                final_response="",
                delegated=False,
            )

    monkeypatch.setattr("agent.workflow_orchestrator.WorkflowOrchestrator", FakeOrchestrator)
    cli = object.__new__(HermesCLI)
    cli.agent = object()
    cli.conversation_history = []

    assert cli._workflow_result_dict("simple task") is None
