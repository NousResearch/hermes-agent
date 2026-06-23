from backend.models.handoff import Handoff
from backend.models.task import Task


def test_task_defaults() -> None:
    task = Task(id="task-1", title="Test", goal="Ship")
    assert task.status == "pending"
    assert task.agent == "codex"
    assert task.tags == []


def test_handoff_defaults() -> None:
    handoff = Handoff(id="handoff-1", from_agent="hermes", to_agent="codex")
    assert handoff.status == "pending"
    assert handoff.payload == {}
    assert handoff.log_refs == []
