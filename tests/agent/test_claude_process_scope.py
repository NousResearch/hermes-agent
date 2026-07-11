from types import SimpleNamespace
import json

import pytest

from agent.claude_process_scope import WorkerProcessBroker


class FakeRegistry:
    def get(self, session_id):
        return {
            "owned": SimpleNamespace(task_id="shared-container", owner_task_id="worker-task"),
            "foreign": SimpleNamespace(task_id="shared-container", owner_task_id="other-task"),
        }.get(session_id)

    def list_sessions(self, owner_task_id=None):
        assert owner_task_id == "worker-task"
        return [{"session_id": "owned"}]

    def poll(self, session_id):
        return {"session_id": session_id, "status": "running"}


@pytest.mark.parametrize(
    "action", ["poll", "log", "wait", "kill", "write", "submit", "close"]
)
def test_worker_process_scope_rejects_every_foreign_session_action(action):
    with pytest.raises(RuntimeError, match="does not belong"):
        WorkerProcessBroker("worker-task", FakeRegistry()).handle(
            {"action": action, "session_id": "foreign"}
        )


def test_worker_process_scope_allows_owned_session_and_strict_list():
    broker = WorkerProcessBroker("worker-task", FakeRegistry())
    owned = broker.handle({"action": "poll", "session_id": "owned"})
    listed = broker.handle({"action": "list"})

    assert '"session_id": "owned"' in owned
    assert '"session_id": "owned"' in listed


def test_model_cannot_forge_worker_authorization_marker():
    broker = WorkerProcessBroker("worker-task", FakeRegistry())
    with pytest.raises(RuntimeError, match="does not belong"):
        broker.handle(
            {
                "action": "poll",
                "session_id": "foreign",
                "_hermes_worker_task_id": "worker-task",
            }
        )


def test_worker_process_broker_controls_own_background_process_end_to_end(tmp_path):
    from tools.process_registry import ProcessRegistry

    registry = ProcessRegistry()
    session = registry.spawn_local(
        "printf owned-output",
        cwd=str(tmp_path),
        task_id="shared-container",
        owner_task_id="worker-a",
    )
    owner = WorkerProcessBroker("worker-a", registry)
    sibling = WorkerProcessBroker("worker-b", registry)

    result = json.loads(owner.handle({"action": "wait", "session_id": session.id}))
    listed = json.loads(owner.handle({"action": "list"}))

    assert result["status"] == "exited"
    assert "owned-output" in result["output"]
    assert [item["session_id"] for item in listed["processes"]] == [session.id]
    with pytest.raises(RuntimeError, match="does not belong"):
        sibling.handle({"action": "log", "session_id": session.id})
    assert json.loads(sibling.handle({"action": "list"})) == {"processes": []}
