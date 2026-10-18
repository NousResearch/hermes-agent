"""Real Relay ATOF export of Hermes's opaque Docker tool correlation."""

from __future__ import annotations

import json

pytest_plugins = ("tests._fixtures.relay_capture",)


def test_atof_tool_end_exports_only_opaque_container_identity(relay_capture, monkeypatch):
    from agent import relay_tools
    from tools.environments.base import BaseEnvironment
    from tools.environments.docker import DockerEnvironment
    from tools.execution_observability import (
        docker_exec_started,
        opaque_docker_resource_id,
    )

    native_id = "a" * 64
    env = object.__new__(DockerEnvironment)
    env._container_id = native_id
    env._persist_across_processes = False

    def docker_execute(_self, *_args, **_kwargs):
        docker_exec_started(native_id)
        return {"output": "ready", "returncode": 0}

    monkeypatch.setattr(BaseEnvironment, "execute", docker_execute)
    capture = relay_capture
    lease = capture.acquire()
    turn = capture.begin(lease)
    result, _ = relay_tools.execute(
        "terminal", {"command": "echo ready"}, lambda _args: env.execute("echo ready"),
        session_id=lease.session_id, tool_call_id="tool-1",
    )
    assert result == {"output": "ready", "returncode": 0}
    capture.coordinator.end_turn(turn, outcome="success")

    events = capture.events()
    assert native_id not in json.dumps(events)
    tool_ends = [
        event
        for event in events
        if event.get("name") == "terminal" and event.get("scope_category") == "end"
    ]
    assert len(tool_ends) == 1
    annotation = tool_ends[0]["category_profile"]["tool_result_annotation"][
        "hermes.execution_environment"
    ]
    assert annotation["resource_id"] == opaque_docker_resource_id(native_id)
    assert annotation["resource_id_status"] == "available"
