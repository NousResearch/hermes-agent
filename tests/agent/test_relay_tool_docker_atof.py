"""Real Relay ATOF export of Hermes's opaque Docker tool correlation."""

from __future__ import annotations

import json

import pytest


def test_atof_tool_end_exports_only_opaque_container_identity(tmp_path, monkeypatch):
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")

    from agent import relay_runtime, relay_tools
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
    atof_dir = tmp_path / "atof"
    atof_dir.mkdir()
    config = tmp_path / "plugins.toml"
    config.write_text(
        f"""version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 4

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = {json.dumps(str(atof_dir))}
filename = "events.jsonl"
mode = "overwrite"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    relay_runtime._reset_for_tests()
    lease = None
    try:
        lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
            profile_key=relay_runtime.current_profile_key(),
            session_id="session-1",
            platform="cli",
        )
        lease.host.retain_managed_execution("test.docker_atof")
        turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
            lease, turn_id="turn-1", task_id="task-1"
        )
        result, _ = relay_tools.execute(
            "terminal",
            {"command": "echo ready"},
            lambda _args: env.execute("echo ready"),
            session_id="session-1",
            tool_call_id="tool-1",
        )
        assert result == {"output": "ready", "returncode": 0}
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
    finally:
        if lease is not None:
            lease.host.release_managed_execution("test.docker_atof")
            relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()

    raw = (atof_dir / "events.jsonl").read_text(encoding="utf-8")
    assert native_id not in raw
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
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
