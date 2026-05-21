import json
import logging

from gateway.builtin_hooks import swarm_operator
from gateway.hooks import HookRegistry


class FailingStore:
    def save_job(self, job):
        raise RuntimeError("disk full")

    def append_event(self, event):  # pragma: no cover - save fails first
        raise AssertionError("should not append")


def _context(tmp_path, enabled=True):
    from agent.swarm_store import SwarmStore

    return {
        "platform": "slack",
        "user_id": "U123",
        "chat_id": "C123",
        "session_id": "S123",
        "message": "Research docs and review code",
        "message_id": "M123",
        "gateway_config": {"swarm_operator": {"enabled": enabled, "dry_run": True, "max_children": 3}},
        "swarm_store": SwarmStore(base_dir=tmp_path),
    }


def test_hook_receives_inbound_metadata_and_message_text(tmp_path):
    swarm_operator.handle("agent:start", _context(tmp_path, enabled=True))

    data = json.loads((tmp_path / "swarm_operator_state.json").read_text())
    job = next(iter(data["jobs"].values()))
    assert job["original_request"] == "Research docs and review code"
    assert job["platform"] == "slack"
    assert job["user_id"] == "U123"
    assert job["chat_id"] == "C123"
    assert job["session_id"] == "S123"
    assert job["metadata"]["message_id"] == "M123"


def test_hook_disabled_does_nothing(tmp_path):
    swarm_operator.handle("agent:start", _context(tmp_path, enabled=False))

    assert not (tmp_path / "swarm_operator_state.json").exists()
    assert not (tmp_path / "swarm_operator_metrics.jsonl").exists()


def test_hook_enabled_dry_run_writes_job_and_metrics_event(tmp_path):
    swarm_operator.handle("agent:start", _context(tmp_path, enabled=True))

    state = json.loads((tmp_path / "swarm_operator_state.json").read_text())
    job = next(iter(state["jobs"].values()))
    assert job["metadata"]["dry_run"] is True
    assert job["routing_plan"]["mode"] == "swarm"

    metrics = [json.loads(line) for line in (tmp_path / "swarm_operator_metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["event_type"] == "shadow_job_recorded"
    assert metrics[-1]["metadata"]["job_id"] == job["job_id"]


def test_hook_honcho_summary_uses_injected_writer_only_when_enabled(tmp_path):
    calls = []
    context = _context(tmp_path, enabled=True)
    context["gateway_config"]["swarm_operator"]["persist_to_honcho"] = True
    context["swarm_honcho_writer"] = lambda payload: calls.append(payload)

    swarm_operator.handle("agent:start", context)

    assert len(calls) == 1
    assert calls[0]["metadata"]["job_id"]
    state = json.loads((tmp_path / "swarm_operator_state.json").read_text())
    job = next(iter(state["jobs"].values()))
    assert job["metadata"]["honcho_summary"]["persisted"] is True


def test_hook_honcho_summary_not_called_when_disabled(tmp_path):
    calls = []
    context = _context(tmp_path, enabled=True)
    context["swarm_honcho_writer"] = lambda payload: calls.append(payload)

    swarm_operator.handle("agent:start", context)

    assert calls == []


def test_hook_never_blocks_normal_flow_on_store_failure(caplog):
    context = _context(tmp_path="/tmp/unused", enabled=True)
    context["swarm_store"] = FailingStore()

    with caplog.at_level(logging.WARNING):
        result = swarm_operator.handle("agent:start", context)

    assert result is None
    assert "swarm operator shadow hook failed" in caplog.text


def test_builtin_hook_registry_registers_swarm_operator_default_disabled(tmp_path):
    registry = HookRegistry(include_builtins=True)
    registry.discover_and_load()

    assert any(hook["name"] == "swarm_operator" for hook in registry.loaded_hooks)
    # Default config omitted means disabled; emitting must not create runtime state.
    import asyncio

    asyncio.run(registry.emit("agent:start", {"message": "hello", "swarm_store": _context(tmp_path)["swarm_store"]}))
    assert not (tmp_path / "swarm_operator_state.json").exists()
