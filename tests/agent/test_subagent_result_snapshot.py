"""A plugin must not be able to mutate the retained terminal receipt."""
import dataclasses
import hashlib
import json
from types import SimpleNamespace

from agent import subagent_lifecycle as lifecycle


def test_result_metadata_mutation_does_not_change_the_retained_receipt(monkeypatch):
    service = lifecycle.SubagentLifecycleService(lambda: SimpleNamespace(session_id="parent"))
    created = lifecycle.time.time()
    handle = lifecycle.SubagentHandle(
        1, "snapshot-child", "parent", None, created, None, None, "leaf", 1,
        service._capability("snapshot-child", "parent", created),
    )
    result = lifecycle.SubagentResult(
        handle, lifecycle.SubagentState.SUCCEEDED, True,
        usage_metadata={"api_calls": 2}, tool_execution_summary={"duration_seconds": 3},
    )
    payload = dataclasses.asdict(result)
    payload.pop("result_hash")
    result = dataclasses.replace(result, result_hash=hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest())
    registry = lifecycle._Registry()
    registry.records[handle.subagent_id] = lifecycle._Record(
        handle, result.terminal_state, created, result=result, completed_at=created,
    )
    monkeypatch.setattr(lifecycle, "_REGISTRY", registry)

    received = service.result(handle)
    received.usage_metadata["api_calls"] = 999
    received.tool_execution_summary["duration_seconds"] = 999

    retained = service.result(handle)
    assert retained.usage_metadata == {"api_calls": 2}
    assert retained.tool_execution_summary == {"duration_seconds": 3}
    assert retained == result
