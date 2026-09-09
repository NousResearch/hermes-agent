"""Provider overrides cannot replace isolated input, tools, or per-preset output limits."""
import asyncio

import pytest

from gateway.side_runs import SideRunService
from hermes_cli.plugin_side_runs import SideRunConfig
from tests.gateway.test_plugin_side_run_failures import fixture_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("tools", [{"type": "function", "function": {"name": "unselected_tool"}}]),
    ("messages", [{"role": "user", "content": "injected history"}]),
    ("input", "injected history"),
    ("max_tokens", 99999),
    ("max_completion_tokens", 99999),
    ("max_output_tokens", 99999),
    ("reasoning", {"effort": "high"}),
])
@pytest.mark.parametrize("nested", [False, True])
async def test_provider_override_cannot_escape_preset(tmp_path, monkeypatch, field, value, nested):
    runner, db, _, event = fixture_runner(tmp_path)
    constructed = []

    class Agent:
        def __init__(self, **kwargs):
            constructed.append(True)
            self.model, self.provider = kwargs["model"], kwargs["provider"]
        def run_conversation(self, **kwargs):
            return {"final_response": "unsafe success"}
        def close(self):
            pass

    override = {field: value}
    if nested:
        override = {"extra_body": override}
    monkeypatch.setattr("run_agent.AIAgent", Agent)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kwargs: {
        "provider": "openai", "request_overrides": override})
    service = SideRunService(runner)
    sid = service.start(event, "test", "prompt", SideRunConfig.from_mapping({
        "provider": "openai", "model": "fixture", "tools": [], "max_tokens": 80}))
    try:
        await asyncio.wait_for(service.wait(), 10)
        assert not constructed
        assert db.get_session(sid)["end_reason"] == "side_run_failed"
    finally:
        runner._shutdown_executor()
        db.close()
