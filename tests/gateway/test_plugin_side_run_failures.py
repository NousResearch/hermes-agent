"""Side-run failure paths must preserve ownership and fail closed without leaking errors."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.side_runs import SideRunService
from hermes_cli.plugin_side_runs import SideRunConfig
from hermes_state import SessionDB


def fixture_runner(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    runner = object.__new__(GatewayRunner)
    runner._session_db = SimpleNamespace(_db=db)
    runner._draining = False
    runner.session_store = SimpleNamespace(_entries={})
    runner._session_key_for_source = lambda source: "parent"
    runner._reply_anchor_for_event = lambda event: "trigger"
    runner._thread_metadata_for_source = lambda *args: {}
    adapter = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True)))
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._adapter_for_source = lambda source: adapter
    source = SessionSource(Platform.TELEGRAM, "room", user_id="owner")
    return runner, db, adapter, MessageEvent(text="/run prompt", source=source)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["resolver", "identity", "transport", "constructor", "conversation"])
async def test_failed_run_is_saved_owned_and_error_is_sanitized(tmp_path, monkeypatch, stage):
    runner, db, adapter, event = fixture_runner(tmp_path)
    constructed, called, closed = [], [], []
    secret = "private-provider-credential-do-not-display"
    def resolve(**kwargs):
        if stage == "resolver":
            raise ValueError(secret)
        if stage == "transport":
            return {"provider": "openai", "command": "external-agent", "base_url": "acp://fixture"}
        return {"provider": "openrouter" if stage == "identity" else "openai"}
    class Agent:
        def __init__(self, **kwargs):
            constructed.append(True)
            if stage == "constructor":
                raise RuntimeError(secret)
            self.model, self.provider = kwargs["model"], kwargs["provider"]
        def run_conversation(self, **kwargs):
            called.append(True)
            raise RuntimeError(secret)
        def close(self):
            closed.append(True)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    monkeypatch.setattr("run_agent.AIAgent", Agent)
    service = SideRunService(runner)
    sid = service.start(event, "test", "prompt", SideRunConfig.from_mapping({"provider": "openai", "model": "fixture"}))
    try:
        await asyncio.wait_for(service.wait(), 10)
        row = db.get_session(sid)
        assert row["user_id"] == "owner"
        assert row["end_reason"] == "side_run_failed"
        assert secret not in str(row)
        assert "failed" in adapter.send.await_args.args[1]
        assert secret not in str(adapter.send.await_args)
        assert not service.runs
        assert bool(constructed) == (stage in {"constructor", "conversation"})
        assert bool(called) == (stage == "conversation")
        assert bool(closed) == (stage == "conversation")
    finally:
        runner._shutdown_executor()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transport", "timeout"])
async def test_approval_missing_transport_or_timeout_denies_and_cleans_up(tmp_path, monkeypatch, failure):
    from tools import approval
    from tools.approval_context import get_current_session_key
    from tools.approval_gateway_wait import _await_gateway_decision
    runner, db, adapter, event = fixture_runner(tmp_path)
    outcomes = []
    class Agent:
        def __init__(self, **kwargs):
            self.provider, self.model = kwargs["provider"], kwargs["model"]
        def run_conversation(self, **kwargs):
            key = get_current_session_key()
            outcomes.append(_await_gateway_decision(key, approval._gateway_notify_cb(key), {"command": "rm fixture", "pattern_key": "fixture"}))
            return {"final_response": "denied"}
        def close(self):
            pass
    monkeypatch.setattr("run_agent.AIAgent", Agent)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kwargs: {"provider": "openai"})
    if failure == "transport":
        adapter.send.return_value = SendResult(success=False)
    else:
        monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 0)
    service = SideRunService(runner)
    sid = service.start(event, "test", "prompt", SideRunConfig.from_mapping({"provider": "openai", "model": "fixture"}))
    key = service.runs[sid].key
    try:
        await asyncio.wait_for(service.wait(), 10)
        assert outcomes and outcomes[0]["resolved"] is False and outcomes[0]["choice"] is None
        assert not approval.list_gateway_approvals(key)
        assert approval._gateway_notify_cb(key) is None
        assert not service.runs
    finally:
        runner._shutdown_executor()
        db.close()
