"""Additional real-wire controls for the ACP authority client."""
import asyncio
from contextlib import suppress
import json
import shlex

import pytest

from acp_adapter.server import HermesACPAgent
from hermes_cli.gateway_client import GatewayClientError
from tests.acp.test_gateway_sessions import daemon, editor, viewer, model_peer  # noqa: F401


@pytest.mark.linux_only
@pytest.mark.asyncio
async def test_profile_mismatch_refuses_before_creation(daemon, tmp_path, monkeypatch):
    import acp_adapter.gateway_server as server
    agent = HermesACPAgent()
    agent._home = tmp_path / "foreign-profile"
    # A real authenticated connection to the wrong profile, not a fake descriptor.
    monkeypatch.setattr(server, "connect_gateway", lambda: viewer(daemon))
    with pytest.raises(GatewayClientError, match="profile_mismatch"):
        await agent.new_session(cwd=str(tmp_path))
    assert agent._gateway is None
    async with viewer(daemon) as ws:
        assert (await ws.rpc("session.list"))["sessions"] == []


@pytest.mark.linux_only
@pytest.mark.asyncio
async def test_real_acp_answer_resolves_the_exact_pending_control(daemon, tmp_path, model_peer):
    from tests.gateway.fixtures.authority_controls_peer import ModelPeer as ApprovalPeer
    target = tmp_path / "native-answer-owned"
    target.mkdir()
    (target / "owned.txt").write_text("disposable")
    model_peer.command = "rm -r -- " + shlex.quote(str(target))
    model_peer.RequestHandlerClass = ApprovalPeer
    async with viewer(daemon) as ws:
        snapshot = await ws.rpc("session.create", source="cli", cwd=str(tmp_path), request_id="native-acp-answer")
        sid = snapshot["session_id"]
        async with editor(daemon, tmp_path) as acp:
            await acp.rpc("initialize", protocolVersion=1, clientCapabilities={})
            await acp.rpc("session/resume", cwd=str(tmp_path), sessionId=sid, mcpServers=[])
            task = asyncio.create_task(acp.rpc("session/prompt", sessionId=sid,
                prompt=[{"type": "text", "text": "Remove owned fixture after consent"}]))
            try:
                async with asyncio.timeout(15):
                    while not any(f.get("method") == "session/request_permission" for f in acp.frames):
                        await asyncio.sleep(.05)
                permission = next(f for f in acp.frames if f.get("method") == "session/request_permission")
                assert target.exists()
                pending = (await ws.rpc("session.resume", session_id=sid))["prompts"][0]
                acp.process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": permission["id"], "result": {
                    "outcome": {"outcome": "selected", "optionId": "allow_once"}}}).encode() + b"\n")
                await acp.process.stdin.drain()
                response = await task
                assert response["result"]["stopReason"] == "end_turn"
                assert not target.exists()
                assert "APPROVAL_FINISHED" in json.dumps(acp.frames)
                print("ACP_NATIVE_ANSWER_RECEIPT=" + json.dumps({"session_id": sid,
                      "prompt_id": pending["prompt_id"], "generation": pending["execution_generation"],
                      "owned_effect_after_acp_answer": True}))
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
