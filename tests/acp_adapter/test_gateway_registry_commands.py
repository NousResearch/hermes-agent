"""Registry commands ACP does not answer itself, over the real ACP stdio wire and a real gateway.

Every registered Hermes command typed in an editor ends the prompt with a reply the user reads:
the owner's own answer (``slash.exec``, ``session.mutate``) or a named refusal. None of them may
fail the JSON-RPC request with ``Internal error`` (#134889 ACP-17).
"""
import json

import pytest

from tests.acp_adapter.test_gateway_sessions import editor, viewer


def _replies(frames, start):
    return "".join(f["params"]["update"]["content"]["text"] for f in frames[start:]
                   if f.get("method") == "session/update"
                   and f["params"]["update"].get("sessionUpdate") == "agent_message_chunk")


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_registered_commands_answer_or_refuse_by_name_never_internal_error(daemon, tmp_path, model_peer):
    async with viewer(daemon) as ws:
        sid = (await ws.rpc("session.create", request_id="acp-registry", source="cli", cwd=str(tmp_path)))["session_id"]
        async with editor(daemon, tmp_path) as acp:
            await acp.rpc("initialize", protocolVersion=1, clientCapabilities={})
            assert "result" in await acp.rpc("session/load", cwd=str(tmp_path), sessionId=sid, mcpServers=[])
            replies = {}
            for text in ("/title Registry probe", "/title", "/yolo", "/skills", "/undo", "/retry", "/bg x",
                         "/rollback", "/branch"):
                start = len(acp.frames)
                reply = await acp.rpc("session/prompt", sessionId=sid, prompt=[{"type": "text", "text": text}])
                assert "error" not in reply, (text, reply)
                assert reply["result"]["stopReason"] == "end_turn", (text, reply)
                replies[text] = _replies(acp.frames, start)
                assert replies[text].strip(), (text, acp.frames[start:])
            assert "Registry probe" in replies["/title Registry probe"], replies
            assert "Registry probe" in replies["/title"], replies
            assert "yolo" in replies["/yolo"].lower() or "YOLO" in replies["/yolo"], replies
            for refused in ("/bg x", "/rollback"):
                name = refused.split()[0]
                assert f"{name} is not available" in replies[refused], replies
            assert "fork" in replies["/branch"].lower(), replies
            # No model turn ran: every command was answered by the owner or refused.
            assert not model_peer.requests, model_peer.requests

            # /retry resends the last user turn as a new turn; /undo removes it and reports it.
            turn = await acp.rpc("session/prompt", sessionId=sid,
                                 prompt=[{"type": "text", "text": "WS_SHARED rewind me"}])
            assert turn["result"]["stopReason"] == "end_turn", turn
            retried = await acp.rpc("session/prompt", sessionId=sid, prompt=[{"type": "text", "text": "/retry"}])
            assert retried["result"]["stopReason"] == "end_turn", retried
            assert len(model_peer.requests) == 2, model_peer.requests
            start = len(acp.frames)
            undone = await acp.rpc("session/prompt", sessionId=sid, prompt=[{"type": "text", "text": "/undo"}])
            assert undone["result"]["stopReason"] == "end_turn", undone
            assert "Removed message: WS_SHARED rewind me" in _replies(acp.frames, start), acp.frames[start:]
        info = await ws.rpc("session.info", session_id=sid)
        assert info.get("title") == "Registry probe", info
        messages = json.dumps((await ws.rpc("session.resume", session_id=sid))["messages"])
        assert "WS_SHARED rewind me" not in messages, messages
    print("ACP_REGISTRY_COMMANDS=" + json.dumps(replies))
