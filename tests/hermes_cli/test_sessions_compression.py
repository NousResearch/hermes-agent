"""The CLI uses the owning gateway's fenced mutation, not a local database writer."""

import argparse
import asyncio
import json

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", [None, "session_busy", "not_found"])
async def test_reset_compression_uses_resolved_session_fences_and_reports_failures(
    monkeypatch, capsys, refusal
):
    from websockets.asyncio.server import serve
    from hermes_cli.sessions_cmd import cmd_sessions
    from hermes_cli.subcommands.sessions import build_sessions_parser

    calls = []

    async def peer(ws):
        async for raw in ws:
            request = json.loads(raw)
            calls.append((request["method"], request["params"]))
            if refusal and request["method"] == (
                "session.resume" if refusal == "not_found" else "session.mutate"
            ):
                reply = {
                    "error": {
                        "code": 4001,
                        "message": refusal,
                        "data": {"reason": refusal},
                    }
                }
            else:
                reply = {
                    "result": {
                        "session_id": "canonical-id",
                        "revision": 7,
                        "execution_generation": 4,
                        "subscription_id": "attachment",
                    }
                }
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], **reply}))

    parser = argparse.ArgumentParser()
    build_sessions_parser(
        parser.add_subparsers(dest="command"), cmd_sessions=cmd_sessions
    )
    args = parser.parse_args(["sessions", "reset-compression", "listed-id"])
    async with serve(peer, "127.0.0.1", 0) as server:
        monkeypatch.setenv(
            "HERMES_TUI_GATEWAY_URL",
            f"ws://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}",
        )
        assert await asyncio.to_thread(args.func, args) == (1 if refusal else 0)
    assert calls[0] == (
        "session.resume",
        {"session_id": "listed-id", "omit_messages": True},
    )
    if refusal == "not_found":
        assert len(calls) == 1
    else:
        method, mutation = calls[1]
        assert method == "session.mutate"
        assert mutation.pop("request_id")
        assert mutation == {
            "session_id": "canonical-id",
            "expected_revision": 7,
            "expected_generation": 4,
            "operation": "compression_model",
            "payload": {},
        }
        assert calls[2] == (
            "session.detach",
            {"session_id": "canonical-id", "subscription_id": "attachment"},
        )
    output = capsys.readouterr()
    assert (
        (refusal in output.err and not output.out)
        if refusal
        else "History and main model kept" in output.out
    )
