"""`hermes sessions interrupt`: a supervisor can /stop a gateway-hosted session's running turn by id."""
import argparse
import asyncio
import json

import pytest


def _parser():
    from hermes_cli import sessions_cmd
    from hermes_cli.subcommands.sessions import build_sessions_parser
    parser = argparse.ArgumentParser()
    build_sessions_parser(parser.add_subparsers(dest="command"), cmd_sessions=sessions_cmd.cmd_sessions)
    return parser


async def _serve(monkeypatch, snapshot, calls):
    from websockets.asyncio.server import serve

    async def peer(ws):
        async for raw in ws:
            request = json.loads(raw)
            calls.append((request["method"], request["params"]))
            if request["method"] == "session.resume" and request["params"].get("session_id") != snapshot["session_id"]:
                reply = {"error": {"code": 4001, "message": "not_found", "data": {"reason": "not_found"}}}
            else:
                reply = {"result": snapshot if request["method"] == "session.resume" else {"execution_state": "idle"}}
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], **reply}))

    server = await serve(peer, "127.0.0.1", 0).__aenter__()
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}")
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["interrupt", "stop"])
async def test_interrupt_sends_fenced_session_interrupt_for_the_running_generation(monkeypatch, capsys, verb):
    snapshot = {"session_id": "local-abc", "stored_session_id": "local-abc",
                "execution_state": "running", "execution_generation": 7}
    calls = []
    server = await _serve(monkeypatch, snapshot, calls)
    try:
        args = _parser().parse_args(["sessions", verb, "local-abc", "--json"])
        assert await asyncio.to_thread(args.func, args) == 0
    finally:
        server.close()
    assert calls == [("session.resume", {"session_id": "local-abc"}),
                     ("session.interrupt", {"session_id": "local-abc", "execution_generation": 7})]
    out = json.loads(capsys.readouterr().out)
    assert out == {"session_id": "local-abc", "execution_state": "running",
                   "execution_generation": 7, "interrupted": True}


@pytest.mark.asyncio
async def test_interrupt_on_an_idle_session_sends_nothing_and_exits_3(monkeypatch, capsys):
    snapshot = {"session_id": "local-abc", "stored_session_id": "local-abc",
                "execution_state": "idle", "execution_generation": 2}
    calls = []
    server = await _serve(monkeypatch, snapshot, calls)
    try:
        args = _parser().parse_args(["sessions", "interrupt", "local-abc"])
        assert await asyncio.to_thread(args.func, args) == 3
    finally:
        server.close()
    assert [m for m, _ in calls] == ["session.resume"]
    assert "Nothing to interrupt" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_interrupt_unknown_session_is_an_error(monkeypatch, capsys):
    snapshot = {"session_id": "local-abc", "stored_session_id": "local-abc",
                "execution_state": "running", "execution_generation": 1}
    calls = []
    server = await _serve(monkeypatch, snapshot, calls)
    try:
        args = _parser().parse_args(["sessions", "interrupt", "nope"])
        assert await asyncio.to_thread(args.func, args) == 1
    finally:
        server.close()
    assert "No session found matching 'nope'" in capsys.readouterr().err
    assert all(m == "session.resume" for m, _ in calls)
