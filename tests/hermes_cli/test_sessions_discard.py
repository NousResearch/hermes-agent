"""`hermes sessions discard`: a script can clear a turn lost in a gateway crash without a REPL."""
import argparse
import asyncio
import json

import pytest


@pytest.mark.asyncio
async def test_sessions_discard_yes_resolves_every_unknown_admission_with_its_fenced_generation(monkeypatch, capsys):
    """The same fenced control as interactive /discard: resume by title, resolve each unknown row with the
    generation the owner stamped on it, and never touch the queued input behind it. Without --yes on a
    non-TTY stdin it refuses, and nothing is resolved."""
    from websockets.asyncio.server import serve

    from hermes_cli import sessions_cmd
    from hermes_cli.subcommands.sessions import build_sessions_parser

    lost = ["admission-lost-a", "admission-lost-b"]
    snapshot = {"session_id": "local-abc", "stored_session_id": "local-abc", "pending": [
        {"admission_id": lost[0], "status": "unknown", "execution_generation": 4},
        {"admission_id": "admission-queued", "status": "queued", "execution_generation": None},
        {"admission_id": lost[1], "status": "unknown", "execution_generation": 5}]}
    calls = []

    async def peer(ws):
        async for raw in ws:
            request = json.loads(raw)
            calls.append((request["method"], request["params"]))
            if request["method"] == "session.resume" and "session_id" in request["params"]:
                reply = {"error": {"code": 4001, "message": "not_found", "data": {"reason": "not_found"}}}
            else:
                reply = {"result": snapshot if request["method"] == "session.resume" else {"status": "terminal"}}
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], **reply}))

    parser = argparse.ArgumentParser()
    build_sessions_parser(parser.add_subparsers(dest="command"), cmd_sessions=sessions_cmd.cmd_sessions)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    async with serve(peer, "127.0.0.1", 0) as server:
        monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}")

        def run(*argv):
            args = parser.parse_args(["sessions", "discard", "my script session", *argv])
            return args.func(args)

        assert await asyncio.to_thread(run) == 2
        assert "--yes" in capsys.readouterr().err
        assert [m for m, _ in calls] == ["session.resume", "session.resume"]
        calls.clear()
        assert await asyncio.to_thread(run, "--yes") == 0

    assert calls == [
        ("session.resume", {"session_id": "my script session"}),
        ("session.resume", {"title": "my script session"}),
        ("prompt.resolve_unknown", {"session_id": "local-abc", "admission_id": lost[0], "execution_generation": 4}),
        ("prompt.resolve_unknown", {"session_id": "local-abc", "admission_id": lost[1], "execution_generation": 5}),
    ]
    assert capsys.readouterr().out.split() == ["Discarded", lost[0], "Discarded", lost[1]]
