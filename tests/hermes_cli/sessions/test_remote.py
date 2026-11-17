"""`hermes sessions send|wait`: a script drives a gateway-hosted session it names by prefix or `-`."""
import argparse
import asyncio
import json
import subprocess

import pytest


def _parser():
    from hermes_cli import sessions_cmd
    from hermes_cli.subcommands.sessions import build_sessions_parser
    parser = argparse.ArgumentParser()
    build_sessions_parser(parser.add_subparsers(dest="command"), cmd_sessions=sessions_cmd.cmd_sessions)
    return parser


def _store_sessions(tmp_path):
    """Two CLI sessions in two git workspaces; the one in `other` is the most recent overall."""
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    here, other = tmp_path / "here", tmp_path / "other"
    db = SessionDB(get_hermes_home() / "state.db")
    try:
        for name, sid in ((here, "local-" + "a" * 64), (other, "local-" + "b" * 64)):
            name.mkdir()
            subprocess.run(["git", "init", "-q", str(name)], check=True)
            db.create_session(sid, "cli", cwd=str(name), git_repo_root=str(name))
            db.append_message(sid, "user", "earlier prompt")
    finally:
        db.close()
    return here


async def _owner(monkeypatch, calls, *, busy_until_info=False, starts_during_attach=False):
    """A fake gateway owner: resume answers for whatever id it is asked, a submit is admitted and
    settles with a reply, and (``busy_until_info``) the session stays busy until a session.info."""
    from websockets.asyncio.server import serve

    async def peer(ws):
        async for raw in ws:
            request = json.loads(raw)
            method, params = request["method"], request["params"]
            calls.append((method, params))
            sid = params.get("session_id")
            result = {}
            if method == "session.resume":
                pending = [{"admission_id": "running", "status": "started"}] if busy_until_info else []
                result = {"session_id": sid, "stored_session_id": sid, "execution_generation": 3,
                          "pending": pending, "prompts": [], "info": {}}
            elif method == "prompt.submit":
                result = {"admission_id": "adm-1", "status": "queued"}
            elif method == "prompt.receipt":
                result = {"admission_id": "adm-1", "status": "terminal", "outcome": "completed",
                          "result": {"final_response": "PONG", "completed": True}}
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}))
            if method == "prompt.submit":
                await ws.send(json.dumps({"jsonrpc": "2.0", "method": "event", "params": {
                    "type": "message.complete", "session_id": sid, "admission_id": "adm-1",
                    "payload": {"text": "PONG", "outcome": "completed", "admission_id": "adm-1"}}}))
            if method == "session.resume" and starts_during_attach and len(calls) == 1:
                # A turn starts on the first session while the second is still attaching.
                await ws.send(json.dumps({"jsonrpc": "2.0", "method": "event", "params": {
                    "type": "session.info", "session_id": sid,
                    "payload": {"pending": [{"admission_id": "late", "status": "started"}]}}}))
            if method == "session.resume" and busy_until_info and len(calls) > 1:
                await ws.send(json.dumps({"jsonrpc": "2.0", "method": "event", "params": {
                    "type": "session.info", "session_id": sid, "payload": {"pending": []}}}))

    server = await serve(peer, "127.0.0.1", 0).__aenter__()
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}")
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("name, expected", [("local-bbbb", "local-" + "b" * 64), ("-", "local-" + "a" * 64)])
async def test_send_wait_resolves_prefix_or_this_workspaces_last_session_and_prints_the_reply(
        monkeypatch, capsys, tmp_path, name, expected):
    """A unique id prefix names that session; `-` names this directory's last CLI session even when
    another workspace's session is newer. The prompt goes in as a finite turn and the reply is printed
    with the finite chat exit code."""
    monkeypatch.chdir(_store_sessions(tmp_path))
    calls = []
    server = await _owner(monkeypatch, calls)
    try:
        args = _parser().parse_args(["sessions", "send", name, "say", "PONG", "--wait"])
        assert await asyncio.to_thread(args.func, args) == 0
    finally:
        server.close()
    assert calls[0] == ("session.resume", {"session_id": expected, "omit_messages": True})
    submit = next(params for method, params in calls if method == "prompt.submit")
    assert (submit["session_id"], submit["text"], submit["finite"]) == (expected, "say PONG", True)
    captured = capsys.readouterr()
    assert captured.out.strip() == "PONG"
    assert f"session_id: {expected}" in captured.err


@pytest.mark.asyncio
async def test_wait_times_out_with_124_on_a_busy_session_and_returns_0_once_it_is_idle(monkeypatch, capsys, tmp_path):
    """`wait` never interrupts: past --timeout it exits 124; a session.info clearing the pending
    rows ends the wait with 0. A turn that starts on the first session while the second is still
    attaching (both snapshots read idle) keeps `--all` waiting."""
    monkeypatch.chdir(_store_sessions(tmp_path))
    calls = []
    server = await _owner(monkeypatch, calls, busy_until_info=True)
    try:
        timed = _parser().parse_args(["sessions", "wait", "local-aaaa", "--timeout", "0.5"])
        assert await asyncio.to_thread(timed.func, timed) == 124
        settled = _parser().parse_args(["sessions", "wait", "local-aaaa", "local-bbbb", "--timeout", "10"])
        assert await asyncio.to_thread(settled.func, settled) == 0
    finally:
        server.close()
    assert not any(method == "session.interrupt" for method, _ in calls)
    assert "Timed out after 0.5s" in capsys.readouterr().err
    raced = []
    server = await _owner(monkeypatch, raced, starts_during_attach=True)
    try:
        both = _parser().parse_args(["sessions", "wait", "local-aaaa", "local-bbbb", "--timeout", "0.5"])
        assert await asyncio.to_thread(both.func, both) == 124
    finally:
        server.close()
