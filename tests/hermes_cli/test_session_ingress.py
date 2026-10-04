"""Queue CLI preserves exact addressing, profile isolation and credential secrecy."""

import json
import sys
import types
from contextlib import closing, nullcontext
from unittest.mock import Mock

import pytest

from hermes_cli import session_ingress
from hermes_state import SessionDB


def test_queue_parser_and_exact_profile_resolution(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    monkeypatch.setitem(sys.modules, "hermes_bootstrap", types.ModuleType("hermes_bootstrap"))
    from hermes_cli.main import _build_cli_parser, _parse_cli_args

    parser, subparsers = _build_cli_parser()
    args = _parse_cli_args(parser, subparsers, ["queue", "--session", "SID", "--message", "TEXT"])
    assert (args.session, args.message, args.func.__name__) == ("SID", "TEXT", "cmd_session_ingress")
    steer = _parse_cli_args(parser, subparsers, ["steer", "--session", "SID", "--message", "TEXT"])
    assert (steer.command, steer.session, steer.message, steer.func) == ("steer", "SID", "TEXT", args.func)
    assert "live/UI ID" in subparsers.choices["queue"].format_help()
    for argv in (["queue", "--message", "TEXT"], ["queue", "--session", "SID"]):
        with pytest.raises(SystemExit) as caught:
            parser.parse_args(argv)
        assert caught.value.code == 2

    homes = [tmp_path / "a", tmp_path / "b"]
    for home in homes:
        monkeypatch.setenv("HERMES_HOME", str(home))
        with closing(SessionDB()) as db:
            for sid, title in [(home.name + "-id", "same name"), ("exact-id", "other"),
                               ("name-id", "exact-id"), ("live-title", "1f508780"),
                               ("title-shadow", "duplicate-live")]:
                db.create_session(sid, "desktop")
                db.set_session_title(sid, title)
            # Older/imported stores can lack the unique-title constraint.
            db._conn.execute("DROP INDEX idx_sessions_title_unique")
            for sid in ["duplicate-1", "duplicate-2"]:
                db.create_session(sid, "desktop")
                db._conn.execute("UPDATE sessions SET title = 'duplicate' WHERE id = ?", (sid,))
            db._conn.commit()

    for home in [homes[0], homes[1], homes[0]]:
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert session_ingress.resolve_live_session("same name", []) == home.name + "-id"
        owners = [
            {"session_id": home.name + "-id", "metadata": {"live_session_id": "1f508780"}},
            {"session_id": "live-name-owner", "metadata": {"live_session_id": "same name"}},
            {"session_id": "live-2", "metadata": {"live_session_id": "duplicate-live"}},
            {"session_id": "live-3", "metadata": {"live_session_id": "duplicate-live"}},
            {"session_id": "collision-owner", "metadata": {"live_session_id": "exact-id"}},
            {"session_id": "collision-2", "metadata": {"live_session_id": "exact-id"}},
        ]
        assert session_ingress.resolve_live_session("exact-id", owners) == "exact-id"
        assert session_ingress.resolve_live_session(home.name + "-id", owners) == home.name + "-id"
        assert session_ingress.resolve_live_session("1f508780", owners) == home.name + "-id"
        assert session_ingress.resolve_live_session("same name", owners) == "live-name-owner"
        with pytest.raises(ValueError, match="Ambiguous.*live-2.*live-3"):
            session_ingress.resolve_live_session("duplicate-live", owners)
        for target in ("1f50878", "1F508780"):
            with pytest.raises(ValueError, match="No session"):
                session_ingress.resolve_live_session(target, owners)
        for target in ("same", "a-", "missing"):
            with pytest.raises(ValueError, match="No session"):
                session_ingress.resolve_live_session(target, [])
        with pytest.raises(ValueError, match="Ambiguous.*duplicate-1.*duplicate-2"):
            session_ingress.resolve_live_session("duplicate", [])
    assert session_ingress.resolve_live_session("lazy", [{"session_id": "lazy"}]) == "lazy"
    # A registry durable ID wins even when other owners use it as their UI ID.
    assert session_ingress.resolve_live_session("lazy", [
        {"session_id": "lazy"},
        {"session_id": "other-1", "metadata": {"live_session_id": "lazy"}},
        {"session_id": "other-2", "metadata": {"live_session_id": "lazy"}},
    ]) == "lazy"


@pytest.mark.parametrize("failure", ["offline", "unsupported", "discovery", "transport", "rpc"])
def test_queue_failures_do_not_leak_credentials(tmp_path, monkeypatch, capsys, failure):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with closing(SessionDB()) as db:
        db.create_session("stored", "desktop")
        db.set_session_title("stored", "name")
    secret = "ws://127.0.0.1:1234/api/ws?token=SECRET-CANARY"
    owners = [] if failure == "offline" else [{"session_id": "stored", "metadata": {
        "live_session_id": None if failure == "unsupported" else "live"}}]
    monkeypatch.setattr(session_ingress, "active_session_registry_snapshot", lambda **kw: owners)

    def discover(_sid):
        if failure == "discovery":
            raise ValueError(secret)
        return secret

    def submit(url, sid, text, action):
        assert action == "queue"
        assert (url, sid, text) == (secret, "live", "SECOND")
        if failure == "transport":
            raise OSError(secret)
        return {"error": {"code": 4001, "message": secret}}

    monkeypatch.setattr(session_ingress, "discover_attach_url", discover)
    monkeypatch.setattr(session_ingress, "submit_live_prompt", submit)
    assert session_ingress.send_live_message("name", "SECOND", "queue") == 1
    out = capsys.readouterr()
    assert not out.out
    assert "SECRET-CANARY" not in out.err and "ws://" not in out.err
    if failure == "offline":
        assert "not live" in out.err and "hermes --resume stored" in out.err


@pytest.mark.parametrize("steer_reply", [
    {"result": {"status": "rejected"}},
    {"error": {"code": 4010, "message": "agent does not support steer"}},
])
@pytest.mark.parametrize("submit_reply", [
    {"result": {"status": "queued"}},
    {"error": {"code": 4010, "message": "submission rejected"}},
])
def test_steer_explicit_rejection_submits_next_turn_once(monkeypatch, steer_reply, submit_reply):
    import websockets.sync.client

    socket = Mock()
    connect = Mock(return_value=nullcontext(socket))
    monkeypatch.setattr(websockets.sync.client, "connect", connect)
    replies = iter([steer_reply, submit_reply])

    def recv(**kwargs):
        request = json.loads(socket.send.call_args.args[0])
        return json.dumps({"id": request["id"], **next(replies)})

    socket.recv.side_effect = recv
    reply = session_ingress.submit_live_prompt("ws://owner", "live", "NEXT", "steer")
    requests = [json.loads(call.args[0]) for call in socket.send.call_args_list]
    assert [request["method"] for request in requests] == ["session.steer", "prompt.submit"]
    assert requests[0]["params"] == {"session_id": "live", "text": "NEXT"}
    assert requests[1]["params"] == {"session_id": "live", "text": "NEXT", "queued": True}
    assert requests[0]["id"] != requests[1]["id"]
    assert reply == {"id": requests[1]["id"], **submit_reply}
    connect.assert_called_once()


@pytest.mark.parametrize("outcome", ["accepted", "unknown_error", "transport_error"])
def test_steer_does_not_fallback_without_explicit_rejection(monkeypatch, outcome):
    import websockets.sync.client

    socket = Mock()
    connect = Mock(return_value=nullcontext(socket))
    monkeypatch.setattr(websockets.sync.client, "connect", connect)

    def recv(**kwargs):
        if outcome == "transport_error":
            raise TimeoutError("submission unconfirmed")
        request = json.loads(socket.send.call_args.args[0])
        payload = ({"error": {"code": 4011, "message": "unknown"}}
                   if outcome == "unknown_error" else {"result": {"status": "queued"}})
        return json.dumps({"id": request["id"], **payload})

    socket.recv.side_effect = recv
    if outcome == "transport_error":
        with pytest.raises(TimeoutError, match="submission unconfirmed"):
            session_ingress.submit_live_prompt("ws://owner", "live", "NEXT", "steer")
    else:
        reply = session_ingress.submit_live_prompt("ws://owner", "live", "NEXT", "steer")
        assert (reply["error"]["code"] == 4011 if outcome == "unknown_error"
                else reply["result"]["status"] == "queued")
    socket.send.assert_called_once()
    assert json.loads(socket.send.call_args.args[0])["method"] == "session.steer"
    connect.assert_called_once()
