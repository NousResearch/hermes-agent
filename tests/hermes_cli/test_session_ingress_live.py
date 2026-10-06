"""Separate CLI process -> real owner registry/HTTP/WS/RPC -> durable queued turns.

The production owner listener, handshake, lease and gateway turn/queue code are
real. Only model execution is deterministic.
"""

import os
import subprocess
import sys
import threading

import pytest
from hermes_cli.active_sessions import active_session_registry_snapshot
from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway import shared_runtime


@pytest.fixture(params=["stdio", "serve", "gated-serve"])
def runtime(tmp_path, monkeypatch, request):
    from tests.tui_gateway.test_queued_prompt_persistence import _desktop_session, _flush_agent
    from agent.turn_context import _stage_turn_user_message

    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(db_path=home / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    server._ensure_session_db_row(session)
    db.set_session_title(key, "exact name")
    agent = _flush_agent(db, key)
    first_started, release_first, second_done = (threading.Event() for _ in range(3))
    corrections, turns = [], []

    def run_conversation(text, *, conversation_history, **kwargs):
        turns.append(text)
        user, _ = _stage_turn_user_message(agent, text, text, None, None, None, None)
        messages = [*conversation_history, user]
        agent._persist_user_message_idx = len(conversation_history)
        agent._flush_messages_to_session_db(messages, conversation_history)
        if text == "FIRST":
            first_started.set()
            assert release_first.wait(15)
        messages.append({"role": "assistant", "content": "reply " + text})
        agent._flush_messages_to_session_db(messages, conversation_history)
        return {"final_response": "reply " + text, "messages": messages}

    agent.run_conversation = run_conversation
    agent.clear_interrupt = lambda: None
    agent.interrupt = lambda *a, **kw: corrections.append("interrupt")
    agent.hard_interrupt = lambda *a, **kw: corrections.append("hard_interrupt")
    agent.steer = lambda *a, **kw: corrections.append("steer") or True
    agent.redirect = lambda *a, **kw: corrections.append("redirect")
    agent._supports_active_turn_redirect = True
    session["agent"] = agent
    session["agent_ready"].set()
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *a: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_start_turn_voice", lambda: (None, False))
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    # Observe settlement after the real turn pipeline has released its slot.
    real_followups = server._run_post_turn_followups

    def followups(rid, sid, session, result, goal):
        real_followups(rid, sid, session, result, goal)
        if turns and turns[-1] == "SECOND" and not session["running"]:
            second_done.set()

    monkeypatch.setattr(server, "_run_post_turn_followups", followups)

    class OwnerWindow:
        _closed = False

        def __init__(self):
            self.received_second = threading.Event()
            self.frames = []

        def write(self, frame):
            self.frames.append(frame)
            params = frame.get("params") or {}
            if params.get("type") == "message.complete" and params.get("payload", {}).get("text") == "reply SECOND":
                self.received_second.set()
            return True

    window = OwnerWindow()
    session["transport"] = window
    if request.param == "stdio":
        shared_runtime.start_shared_runtime()
        stop_runtime = shared_runtime.stop_shared_runtime
    else:
        import socket
        import uvicorn
        from hermes_cli import web_server, mcp_startup

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        origin = f"http://127.0.0.1:{listener.getsockname()[1]}"
        monkeypatch.setattr(shared_runtime, "_origin", None)
        monkeypatch.setattr(web_server.app.state, "session_attach_origin", origin, raising=False)
        monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(web_server.app.state, "auth_required", request.param == "gated-serve", raising=False)
        monkeypatch.setattr(web_server, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
        monkeypatch.setattr(mcp_startup, "start_deferred_mcp_discovery_now", lambda: None)
        # Real production routes/auth/middleware, with lifespan off: no boot probes,
        # cron, external integrations or native notification devices.
        owner_server = uvicorn.Server(uvicorn.Config(
            web_server.app, lifespan="off", log_level="error", log_config=None,
            access_log=False, proxy_headers=False, ws_ping_interval=None, timeout_graceful_shutdown=1))
        owner_thread = threading.Thread(target=lambda: owner_server.run(sockets=[listener]), daemon=True)
        owner_thread.start()

        def stop_runtime():
            owner_server.should_exit = True
            owner_thread.join(3)

    assert server._ensure_active_session_slot(sid, session) is None
    lease = session["active_session_lease"]
    requests, handshakes = [], []
    real_attach_reply = shared_runtime._attach_reply

    def record_attach(session_id, lease_id, profile_home, websocket_url):
        handshakes.append({"session_id": session_id, "lease_id": lease_id, "profile_home": profile_home})
        return real_attach_reply(session_id, lease_id, profile_home, websocket_url)

    monkeypatch.setattr(shared_runtime, "_attach_reply", record_attach)
    real_submit = server._methods["prompt.submit"]

    def record_submit(rid, params):
        requests.append(params.copy())
        return real_submit(rid, params)

    monkeypatch.setitem(server._methods, "prompt.submit", record_submit)
    def cli(target=key, action="queue"):
        # Skip installation bootstrap (this checkout lives under the real home),
        # but exercise the actual CLI entry point in another process.
        code = """import sys, types
sys.modules['hermes_bootstrap'] = types.ModuleType('hermes_bootstrap')
from hermes_cli.main import main
main()
"""
        return subprocess.run([sys.executable, "-c", code, action, "--session", target,
                               "--message", "SECOND"], env=dict(os.environ), capture_output=True,
                              text=True, timeout=20)

    yield locals()
    release_first.set()
    run_thread = session.get("_run_thread")
    if run_thread is not None:
        run_thread.join(10)
    stop_runtime()
    server._sessions.pop(sid, None)
    lease.release()
    db.close()


@pytest.mark.parametrize("mode", ["steer", "interrupt", "idle"])
def test_queue_busy_and_idle_delivery(runtime, monkeypatch, mode):
    r = runtime
    session, db, sid, key = (r[k] for k in ["session", "db", "sid", "key"])
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: mode)
    if mode != "idle":
        response = server.handle_request({"id": "first", "method": "prompt.submit", "params": {
            "session_id": sid, "text": "FIRST"}})
        assert response["result"]["status"] == "streaming", response
        assert r["first_started"].wait(10)
    r["requests"].clear()
    target = {"idle": "exact name", "interrupt": key, "steer": sid}[mode]
    result = r["cli"](target)
    assert result.returncode == 0, result.stderr
    assert "SECRET-CANARY" not in result.stdout + result.stderr
    assert r["requests"] == [{"session_id": sid, "text": "SECOND", "queued": True}]
    assert r["handshakes"] == [{"session_id": key, "lease_id": r["lease"].lease_id,
                                 "profile_home": str(r["home"].resolve())}]
    assert r["corrections"] == []
    if mode != "idle":
        assert session["running"] and r["turns"] == ["FIRST"]
        assert any(row["content"] == "SECOND" for row in db.get_messages(key))
        r["release_first"].set()
    assert r["second_done"].wait(10)
    expected = ([] if mode == "idle" else [("user", "FIRST"), ("assistant", "reply FIRST")])
    expected += [("user", "SECOND"), ("assistant", "reply SECOND")]
    assert [(row["role"], row["content"]) for row in db.get_messages_as_conversation(key)] == expected
    assert active_session_registry_snapshot(r["home"])[0]["lease_id"] == r["lease"].lease_id
    assert r["window"].received_second.wait(10)
    frames = [f["params"] for f in r["window"].frames]
    user = next(f for f in frames if f["type"] == "message.user" and f["payload"]["message"]["text"] == "SECOND")
    canonical = [row for row in db.get_messages(key) if row["content"] == "SECOND"]
    assert len(canonical) == 1
    assert user["payload"]["message"]["row_id"] == canonical[0]["id"]
    assert user["payload"]["message"]["message_uid"] == canonical[0]["message_uid"]
    assert frames.index(user) < next(i for i, f in enumerate(frames) if f["type"] == "message.complete" and f["payload"]["text"] == "reply SECOND")


@pytest.mark.parametrize("field", ["session_id", "lease_id", "profile_home", "websocket_url"])
def test_cli_fences_handshake_identity_and_endpoint(runtime, monkeypatch, field):
    r = runtime
    real_reply = shared_runtime._attach_reply

    def mismatch(*args):
        reply = real_reply(*args)
        reply[field] = "ws://example.com/api/ws?token=SECRET-CANARY" if field == "websocket_url" else "wrong"
        return reply

    monkeypatch.setattr(shared_runtime, "_attach_reply", mismatch)
    result = r["cli"]()
    if field == "websocket_url" and r["request"].param != "stdio":
        # Serve constructs its own final URL AFTER lease validation; it never
        # trusts a URL supplied by the identity helper.
        assert result.returncode == 0, result.stderr
        assert r["requests"] == [{"session_id": r["sid"], "text": "SECOND", "queued": True}]
        assert r["second_done"].wait(10)
    else:
        assert result.returncode == 1
        assert "attachment failed" in result.stderr
        assert r["requests"] == []
    assert "SECRET-CANARY" not in result.stdout + result.stderr
    assert active_session_registry_snapshot(r["home"])[0]["lease_id"] == r["lease"].lease_id


@pytest.mark.parametrize("busy", [False, True])
def test_steer_uses_primitive_and_only_rejected_idle_input_becomes_next_turn(runtime, busy):
    r = runtime
    if busy:
        response = server.handle_request({"id": "first", "method": "prompt.submit", "params": {
            "session_id": r["sid"], "text": "FIRST"}})
        assert response["result"]["status"] == "streaming"
        assert r["first_started"].wait(10)
    r["requests"].clear()
    result = r["cli"](action="steer")
    assert result.returncode == 0, result.stderr
    if busy:
        assert r["corrections"] == ["steer"]
        assert r["requests"] == []
        assert r["turns"] == ["FIRST"]
        assert r["session"].get("queued_prompt") is None
    else:
        assert r["corrections"] == []
        assert r["requests"] == [{"session_id": r["sid"], "text": "SECOND", "queued": True}]
        assert r["second_done"].wait(10)
        assert r["turns"] == ["SECOND"]


@pytest.mark.parametrize("state", ["running", "idle", "building"])
def test_unsupported_steer_refuses_without_submitting_a_next_turn(runtime, state):
    r = runtime
    session = r["session"]
    if state == "running":
        response = server.handle_request({"id": "first", "method": "prompt.submit", "params": {
            "session_id": r["sid"], "text": "FIRST"}})
        assert response["result"]["status"] == "streaming"
        assert r["first_started"].wait(10)
    if state == "building":
        session["agent"] = None
        session["running"] = True
    else:
        del session["agent"].steer
    # Confirm the actual RPC's capability refusal, independently of CLI policy.
    response = server.handle_request({"id": "unsupported", "method": "session.steer", "params": {
        "session_id": r["sid"], "text": "SECOND"}})
    assert response["error"]["code"] == 4010
    assert response["error"]["message"] == "agent does not support steer"
    r["requests"].clear()
    result = r["cli"](action="steer")
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "4010" in result.stderr and "No prompt was queued" in result.stderr
    assert "hermes queue" in result.stderr
    assert not result.stdout
    assert r["requests"] == []
    assert session.get("queued_prompt") is None
    assert r["corrections"] == []
    assert not any(row["content"] == "SECOND" for row in r["db"].get_messages(r["key"]))
    if state == "building":
        session["agent"] = r["agent"]
        session["running"] = False
