"""Plugin wakes restore closed local conversations without reopening a viewer (#150831).

Real gateway processes, websocket clients, loaded plugins and SQLite admission ledgers; only the
model is a loopback fixture. A restart must not change the destination, history or profile boundary.
"""
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import asyncio
import json
from pathlib import Path
import signal
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon
from tests.gateway.test_cli_plugin_command_in_gateway import _Wire
from tests.gateway.test_plugin_keyless_inject import _probe_plugin, admissions, keyless_gateway, landed_once, noted
from tests.gateway.test_pre_command_in_gateway import _ProfileWire
from tests.waiting import wait_until


def _stored(home, session_id):
    with sqlite3.connect(home / "state.db") as db:
        return db.execute("SELECT session_key, end_reason FROM sessions WHERE id=?", (session_id,)).fetchone()


@contextmanager
def _environment(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, work, log = (tmp_path / name for name in ("state", "user", "work", "inject.jsonl"))
    for directory in (home, user, work):
        directory.mkdir(mode=0o700)
    _probe_plugin(home, log)
    peer = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{peer.server_port}/v1"
    (home / "config.yaml").write_text(json.dumps({
        "gateway": {"multiplex_profiles": False},
        "model": {"provider": "custom", "default": "plugin-model", "base_url": url},
        "auxiliary": {"title_generation": {"enabled": False}},
        "plugins": {"enabled": ["probe-inj"], "entries": {"probe-inj": {"allow_gateway_injection": True}}},
    }))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=url, PYTHONUNBUFFERED="1")
    try:
        yield root, home, work, log, env, peer
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


def _close_target(wire, home, work, peer, source, text="PROMISE_CONTEXT"):
    target = wire.call("session.create", request_id="target", source=source, toolsets=[], cwd=str(work))
    sid = target["session_id"]
    wire.call("prompt.submit", session_id=sid, input_id="warm", text=text)
    assert landed_once(home, peer, text) == sid
    key = _stored(home, sid)[0]
    closed = wire.call("session.detach", session_id=sid, subscription_id=target["subscription_id"],
                       end_reason="tui_shutdown")
    assert closed["detached"] is True
    assert _stored(home, sid)[1] == "tui_shutdown"
    return sid, key


def _follow_up(wire, home, work, log, peer, sid, key):
    controller = wire.call("session.create", request_id="controller", source="gui", toolsets=[],
                           cwd=str(work))["session_id"]
    # No target-session read/resume RPC may restore the target before injection.
    wire.call("slash.exec", session_id=controller, command=f"/inj-key {key} FOLLOW_UP")
    assert noted(log, "keyed", "FOLLOW_UP") is True
    assert landed_once(home, peer, "FOLLOW_UP") == sid
    request = next(r for r in peer.requests if any(m.get("content") == "FOLLOW_UP" for m in r["messages"]))
    assert "RECOVERY_ACK_PROMISE_CONTEXT" in json.dumps(request["messages"])
    resumed = wire.call("session.resume", session_id=sid)
    assert "RECOVERY_ACK_FOLLOW_UP" in json.dumps(resumed["messages"])

    # A restored conversation still uses the FIFO: a second wake cannot interrupt its current turn.
    wire.call("prompt.submit", session_id=sid, input_id="busy", text="BLOCK_STARTED")
    assert peer.blocked.wait(20)
    wire.call("slash.exec", session_id=controller, command=f"/inj-key {key} QUEUED_FOLLOW_UP")
    assert noted(log, "keyed", "QUEUED_FOLLOW_UP") is True
    wait_until(lambda: admissions(home, "QUEUED_FOLLOW_UP") == [(sid, "queued")], timeout=20,
               what="the follow-up queued behind the running turn")
    peer.release.set()
    assert landed_once(home, peer, "QUEUED_FOLLOW_UP") == sid
    resumed = wire.call("session.resume", session_id=sid)
    history = json.dumps(resumed["messages"])
    assert "RECOVERY_ACK_FOLLOW_UP" in history and "RECOVERY_ACK_QUEUED_FOLLOW_UP" in history


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("source", ["cli", "tui", "gui"])
@pytest.mark.parametrize("restart", [False, True])
def test_plugin_wakes_a_closed_conversation_without_a_viewer(tmp_path, source, restart):
    with _environment(tmp_path) as (root, home, work, log, env, peer):
        with daemon(root, home, env, barrier=False) as (proc, desc):
            wire = _Wire(home, desc)
            try:
                sid, key = _close_target(wire, home, work, peer, source)
                if not restart:
                    _follow_up(wire, home, work, log, peer, sid, key)
            finally:
                wire.close()
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
        if restart:
            with daemon(root, home, env, barrier=False) as (proc, desc):
                wire = _Wire(home, desc)
                try:
                    _follow_up(wire, home, work, log, peer, sid, key)
                finally:
                    wire.close()
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=20)


@pytest.mark.platforms("posix")
def test_retired_injection_preserves_profile_isolation_across_a_b_a(tmp_path):
    with keyless_gateway(tmp_path, beta=True) as (desc, home, work, log, peer):
        beta_home, beta_log = (home / "profiles" / "beta").resolve(), tmp_path / "beta.jsonl"
        alpha, beta = _Wire(home, desc), _ProfileWire(home, desc, beta_home)
        try:
            a_sid, a_key = _close_target(alpha, home, work, peer, "gui", "ALPHA_CONTEXT")
            b_sid, b_key = _close_target(beta, beta_home, work, peer, "gui", "BETA_CONTEXT")
            a_control = alpha.call("session.create", request_id="control", source="gui", toolsets=[],
                                   cwd=str(work))["session_id"]
            b_control = beta.call("session.create", request_id="control", source="gui", toolsets=[],
                                  cwd=str(work))["session_id"]
            for wire, control, target, output, text in (
                (alpha, a_control, b_key, log, "ALPHA_TO_BETA"),
                (beta, b_control, a_key, beta_log, "BETA_TO_ALPHA"),
            ):
                wire.call("slash.exec", session_id=control, command=f"/inj-key {target} {text}")
                assert noted(output, "keyed", text) is False
                assert admissions(home, text) == [] and admissions(beta_home, text) == []
            for wire, control, own_home, own_log, sid, key, text, foreign in (
                (alpha, a_control, home, log, a_sid, a_key, "ALPHA_WAKE", "BETA_CONTEXT"),
                (beta, b_control, beta_home, beta_log, b_sid, b_key, "BETA_WAKE", "ALPHA_CONTEXT"),
                (alpha, a_control, home, log, a_sid, a_key, "ALPHA_AGAIN", "BETA_CONTEXT"),
            ):
                wire.call("slash.exec", session_id=control, command=f"/inj-key {key} {text}")
                assert noted(own_log, "keyed", text) is True
                assert landed_once(own_home, peer, text) == sid
                request = next(r for r in peer.requests if any(m.get("content") == text for m in r["messages"]))
                assert foreign not in json.dumps(request["messages"])
        finally:
            alpha.close()
            beta.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "malformed", "profile", "route", "suspended"])
async def test_invalid_stored_identity_is_refused_without_restoring(tmp_path, monkeypatch, damage):
    from hermes_state_local import POLICY_PREFIX, local_receipt
    from tests.gateway.test_local_closed_retirement import _close, _create, _owner
    from tests.gateway.test_plugin_origin_injection_authority import _context, _plugin_rows, _publish
    from tests.waiting import async_wait_until

    runner, authority, db = await _owner(tmp_path, monkeypatch)
    try:
        ref = _create(authority)
        key = authority.sessions[ref.session_id].route
        await _close(authority, ref)
        await async_wait_until(lambda: ref.session_id not in authority.sessions, timeout=10,
                               what="the closed conversation leaving memory")
        receipt = local_receipt(db, ref.session_id)
        if damage == "profile":
            receipt["profile_id"] = str(tmp_path / "other-profile")
        elif damage == "route":
            receipt["route"] += ":wrong"
        meta_key = POLICY_PREFIX + ref.session_id
        if damage == "suspended":
            assert runner.session_store.suspend_session(key)
        elif damage == "missing":
            db._execute_write(lambda conn: conn.execute("DELETE FROM state_meta WHERE key=?", (meta_key,)))
        else:
            value = "{}" if damage == "malformed" else json.dumps(receipt)
            db._execute_write(lambda conn: conn.execute("UPDATE state_meta SET value=? WHERE key=?", (value, meta_key)))
        ctx, manager = _context(tmp_path)
        _publish(runner, manager)
        assert await asyncio.to_thread(ctx.inject_message, "REFUSED", session_key=key) is False
        assert await runner._dispatch_plugin_message_injection(
            session_key=key, content="REFUSED", plugin_id="desk", plugin_home=tmp_path) is False
        assert ref.session_id not in authority.sessions
        assert _plugin_rows(authority, ref.session_id, "REFUSED") == []
    finally:
        db.close()
