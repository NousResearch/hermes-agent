"""Edges of the idle-released Desktop lease: every refusal answers 4090, changes nothing, and never clears
another turn's ``running`` or leaks a transcript claim.

Real SQLite SessionDB, real active-session registry, real reaper and real JSON-RPC handlers. Interleavings
are pinned at production seams (the refusal ``error`` frame, the supervisor's late-handler registration).
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from hermes_cli.active_sessions import try_acquire_active_session
from hermes_state import SessionDB
from tui_gateway import server


class _LiveSocket:
    _closed = False

    def write(self, frame):
        return True


class _Agent:
    def __init__(self, sid):
        self.session_id, self.model, self.provider, self.base_url, self.api_key = sid, "old/model", "prov", "", ""
        self.reasoning_config = None
        self.switched = []

    def clear_interrupt(self):
        return None

    def switch_model(self, **kw):
        self.switched.append(kw["new_model"])
        self.model = kw["new_model"]


def _make(monkeypatch, sid):
    from hermes_constants import get_hermes_home
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    db = SessionDB(db_path=get_hermes_home() / "state.db")
    monkeypatch.setattr(server, "_db", db)
    db.create_session(sid, source="desktop")
    for n in range(3):
        db.append_message(sid, "user", f"question {n}")
        db.append_message(sid, "assistant", f"answer {n}")
    ready = threading.Event()
    ready.set()
    long_ago = time.time() - 3600
    record = {
        "session_key": sid, "source": "desktop", "history_version": 0, "history_lock": threading.Lock(),
        "history": db.get_messages_as_conversation(sid, include_row_ids=True), "agent_ready": ready,
        "running": False, "created_at": long_ago, "last_active": long_ago, "transport": _LiveSocket(),
        "agent": _Agent(sid), "attached_images": [], "cols": 80}
    server._sessions[sid] = record
    assert server._ensure_active_session_slot(sid, record) is None
    return db, record


@pytest.fixture
def chat(monkeypatch):
    """Idle Desktop record still holding its lease (nobody else around)."""
    sid = "idle-edge-chat"
    db, record = _make(monkeypatch, sid)
    leases = []

    def compete():
        lease, _ = try_acquire_active_session(session_id=sid, surface="telegram", config={})
        if lease is not None:
            leases.append(lease)
        return lease
    try:
        yield SimpleNamespace(sid=sid, record=record, db=db, compete=compete)
    finally:
        for lease in leases:
            lease.release()
        server._release_active_session_slot(record)
        server._sessions.pop(sid, None)
        db.close()


@pytest.fixture
def taken(chat):
    """The real reaper released the idle lease; Telegram claimed the chat and extended it."""
    server._reap_idle_sessions()
    assert chat.record.get("active_session_lease") is None
    lease = chat.compete()
    assert lease is not None
    chat.db.append_message(chat.sid, "user", "telegram message")
    chat.db.append_message(chat.sid, "assistant", "telegram answer")
    chat.lease = lease
    return chat


def _transcript(chat):
    return [m["content"] for m in chat.db.get_messages_as_conversation(chat.sid)]


# ── SessionNotOwned on the compute-host paths answers 4090 (Greptile 4569456266)
def _compute_host(monkeypatch):
    sent = []
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: True)
    monkeypatch.setattr(server, "_send_compute_host_control", lambda sid, **kw: sent.append(kw) or {"type": "control.ack"})
    return sent


def test_g1_compute_host_session_compress_answers_4090(taken, monkeypatch):
    sent = _compute_host(monkeypatch)
    resp = server.handle_request({"id": "r", "method": "session.compress", "params": {"session_id": taken.sid}})
    err = resp.get("error") or {}
    assert {"code": err.get("code"), "reason": (err.get("data") or {}).get("reason"), "sent": sent} == {
        "code": 4090, "reason": "SESSION_NOT_OWNED", "sent": []}, err


def test_g2_compute_host_command_dispatch_compress_answers_4090(taken, monkeypatch):
    sent = _compute_host(monkeypatch)
    resp = server.handle_request({"id": "r", "method": "command.dispatch",
                                  "params": {"session_id": taken.sid, "name": "compress", "arg": ""}})
    err = resp.get("error") or {}
    assert {"code": err.get("code"), "sent": sent} == {"code": 4090, "sent": []}, resp


def test_g3_compute_host_slash_exec_model_answers_4090(taken, monkeypatch):
    sent = _compute_host(monkeypatch)

    class _Worker:
        def run(self, cmd):
            return "model switched (worker)"

        def pop_seed(self):
            return ""
    monkeypatch.setattr(server, "_slash_worker_for", lambda rid, sid, session: (_Worker(), None))
    resp = server.handle_request({"id": "r", "method": "slash.exec",
                                  "params": {"session_id": taken.sid, "command": "/model other/model"}})
    err = resp.get("error") or {}
    assert {"code": err.get("code"), "result": resp.get("result"), "sent": sent} == {
        "code": 4090, "result": None, "sent": []}, resp


# ── A refused notification turn releases only its own claim on ``running``
def _user_turn_in_refusal_gap(chat, monkeypatch, competitor):
    user_turn = []

    def emit(event, sid, payload=None, **_kw):
        if event == "error" and not user_turn:  # admission already set running=False here
            competitor.release()
            err, _ = server._lock_in_submit_turn("user", chat.sid, chat.record, "hello", {}, False, None, None, None)
            user_turn.append(err)
    monkeypatch.setattr(server, "_emit", emit)
    return user_turn


def test_n1_refused_heartbeat_tick_never_clears_a_user_turn(taken, monkeypatch):
    from hermes_cli.heartbeat import HeartbeatManager, save_heartbeat
    mgr = HeartbeatManager(taken.sid)
    mgr.set("check the build", interval_seconds=600)
    mgr.state.created_at = time.time() - 3600
    save_heartbeat(taken.sid, mgr.state)
    user_turn = _user_turn_in_refusal_gap(taken, monkeypatch, taken.lease)
    try:
        server._maybe_fire_tui_heartbeat_tick(taken.sid, taken.record)
        assert {"user_turn_admitted": user_turn == [None], "running": taken.record["running"]} == {
            "user_turn_admitted": True, "running": True}
    finally:
        taken.record["running"] = False
        HeartbeatManager(taken.sid).clear()


def test_n2_refused_notification_submit_never_clears_a_user_turn(taken, monkeypatch):
    user_turn = _user_turn_in_refusal_gap(taken, monkeypatch, taken.lease)
    assert server._notif_claim_turn(taken.record)
    try:
        accepted = server._notif_submit("__notif__1", taken.sid, taken.record, "process finished", "probe")
        assert {"accepted": accepted, "user_turn_admitted": user_turn == [None],
                "running": taken.record["running"]} == {"accepted": False, "user_turn_admitted": True, "running": True}
    finally:
        taken.record["running"] = False


# ── The same race against a holder that never released (no idle reap involved)
@pytest.fixture
def foreign(monkeypatch):
    sid = "idle-edge-foreign"
    from hermes_constants import get_hermes_home
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    db = SessionDB(db_path=get_hermes_home() / "state.db")
    monkeypatch.setattr(server, "_db", db)
    db.create_session(sid, source="desktop")
    ready = threading.Event()
    ready.set()
    record = {
        "session_key": sid, "source": "desktop", "history_version": 0, "history_lock": threading.Lock(),
        "history": [], "agent_ready": ready, "running": False, "created_at": time.time(), "last_active": time.time(),
        "transport": _LiveSocket(), "agent": _Agent(sid), "attached_images": [], "cols": 80}
    server._sessions[sid] = record
    lease, _ = try_acquire_active_session(session_id=sid, surface="telegram", config={})
    assert lease is not None
    try:
        yield SimpleNamespace(sid=sid, record=record, db=db, lease=lease)
    finally:
        lease.release()
        server._release_active_session_slot(record)
        server._sessions.pop(sid, None)
        db.close()


def _user_turn_after_refusal(chat, monkeypatch):
    user_turn = []

    def emit(event, sid, payload=None, **_kw):
        if event == "error" and not user_turn:
            chat.lease.release()
            # The user's own turn claims running in the gap (any admitted turn source does the same).
            with chat.record["history_lock"]:
                chat.record["running"] = True
                chat.record.pop("_notif_turn_claim", None)
            user_turn.append(True)
    monkeypatch.setattr(server, "_emit", emit)
    return user_turn


def test_m1_heartbeat_refusal_keeps_running_of_a_turn_started_in_the_gap(foreign, monkeypatch):
    from hermes_cli.heartbeat import HeartbeatManager, save_heartbeat
    mgr = HeartbeatManager(foreign.sid)
    mgr.set("check the build", interval_seconds=600)
    mgr.state.created_at = time.time() - 3600
    save_heartbeat(foreign.sid, mgr.state)
    user_turn = _user_turn_after_refusal(foreign, monkeypatch)
    try:
        server._maybe_fire_tui_heartbeat_tick(foreign.sid, foreign.record)
        assert {"gap_turn": user_turn, "running": foreign.record["running"]} == {"gap_turn": [True], "running": True}
    finally:
        foreign.record["running"] = False
        HeartbeatManager(foreign.sid).clear()


def test_m2_notif_submit_refusal_keeps_running_of_a_turn_started_in_the_gap(foreign, monkeypatch):
    user_turn = _user_turn_after_refusal(foreign, monkeypatch)
    assert server._notif_claim_turn(foreign.record)
    try:
        server._notif_submit("__notif__1", foreign.sid, foreign.record, "process finished", "probe")
        assert {"gap_turn": user_turn, "running": foreign.record["running"]} == {"gap_turn": [True], "running": True}
    finally:
        foreign.record["running"] = False


# ── Refused gated writers change nothing: no side effect runs before the ownership gate
def test_r1_rollback_restore_on_a_taken_chat_restores_nothing_and_answers_4090(taken, monkeypatch):
    restored = []

    class _Mgr:
        enabled = True

        def unsupported_backend_reason(self, task_id):
            return None

        def list_checkpoints(self, cwd):
            return []

        def restore(self, cwd, commit, file_path=None, safe=True):
            restored.append(commit)
            return {"success": True}
    taken.record["agent"]._checkpoint_mgr = _Mgr()
    resp = server.handle_request({"id": "r", "method": "rollback.restore",
                                  "params": {"session_id": taken.sid, "hash": "abcdef1234567"}})
    err = resp.get("error") or {}
    assert {"code": err.get("code"), "restored": restored, "transcript_tail": _transcript(taken)[-2:]} == {
        "code": 4090, "restored": [], "transcript_tail": ["telegram message", "telegram answer"]}, resp


@pytest.mark.parametrize("command", ["/model other/model", "/compress"])
def test_r3_in_process_slash_writer_on_a_taken_chat_never_runs_the_worker(taken, monkeypatch, command):
    ran = []

    class _Worker:
        def run(self, cmd):  # the worker's CLI writes the session row before the live mirror's gate
            ran.append(cmd)
            return "done"

        def pop_seed(self):
            return ""
    monkeypatch.setattr(server, "_slash_worker_for", lambda rid, sid, session: (_Worker(), None))
    resp = server.handle_request({"id": "r", "method": "slash.exec",
                                  "params": {"session_id": taken.sid, "command": command}})
    err = resp.get("error") or {}
    assert {"code": err.get("code"), "ran": ran} == {"code": 4090, "ran": []}, resp


def test_r2_model_reset_on_a_taken_chat_keeps_the_pin_it_refuses_to_drop(taken, monkeypatch):
    from hermes_cli.model_switch import ModelSwitchResult
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: ModelSwitchResult(
        success=True, new_model="cfg/model", target_provider="prov"))
    monkeypatch.setattr(server, "_expensive_model_confirm", lambda *a, **k: None)
    monkeypatch.setattr(server, "_merge_preflight_warning", lambda *a, **k: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("cfg/model", "prov"))
    pin = {"model": "old/model", "provider": "prov", "base_url": "", "api_key": "", "api_mode": ""}
    taken.record["model_override"] = dict(pin)
    resp = server.handle_request({"id": "r", "method": "config.set", "params": {
        "session_id": taken.sid, "key": "model", "value": "reset"}})
    err = resp.get("error") or {}
    assert {"code": err.get("code"), "pin": taken.record.get("model_override")} == {"code": 4090, "pin": pin}, resp


# ── A /compress claim is released however late, or never, the host's ack arrives ──────────────────────────

def _compress_supervisor(chat, monkeypatch, tmp_path):
    from tui_gateway.host_supervisor import HostSupervisor
    sup = HostSupervisor(registry_path=tmp_path / "idle-edge-host.json", autostart=False)
    frames = []
    monkeypatch.setattr(sup, "start", lambda: None)
    monkeypatch.setattr(sup, "_send_frame", frames.append)
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *a, **k: sup)
    monkeypatch.setattr(server, "_compute_host_compress_wait_seconds", lambda *a, **k: 0.01)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: True)
    return sup, frames


def test_c1_compress_claim_released_when_the_ack_lands_as_the_waiter_gives_up(chat, monkeypatch, tmp_path):
    import queue as stdlib_queue
    from tui_gateway import host_supervisor
    sup, frames = _compress_supervisor(chat, monkeypatch, tmp_path)

    class _ReaderWinsAfterTimeout(stdlib_queue.Queue):
        def get(self, block=True, timeout=None):
            if not block:
                return super().get(block=False)
            # The reader thread's control.ack lands just after the wait timed out, before the waiter retires it.
            request_id = frames[-1]["request_id"]
            sup._deliver_control_frame(request_id, {"type": "control.ack", "request_id": request_id, "output": "ok"})
            raise stdlib_queue.Empty
    monkeypatch.setattr(host_supervisor, "queue", SimpleNamespace(
        Queue=_ReaderWinsAfterTimeout, Empty=stdlib_queue.Empty, Full=stdlib_queue.Full))

    status = server._compute_host_slash(chat.sid, chat.record, "compress", "/compress")[0]
    released = server._release_idle_session_leases(time.time() + 1200)
    assert {"status": status, "frames": len(frames), "released_20min_later": released} == {
        "status": "ok", "frames": 1, "released_20min_later": 1}


def test_c2_compress_claim_released_when_its_late_handler_is_evicted(chat, monkeypatch, tmp_path):
    from tui_gateway import host_supervisor
    sup, frames = _compress_supervisor(chat, monkeypatch, tmp_path)
    assert server._compute_host_slash(chat.sid, chat.record, "compress", "/compress")[0] == "pending"
    for n in range(host_supervisor._LATE_CONTROL_MAX):  # later compressions on other chats fill the cap
        sup._register_late_control_handler(f"other-{n}", lambda _frame: None)
    assert frames[0]["request_id"] not in sup._late_control_handlers
    assert server._release_idle_session_leases(time.time() + 1200) == 1
