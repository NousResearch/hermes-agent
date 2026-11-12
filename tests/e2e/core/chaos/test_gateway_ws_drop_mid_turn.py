"""Network drop mid-turn on the gateway's session WebSocket: the turn finishes, the client comes back.

Refs #145171 (EP-2). Contract: closing a client removes a viewer; it never stops accepted work.
A real ``hermes gateway run`` in a hermetic home, a loopback model that holds each turn open until
the test releases it (so "mid-turn" is deterministic), and a client on the gateway's ticketed
``/api/ws`` that loses its connection with no WebSocket close frame (a FIN, then an RST).
``sessions.orphan_end_grace_s`` is 1 s, so the owner's "last viewer vanished" close fires while the
turn is still held at the model; its persisted record is the proof the drop was seen mid-turn.

Two scenarios, one fresh session each:

* ``absent``: the client drops, the viewer close fires, the model is released with nobody attached;
  the turn completes exactly once, the deferred close then ends the row, and a new connection's
  ``session.resume`` shows the reply.
* ``returning``: the client drops and a new connection reattaches while the turn is still held;
  ``session.resume`` reports it running, and the completion arrives live on the new connection.

Either way, the returning client's next message in the same session runs and answers.

Boot failures and timeouts fail loudly with the gateway log attached.
"""

from __future__ import annotations

import json
import signal
import socket
import sqlite3
import struct
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

import pytest
from websockets.sync.client import connect

from hermes_agent.gateway.control_socket import resolve_client_socket_path
from hermes_agent.hermes_state_mutation_retirement import VIEWER_CLOSE_PREFIX
from tests.e2e.core.chaos._helpers import (
    hermetic_env, kill_tagged, new_tag, persisted_messages, python_exe, write_chaos_home)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text
from tests.waiting import wait_until

# Verified on Linux only. The gateway child is spawned and reaped here (fake HOME, no systemd bus).
pytestmark = [pytest.mark.platforms("linux"), pytest.mark.spawns_gateway_lookalike]

BOOT_S = 180.0
TURN_S = 90.0
PHASES = ("absent", "returning")


def _reply(phase: str) -> str:
    return f"netdrop-reply-{phase}"


def _last_user(record: dict[str, Any]) -> str:
    messages = record["body"].get("messages") or []
    content = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
    if isinstance(content, list):
        return " ".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return str(content or "")


class _Model(FakeLLMServer):
    """Loopback model: a prompt naming ``hold-<phase>`` blocks until that phase is released."""

    def __init__(self) -> None:
        super().__init__(self._main)
        self.release = {phase: threading.Event() for phase in PHASES}
        self.calls = {phase: 0 for phase in PHASES}
        self._seen = threading.Condition()

    def _main(self, record: dict[str, Any]) -> Text:
        text = _last_user(record)
        phase = next((p for p in PHASES if f"hold-{p}" in text), None)
        if phase is None:
            return Text("netdrop-other")
        with self._seen:
            self.calls[phase] += 1
            self._seen.notify_all()
        self.release[phase].wait(TURN_S * 2)
        return Text(_reply(phase))

    def wait_called(self, phase: str) -> None:
        with self._seen:
            if not self._seen.wait_for(lambda: self.calls[phase] >= 1, timeout=TURN_S):
                raise AssertionError(f"{phase}: the turn never reached the model")


class _Viewer:
    """One ticketed connection to the gateway's ``/api/ws``; a reader thread records every frame."""

    def __init__(self, home: Path, desc: dict) -> None:
        grant = _control(home, "session-ticket", {"profile_id": str(home.resolve()),
                                                  "instance_id": desc["instance_id"], "purpose": "interactive"})
        url = desc["api_origin"].replace("http:", "ws:") + "/api/ws"
        self._ws = connect(url, subprotocols=["hermes-gateway-v1", "hermes-gateway-ticket." + grant["ticket"]],
                           open_timeout=60, max_size=None, ping_interval=None, close_timeout=2)
        self.frames: list[dict] = []
        self._cond = threading.Condition()
        self._next = 0
        self._thread = threading.Thread(target=self._read, name="netdrop-ws", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        while True:
            try:
                raw = self._ws.recv()
            except Exception:  # the socket is gone (dropped or closed): stop reading
                return
            for line in str(raw).splitlines():
                if line.strip():
                    with self._cond:
                        self.frames.append(json.loads(line))
                        self._cond.notify_all()

    def wait_for(self, pred: Callable[[dict], bool], what: str, since: int = 0) -> dict:
        def first() -> dict | None:
            return next((f for f in self.frames[since:] if pred(f)), None)
        with self._cond:
            frame = self._cond.wait_for(first, timeout=TURN_S)
        if frame is None:
            raise AssertionError(f"timed out after {TURN_S:.0f}s waiting for {what}")
        return frame

    def call(self, method: str, **params: Any) -> dict:
        self._next += 1
        rid = self._next
        self._ws.send(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}))
        reply = self.wait_for(lambda f: f.get("id") == rid and "method" not in f, f"{method} reply")
        assert "result" in reply, f"{method}: {reply}"
        return reply["result"]

    def drop(self) -> None:
        """Drop the connection without a WebSocket close frame.

        ``SHUT_RD`` wakes websockets' reader thread blocked in ``recv``. It takes the EOF as an
        abnormal closure (1006, no close frame), half-closes the socket (a FIN) and closes it, which
        ``SO_LINGER 0`` turns into a trailing RST. Closing the fd under the blocked ``recv`` instead
        defers the real close until the gateway next sends a frame."""
        sock = self._ws.socket
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        sock.shutdown(socket.SHUT_RD)
        self._thread.join(timeout=10)
        sock.close()

    def close(self) -> None:
        try:
            self._ws.close()
        except OSError:
            pass


def _control(home: Path, verb: str, params: dict | None = None) -> dict:
    path = resolve_client_socket_path(home)
    if path is None:
        raise OSError("control socket absent")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(10)
        peer.connect(str(path))
        peer.sendall(json.dumps({"protocol": 1, "id": 1, "verb": verb, "params": params or {}}).encode() + b"\n")
        with peer.makefile("rb") as stream:
            reply = json.loads(stream.readline())
    assert reply.get("ok"), reply
    return reply["result"]


def _identify(home: Path) -> dict | None:
    try:
        desc = _control(home, "identify")
    except (OSError, ValueError, AssertionError):
        return None
    return desc if desc.get("state") == "ready" else None


def _poll(pred: Callable[[], Any], timeout: float, what: str, log: Path) -> Any:
    try:
        return wait_until(pred, timeout=timeout, interval=0.1, what=what)
    except AssertionError as exc:
        raise AssertionError(f"{exc}\n{log.read_text(errors='replace')[-4000:]}") from None


def _read(db: Path, sql: str, args: tuple) -> list[tuple]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10) as conn:
        return conn.execute(sql, args).fetchall()


def _close_armed(db: Path, sid: str) -> bool:
    return bool(_read(db, "SELECT 1 FROM state_meta WHERE key=?", (VIEWER_CLOSE_PREFIX + sid,)))


def _replies(db: Path, sid: str, phase: str) -> int:
    return sum(m["role"] == "assistant" and _reply(phase) in (m["content"] or "") for m in persisted_messages(db, sid))


def _end_reason(db: Path, sid: str) -> str | None:
    rows = _read(db, "SELECT end_reason FROM sessions WHERE id=? AND ended_at IS NOT NULL", (sid,))
    return rows[0][0] if rows else None


def _admissions(db: Path, sid: str) -> list[tuple[str, str | None]]:
    return _read(db, "SELECT status, outcome FROM session_admissions WHERE target_session_id=? ORDER BY seq", (sid,))


def _settled(db: Path, sid: str, phase: str) -> bool:
    """Every admission is terminal and a completed one has persisted its reply."""
    rows = _admissions(db, sid)
    return bool(rows) and all(st == "terminal" for st, _ in rows) and (
        _replies(db, sid, phase) > 0 or all(out != "completed" for _, out in rows))


def _closed_mid_turn(db: Path, sid: str, phase: str, log: Path) -> bool:
    """Wait for the owner to close the dropped viewer (its close record) or for the turn to end
    anyway; True when the close landed while the turn was still running."""
    _poll(lambda: _close_armed(db, sid) or any(st == "terminal" for st, _ in _admissions(db, sid)),
          TURN_S, f"{phase}: the owner closed the dropped viewer", log)
    return _close_armed(db, sid) and not any(st == "terminal" for st, _ in _admissions(db, sid))


def _start(viewer: _Viewer, model: _Model, phase: str) -> str:
    sid = viewer.call("session.create", request_id=f"netdrop-{phase}", source="cli")["session_id"]
    viewer.call("prompt.submit", session_id=sid, text=f"hold-{phase}", input_id=f"netdrop-{phase}-0")
    model.wait_called(phase)
    return sid


def _follow_up(viewer: _Viewer, db: Path, sid: str, phase: str, log: Path) -> dict:
    """The returning client's next message in the same session runs and answers."""
    since = len(viewer.frames)
    viewer.call("prompt.submit", session_id=sid, text=f"next-{phase}", input_id=f"netdrop-{phase}-1")
    done = viewer.wait_for(lambda f: f.get("method") == "event" and (f.get("params") or {}).get("session_id") == sid
                           and f["params"].get("type") in {"message.complete", "error"}, f"{phase}: follow-up", since)
    _poll(lambda: all(st == "terminal" for st, _ in _admissions(db, sid)), TURN_S, f"{phase}: follow-up settled", log)
    return {"done": done, "admissions": _admissions(db, sid)}


@pytest.fixture(scope="module")
def scenario(tmp_path_factory: pytest.TempPathFactory):
    root, tag, model = tmp_path_factory.mktemp("netdrop"), new_tag(), _Model()
    model.start()
    home, hermes_home = write_chaos_home(
        root, model.base_url, stale_timeout=600, request_timeout=600,
        extra=("auxiliary:\n  title_generation:\n    enabled: false\n"
               "sessions:\n  orphan_end_grace_s: 1\n"))
    db, log = hermes_home / "state.db", root / "gateway.log"
    env = hermetic_env(home, hermes_home, tag)
    for key in ("NOTIFY_SOCKET", "INVOCATION_ID", "XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
                "XDG_CACHE_HOME", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        env.pop(key, None)
    # Host rendezvous stays inside the tmp root: the developer's live gateway is never seen.
    env.update(HERMES_GATEWAY_LOCK_DIR=str(root / "locks"), HERMES_GATEWAY_MAX_STARTS="0")
    out, viewers, facts = open(log, "wb"), [], {}
    gateway = subprocess.Popen([python_exe(), "-m", "hermes_cli.main", "gateway", "run", "--quiet"],
                               cwd=str(home), env=env, stdin=subprocess.DEVNULL, stdout=out,
                               stderr=subprocess.STDOUT)

    def viewer() -> _Viewer:
        viewers.append(_Viewer(hermes_home, desc))
        return viewers[-1]
    try:
        desc = _poll(lambda: _identify(hermes_home) or (gateway.poll() is not None and {"exited": True}),
                     BOOT_S, "gateway identify ready", log)
        assert "exited" not in desc, f"gateway exited rc={gateway.returncode}\n{log.read_text(errors='replace')}"

        # absent: drop mid-turn, the viewer close fires, the turn finishes with nobody attached.
        first = viewer()
        sid = _start(first, model, "absent")
        first.drop()
        closed = _closed_mid_turn(db, sid, "absent", log)
        model.release["absent"].set()
        _poll(lambda: _settled(db, sid, "absent"), TURN_S, "absent: turn settled with no viewer", log)
        # The close that fired mid-turn waited for the turn, then ended the row.
        _poll(lambda: not _close_armed(db, sid) and _end_reason(db, sid), TURN_S, "absent: the deferred close ran", log)
        facts["absent"] = {"sid": sid, "closed_mid_turn": closed, "admissions": _admissions(db, sid), "replies": _replies(db, sid, "absent"),
                           "calls": model.calls["absent"], "end_reason": _end_reason(db, sid)}
        back = viewer()
        facts["absent"]["resume"] = back.call("session.resume", session_id=sid)
        facts["absent"]["follow_up"] = _follow_up(back, db, sid, "absent", log)

        # returning: drop mid-turn, reattach while the turn is still held, then release.
        first = viewer()
        sid = _start(first, model, "returning")
        first.drop()
        closed = _closed_mid_turn(db, sid, "returning", log)
        second = viewer()
        resumed = second.call("session.resume", session_id=sid)
        model.release["returning"].set()
        done = closed and second.wait_for(
            lambda f: f.get("method") == "event" and (f.get("params") or {}).get("session_id") == sid
            and f["params"].get("type") in {"message.complete", "error"}, "returning: completion on the new viewer")
        _poll(lambda: _settled(db, sid, "returning"), TURN_S, "returning: turn settled", log)
        facts["returning"] = {"sid": sid, "closed_mid_turn": closed, "resume": resumed, "done": done,
                              "admissions": _admissions(db, sid), "replies": _replies(db, sid, "returning"), "calls": model.calls["returning"]}
        facts["returning"]["follow_up"] = _follow_up(second, db, sid, "returning", log)
        yield facts
    finally:
        for event in model.release.values():
            event.set()
        for ws in viewers:
            ws.close()
        if gateway.poll() is None:
            gateway.send_signal(signal.SIGTERM)
            try:
                gateway.wait(timeout=60)
            except subprocess.TimeoutExpired:
                gateway.kill()
                gateway.wait(timeout=10)
        kill_tagged(tag)
        out.close()
        model.stop()


def test_dropped_viewer_never_stops_the_accepted_turn(scenario) -> None:
    s = scenario["absent"]
    assert s["admissions"] == [("terminal", "completed")], f"the dropped viewer's turn ended {s['admissions']}"
    assert s["closed_mid_turn"], "the owner never closed the dropped viewer while its turn was running"
    assert s["end_reason"] == "ws_orphan_reap", f"the deferred close ended the row with {s['end_reason']!r}"
    assert s["calls"] == 1, f"the turn reached the model {s['calls']} times; a drop must not rerun it"
    assert s["replies"] == 1, f"{s['replies']} assistant replies persisted for {s['sid']}, wanted exactly 1"


@pytest.mark.parametrize("phase", PHASES)
def test_the_session_takes_the_next_message_after_the_drop(scenario, phase) -> None:
    follow_up = scenario[phase]["follow_up"]
    assert follow_up["done"]["params"]["type"] == "message.complete", follow_up["done"]
    assert "netdrop-other" in json.dumps(follow_up["done"]), follow_up["done"]
    assert follow_up["admissions"] == [("terminal", "completed")] * 2, follow_up["admissions"]


def test_reattach_after_drop_shows_the_finished_turn(scenario) -> None:
    s = scenario["absent"]
    resume = s["resume"]
    assert resume["running"] is False, resume
    texts = [str(m.get("content") or "") for m in resume["messages"] if m.get("role") == "assistant"]
    assert any(_reply("absent") in t for t in texts), f"session.resume history lacks the reply: {texts}"


def test_reattach_mid_turn_receives_the_completion_live(scenario) -> None:
    s = scenario["returning"]
    assert s["admissions"] == [("terminal", "completed")], f"the dropped viewer's turn ended {s['admissions']}"
    assert s["closed_mid_turn"], "the owner never closed the dropped viewer while its turn was running"
    assert s["resume"]["running"] is True, f"session.resume after the drop did not see the held turn: {s['resume']}"
    assert s["done"]["params"]["type"] == "message.complete", s["done"]
    assert _reply("returning") in json.dumps(s["done"]), s["done"]
    assert s["calls"] == 1, f"the turn reached the model {s['calls']} times; a reattach must not rerun it"
    assert s["replies"] == 1, f"{s['replies']} assistant replies persisted for {s['sid']}, wanted exactly 1"
