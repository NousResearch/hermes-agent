"""A real owner (SessionAuthority, AuthorityConnection, FanoutTransport overflow, the drain's publish sequence)
for one session ``s`` and one viewer whose socket stops reading while paused (Ctrl-Z). #140292 raised the frame
cap to 16384 and kept the overflow path; ``frame_cap`` pins the old 256, as its own overflow probe does."""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

DELTAS = [f"{i:07d}|" for i in range(600)]
FINAL = "".join(DELTAS)


@pytest.fixture(autouse=True)
def frame_cap(monkeypatch):
    from tui_gateway.transport import FanoutTransport
    monkeypatch.setattr(FanoutTransport, "_MAX_PENDING_FRAMES", 256)


class Owner:
    """``pause``: the next turn pauses the viewer throughout (``"ended"``) or until delta 400 (``"live"``)."""

    def __init__(self, db_path, pause="ended"):
        from gateway.session_authority import LiveSession, SessionAuthority
        from gateway.session_contract import SessionRef
        from gateway.session_controls import AuthorityConnection
        from hermes_state import SessionDB
        from hermes_state_runtime import begin_runtime_epoch
        self.loop, self.events, self.running = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()
        self.pause, self.gaps, self.turns, self.ref = pause, [], [], SessionRef("owned", "s")
        self.running.set()
        self.db = SessionDB(db_path=db_path)
        self.db.create_session("s", source="test")
        self.epoch = begin_runtime_epoch(self.db, instance_id="owner")
        self.authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id="owned",
                                          instance_id="owner", db=self.db, epoch=self.epoch)
        self.live = self.authority.sessions["s"] = LiveSession(None, "route")
        retire = self.authority._retire_overflowed
        self.authority._retire_overflowed = lambda sid, peer: (self.gaps.append(sid), retire(sid, peer))
        self.conn = AuthorityConnection(self.authority, self, {"user_id": "viewer"})

    def write(self, frame):  # the viewer's socket
        self.running.wait(30)
        self.loop.call_soon_threadsafe(self.events.put_nowait, frame)
        return True

    def close(self):
        return None

    async def rpc(self, method, _timeout=None, **params):
        from hermes_state_runtime import admit_session_input, claim_session_input
        if method == "prompt.submit":
            admit_session_input(self.db, epoch=self.epoch, principal_id=self.conn.actor.subject, session_id="s",
                                request_id=params.get("input_id") or params.get("submission_id"),
                                payload={"text": params["text"]})
            row = claim_session_input(self.db, epoch=self.epoch, session_id="s")
            self.turns.append(threading.Thread(target=self._turn, args=(row, self.pause), daemon=True))
            self.pause = None
            self.turns[-1].start()
            return {"admission_id": row["admission_id"], "status": "queued"}
        verb = {"session.resume": "resume", "prompt.receipt": "receipt", "session.detach": "detach"}.get(method)
        return await getattr(self.conn, verb)(self.ref, {} if verb == "resume" else params) if verb else {}

    def _turn(self, row, pause):
        from gateway.session_settlement_recovery import commit_and_publish
        self.authority._stamp_claim(self.ref, self.live, row)
        if pause:
            self.running.clear()  # Ctrl-Z mid-turn
        for index, delta in enumerate(DELTAS):
            if pause == "live" and index == 400:
                self.running.set()  # fg mid-turn
                deadline = time.monotonic() + 5
                while not self.live.subscribers and time.monotonic() < deadline:
                    time.sleep(0.01)  # health: allow HX013 -- the owner thread waits for the re-attach
            self.authority.publish_execution("s", row["generation"], "message.delta", {"text": delta})
        with self.live.event_stream.lock:
            self.authority.executing.pop("s", None)
        commit_and_publish(self.authority, self.live, self.ref, row, FINAL, "completed", None, {})
        self.running.set()

    def finish(self):
        self.running.set()
        for turn in self.turns:
            turn.join(10)
        self.db.close()
