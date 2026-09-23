"""C14 — multi-client session ownership, routing and switching over the REAL backend.

Class: a message persisted to a stale session after a switch, session RPCs collapsing onto the
wrong connection, events rendered twice or leaking into another chat, a zombie lease locking a
chat ("already has a live owner"), a ws drop orphaning the in-flight turn, client absence killing
the turn, an active session hard-deleted.

Harness: one real ``hermes serve`` (the Desktop's backend argv) per module, 3 Desktop-shaped
WebSocket clients, and a seeded fuzzer that interleaves create / switch(resume) / prompt (fast and
slow streams) / interrupt mid-stream / delete (inactive and active) / close / abrupt ws drop +
reconnect + resume / owner death + takeover / client absence. The fake provider echoes a per-prompt
canary so every assistant reply is attributable. A reference model tracks what each client has
selected and sent; invariants are checked after every step and at the end:

1. every accepted prompt is persisted to the session its sender had selected, exactly once, in order;
2. each event reaches a subscribed connection at most once (per-session ``seq`` strictly increasing
   per connection), never reaches a connection that did not attach to that session, and every
   assistant canary appears only in its own session's events;
3. after the owning connection dies, the session is released in bounded time and another client can
   resume and prompt it (no zombie lease), and no session key ever has two live runtimes;
4. deleting a live session is refused and destroys nothing;
5. a client that reconnects after a mid-turn drop gets the turn's completion (live event or history),
   and neither a drop nor an absence interrupts the turn.
"""

from __future__ import annotations

import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pytest

from tests.e2e.core.terminal._gateway_client import Backend, RpcError, WSClient, etype, poll_until
from tests.fakes.fake_llm_provider import Text

# Not marked ``integration`` (that marker means external services and is deselected by addopts): the backend is
# loopback-only. ``tests/e2e`` is outside default discovery; run with ``scripts/run_tests.sh --include-integration``.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group lifecycle for the spawned backend")

CANARY_RE = re.compile(r"cnry-s\d+-\d{3}")
FILLER = " ".join(["stream"] * 45)
ABSENCE_S = 3.0  # far below the 30 s WS send deadline and the reaper's activity-stale threshold
GRACE_S = 2  # HERMES_TUI_WS_ORPHAN_REAP_GRACE_S: short so owner-death paths finish inside the test
STEP_TIMEOUT = 90.0


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return str(content or "")


def _responder(record: dict):
    """Echo the newest canary: ``ACK <canary> ... DONE``; prompts containing ``slow`` stream for ~3 s, so a
    reply that ends in ``DONE`` proves the turn was not cut short."""
    messages = record["body"].get("messages") or []
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
    text = _text_of(last_user)
    canaries = CANARY_RE.findall(text)
    canary = canaries[-1] if canaries else "none"
    if "slow" in text:
        return Text(f"ACK {canary} {FILLER} DONE", chunk_chars=6, delay_per_chunk=0.07)
    return Text(f"ACK {canary} DONE")


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    root = tmp_path_factory.mktemp("c14")
    be = Backend(root, _responder, extra_config="display:\n  busy_input_mode: queue\n",
                 env={"HERMES_TUI_WS_ORPHAN_REAP_GRACE_S": str(GRACE_S)})
    be.start()
    try:
        yield be
    finally:
        be.stop()


@dataclass
class Turn:
    canary: str
    key: str
    sid: str
    client: str
    slow: bool
    attached: set[int]  # id() of connections attached to ``sid`` when the prompt was accepted
    interrupted: bool = False
    frame_idx: int = 0  # the submitting connection's frame count when the accepted prompt.submit went out


@dataclass
class Sess:
    sid: str | None
    prompts: list[str] = field(default_factory=list)
    deleted: bool = False
    closed: bool = False


@dataclass
class Slot:
    name: str
    conn: WSClient | None = None
    key: str | None = None
    sid: str | None = None


class Model:
    def __init__(self, backend: Backend, seed: int) -> None:
        self.be = backend
        self.seed = seed
        self.rng = random.Random(seed)
        self.slots = {n: Slot(n) for n in ("A", "B", "C")}
        self.sessions: dict[str, Sess] = {}
        self.sid_key: dict[str, str] = {}
        self.turns: dict[str, Turn] = {}
        self.conns: list[WSClient] = []
        # id(conn) -> sid -> frame index at the moment the attaching RPC was sent
        self.attach_idx: dict[int, dict[str, int]] = defaultdict(dict)
        self.n = 0
        self.log: list[str] = []

    # -- plumbing ----------------------------------------------------------------------------
    def connect(self, slot: Slot) -> WSClient:
        conn = self.be.connect(f"{slot.name}{len(self.conns)}-s{self.seed}")
        self.conns.append(conn)
        slot.conn = conn
        return conn

    def live_conns(self) -> list[WSClient]:
        return [c for c in self.conns if not c.dropped and not c.closed]

    def any_conn(self) -> WSClient:
        for slot in self.slots.values():
            if slot.conn is not None and not slot.conn.dropped:
                return slot.conn
        return self.connect(self.slots["A"])

    def attached_live(self, sid: str) -> list[WSClient]:
        return [c for c in self.live_conns() if sid in self.attach_idx[id(c)]]

    def mark_attach(self, conn: WSClient, sid: str, idx: int) -> None:
        self.attach_idx[id(conn)].setdefault(sid, idx)

    def active_rows(self) -> list[dict]:
        return self.any_conn().call("session.active_list").get("sessions") or []

    def wait_idle(self, key: str) -> None:
        def idle() -> bool:
            rows = [r for r in self.active_rows() if r.get("session_key") == key]
            return all(r.get("status") == "idle" for r in rows)
        poll_until(idle, timeout=STEP_TIMEOUT, interval=0.1, what=f"session {key} idle")

    def resume(self, slot: Slot, key: str) -> str:
        conn = slot.conn
        assert conn is not None
        old = self.sessions[key].sid
        holders = [c for c in self.attached_live(old)] if old else []

        def attempt():
            idx = len(conn.snapshot())
            frame = conn.request("session.resume", {"session_id": key, "cols": 100})
            if "error" in frame:
                if frame["error"].get("code") == 4009:  # client-gone interrupt settling: documented retry
                    return None
                raise RpcError("session.resume", frame["error"])
            return idx, frame["result"]

        idx, result = poll_until(attempt, timeout=STEP_TIMEOUT, interval=0.2, what=f"resume {key}")
        sid = result["session_id"]
        if holders:
            assert sid == old, (
                f"resume of {key} minted runtime {sid} while connection(s) "
                f"{[c.name for c in holders]} still hold live runtime {old}: two runtimes for one chat")
        self.sessions[key].sid = sid
        self.sessions[key].closed = False
        self.sid_key[sid] = key
        self.mark_attach(conn, sid, idx)
        slot.key, slot.sid = key, sid
        return sid

    def submit(self, slot: Slot, *, slow: bool) -> Turn:
        conn, key = slot.conn, slot.key
        assert conn is not None and key is not None
        self.wait_idle(key)
        self.n += 1
        canary = f"cnry-s{self.seed}-{self.n:03d}"
        text = f"{'slow ' if slow else ''}please ack {canary}"
        refusals: list[str] = []
        sent_at = [0]

        def attempt():
            sent_at[0] = len(conn.snapshot())
            frame = conn.request("prompt.submit", {"session_id": slot.sid, "text": text})
            if "error" not in frame:
                return frame["result"]
            err = frame["error"]
            code = err.get("code")
            if code == 4009:  # settling fence
                return None
            if code in (4001, 4007):  # runtime reaped under us: the client re-resumes, as Desktop does
                self.resume(slot, key)
                return None
            if code == 4090:  # ownership refusal: retried until the deadline, then a zombie lease
                refusals.append(str(err.get("message")))
                return None
            raise RpcError("prompt.submit", err)

        try:
            result = poll_until(attempt, timeout=45.0, interval=0.25, what=f"prompt.submit to {key}")
        except AssertionError as exc:
            raise AssertionError(f"{exc}; ownership refusals: {refusals[-3:]}") from None
        assert result.get("status") == "streaming", f"prompt to idle {key} was not started: {result}"
        sid = slot.sid
        assert sid is not None
        turn = Turn(canary, key, sid, slot.name, slow, {id(c) for c in self.attached_live(sid)}, frame_idx=sent_at[0])
        self.turns[canary] = turn
        self.sessions[key].prompts.append(canary)
        return turn

    def wait_first_delta(self, turn: Turn, conn: WSClient) -> None:
        # From this turn's submit on: an earlier turn's delta on the same sid is not this turn streaming.
        conn.wait_for(lambda f: etype(f) == "message.delta" and f["params"].get("session_id") == turn.sid,
                      timeout=STEP_TIMEOUT, start=turn.frame_idx, what=f"first delta of {turn.canary}")

    def db_has_reply(self, turn: Turn) -> bool:
        return bool(self.be.db_rows(
            "SELECT 1 FROM messages WHERE session_id=? AND role='assistant' AND content LIKE ?",
            (turn.key, f"%ACK {turn.canary} %DONE%")))

    def complete_frames(self, conn: WSClient, turn: Turn) -> list[dict]:
        return [f for f in conn.events(turn.sid, "message.complete")
                if f"ACK {turn.canary}" in str((f["params"].get("payload") or {}).get("text") or "")]

    # -- ops ---------------------------------------------------------------------------------
    def op_create(self, slot: Slot) -> None:
        conn = slot.conn or self.connect(slot)
        idx = len(conn.snapshot())
        result = conn.call("session.create", cols=100, source="desktop")
        sid, key = result["session_id"], result["stored_session_id"]
        self.sessions[key] = Sess(sid)
        self.sid_key[sid] = key
        self.mark_attach(conn, sid, idx)
        slot.key, slot.sid = key, sid

    def resumable(self) -> list[str]:
        out = []
        for key, sess in self.sessions.items():
            if sess.deleted:
                continue
            if sess.prompts or (sess.sid and self.attached_live(sess.sid) and not sess.closed):
                out.append(key)
        return out

    def op_switch(self, slot: Slot) -> None:
        if slot.conn is None:
            self.connect(slot)
        others = [k for k in self.resumable() if k != slot.key]
        if not others:
            return self.op_create(slot)
        self.resume(slot, self.rng.choice(sorted(others)))

    def ensure_selected(self, slot: Slot) -> None:
        if slot.conn is None:
            self.op_reconnect(slot)
        if slot.key is None or self.sessions[slot.key].deleted or self.sessions[slot.key].closed:
            self.op_create(slot)

    def op_prompt(self, slot: Slot, slow: bool = False) -> Turn:
        self.ensure_selected(slot)
        return self.submit(slot, slow=slow)

    def op_interrupt_mid_turn(self, slot: Slot) -> None:
        turn = self.op_prompt(slot, slow=True)
        assert slot.conn is not None
        self.wait_first_delta(turn, slot.conn)
        slot.conn.call("session.interrupt", session_id=turn.sid)
        turn.interrupted = True
        self.wait_idle(turn.key)

    def op_delete_active(self, slot: Slot) -> None:
        self.ensure_selected(slot)
        if not self.sessions[slot.key].prompts:
            self.submit(slot, slow=False)
        key = slot.key
        assert slot.conn is not None and key is not None
        before = self.be.db_rows("SELECT COUNT(*) FROM messages WHERE session_id=?", (key,))
        frame = slot.conn.request("session.delete", {"session_id": key})
        assert "error" in frame, f"session.delete of LIVE session {key} succeeded: {frame}"
        assert self.be.db_rows("SELECT COUNT(*) FROM sessions WHERE id=?", (key,)) == [(1,)], (
            f"refused delete still removed live session {key}")
        after = self.be.db_rows("SELECT COUNT(*) FROM messages WHERE session_id=?", (key,))
        assert after >= before, f"refused delete of {key} dropped messages: {before} -> {after}"

    def op_close_and_delete(self, slot: Slot) -> None:
        # A private chat: create, prompt, close (the runtime ends), then delete the inactive row.
        self.ensure_selected(slot)
        self.op_create(slot)
        turn = self.submit(slot, slow=False)
        key, sid = turn.key, turn.sid
        assert slot.conn is not None
        self.wait_idle(key)
        poll_until(lambda: self.db_has_reply(turn), timeout=STEP_TIMEOUT, what=f"reply persisted for {turn.canary}")
        slot.conn.call("session.close", session_id=sid)
        self.sessions[key].closed = True
        self.sessions[key].sid = None
        slot.key = slot.sid = None
        poll_until(lambda: not any(r.get("session_key") == key for r in self.active_rows()),
                   timeout=STEP_TIMEOUT, what=f"closed session {key} gone from the live list")
        slot.conn.call("session.delete", session_id=key)
        self.sessions[key].deleted = True
        assert self.be.db_rows("SELECT COUNT(*) FROM sessions WHERE id=?", (key,)) == [(0,)]
        assert self.be.db_rows("SELECT COUNT(*) FROM messages WHERE session_id=?", (key,)) == [(0,)]

    def op_drop_mid_turn(self, slot: Slot) -> None:
        turn = self.op_prompt(slot, slow=True)
        assert slot.conn is not None
        self.wait_first_delta(turn, slot.conn)
        slot.conn.drop()
        conn = self.connect(slot)
        self.resume(slot, turn.key)
        poll_until(lambda: self.complete_frames(conn, turn) or self.db_has_reply(turn),
                   timeout=STEP_TIMEOUT, what=f"completion of in-flight {turn.canary} after reconnect")
        self.wait_idle(turn.key)
        assert self.db_has_reply(turn), f"turn {turn.canary} was cut short by the ws drop (no full reply persisted)"
        for frame in self.complete_frames(conn, turn):
            assert (frame["params"].get("payload") or {}).get("status", "complete") == "complete", frame

    def orphan(self, slot: Slot, *, running: bool) -> None:
        """The owner of a private chat dies (no reconnect); another client must take it over."""
        self.ensure_selected(slot)
        self.op_create(slot)
        self.submit(slot, slow=False)  # a durable row: the chat is worth taking over
        turn = self.submit(slot, slow=True) if running else None
        key, sid = slot.key, slot.sid
        assert slot.conn is not None and key is not None and sid is not None
        if turn is not None:
            self.wait_first_delta(turn, slot.conn)
        slot.conn.drop()
        slot.conn = None
        assert not self.attached_live(sid)
        # Nobody else watches: the runtime must be released in bounded time (grace + turn end).
        poll_until(lambda: not any(r.get("session_key") == key for r in self.active_rows()),
                   timeout=STEP_TIMEOUT, interval=0.2, what=f"orphaned {key} released after its owner died")
        if turn is not None:
            assert self.db_has_reply(turn), f"owner death cut fresh turn {turn.canary} short (no full reply)"
        taker = self.rng.choice([s for s in self.slots.values() if s is not slot])
        if taker.conn is None:
            self.connect(taker)
        self.resume(taker, key)
        self.submit(taker, slow=False)  # 45 s of 4090 refusals here = zombie lease

    def op_orphan_idle(self, slot: Slot) -> None:
        self.orphan(slot, running=False)

    def op_orphan_running(self, slot: Slot) -> None:
        self.orphan(slot, running=True)

    def op_shared_owner_drop(self, slot: Slot) -> None:
        """Two windows on one chat; the prompting one dies mid-stream. The watcher keeps the turn."""
        self.ensure_selected(slot)
        key = slot.key
        assert key is not None
        if not self.sessions[key].prompts:
            self.submit(slot, slow=False)
        watcher = self.rng.choice([s for s in self.slots.values() if s is not slot])
        if watcher.conn is None:
            self.connect(watcher)
        self.resume(watcher, key)
        turn = self.submit(slot, slow=True)
        assert watcher.conn is not None and id(watcher.conn) in turn.attached
        self.wait_first_delta(turn, watcher.conn)
        assert slot.conn is not None
        slot.conn.drop()
        slot.conn = None
        poll_until(lambda: self.complete_frames(watcher.conn, turn), timeout=STEP_TIMEOUT,
                   what=f"watcher {watcher.name} gets {turn.canary} after the prompting window died")
        (frame,) = self.complete_frames(watcher.conn, turn)
        assert (frame["params"].get("payload") or {}).get("status", "complete") == "complete", frame
        assert self.db_has_reply(turn)
        self.submit(watcher, slow=False)

    def op_absence(self, slot: Slot) -> None:
        turn = self.op_prompt(slot, slow=True)
        conn = slot.conn
        assert conn is not None
        self.wait_first_delta(turn, conn)
        conn.pause()
        time.sleep(ABSENCE_S)  # the scenario itself (laptop lid closed), not a synchronisation wait
        conn.resume()
        poll_until(lambda: self.complete_frames(conn, turn), timeout=STEP_TIMEOUT,
                   what=f"{turn.canary} completion after a {ABSENCE_S}s absence")
        (frame,) = self.complete_frames(conn, turn)
        assert (frame["params"].get("payload") or {}).get("status", "complete") == "complete", (
            f"client absence ended turn {turn.canary}: {frame['params'].get('payload')}")

    def op_reconnect(self, slot: Slot) -> None:
        if slot.conn is not None and not slot.conn.dropped:
            slot.conn.drop()
        self.connect(slot)
        key = slot.key
        slot.sid = None
        if key and key in self.resumable():
            self.resume(slot, key)
        else:
            slot.key = None

    # -- invariants --------------------------------------------------------------------------
    def check_events(self) -> None:
        for conn in self.conns:
            seqs: dict[str, list[int]] = defaultdict(list)
            for idx, frame in enumerate(conn.snapshot()):
                if frame.get("method") != "event":
                    continue
                params = frame["params"]
                sid = params.get("session_id") or ""
                if not sid:
                    continue
                first = self.attach_idx[id(conn)].get(sid)
                assert first is not None and idx >= first, (
                    f"{conn.name} received {params.get('type')} for session {sid} "
                    f"({self.sid_key.get(sid, 'foreign')}) it never attached to")
                if isinstance(params.get("seq"), int):
                    seqs[sid].append(params["seq"])
                if params.get("type") == "message.complete":
                    text = str((params.get("payload") or {}).get("text") or "")
                    for canary in CANARY_RE.findall(text):
                        owner = self.turns.get(canary)
                        assert owner is not None and owner.key == self.sid_key.get(sid), (
                            f"{conn.name}: reply for {canary} (sent to "
                            f"{owner.key if owner else '?'}) delivered as session {self.sid_key.get(sid)}")
            for sid, values in seqs.items():
                dupes = [s for s, n in Counter(values).items() if n > 1]
                assert not dupes, f"{conn.name} got session {sid} events twice: duplicate seq {dupes[:5]}"
                assert values == sorted(values), f"{conn.name} got session {sid} events out of order"

    def check_single_runtime(self) -> None:
        keys = Counter(r.get("session_key") for r in self.active_rows())
        mine = {k: n for k, n in keys.items() if k in self.sessions and n > 1}
        assert not mine, f"chat(s) with more than one live runtime: {mine}"

    def check_final(self) -> None:
        for key in list(self.sessions):
            if not self.sessions[key].deleted:
                self.wait_idle(key)
        prefix = f"cnry-s{self.seed}-"
        rows = self.be.db_rows(
            "SELECT session_id, role, content FROM messages WHERE content LIKE ? ORDER BY id", (f"%{prefix}%",))
        user_by_key: dict[str, list[str]] = defaultdict(list)
        replies: Counter = Counter()
        full: Counter = Counter()
        for key, role, content in rows:
            found = [c for c in CANARY_RE.findall(str(content or "")) if c.startswith(prefix)]
            if role == "user":
                user_by_key[key].extend(found)
            elif role == "assistant":
                for canary in found:
                    replies[(key, canary)] += 1
                    full[(key, canary)] += str(content).rstrip().endswith("DONE")
        for key, sess in self.sessions.items():
            if sess.deleted:
                assert not user_by_key.get(key), f"deleted session {key} still has rows"
                continue
            assert user_by_key.get(key, []) == sess.prompts, (
                f"session {key}: persisted prompts {user_by_key.get(key, [])} != sent {sess.prompts}")
        stray = set(user_by_key) - set(self.sessions)
        assert not stray, f"prompts persisted to sessions nobody selected: {stray}"
        for canary, turn in self.turns.items():
            if self.sessions[turn.key].deleted:
                continue
            owner_replies = replies.get((turn.key, canary), 0)
            elsewhere = {k: n for (k, c), n in replies.items() if c == canary and k != turn.key}
            assert not elsewhere, f"reply to {canary} persisted into other session(s) {elsewhere}"
            if turn.interrupted:
                assert owner_replies <= 1, f"interrupted {canary} has {owner_replies} replies"
            else:
                assert owner_replies == 1, f"{canary} in {turn.key}: {owner_replies} persisted replies (want 1)"
                assert full[(turn.key, canary)] == 1, f"{canary} in {turn.key}: reply was cut short"
        survivors = {id(c) for c in self.live_conns()}
        for canary, turn in self.turns.items():
            if turn.interrupted or self.sessions[turn.key].deleted:
                continue
            for conn in self.conns:
                if id(conn) in turn.attached and id(conn) in survivors:
                    got = len(self.complete_frames(conn, turn))
                    assert got == 1, f"{conn.name} saw {got} completions of {canary} (want exactly 1)"


MANDATORY = ["prompt", "prompt_slow", "interrupt_mid_turn", "switch", "delete_active", "close_and_delete",
             "drop_mid_turn", "orphan_idle", "orphan_running", "shared_owner_drop", "absence", "reconnect"]
EXTRA = ["prompt", "prompt_slow", "switch", "switch", "prompt", "create", "delete_active"]


@pytest.mark.parametrize("seed", [11, 23, 37])
def test_multiclient_session_model(backend: Backend, seed: int) -> None:
    model = Model(backend, seed)
    rng = model.rng
    for slot in model.slots.values():  # every client starts with its own chat
        model.connect(slot)
        model.op_create(slot)
    plan = MANDATORY + [rng.choice(EXTRA) for _ in range(6)]
    rng.shuffle(plan)
    try:
        for step, op in enumerate(plan):
            slot = model.slots[rng.choice(sorted(model.slots))]
            model.log.append(f"{step}:{op}:{slot.name}")
            t0 = time.monotonic()
            if op == "prompt_slow":
                model.op_prompt(slot, slow=True)
            else:
                getattr(model, f"op_{op}")(slot)
            model.check_events()
            model.check_single_runtime()
            model.log.append(f"{time.monotonic() - t0:.1f}s")
        model.check_final()
        model.check_events()
        print(f"seed={seed} steps={model.log} turns={len(model.turns)} sessions={len(model.sessions)}")
    except AssertionError as exc:
        raise AssertionError(f"seed={seed} steps={model.log}\n{exc}\n{backend.logs(25)}") from None
    finally:
        for conn in model.conns:
            conn.close()
