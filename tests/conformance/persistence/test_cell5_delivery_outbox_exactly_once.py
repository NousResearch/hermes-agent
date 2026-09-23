"""Cell 5 — effect exactly-once across the delivery-outbox boundary.

Contract clause (arXiv:2608.03836, "effect exactly-once"): a crash between
the provider send and its durable record must not double-deliver on catch-up
(gateway reboot / boot redelivery sweep). Distinct from cell 2's consume-once:
cell 2 pins that exactly one claimant WINS a parked row; this cell pins the
observable SIDE EFFECT — how many copies of a reply the platform received.

Hermes' outbox is ``gateway/delivery_ledger.py``. Because a send and its
``mark_delivered`` can never be one atomic step, the ledger promises honest
at-least-once rather than a silent resend: a row that was never sent
('pending') is redelivered plainly; a row whose send may have landed
('attempting' / 'failed') is redelivered WITH a visible recovery marker. The
effect-level invariant that follows, per obligation id, is:

* at most ONE unmarked copy ever reaches the platform (every extra copy is
  labelled as a possible duplicate);
* at least one copy reaches it (nothing is lost);
* after a clean boot sweep the ledger row is terminal (delivered/abandoned);
* further reboots deliver nothing new (no re-send once recorded);
* N gateways rebooting concurrently over the same rows never both send one
  (claim exclusivity rests on the owner-stamp CAS in ``sweep_recoverable``).

What is real
------------
* The ledger module, on a real ``state.db`` under an isolated
  ``HOME=<tmp>/home``, ``HERMES_HOME=<tmp>/home/.hermes`` (every child).
* The turn-side checkpoints: ``BasePlatformAdapter._record_delivery_obligation``
  (record + mark_attempting) and ``_finalize_delivery_obligation``
  (mark_delivered) — the exact methods the adapter's final-send path calls.
  The "recorded, crashed before attempting" kill point calls the ledger's
  ``record_obligation`` directly (there is no seam between the two writes).
* The boot sweep: ``GatewayRunner._claim_pending_obligations`` (ledger claim +
  real ``SessionStore`` resume-flag clear) then
  ``GatewayRunner._redeliver_claimed_obligations`` (marker choice, send,
  mark_delivered/mark_failed) — the two halves ``_await_startup_boot_sends``
  runs at startup, on a runner built with ``object.__new__`` exactly as the
  gateway unit suite does (no connect/auth/network start-up).
* Crashes are real ``SIGKILL``s to separate interpreters, asserted alive at
  kill time; the killed owner's pid is genuinely dead for ``_owner_alive``.

What is fake (the seam)
-----------------------
* The transport only: ``JournalAdapter`` is a real ``BasePlatformAdapter``
  subclass whose ``send`` appends the delivered content to a journal file and
  ``fsync``s it BEFORE returning success. The journal is therefore ground
  truth for "the platform accepted this" even across SIGKILL. With
  ``CELL5_HANG_AFTER_SEND`` set, ``send`` journals, then parks until killed —
  this is the precise "provider accepted, durable record not yet written"
  window.
* Concurrency scheduling (cell c only): each rebooter's first
  ``delivery_ledger._owner_alive`` call — which runs after the sweep's SELECT
  and before any claim UPDATE — waits on a file barrier until every rebooter
  has SELECTed. It only delays, never changes a decision; it forces the
  worst-case interleave (all sweepers holding the same stale snapshot) so
  exclusivity is proven by the CAS rather than by lucky timing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conformance.persistence._harness import (
    CHILD_DEADLINE,
    REPO_ROOT,
    integrity_ok,
    kill9_and_reap,
    reap,
    wait_for,
)

SESSION_KEY = "agent:main:telegram:dm:cell5"
CHAT_ID = "4242"
N_OBLIGATIONS = 3  # crash-window obligations per kill point (plus one delivered control)
N_CONCURRENT_OBLIGATIONS = 12
N_REBOOTERS = 3

PRELUDE = r'''
import asyncio, json, os, sys, time
from pathlib import Path
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult

JOURNAL = Path(os.environ["CELL5_JOURNAL"])
ROLE = os.environ["CELL5_ROLE"]
HANG_AFTER_SEND = os.environ.get("CELL5_HANG_AFTER_SEND")
CHAT_ID = os.environ["CELL5_CHAT_ID"]
SESSION_KEY = os.environ["CELL5_SESSION_KEY"]


def park(signal_path):
    """Announce the kill window, then block until the parent SIGKILLs us."""
    Path(signal_path).touch()
    end = time.monotonic() + 55
    while time.monotonic() < end:
        time.sleep(0.05)
    sys.exit(4)  # never killed: the parent's harness failed, surface it


def journal_send(content):
    line = (json.dumps({"pid": os.getpid(), "role": ROLE, "content": content}) + "\n").encode()
    fd = os.open(str(JOURNAL), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, line)
        os.fsync(fd)  # accepted == durable in the journal BEFORE send returns
    finally:
        os.close(fd)


class JournalAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="cell5"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        journal_send(content)
        if HANG_AFTER_SEND:
            park(HANG_AFTER_SEND)  # provider accepted; mark_delivered never runs
        return SendResult(success=True, message_id=str(time.time_ns()))

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def body(tag):
    return f"cell5 reply {tag}"
'''

# The gateway turn: one fully delivered control reply, then N replies driven to
# the kill point, then park for SIGKILL.
SEEDER = PRELUDE + r'''
import gateway.delivery_ledger as dl
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

kinds = json.loads(os.environ["CELL5_KINDS"])
adapter = JournalAdapter()


def event(tag):
    src = SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT_ID, chat_type="dm", user_id="u1")
    return MessageEvent(text=f"question {tag}", source=src, message_id=f"m-{tag}")


async def main():
    ids = {}
    ev, text = event("control"), body("control")
    oid = await adapter._record_delivery_obligation(ev, SESSION_KEY, text, adapter, False)
    assert oid, "ledger refused to record the control obligation"
    result = await adapter.send(CHAT_ID, text)
    await adapter._finalize_delivery_obligation(oid, result, ev, adapter)
    ids[body("control")] = {"id": oid, "kind": "delivered"}
    for i, kind in enumerate(kinds):
        ev, text = event(i), body(i)
        if kind == "pending":
            # Recorded, crashed before mark_attempting.
            oid = dl.compute_obligation_id(SESSION_KEY, ev.message_id, text)
            dl.record_obligation(obligation_id=oid, session_key=SESSION_KEY, platform="telegram",
                                 chat_id=CHAT_ID, thread_id=None, content=text)
        else:
            oid = await adapter._record_delivery_obligation(ev, SESSION_KEY, text, adapter, False)
            assert oid, "ledger refused to record an obligation"
            if kind == "sent":
                await adapter.send(CHAT_ID, text)  # accepted; finalize never runs
        ids[text] = {"id": oid, "kind": kind}
    out = Path(os.environ["CELL5_IDS"])
    tmp = out.with_suffix(".part")
    tmp.write_text(json.dumps(ids))
    os.replace(tmp, out)
    park(os.environ["CELL5_READY"])


asyncio.run(main())
'''

# A gateway boot: the real startup claim half, then the real redelivery half.
REBOOTER = PRELUDE + r'''
import gateway.delivery_ledger as dl
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.session import SessionStore

go = os.environ.get("CELL5_GO")
if go:
    Path(os.environ["CELL5_READY"]).touch()
    end = time.monotonic() + 55
    while not Path(go).exists():
        if time.monotonic() > end:
            sys.exit(3)
        time.sleep(0.005)

barrier_dir = os.environ.get("CELL5_SWEEP_BARRIER")
if barrier_dir:
    parties = int(os.environ["CELL5_SWEEP_PARTIES"])
    _real_owner_alive = dl._owner_alive
    _hit = []

    def _gated_owner_alive(pid, started_at):
        # First call happens after the sweep's SELECT, before any claim UPDATE.
        if not _hit:
            _hit.append(1)
            Path(barrier_dir, f"selected-{os.getpid()}").touch()
            end = time.monotonic() + 30
            while (len(list(Path(barrier_dir).glob("selected-*"))) < parties
                   and time.monotonic() < end):
                time.sleep(0.002)
        return _real_owner_alive(pid, started_at)

    dl._owner_alive = _gated_owner_alive

home = Path(os.environ["HERMES_HOME"])
runner = object.__new__(GatewayRunner)
runner.adapters = {Platform.TELEGRAM: JournalAdapter()}
runner._profile_adapters = {}
runner.session_store = SessionStore(home / "sessions", GatewayConfig())


async def main():
    claimed = await runner._claim_pending_obligations()
    n = await runner._redeliver_claimed_obligations(claimed)
    print(json.dumps({"claimed": [r["obligation_id"] for r in claimed], "redelivered": n}))


asyncio.run(main())
'''


# --------------------------------------------------------------------------- harness


def _strip_marker(content: str) -> tuple[str, bool]:
    """(reply body, carried a recovery marker?) for one journaled send."""
    from gateway.delivery_ledger import FLOOD_MARKER, RECONNECTED_MARKER, RECOVERED_MARKER

    for marker in (RECOVERED_MARKER, RECONNECTED_MARKER, FLOOD_MARKER):
        if content.startswith(marker):
            return content[len(marker):], True
    return content, False


class Cell:
    """One isolated gateway install: temp home, state.db, transport journal."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path
        self.home = tmp_path / "home"
        self.hermes_home = self.home / ".hermes"
        self.hermes_home.mkdir(parents=True)
        (self.hermes_home / "config.yaml").write_text(
            "gateway:\n  delivery_ledger: true\n", encoding="utf-8"
        )
        self.journal = tmp_path / "transport.jsonl"
        self.ids_file = tmp_path / "ids.json"
        self.db_path = self.hermes_home / "state.db"
        (tmp_path / "tmp").mkdir()
        self._boots = 0

    def spawn(self, script: str, role: str, extra: dict | None = None) -> subprocess.Popen:
        env = {k: v for k, v in os.environ.items()
               if not (k.endswith("_API_KEY") or k.endswith("_BOT_TOKEN"))}
        inherited = env.get("PYTHONPATH")
        env.update({
            "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{inherited}" if inherited else str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1",
            "HOME": str(self.home),
            "HERMES_HOME": str(self.hermes_home),
            "TMPDIR": str(self.root / "tmp"),
            # HOME/.hermes IS the temp home here, so the live-system guard would
            # read it as production; bypass it in the CHILD only.
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
            "CELL5_JOURNAL": str(self.journal),
            "CELL5_ROLE": role,
            "CELL5_CHAT_ID": CHAT_ID,
            "CELL5_SESSION_KEY": SESSION_KEY,
        })
        env.pop("CELL5_HANG_AFTER_SEND", None)
        env.update(extra or {})
        return subprocess.Popen(
            [sys.executable, "-c", script], cwd=str(REPO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def seed_and_crash(self, kinds: list[str]) -> dict:
        ready = self.root / "seed-ready"
        child = self.spawn(SEEDER, "turn", {
            "CELL5_KINDS": json.dumps(kinds), "CELL5_IDS": str(self.ids_file),
            "CELL5_READY": str(ready),
        })
        self._kill_at(child, ready, "turn process at its kill point")
        return json.loads(self.ids_file.read_text())

    def crashing_boot(self) -> None:
        """A boot whose redelivery send is accepted, then SIGKILL before mark_delivered."""
        self._boots += 1
        sent = self.root / f"boot{self._boots}-sent"
        child = self.spawn(REBOOTER, f"boot-{self._boots}", {"CELL5_HANG_AFTER_SEND": str(sent)})
        self._kill_at(child, sent, f"boot {self._boots} inside its redelivery send")

    def boot(self) -> dict:
        self._boots += 1
        rc, out, err = reap(self.spawn(REBOOTER, f"boot-{self._boots}"))
        assert rc == 0, f"boot {self._boots} crashed rc={rc}: {err[-2000:]}"
        return json.loads(out.strip().splitlines()[-1])

    @staticmethod
    def _kill_at(child: subprocess.Popen, signal_file: Path, what: str) -> None:
        try:
            wait_for(signal_file.exists, what=what, child=child)
            # The kill must hit a LIVE process parked in the window.
            assert child.poll() is None, f"{what}: exited early rc={child.returncode}"
        finally:
            kill9_and_reap(child)
        assert child.returncode == -9, f"{what}: not SIGKILLed (rc={child.returncode})"

    def journal_entries(self) -> list[dict]:
        if not self.journal.exists():
            return []
        # Strict parse: every kill lands AFTER the fsync'd line (signal file is
        # touched post-fsync), so a torn line would be a harness bug.
        return [json.loads(line) for line in self.journal.read_text().splitlines()]

    def copies(self, ids: dict) -> dict[str, dict]:
        """Per obligation id: unmarked/marked copy counts and which roles sent them."""
        by_body = {b: v["id"] for b, v in ids.items()}
        out = {v["id"]: {"unmarked": 0, "marked": 0, "roles": []} for v in ids.values()}
        for entry in self.journal_entries():
            content, marked = _strip_marker(entry["content"])
            assert content in by_body, f"transport received an unknown payload: {entry!r}"
            c = out[by_body[content]]
            c["marked" if marked else "unmarked"] += 1
            c["roles"].append(entry["role"])
        return out

    def ledger(self) -> dict[str, tuple[str, int]]:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            return {oid: (state, attempts) for oid, state, attempts in conn.execute(
                "SELECT obligation_id, state, attempts FROM delivery_obligations")}
        finally:
            conn.close()


def _assert_effect_exactly_once(cell: Cell, ids: dict, label: str) -> dict:
    copies = cell.copies(ids)
    ledger = cell.ledger()
    for body, meta in ids.items():
        oid, c = meta["id"], copies[meta["id"]]
        assert c["unmarked"] <= 1, (
            f"[{label}] {body!r} ({meta['kind']}): {c['unmarked']} UNMARKED copies reached the "
            f"platform — a silent duplicate (roles={c['roles']})"
        )
        assert c["unmarked"] + c["marked"] >= 1, f"[{label}] {body!r}: reply lost (0 copies)"
        assert oid in ledger, f"[{label}] {body!r}: ledger row vanished before terminal"
        state, attempts = ledger[oid]
        assert state in ("delivered", "abandoned"), (
            f"[{label}] {body!r}: ledger row not terminal after a clean boot sweep "
            f"(state={state}, attempts={attempts})"
        )
    return copies


def _assert_reboots_are_noops(cell: Cell, ids: dict, label: str, n: int = 2) -> None:
    before = cell.journal_entries()
    ledger_before = cell.ledger()
    for _ in range(n):
        result = cell.boot()
        assert result["claimed"] == [] and result["redelivered"] == 0, (
            f"[{label}] a later reboot re-claimed/re-sent settled obligations: {result}"
        )
    assert cell.journal_entries() == before, f"[{label}] a later reboot re-sent a reply"
    assert cell.ledger() == ledger_before, f"[{label}] a later reboot mutated settled rows"


# Kill point -> (kinds seeded by the turn process, reboot #1 crashes inside its resend?)
KILL_POINTS = {
    # recorded 'pending', crashed before mark_attempting: never sent.
    "recorded-unsent": (["pending"] * N_OBLIGATIONS, False),
    # (a) recorded + attempting, send NOT issued.
    "attempting-unsent": (["attempting"] * N_OBLIGATIONS, False),
    # (b) send accepted by the transport, SIGKILL before mark_delivered.
    "sent-unacked": (["sent"] * N_OBLIGATIONS, False),
    # (b') the REBOOTING gateway is itself killed inside its redelivery send.
    "resend-crash-attempting": (["attempting"] * N_OBLIGATIONS, True),
}

# Expected copies per kind after the first clean sweep (deterministic).
EXPECTED = {
    "delivered": (1, 0),   # control: sent + marked delivered, never resent
    "pending": (1, 0),     # never sent -> one plain redelivery
    "attempting": (0, 1),  # may have landed -> one marked redelivery
    "sent": (1, 1),        # landed unmarked -> one marked (honest) duplicate
}


@pytest.mark.parametrize("kill_point", list(KILL_POINTS))
def test_crash_between_send_and_record_never_double_delivers(tmp_path, kill_point):
    kinds, resend_crash = KILL_POINTS[kill_point]
    cell = Cell(tmp_path)
    ids = cell.seed_and_crash(kinds)
    assert len(ids) == N_OBLIGATIONS + 1

    crashed_resend = None
    if resend_crash:
        cell.crashing_boot()
        resent = [e for e in cell.journal_entries() if e["role"] == "boot-1"]
        assert len(resent) == 1, f"crashing boot journaled {len(resent)} sends (want 1)"
        crashed_resend, _ = _strip_marker(resent[0]["content"])

    first = cell.boot()
    copies = _assert_effect_exactly_once(cell, ids, kill_point)

    owed = {m["id"] for m in ids.values() if m["kind"] != "delivered"}
    assert set(first["claimed"]) == owed, (
        f"[{kill_point}] first clean boot claimed {sorted(first['claimed'])}, owed {sorted(owed)}"
    )
    for body, meta in ids.items():
        unmarked, marked = EXPECTED[meta["kind"]]
        if body == crashed_resend:
            marked += 1  # the killed boot's accepted resend: marked, so honest
        got = copies[meta["id"]]
        assert (got["unmarked"], got["marked"]) == (unmarked, marked), (
            f"[{kill_point}] {body!r} ({meta['kind']}): copies unmarked/marked="
            f"{got['unmarked']}/{got['marked']}, want {unmarked}/{marked} (roles={got['roles']})"
        )
        if meta["kind"] != "delivered":
            assert cell.ledger()[meta["id"]][0] == "delivered", (
                f"[{kill_point}] {body!r}: redelivered but ledger says {cell.ledger()[meta['id']]}"
            )

    _assert_reboots_are_noops(cell, ids, kill_point)
    assert integrity_ok(cell.db_path)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FIRE (cell 5): sweep_recoverable claims a 'pending' row without flipping it to "
        "'attempting', and _redeliver_claimed_obligations never marks it attempting before the "
        "send — a boot killed inside that resend leaves it 'pending', so the next boot resends it "
        "UNMARKED (2 unmarked copies). strict: flips red when fixed so the xfail is removed."
    ),
)
def test_boot_killed_inside_plain_redelivery_never_double_delivers(tmp_path):
    cell = Cell(tmp_path)
    ids = cell.seed_and_crash(["pending"] * N_OBLIGATIONS)
    cell.crashing_boot()
    cell.boot()
    _assert_effect_exactly_once(cell, ids, "resend-crash-recorded")
    _assert_reboots_are_noops(cell, ids, "resend-crash-recorded")


def test_concurrent_reboots_claim_each_obligation_once(tmp_path):
    """(c) N gateways reboot at once over the same dead-owner rows."""
    cell = Cell(tmp_path)
    half = N_CONCURRENT_OBLIGATIONS // 2
    ids = cell.seed_and_crash(["pending"] * half + ["attempting"] * half)

    go = tmp_path / "go"
    barrier = tmp_path / "sweep-barrier"
    barrier.mkdir()
    readies, children = [], []
    for i in range(N_REBOOTERS):
        ready = tmp_path / f"reboot-ready-{i}"
        readies.append(ready)
        children.append(cell.spawn(REBOOTER, f"concurrent-{i}", {
            "CELL5_GO": str(go), "CELL5_READY": str(ready),
            "CELL5_SWEEP_BARRIER": str(barrier), "CELL5_SWEEP_PARTIES": str(N_REBOOTERS),
        }))
    try:
        wait_for(lambda: all(r.exists() for r in readies), what="all rebooters at the barrier")
    except BaseException:
        for c in children:
            kill9_and_reap(c)
        raise
    go.touch()
    results = [reap(c, deadline=CHILD_DEADLINE) for c in children]
    assert all(rc == 0 for rc, _, _ in results), (
        f"rebooter crashed: {[(rc, e[-800:]) for rc, _, e in results if rc]}"
    )
    # Worst-case interleave actually happened: every sweeper SELECTed the same
    # stale rows before any of them claimed one.
    assert len(list(barrier.glob("selected-*"))) == N_REBOOTERS, "sweep barrier not reached by all"

    outs = [json.loads(o.strip().splitlines()[-1]) for _, o, _ in results]
    claims = [oid for out in outs for oid in out["claimed"]]
    owed = {m["id"] for m in ids.values() if m["kind"] != "delivered"}
    assert sorted(claims) == sorted(owed), (
        f"claim exclusivity violated: {len(claims)} claims for {len(owed)} obligations "
        f"(per rebooter: {[len(o['claimed']) for o in outs]})"
    )

    copies = _assert_effect_exactly_once(cell, ids, "concurrent")
    for body, meta in ids.items():
        c = copies[meta["id"]]
        assert c["unmarked"] + c["marked"] == 1, (
            f"[concurrent] {body!r} ({meta['kind']}) delivered {c['unmarked'] + c['marked']}x "
            f"by concurrent rebooters (roles={c['roles']})"
        )
        assert (c["unmarked"], c["marked"]) == EXPECTED[meta["kind"]], (
            f"[concurrent] {body!r} ({meta['kind']}): wrong marker shape {c}"
        )
    _assert_reboots_are_noops(cell, ids, "concurrent", n=1)
    assert integrity_ok(cell.db_path)
