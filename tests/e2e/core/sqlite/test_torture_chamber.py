"""SQLite torture chamber: state.db integrity under real multi-process load (issue class C1).

One ``state.db`` per journal mode Hermes deploys — WAL, and DELETE (what it runs on a WAL-reset-vulnerable
SQLite and on network/FUSE homes; selected by the production ``apply_wal_with_fallback`` through a pinned
version probe in every child, see ``_helpers``); every role is its own OS process running the production
``SessionDB``:
a gateway-like writer, a TUI-like writer, a dashboard-like reader that opens at startup and lives across
episodes, short-lived openers (production ``SessionDB`` and bare ``sqlite3``, plus the real
``hermes sessions list`` / ``sessions stats`` CLI), FTS rebuild/optimize maintenance, and
``repair_state_db_schema``. Each episode injects one fault class — ``kill -9`` mid-write, SIGTERM graceful
close, POSIX lock cancellation by a stray in-process open/close, chmod flips, concurrent FTS rebuilds,
repair against a live and an offline store, FTS corruption, a whole-fleet SIGKILL — and then asserts the
SAME invariants:

* ``PRAGMA integrity_check`` is ``ok``; the store is still in the arm's journal mode and every SessionDB
  role really ran in it;
* no child ever held a ``(deleted)`` ``state.db`` descriptor — nor, in WAL mode, ``-wal``/``-shm``
  (``/proc/<pid>/fd`` scan);
* every acknowledged append is stored exactly once (per-writer intent/ack journals), an in-flight append at
  most once, and no row exists that no writer intended;
* canonical row counts only grow (no compaction runs here), and repair never lowers them;
* FTS mirrors the canonical rows (docsize == source rows, FTS5 ``integrity-check``) and session search
  finds acked messages exactly once;
* no role hit an error (DELETE arm: readers block on writes by design, so a SQLITE_BUSY refusal of a read,
  open or FTS pass is waited out and counted, never an integrity failure); the long-lived reader's and the
  open/close churner's fd counts stay bounded.

Randomness (ack thresholds, kill points) is seeded per episode; the seed is in every failure message and
``HERMES_SQLITE_TORTURE_SEED`` replays a run.
"""

from __future__ import annotations

import contextlib
import os
import random
import sqlite3
import sys
import time

import pytest

from tests.conformance.persistence._harness import wait_for
from tests.e2e.core.sqlite._helpers import (
    JOURNAL_MODES,
    SEED_ENV,
    Chamber,
    acked_tokens,
    base_seed,
    busy_summary,
    counts,
    episode_seed,
    exactly_once_problems,
    fts_problems,
    integrity_rows,
    journal_mode_problems,
    sample,
    skip_unless_deployable,
)

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc fd scan and POSIX lock semantics"),
]

READER_FD_SLACK = 6
CHURN_FD_SLACK = 2


@pytest.fixture(scope="module", params=JOURNAL_MODES)
def chamber(request, tmp_path_factory):
    skip_unless_deployable(request.param)
    ch = Chamber(tmp_path_factory.mktemp(f"chamber-{request.param}"), journal=request.param)
    _ensure_reader(ch)
    yield ch
    ch.shutdown()


def _ensure_reader(ch: Chamber) -> None:
    if ch.reader_name and ch.procs[ch.reader_name].poll() is None:
        return
    ch.reader_seq += 1
    ch.reader_name = f"reader{ch.reader_seq}"
    ch.spawn("reader", ch.reader_name, pace=0.02)
    ch.wait_event(ch.reader_name, "ready")


def _stop_reader(ch: Chamber) -> None:
    if ch.reader_name and ch.procs[ch.reader_name].poll() is None:
        assert ch.stop(ch.reader_name) == 0, ch.stderr(ch.reader_name)


def _writers(ch: Chamber, ep: str, **kw) -> tuple[str, str]:
    gw, tui = f"{ep}-gw", f"{ep}-tui"
    ch.spawn("writer", gw, tag="g", sessions=["gw-a", "gw-b", "gw-c"], source="telegram", batch_every=7, **kw)
    ch.spawn("writer", tui, tag="t", sessions=["tui-a"], source="tui", batch_every=5, **kw)
    ch.wait_event(gw, "ready")
    ch.wait_event(tui, "ready")
    return gw, tui


def _background(ch: Chamber, ep: str, *, openers: int = 4, churn: int = 40) -> list[str]:
    names = []
    if churn:
        ch.spawn("churn", f"{ep}-churn", iterations=churn)
        names.append(f"{ep}-churn")
    for i in range(openers):
        name = f"{ep}-open{i}"
        ch.spawn("opener", name, raw=bool(i % 2))
        names.append(name)
    return names


def _reap_all(ch: Chamber, names: list[str]) -> None:
    for name in names:
        proc = ch.procs[name]
        if proc.returncode is None:
            rc = ch.reap(name)
            errors = [f"{e.get('error')}\n{e.get('tb', '')}" for e in ch.events(name) if e.get("event") == "error"]
            assert rc == 0, f"{name} exited {rc}: {errors}\n{ch.stderr(name)}"


def _stop_all(ch: Chamber, names: list[str]) -> None:
    for name in names:
        if ch.procs[name].poll() is None:
            ch.request_stop(name)
    _reap_all(ch, names)


# -- episodes --------------------------------------------------------------------------------------------


def ep_baseline(ch, ep, rng):
    gw, tui = _writers(ch, ep)
    bg = _background(ch, ep)
    ch.spawn_cli(f"{ep}-cli-list", "sessions", "list")
    ch.spawn_cli(f"{ep}-cli-stats", "sessions", "stats")
    ch.wait_acks(gw, rng.randint(60, 120))
    ch.wait_acks(tui, rng.randint(40, 80))
    _reap_all(ch, bg + [f"{ep}-cli-list", f"{ep}-cli-stats"])
    _stop_all(ch, [gw, tui])
    return [gw, tui]


def ep_kill9_writer_midwrite(ch, ep, rng):
    gw, tui = _writers(ch, ep)
    ch.spawn("churn", f"{ep}-churn", iterations=400)
    ch.wait_acks(gw, rng.randint(20, 60))
    ch.kill9(gw)  # gateway dies mid-append: the in-flight row is either whole or absent
    ch.kill9(f"{ep}-churn")  # and a CLI dies holding the file open
    gw2 = f"{ep}-gw2"
    ch.spawn("writer", gw2, tag="g", sessions=["gw-a", "gw-b"], source="telegram", batch_every=3)
    bg = _background(ch, ep, churn=0)
    ch.wait_acks(gw2, rng.randint(30, 60))
    ch.wait_acks(tui, 40)
    _reap_all(ch, bg)
    _stop_all(ch, [gw2, tui])
    return [gw, tui, gw2]


def ep_sigterm_graceful_close(ch, ep, rng):
    gw, tui = _writers(ch, ep)
    bg = _background(ch, ep, openers=6, churn=60)
    ch.wait_acks(gw, rng.randint(30, 80))
    ch.sigterm(gw)
    ch.sigterm(tui)  # both close (checkpoint-on-close) while the churner and openers are mid-cycle
    for name in (gw, tui):
        assert ch.reap(name) == 0, ch.stderr(name)
        assert any(e.get("event") == "closed" for e in ch.events(name)), f"{name} never closed cleanly"
    gw2 = f"{ep}-gw2"
    ch.spawn("writer", gw2, tag="g", sessions=["gw-a"], source="telegram")
    ch.wait_acks(gw2, 30)
    _reap_all(ch, bg)
    _stop_all(ch, [gw2])
    return [gw, tui, gw2]


def ep_lock_cancellation(ch, ep, rng):
    """Plugins, tool reads of ~/.hermes and header probes open+close state.db inside the writer process,
    which cancels its POSIX locks; a sibling's last close must still not end the writer's WAL generation.
    Field topology: the gateway (+ a TUI) are the only long-lived holders — no dashboard reader pinning the
    file — so a short-lived CLI's close is the last close SQLite can see."""
    _stop_reader(ch)
    gw, tui = _writers(ch, ep, stray_every=rng.randint(2, 4))
    names = []
    target_gw, target_tui = rng.randint(80, 140), rng.randint(60, 100)
    wave = 0
    while len(ch.journal(gw)[1]) < target_gw or len(ch.journal(tui)[1]) < target_tui:
        batch = _background(ch, f"{ep}-w{wave}", openers=4, churn=6)
        _reap_all(ch, batch)
        names += batch
        wave += 1
        assert wave < 200, "writers made no progress"
        for w in (gw, tui):
            rc = ch.procs[w].poll()
            assert rc is None, (f"{w} died (rc={rc}) after {len(ch.journal(w)[1])} acks: {ch.events(w)[-1:]}\n"
                                f"deleted sidecars held: {ch.deleted_hits_snapshot()}\n{ch.stderr(w)}")
    _stop_all(ch, [gw, tui])
    _ensure_reader(ch)
    return [gw, tui]


def ep_chmod_flip(ch, ep, rng):
    gw, tui = _writers(ch, ep)
    files = [ch.db, ch.db.with_name(ch.db.name + "-wal"), ch.db.with_name(ch.db.name + "-shm")]
    try:
        for i in range(rng.randint(3, 5)):  # each flip is the same fault; process start-up dominates
            for f in files:
                if f.exists():
                    os.chmod(f, 0o644)  # a permissive mode the next SessionDB open tightens
            ch.spawn("opener", f"{ep}-tighten{i}")
            _reap_all(ch, [f"{ep}-tighten{i}"])
            os.chmod(ch.db, 0o400)  # read-only main file: new openers may refuse, must not damage
            ch.spawn("opener", f"{ep}-ro{i}", may_fail=True)
            ch.spawn("opener", f"{ep}-rorw{i}", may_fail=True, raw=True)
            _reap_all(ch, [f"{ep}-ro{i}", f"{ep}-rorw{i}"])
            os.chmod(ch.db, 0o600)
    finally:
        for f in files:
            if f.exists():
                os.chmod(f, 0o600)
    n_gw = len(ch.journal(gw)[1])
    ch.wait_acks(gw, n_gw + 20)  # still writing after every permission flip
    _stop_all(ch, [gw, tui])
    return [gw, tui]


def ep_fts_rebuild_concurrent(ch, ep, rng):
    gw, tui = _writers(ch, ep)
    ch.wait_acks(gw, 20)
    ch.spawn("fts", f"{ep}-fts-a")
    ch.spawn("fts", f"{ep}-fts-b")
    ch.spawn("fts", f"{ep}-fts-killed")
    ch.wait_event(f"{ep}-fts-killed", "ready")
    time.sleep(rng.uniform(0.0, 0.05))  # fault-injection jitter, not synchronization
    ch.kill9(f"{ep}-fts-killed")  # maintenance killed mid-rebuild
    _reap_all(ch, [f"{ep}-fts-a", f"{ep}-fts-b"])
    rebuilt = [e for n in ("a", "b") for e in ch.events(f"{ep}-fts-{n}") if e.get("event") == "stats"]
    assert len(rebuilt) == 2, rebuilt
    ch.wait_acks(gw, len(ch.journal(gw)[1]) + 20)
    _stop_all(ch, [gw, tui])
    return [gw, tui]


def _repair(ch: Chamber, name: str) -> dict:
    ch.spawn("repair", name)
    _reap_all(ch, [name])
    stats = [e for e in ch.events(name) if e.get("event") == "stats"]
    assert stats, ch.events(name)
    s = stats[0]
    assert s["after"] >= s["before"], f"repair lowered canonical rows: {s}"
    return s


def ep_repair_live_then_offline(ch, ep, rng):
    gw, tui = _writers(ch, ep)
    ch.wait_acks(gw, rng.randint(20, 50))
    _repair(ch, f"{ep}-repair-live")  # must not cost an acked row (checked by the episode invariants)
    ch.wait_acks(gw, len(ch.journal(gw)[1]) + 20)
    _stop_all(ch, [gw, tui])
    _stop_reader(ch)
    _repair(ch, f"{ep}-repair-offline")
    _ensure_reader(ch)
    return [gw, tui]


def ep_fts_corruption_fail_open(ch, ep, rng):
    """A corrupt FTS index must not cost a canonical write; repair + the next open restore search."""
    _stop_reader(ch)
    conn = sqlite3.connect(str(ch.db), timeout=30.0)
    try:
        conn.execute("UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'")
        conn.commit()
    finally:
        conn.close()
    gw, tui = _writers(ch, ep)
    ch.wait_acks(gw, rng.randint(20, 50))
    ch.wait_acks(tui, 20)
    _repair(ch, f"{ep}-repair-live")  # a repair racing live writers must not overwrite their acked rows
    _stop_all(ch, [gw, tui])
    _repair(ch, f"{ep}-repair-offline")
    ch.spawn("opener", f"{ep}-reopen")  # the next SessionDB open finishes the stale-FTS recovery
    _reap_all(ch, [f"{ep}-reopen"])
    _ensure_reader(ch)
    return [gw, tui]


def ep_kill9_everything(ch, ep, rng):
    """Update fleet-restart / OOM-killer shape: every process on the file dies at once, mid-write."""
    gw, tui = _writers(ch, ep, stray_every=5)
    ch.spawn("churn", f"{ep}-churn", iterations=400)
    ch.spawn("fts", f"{ep}-fts")
    ch.wait_acks(gw, rng.randint(30, 90))
    for name in (gw, tui, f"{ep}-churn", f"{ep}-fts", ch.reader_name):
        if ch.procs[name].poll() is None:
            ch.kill9(name)
    gw2 = f"{ep}-gw2"
    ch.spawn("writer", gw2, tag="g", sessions=["gw-a", "tui-a"], source="telegram", batch_every=4)
    _ensure_reader(ch)
    ch.wait_acks(gw2, 30)
    _stop_all(ch, [gw2])
    return [gw, tui, gw2]


EPISODES = {
    "baseline": ep_baseline,
    "kill9_writer_midwrite": ep_kill9_writer_midwrite,
    "sigterm_graceful_close": ep_sigterm_graceful_close,
    "lock_cancellation": ep_lock_cancellation,
    "chmod_flip": ep_chmod_flip,
    "fts_rebuild_concurrent": ep_fts_rebuild_concurrent,
    "repair_live_then_offline": ep_repair_live_then_offline,
    "fts_corruption_fail_open": ep_fts_corruption_fail_open,
    "kill9_everything": ep_kill9_everything,
}


def _reader_fd_problems(ch: Chamber) -> list[str]:
    fds = [e["fds"] for e in ch.events(ch.reader_name) if e.get("fds") is not None]
    if len(fds) < 4:
        return []
    warm = max(fds[:3])  # after the first passes the read pool has settled
    if max(fds) > warm + READER_FD_SLACK:
        return [f"{ch.reader_name} fd count grew {warm} -> {max(fds)} over {len(fds)} samples ({fds[-5:]})"]
    return []


def _churn_fd_problems(ch: Chamber, ep: str) -> list[str]:
    out = []
    for name in ch.procs:
        if not name.startswith(ep) or "churn" not in name:
            continue
        for e in ch.events(name):
            if e.get("event") == "stats" and e.get("warm_fds") is not None:
                if e["end_fds"] > e["warm_fds"] + CHURN_FD_SLACK:
                    out.append(f"{name}: fds {e['warm_fds']} after warmup -> {e['end_fds']} after "
                               f"{e['iterations']} open/close cycles")
    return out


@pytest.mark.parametrize("episode", list(EPISODES))
def test_torture_episode(chamber, episode):
    seed = episode_seed(episode)
    rng = random.Random(seed)
    ctx = (f"[journal={chamber.mode} episode={episode} seed={seed} base={base_seed()}; "
           f"replay: {SEED_ENV}={base_seed()}]")
    _ensure_reader(chamber)
    before = counts(chamber.db) if chamber.db.exists() else {"__total__": 0}
    hits_mark = chamber.deleted_hits_mark()  # the chamber is shared: an earlier episode's hit is not this one's
    started = time.monotonic()

    try:
        runs = EPISODES[episode](chamber, episode, rng)
    finally:
        # Nothing but the long-lived reader may still be running on the file. Stop the rest either
        # way: the chamber is shared, so a failed episode's writers must not fail every later one.
        stragglers = [n for n, _p in chamber.live() if n != chamber.reader_name]
        for name in stragglers:
            with contextlib.suppress(AssertionError):
                chamber.stop(name, deadline=30.0)
    assert not stragglers, f"{ctx} roles still running: {stragglers}"
    problems: list[str] = []
    problems += [f"{n}: {e.get('error')}\n{e.get('tb', '')}" for n, e in chamber.errors()]
    problems += [f"{name} (pid {pid}) held {link}" for name, pid, link in chamber.deleted_hits_snapshot(hits_mark)]
    rows = integrity_rows(chamber.db)
    if rows != ["ok"]:
        problems.append(f"integrity_check: {rows[:5]}")
    problems += journal_mode_problems(chamber)
    problems += exactly_once_problems(chamber)
    after = counts(chamber.db)
    for sid, n in before.items():
        if after.get(sid, 0) < n:
            problems.append(f"canonical rows went down for {sid}: {n} -> {after.get(sid, 0)}")
    episode_acks = acked_tokens(chamber, runs)
    if not episode_acks:
        problems.append("episode acknowledged no appends")
    problems += fts_problems(chamber.db, sample(rng, episode_acks, 12))
    problems += _reader_fd_problems(chamber)
    problems += _churn_fd_problems(chamber, episode)
    busy = busy_summary(chamber, episode)
    assert not problems, f"{ctx} ({time.monotonic() - started:.1f}s, busy waits {busy})\n" + "\n".join(problems)


def test_long_lived_reader_saw_every_episode_grow_only(chamber):
    """The dashboard reader lives across episodes: it never saw a session shrink and never errored."""
    wait_for(lambda: any(e.get("event") == "stats" for e in chamber.events(chamber.reader_name)),
             what="reader stats")
    errors = [(n, e) for n, e in chamber.errors() if n.startswith("reader")]
    assert not errors, errors
    assert not _reader_fd_problems(chamber)
