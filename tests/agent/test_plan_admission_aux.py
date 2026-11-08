"""Auxiliary/MoA seam gating for agent.plan_admission (rebase follow-up commit).

The main-transport gate (tests/agent/test_plan_admission.py) proves the two
chat-completion choke points. This module proves the auxiliary and MoA seams
hold the SAME host-wide slot: an async admission twin that waits cooperatively
(never blocking the event loop), fail-closed bounded waits, cross-thread
serialisation behind a sync holder, same-thread re-entry, and wiring coverage
for every aux/MoA call site. Invented data only; no network, no real provider
calls.
"""
import asyncio
import json
import os
import subprocess
import sys
import textwrap
import threading
import types
import time
from pathlib import Path

import pytest

from agent import plan_admission as pa

PLAN_A = "https://api.z.ai/api/coding/paas/v4"
NON_PLAN = "https://auxiliary-provider.example/v1"

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def admission_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "_admission_dir", lambda: str(tmp_path / "admission"))
    yield str(tmp_path / "admission")


HOLDER = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {root!r})
    from agent import plan_admission as pa
    pa._admission_dir = lambda: {adir!r}
    with pa.plan_admission({base_url!r}, model="glm-5.3", purpose="test_holder", wait_seconds={wait}):
        print("HELD", flush=True)
        time.sleep({hold})
    """
)


def _spawn_holder(adir, base_url=PLAN_A, hold=1.2, wait=5.0):
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLDER.format(root=str(ROOT), adir=adir, base_url=base_url, hold=hold, wait=wait)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    line = proc.stdout.readline().strip()
    assert line == "HELD", f"holder failed to start: {{line!r}}".format(line=line)
    return proc


def _log_rows(adir):
    path = Path(adir) / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------- async basics


def test_async_admission_acquires_when_free(admission_dir):
    async def run():
        async with pa.plan_admission_async(PLAN_A, model="glm-5.3", purpose="test", wait_seconds=2) as token:
            assert token is None  # acquired: inside the hold
            rows = [r for r in _log_rows(admission_dir) if r["event"] == "acquire"]
            assert rows and rows[-1]["purpose"] == "test"

    asyncio.run(run())


def test_async_admission_noop_for_non_plan_route(admission_dir):
    async def run():
        started = time.monotonic()
        async with pa.plan_admission_async(NON_PLAN, model="m", purpose="test", wait_seconds=5):
            pass
        assert time.monotonic() - started < 0.5
        assert not _log_rows(admission_dir)  # never touches the lock or the log

    asyncio.run(run())


def test_async_admission_fail_closed_while_held(admission_dir):
    holder = _spawn_holder(admission_dir, hold=3.0)

    async def run():
        with pytest.raises(pa.PlanAdmissionBusy):
            async with pa.plan_admission_async(PLAN_A, model="glm-5.3", purpose="contender", wait_seconds=0.8):
                pytest.fail("contender acquired while a holder owns the slot")

    try:
        asyncio.run(run())
    finally:
        holder.kill()
        holder.wait()
    busy = [r for r in _log_rows(admission_dir) if r["event"] == "busy"]
    assert busy and busy[-1]["purpose"] == "contender"


def test_async_admission_waits_then_acquires_after_release(admission_dir):
    holder = _spawn_holder(admission_dir, hold=1.2)

    async def run():
        t0 = time.monotonic()
        async with pa.plan_admission_async(PLAN_A, model="glm-5.3", purpose="late", wait_seconds=6):
            elapsed = time.monotonic() - t0
        return elapsed

    try:
        elapsed = asyncio.run(run())
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait()
    assert elapsed >= 1.0, "contender acquired before the holder released"
    acquire = [r for r in _log_rows(admission_dir) if r["event"] == "acquire" and r["purpose"] == "late"]
    assert acquire and acquire[-1]["wait_s"] >= 1.0


def test_event_loop_stays_responsive_while_contending(admission_dir):
    """The whole point of the async twin: waiting must not freeze the loop."""
    holder = _spawn_holder(admission_dir, hold=1.5)

    async def run():
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        beat = asyncio.ensure_future(heartbeat())
        t0 = time.monotonic()
        async with pa.plan_admission_async(PLAN_A, model="glm-5.3", purpose="responsive", wait_seconds=10):
            elapsed = time.monotonic() - t0
        beat.cancel()
        try:
            await beat
        except asyncio.CancelledError:
            pass
        return ticks, elapsed

    try:
        ticks, elapsed = asyncio.run(run())
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait()
    assert elapsed >= 1.0
    assert ticks >= 10, f"event loop starved during contended wait (only {{ticks}} ticks)".format(ticks=ticks)


# ------------------------------------------------------- cross-thread + re-entry


def test_async_contender_serialises_behind_sync_holder_same_process(admission_dir):
    """A sync hold in a worker thread must gate an async contender on the loop."""
    release = threading.Event()

    def sync_holder():
        with pa.plan_admission(PLAN_A, model="glm-5.3", purpose="sync_holder"):
            release.wait(5.0)

    worker = threading.Thread(target=sync_holder)
    worker.start()
    time.sleep(0.3)  # let the worker take the slot

    async def run():
        async def contender():
            async with pa.plan_admission_async(PLAN_A, model="glm-5.3", purpose="async_contender", wait_seconds=5):
                return time.monotonic()

        task = asyncio.ensure_future(contender())
        await asyncio.sleep(0.6)  # contender is now polling behind the worker's hold
        assert not task.done(), "async contender must not acquire while the sync holder owns the slot"
        release.set()
        acquired_at = await task
        return acquired_at

    try:
        acquired_at = asyncio.run(run())
    finally:
        release.set()
        worker.join(6)
    assert time.monotonic() - acquired_at < 2.0
    purposes = [r["purpose"] for r in _log_rows(admission_dir) if r["event"] == "acquire"]
    assert purposes == ["sync_holder", "async_contender"]  # strict order: holder first


def test_sync_and_async_share_thread_local_reentry(admission_dir):
    async def run():
        with pa.plan_admission(PLAN_A, model="glm-5.3", purpose="outer_sync"):
            t0 = time.monotonic()
            async with pa.plan_admission_async(PLAN_A, model="glm-5.3", purpose="inner_async", wait_seconds=2):
                inner_elapsed = time.monotonic() - t0
        return inner_elapsed

    inner_elapsed = asyncio.run(run())
    assert inner_elapsed < 0.3, "same-thread re-entry must reuse the outer hold, not re-lock"
    acquires = [r for r in _log_rows(admission_dir) if r["event"] == "acquire"]
    assert len(acquires) == 1 and acquires[0]["purpose"] == "outer_sync"


# ---------------------------------------------------------------------- wiring


def test_wiring_covers_aux_and_moa_seams():
    aux_src = (ROOT / "agent" / "auxiliary_client.py").read_text(encoding="utf-8")
    helpers_src = (ROOT / "agent" / "chat_completion_helpers.py").read_text(encoding="utf-8")
    gate_src = (ROOT / "agent" / "plan_admission.py").read_text(encoding="utf-8")

    # The async twin exists with the same fail-closed contract.
    assert "async def plan_admission_async" in gate_src
    assert "PlanAdmissionBusy" in gate_src.split("plan_admission_async")[-1]

    # Every ungated aux entry point is wrapped by a gated namesake.
    for gated, ungated, purpose in [
        ("def _relay_sync_stream(", "_relay_sync_stream_ungated", "aux_relay"),
        ("def _create_with_progress(", "_create_with_progress_ungated", "aux_progress"),
        ("async def _acreate_with_stream(", "_acreate_with_stream_ungated", "aux_async_stream"),
        ("async def _acreate_with_progress(", "_acreate_with_progress_ungated", "aux_async_progress"),
    ]:
        assert gated in aux_src, gated
        assert ungated in aux_src, ungated
        assert purpose in aux_src, purpose
        # The gated name appears BEFORE its ungated twin (wrapper first, body
        # after), and the wrapper's forwarding call sits BETWEEN them.
        wrapper_line = aux_src[: aux_src.index(gated)].count("\n") + 1
        body_line = aux_src[: aux_src.index("def " + ungated)].count("\n") + 1
        assert wrapper_line < body_line, gated
        call_refs = [i for i, line in enumerate(aux_src.splitlines()) if (ungated + "(") in line and "def " not in line]
        assert call_refs, ungated + " never forwarded"
        assert any(wrapper_line < idx < body_line for idx in call_refs), ungated + " forwarded outside its wrapper"

    # The MoA facade return and the MoA/Codex stream branch are gated too.
    assert "chat_moa_facade" in helpers_src
    assert "aux_moa_codex" in aux_src

    # No leftover direct aux entry: the ungated bodies are only referenced by
    # their wrappers inside this module.
    for ungated in ("_relay_sync_stream_ungated", "_create_with_progress_ungated",
                    "_acreate_with_stream_ungated", "_acreate_with_progress_ungated"):
        module_refs = [line for line in aux_src.splitlines() if ungated in line and "def " not in line]
        call_refs = [line for line in module_refs if (ungated + "(") in line]
        assert call_refs, ungated + " never forwarded"


# ------------------------------------------------- aux_relay consumption window
# Review pass 2 (probe P7) proved the relay wrapper released the slot when it
# RETURNED the stream, before the caller consumed it. These tests pin the fix:
# the slot stays held across every chunk and frees only on
# exhaustion / close / error / complete-object.

class _LazyPlanStream:
    def __init__(self, probe=None, fail_on=None):
        self._probe = probe
        self._fail_on = fail_on
        self._i = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= 3:
            raise StopIteration
        self._i += 1
        if self._probe is not None:
            self._probe(self._i)
        if self._fail_on is not None and self._i == self._fail_on:
            raise RuntimeError("mid-stream failure")
        return f"chunk{self._i}"

    def close(self):
        self.closed = True


def _lock_probe(adir):
    """Return a probe that records whether the host slot is FREE at call time."""
    import fcntl

    def probe(_i):
        fd = os.open(pa.lock_path(), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return "free"
        except OSError:
            return "held"
        finally:
            os.close(fd)
    return probe


def _relay_with_stub_stream(admission_dir, monkeypatch, stream):
    import agent.auxiliary_client as ac

    monkeypatch.setattr(
        ac, "_relay_sync_stream_ungated",
        lambda client, kwargs, provider=None, api_mode=None: stream,
    )
    client = types.SimpleNamespace(base_url=PLAN_A)
    return ac._relay_sync_stream(client, {"model": "glm-5.3"})


def test_aux_relay_holds_slot_across_stream_consumption(admission_dir, monkeypatch):
    import agent.auxiliary_client as ac  # noqa: F401
    probe = _lock_probe(admission_dir)
    states = []
    stream = _LazyPlanStream(probe=lambda i: states.append(probe(i)))
    out = _relay_with_stub_stream(admission_dir, monkeypatch, stream)
    chunks = list(out)
    assert chunks == ["chunk1", "chunk2", "chunk3"]
    assert states == ["held", "held", "held"], f"slot freed mid-consume: {states}"
    # After exhaustion the slot is free again for the next process.
    assert probe(0) == "free"


def test_aux_relay_complete_object_releases_before_return(admission_dir, monkeypatch):
    import agent.auxiliary_client as ac  # noqa: F401
    complete = types.SimpleNamespace(choices=[])
    out = _relay_with_stub_stream(admission_dir, monkeypatch, complete)
    assert out is complete  # raw object back, not a proxy
    assert _lock_probe(admission_dir)(0) == "free"


def test_aux_relay_close_releases_and_delegates(admission_dir, monkeypatch):
    import agent.auxiliary_client as ac  # noqa: F401
    stream = _LazyPlanStream()
    out = _relay_with_stub_stream(admission_dir, monkeypatch, stream)
    assert not stream.closed
    out.close()
    assert stream.closed  # delegated to the inner stream
    assert _lock_probe(admission_dir)(0) == "free"


def test_aux_relay_error_mid_consume_releases(admission_dir, monkeypatch):
    import agent.auxiliary_client as ac  # noqa: F401
    probe = _lock_probe(admission_dir)
    states = []
    stream = _LazyPlanStream(probe=lambda i: states.append(probe(i)), fail_on=2)
    out = _relay_with_stub_stream(admission_dir, monkeypatch, stream)
    with pytest.raises(RuntimeError):
        list(out)
    assert states[0] == "held"
    assert _lock_probe(admission_dir)(0) == "free"  # released despite the error


# ------------------------------------------------- stranger-thread release
# Review pass 3 (probe P11/P14, HIGH): releasing the gate from a thread other
# than the opener left the opener's thread-local depth stuck >0, so every
# later plan-route call on the opener thread silently skipped the flock. The
# registry-based nesting makes release thread-agnostic; this test pins it.

def test_stranger_thread_close_does_not_poison_opener(admission_dir, monkeypatch):
    import fcntl

    import agent.auxiliary_client as ac  # noqa: F401

    stream = _LazyPlanStream()
    out = _relay_with_stub_stream(admission_dir, monkeypatch, stream)
    closed_from_stranger = threading.Event()

    def stranger():
        out.close()  # monitor-style close from a non-opener thread
        closed_from_stranger.set()

    worker = threading.Thread(target=stranger)
    worker.start()
    worker.join(5)
    assert closed_from_stranger.is_set() and stream.closed
    assert _lock_probe(admission_dir)(0) == "free"  # slot genuinely released

    # The opener thread's NEXT gated call must truly acquire the flock again:
    # while it holds, a second fd must be refused (a depth-poisoned thread
    # would pass through with no lock and this probe would see "free").
    probe_states = []

    def opener_next():
        with pa.plan_admission(PLAN_A, model="glm-5.3", purpose="opener_next", wait_seconds=3):
            fd = os.open(pa.lock_path(), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                probe_states.append("free")
            except OSError:
                probe_states.append("held")
            finally:
                os.close(fd)

    opener_next()
    assert probe_states == ["held"], f"opener thread not re-acquiring: {probe_states}"

    # And the registry must be clean afterwards (no phantom holds).
    assert pa._nest_depth(threading.get_ident()) == 0
