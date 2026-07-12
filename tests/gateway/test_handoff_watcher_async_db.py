"""Regression test for #40695 (salvage of keystone PR #40782).

The Discord gateway heartbeat was stalling because the handoff watcher
(``GatewayRunner._handoff_watcher``) polled the synchronous, blocking
SQLite-backed ``SessionDB`` directly on the asyncio event loop every 2s
('Shard ID None heartbeat blocked for more than N seconds').

The fix routes every blocking ``SessionDB`` call in the watcher through the
``AsyncSessionDB`` facade, which offloads each call via ``asyncio.to_thread`` so
the SQLite I/O runs on a worker thread and never blocks the event loop / Discord
heartbeat.

These tests assert that behaviour contract. They are mutation-survivable:
reverting any ``await self._session_db.<call>(...)`` back to a direct synchronous
call on the loop makes the relevant assertion fail.
"""

import asyncio
import types

import pytest

import gateway.run as run


class _RecordingSessionDB:
    """SessionDB stand-in that records the thread each method runs on.

    If the watcher calls these methods directly on the event loop (the bug),
    they run on the loop thread. If they are wrapped in ``asyncio.to_thread``
    (the fix), they run on a *different* worker thread.
    """

    def __init__(self, loop_thread_ident):
        self._loop_thread_ident = loop_thread_ident
        self.threads = {}
        self.calls = []

    def _record(self, name):
        import threading

        self.threads.setdefault(name, []).append(threading.get_ident())
        self.calls.append(name)

    def ran_off_loop(self, name):
        """True iff every call to ``name`` ran on a non-loop thread."""
        idents = self.threads.get(name, [])
        return bool(idents) and all(i != self._loop_thread_ident for i in idents)

    def list_pending_handoffs(self):
        self._record("list_pending_handoffs")
        return [{"id": "sess-1"}]

    def claim_handoff(self, session_id):
        self._record("claim_handoff")
        return True

    def complete_handoff(self, session_id):
        self._record("complete_handoff")

    def fail_handoff(self, session_id, error):
        self._record("fail_handoff")


def _make_fake_runner(session_db, *, fail_process=False):
    """Build a minimal object that exposes exactly what the loop body touches.

    The watcher now talks to the SessionDB through the AsyncSessionDB facade,
    so wrap the recording stand-in the same way the gateway does.
    """
    from hermes_state import AsyncSessionDB

    fake = types.SimpleNamespace()
    fake._session_db = AsyncSessionDB(session_db)
    # _running yields True for the first loop check, then False so the loop
    # exits after a single tick.
    states = iter([True, False])

    class _Running:
        def __bool__(_self):
            try:
                return next(states)
            except StopIteration:
                return False

    fake._running = _Running()

    async def _process_handoff(row):
        if fail_process:
            raise RuntimeError("boom")

    fake._process_handoff = _process_handoff
    return fake


async def _run_one_tick(fake, monkeypatch):
    """Run the watcher for a single tick with sleeps neutralised."""

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(run.asyncio, "sleep", _no_sleep)
    # Bind the real (patched) method onto our minimal stand-in.
    coro = run.GatewayRunner._handoff_watcher(fake, interval=0.0)
    await asyncio.wait_for(coro, timeout=5)


@pytest.mark.asyncio
async def test_watcher_offloads_db_calls_to_threads(monkeypatch):
    """The success path must run list_pending/claim/complete off the loop."""
    import threading

    loop_ident = threading.get_ident()
    db = _RecordingSessionDB(loop_ident)
    fake = _make_fake_runner(db, fail_process=False)

    await _run_one_tick(fake, monkeypatch)

    # Sanity: the watcher actually exercised the calls this tick.
    assert "list_pending_handoffs" in db.calls
    assert "claim_handoff" in db.calls
    assert "complete_handoff" in db.calls

    # Contract: each blocking SessionDB call ran on a worker thread, NOT the
    # asyncio event-loop thread. Reverting a to_thread wrap makes the
    # corresponding call run on the loop thread and this fails.
    assert db.ran_off_loop("list_pending_handoffs")
    assert db.ran_off_loop("claim_handoff")
    assert db.ran_off_loop("complete_handoff")


@pytest.mark.asyncio
async def test_watcher_offloads_fail_handoff_to_thread(monkeypatch):
    """The error path must run fail_handoff off the loop too."""
    import threading

    loop_ident = threading.get_ident()
    db = _RecordingSessionDB(loop_ident)
    fake = _make_fake_runner(db, fail_process=True)

    await _run_one_tick(fake, monkeypatch)

    assert "fail_handoff" in db.calls
    assert db.ran_off_loop("fail_handoff")


@pytest.mark.asyncio
async def test_watcher_wraps_calls_via_asyncio_to_thread(monkeypatch):
    """Explicitly assert the offload goes through asyncio.to_thread.

    Patches the AsyncSessionDB facade's ``asyncio.to_thread`` (it lives in
    hermes_state) and records which SessionDB callables were handed to it.
    Mutation-survivable: dropping any await removes its callable from the set.
    """
    import hermes_state

    db = _RecordingSessionDB(loop_thread_ident=-1)
    fake = _make_fake_runner(db, fail_process=False)

    wrapped = []
    real_to_thread = hermes_state.asyncio.to_thread

    async def _spy_to_thread(func, *args, **kwargs):
        wrapped.append(getattr(func, "__name__", repr(func)))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(hermes_state.asyncio, "to_thread", _spy_to_thread)

    await _run_one_tick(fake, monkeypatch)

    assert "list_pending_handoffs" in wrapped
    assert "claim_handoff" in wrapped
    assert "complete_handoff" in wrapped


@pytest.mark.asyncio
async def test_writer_policy_disables_legacy_handoff_watcher(monkeypatch):
    import threading

    db = _RecordingSessionDB(threading.get_ident())
    fake = _make_fake_runner(db)
    monkeypatch.setattr(
        run,
        "_legacy_handoff_forbidden_by_writer_policy",
        lambda: True,
    )

    await run.GatewayRunner._handoff_watcher(fake, interval=0.0)

    assert db.calls == []


@pytest.mark.asyncio
async def test_writer_policy_denies_direct_legacy_handoff_processing(monkeypatch):
    monkeypatch.setattr(
        run,
        "_legacy_handoff_forbidden_by_writer_policy",
        lambda: True,
    )

    with pytest.raises(RuntimeError, match="legacy handoff is disabled"):
        await run.GatewayRunner._process_handoff(types.SimpleNamespace(), {})
