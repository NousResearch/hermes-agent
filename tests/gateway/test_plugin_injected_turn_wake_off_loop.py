"""The arrival signal that ends a plugin turn's background wait (``SessionAuthority._schedule``) runs
on the gateway loop: its ledger read must run in a worker, never block the loop, and still end the
wait; called from a plain thread it ends the wait before returning."""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until


def _person_queued():
    from persistence.authority.admissions import AdmissionRecord
    return AdmissionRecord(seq=1, admission_id="a1", request_id="r1", principal_id="messaging:42",
                           target_session_id="sid", payload={}, intent="queue", status="queued", outcome=None,
                           owner_epoch=None, generation=None)


def _waiting_session(monkeypatch):
    from gateway.session_contract import SessionRef

    readers = []

    def list_admissions(*, session_id, pending_only=True):
        readers.append(threading.current_thread())
        return [_person_queued()]

    live = SimpleNamespace(background_wait=threading.Event())
    authority = SimpleNamespace(sessions={"sid": live},
                                persistence=SimpleNamespace(admissions=SimpleNamespace(list=list_admissions)))
    return authority, SessionRef("default", "sid"), live.background_wait, readers


@pytest.mark.asyncio
async def test_the_wake_on_the_loop_reads_the_ledger_off_the_loop(monkeypatch):
    from gateway.run_plugin_injection import wake_background_wait

    authority, ref, wait, readers = _waiting_session(monkeypatch)
    wake_background_wait(authority, ref)
    await async_wait_until(wait.is_set, timeout=5, what="the wait ended")

    assert readers and threading.current_thread() not in readers


def test_the_wake_from_a_plain_thread_ends_the_wait_before_returning(monkeypatch):
    from gateway.run_plugin_injection import wake_background_wait

    authority, ref, wait, _readers = _waiting_session(monkeypatch)
    worker = threading.Thread(target=wake_background_wait, args=(authority, ref))
    worker.start()
    worker.join(5)

    assert wait.is_set()
