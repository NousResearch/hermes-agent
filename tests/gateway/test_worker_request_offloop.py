"""The claim/verify legs of worker_request must not run on the gateway event loop.

A synchronous psutil probe or SQLite read inside _claim/_verify stalls the loop while
several workers save at once; the worker misses its 20 s worker.persist reply window and
exits, which is the "Kanban execution unknown; retained by owner" crash (#145022).
"""

import asyncio
import threading

import pytest

from gateway import session_authorities
from gateway import session_worker


class _FakeActor:
    profile_id = "profile"
    subject = "worker-subject"
    capabilities = frozenset({"worker:adopt"})


class _FakeAuthority:
    profile_id = "profile"
    epoch = 7
    db = object()

    def _require_admission_open(self):
        pass

    def authorize(self, actor, ref, capability):
        pass


class _FakeConnection:
    authority = _FakeAuthority()
    actor = _FakeActor()


def _params(operation):
    params = {key: "x" for key in session_worker._SCOPE}
    params.update(
        profile_id="profile",
        session_id="sid",
        execution_id="exec",
        generation=1,
        pid=1,
        birth=1,
        secret="s" * 32,
    )
    extra = {
        "register": {"kind": "compute"},
        "adopt": {},
        "persist": {
            "epoch": 1,
            "sequence": 1,
            "operation": "execution.finish",
            "payload": {},
        },
    }
    params.update(extra[operation])
    return params


@pytest.mark.asyncio
async def test_claim_and_verify_run_off_the_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    seen = {}

    def record(name):
        def recorded(*args, **kwargs):
            seen[name] = threading.get_ident()
            return f"{name}-result"

        return recorded

    monkeypatch.setattr(session_worker, "_claim", record("claim"))
    monkeypatch.setattr(session_worker, "_verify", record("verify"))
    monkeypatch.setattr(
        session_worker, "register_worker_execution", lambda *a, **k: "registered"
    )
    monkeypatch.setattr(
        session_worker, "adopt_worker_execution", lambda *a, **k: "adopted"
    )
    monkeypatch.setattr(session_authorities, "require_serving", lambda authority: None)

    connection = _FakeConnection()

    async def drive(operation):
        return await session_worker.worker_request(
            connection, "ref", _params(operation), operation=operation
        )

    register_result, adopt_result = await asyncio.gather(
        drive("register"), drive("adopt")
    )
    assert register_result == "registered"
    assert adopt_result == "adopted"
    assert seen["claim"] != loop_thread, "_claim blocked the event loop"
    assert seen["verify"] != loop_thread, "_verify blocked the event loop"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["child.reserve", "delegation.read"])
@pytest.mark.parametrize("blocked_leg", ["claim", "verify"])
async def test_delegation_keeps_loop_responsive_during_worker_authentication(
    monkeypatch, operation, blocked_leg
):
    from gateway import session_worker_child, session_worker_delegation

    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    entered = asyncio.Event()
    release = threading.Event()

    def leg(name):
        def run(*args):
            if name == blocked_leg:
                assert threading.get_ident() != loop_thread, f"{name} blocked the event loop"
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(30), f"{name} was not released"
            return "claim"
        return run

    monkeypatch.setattr(session_worker, "_claim", leg("claim"))
    monkeypatch.setattr(session_worker, "_verify", leg("verify"))
    monkeypatch.setattr(session_authorities, "require_serving", lambda authority: None)
    monkeypatch.setattr(session_worker_child, "delegation_limits", lambda *args: (1, 1))
    monkeypatch.setattr(session_worker_child, "apply_worker_receipt", lambda *args, **kwargs: "saved")
    monkeypatch.setattr(session_worker_delegation, "persist_worker_delegation", lambda *args, **kwargs: "saved")
    params = _params("persist") | {"operation": operation}
    request = asyncio.create_task(session_worker.worker_request(_FakeConnection(), "ref", params, operation="persist"))
    entered_task = asyncio.create_task(entered.wait())
    done = set()
    try:
        done, _ = await asyncio.wait({entered_task, request}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
        if request in done:
            await request
        assert entered_task in done, "authentication did not start"
    finally:
        release.set()
        entered_task.cancel()
        await asyncio.gather(entered_task, return_exceptions=True)
        if entered_task not in done:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
    assert await asyncio.wait_for(request, timeout=10) == "saved"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["child.reserve", "delegation.read"])
async def test_retirement_during_verification_refuses_worker_write(monkeypatch, operation):
    from types import SimpleNamespace
    from hermes_state_runtime import RuntimeStoreError

    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    entered = asyncio.Event()
    release = threading.Event()
    authority = SimpleNamespace(profile_id="profile", retired=False)
    connection = SimpleNamespace(authority=authority, actor=_FakeActor())

    def verify(*args):
        assert threading.get_ident() != loop_thread, "verify blocked the event loop"
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(30), "verify was not released"

    monkeypatch.setattr(session_worker, "_claim", lambda *args: "claim")
    monkeypatch.setattr(session_worker, "_verify", verify)
    params = _params("persist") | {"operation": operation}
    request = asyncio.create_task(session_worker.worker_request(connection, "ref", params, operation="persist"))
    entered_task = asyncio.create_task(entered.wait())
    done = set()
    try:
        done, _ = await asyncio.wait({entered_task, request}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
        if request in done:
            await request
        assert entered_task in done, "verification did not start"
        authority.retired = True
    finally:
        release.set()
        entered_task.cancel()
        await asyncio.gather(entered_task, return_exceptions=True)
        if entered_task not in done:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
    with pytest.raises(RuntimeStoreError, match="runtime_retired"):
        await asyncio.wait_for(request, timeout=10)
