"""The claim/verify legs of worker_request must not run on the gateway event loop.

A synchronous psutil probe or SQLite read inside _claim/_verify stalls the loop while
several workers save at once; the worker misses its 20 s worker.persist reply window and
exits, which is the "Kanban execution unknown; retained by owner" crash (#145022).
"""

import asyncio
import dataclasses
import threading
from types import SimpleNamespace

import pytest

from gateway import session_worker
from persistence.core.errors import AuthorityRefusal


class _FakeActor:
    profile_id = "profile"
    subject = "worker-subject"
    capabilities = frozenset({"worker:adopt"})


class _FakeAuthority:
    profile_id = "profile"
    epoch = 7

    def _require_admission_open(self):
        pass

    def authorize(self, actor, ref, capability):
        pass

    def wake_after_worker(self, session_id):
        pass


class _FakeConnection:
    authority = _FakeAuthority()
    actor = _FakeActor()


@dataclasses.dataclass
class _Row:
    execution_id: str
    marker: str


class _Store:
    """Stand-in for the profile's persistence handle: records every write the request makes."""

    def __init__(self):
        self.writes = []

        writes = self.writes

        class Workers:
            def persist_operation(self, **kwargs):
                writes.append("persist")
                return {"value": kwargs["execution_id"]}

            def reserve_child(self, **kwargs):
                writes.append("child")
                return SimpleNamespace(child=None, refused="caps")

        class Aio:
            async def register(self, **kwargs):
                writes.append("register")
                return _Row(kwargs["execution_id"], "registered")

            async def adopt(self, **kwargs):
                writes.append("adopt")
                return _Row(kwargs["execution_id"], "adopted")

        self.worker_executions, self.aio = Workers(), SimpleNamespace(worker_executions=Aio())


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


def _patch_store(monkeypatch, store):
    """Every worker route resolves its handle through a module-level or function-level
    ``authority_persistence`` import that reads the attribute at call time."""
    from gateway import session_persistence, session_worker_child
    for module in (session_worker, session_worker_child, session_persistence):
        monkeypatch.setattr(module, "authority_persistence", lambda authority: store)


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
    _patch_store(monkeypatch, _Store())

    connection = _FakeConnection()

    async def drive(operation):
        return await session_worker.worker_request(
            connection, "ref", _params(operation), operation=operation
        )

    register_result, adopt_result = await asyncio.gather(
        drive("register"), drive("adopt")
    )
    assert register_result["marker"] == "registered"
    assert adopt_result["marker"] == "adopted"
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
    _patch_store(monkeypatch, _Store())
    monkeypatch.setattr(session_worker_child, "delegation_limits", lambda *args: (1, 1))
    monkeypatch.setattr(session_worker_delegation, "persist_worker_delegation", lambda *args, **kwargs: "saved")
    expected = {"child.reserve": {"refused": "caps"}, "delegation.read": "saved"}[operation]
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
    assert await asyncio.wait_for(request, timeout=10) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["child.reserve", "delegation.read"])
async def test_retirement_during_verification_refuses_worker_write(monkeypatch, operation):
    from gateway import session_worker_child, session_worker_delegation

    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    entered = asyncio.Event()
    release = threading.Event()
    authority = SimpleNamespace(profile_id="profile", retired=False)
    connection = SimpleNamespace(authority=authority, actor=_FakeActor())
    store = _Store()

    def verify(*args):
        assert threading.get_ident() != loop_thread, "verify blocked the event loop"
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(30), "verify was not released"

    monkeypatch.setattr(session_worker, "_claim", lambda *args: "claim")
    monkeypatch.setattr(session_worker, "_verify", verify)
    _patch_store(monkeypatch, store)
    monkeypatch.setattr(session_worker_child, "delegation_limits", lambda *args: (1, 1))
    monkeypatch.setattr(session_worker_delegation, "persist_worker_delegation", lambda *args, **kwargs: "saved")
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
    with pytest.raises(AuthorityRefusal, match="runtime_retired"):
        await asyncio.wait_for(request, timeout=10)
    assert store.writes == [], f"retired authority accepted a {operation} write: {store.writes}"


class _RetirableAuthority:
    profile_id = "profile"
    epoch = 7

    def __init__(self):
        self.retired = False

    def _require_admission_open(self):
        pass

    def authorize(self, actor, ref, capability):
        pass

    def wake_after_worker(self, session_id):
        pass


async def _drive_until_refused(monkeypatch, operation, blocked_leg):
    """Park one off-loop leg, retire the authority while it is parked, and return the
    request task plus the write-recording store so the caller can assert refusal."""
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    entered = asyncio.Event()
    release = threading.Event()
    authority = _RetirableAuthority()
    connection = SimpleNamespace(authority=authority, actor=_FakeActor())

    def leg(name):
        def run(*args):
            assert threading.get_ident() != loop_thread, f"{name} blocked the event loop"
            if name == blocked_leg:
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(30), f"{name} was not released"
            return "claim"
        return run

    monkeypatch.setattr(session_worker, "_claim", leg("claim"))
    monkeypatch.setattr(session_worker, "_verify", leg("verify"))
    store = _Store()
    _patch_store(monkeypatch, store)

    request = asyncio.create_task(session_worker.worker_request(
        connection, "ref", _params(operation), operation=operation))
    entered_task = asyncio.create_task(entered.wait())
    done = set()
    try:
        done, _ = await asyncio.wait({entered_task, request}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
        if request in done:
            await request
        assert entered_task in done, f"{blocked_leg} did not start"
        authority.retired = True
    finally:
        release.set()
        entered_task.cancel()
        await asyncio.gather(entered_task, return_exceptions=True)
        if entered_task not in done:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
    return request, store


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["adopt", "persist"])
@pytest.mark.parametrize("blocked_leg", ["claim", "verify"])
async def test_retirement_during_authentication_refuses_ordinary_worker_write(
    monkeypatch, operation, blocked_leg
):
    """Moving _claim/_verify off-loop added yield points between the serving check at the
    top of worker_request and the adopt/persist writes; a retirement while either leg is
    parked must still refuse the write, the way the child/delegation routes already do."""
    request, store = await _drive_until_refused(monkeypatch, operation, blocked_leg)
    with pytest.raises(AuthorityRefusal, match="runtime_retired"):
        await asyncio.wait_for(request, timeout=10)
    assert store.writes == [], f"retired authority accepted a {operation} write: {store.writes}"


@pytest.mark.asyncio
async def test_retirement_during_claim_refuses_register_write(monkeypatch):
    """The register leg writes after only the claim await; same fence, same refusal."""
    request, store = await _drive_until_refused(monkeypatch, "register", "claim")
    with pytest.raises(AuthorityRefusal, match="runtime_retired"):
        await asyncio.wait_for(request, timeout=10)
    assert store.writes == [], f"retired authority accepted a register write: {store.writes}"
