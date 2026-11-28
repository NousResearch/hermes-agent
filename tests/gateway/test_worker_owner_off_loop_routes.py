"""Regression for #145022: the owner must not run a worker's _claim/_verify on its event loop.

While one worker's claim (or verify) is blocked in the database, a second worker's worker.persist must still be
answered. Event-based, no sleeps: request A's _claim/_verify waits (up to 5 s) for request B to finish. With
_claim/_verify on the loop, B cannot run until A gives up, so the wait times out. Covers every route that calls
_claim/_verify: register, adopt, persist, child.reserve and delegation.*. Also: a retirement during the
off-loop leg refuses with runtime_retired before any write.
"""
import asyncio
import dataclasses
import threading
import types

import pytest

from gateway import session_persistence, session_worker, session_worker_child, session_worker_delegation
from persistence.core.errors import AuthorityRefusal

ADOPT = frozenset({"worker:adopt"})

class _Authority:
    profile_id, epoch, runner = "p", 7, None

    def __init__(self):
        self.retired, self.sessions = False, {}

    def _require_admission_open(self):
        pass

    def authorize(self, actor, ref, verb):
        if verb not in actor.capabilities:
            raise AuthorityRefusal("permission_denied")

    def wake_after_worker(self, session_id):
        pass

    def logical_owner(self, sid):
        return sid

def _connection():
    return types.SimpleNamespace(authority=_Authority(),
                                 actor=types.SimpleNamespace(profile_id="p", subject="w", capabilities=ADOPT))

@dataclasses.dataclass
class _Row:
    execution_id: str

class _Store:
    def __init__(self):
        self.writes = []
        writes = self.writes

        class Workers:
            def persist_operation(self, **kw):
                writes.append(kw["execution_id"])
                return {"value": kw["execution_id"]}

            def reserve_child(self, **kw):
                writes.append(kw["execution_id"])
                return types.SimpleNamespace(child=None, refused="caps")

        class Aio:
            async def adopt(self, **kw):
                writes.append(kw["execution_id"])
                return _Row(kw["execution_id"])

            async def register(self, **kw):
                writes.append(kw["execution_id"])
                return _Row(kw["execution_id"])

        self.worker_executions, self.aio = Workers(), types.SimpleNamespace(worker_executions=Aio())

def _params(route, execution_id):
    scope = {"profile_id": "p", "session_id": "s", "execution_id": execution_id, "generation": 1,
             "pid": 4242, "birth": 1.0, "secret": "s" * 32}
    receipt = {"epoch": 7, "sequence": 1, "payload": {}}
    return {"register": ("register", scope | {"kind": "compute"}),
            "adopt": ("adopt", scope),
            "persist": ("persist", scope | receipt | {"operation": "execution.finish"}),
            "child": ("persist", scope | receipt | {"operation": "child.reserve"}),
            "delegation": ("persist", scope | receipt | {"operation": "delegation.dispatch"})}[route]

ROUTES = ["register", "adopt", "persist", "child", "delegation"]

@pytest.fixture
def store(monkeypatch):
    st = _Store()
    for module in (session_worker, session_worker_child, session_persistence):
        monkeypatch.setattr(module, "authority_persistence", lambda authority: st)
    monkeypatch.setattr(session_worker_child, "delegation_limits", lambda *a, **k: (1, 1))
    monkeypatch.setattr(session_worker_delegation, "persist_worker_delegation",
                        lambda _s, **kw: st.writes.append(kw["execution_id"]) or {"value": None})
    return st

async def _request(route, connection, execution_id):
    operation, params = _params(route, execution_id)
    return await session_worker.worker_request(connection, "ref", params, operation=operation)

@pytest.mark.asyncio
@pytest.mark.parametrize("leg", ["claim", "verify"])
@pytest.mark.parametrize("route", ROUTES)
async def test_blocked_claim_or_verify_does_not_stall_other_workers(route, leg, store, monkeypatch):
    if route == "register" and leg == "verify":
        pytest.skip("a registration is never verified")
    other_answered = threading.Event()
    waited = {}

    def blocking(name):
        def fn(connection, ref, params, *claim):
            if name == leg and params["execution_id"] == "A":
                waited["A"] = other_answered.wait(5)  # stands in for a slow state.db read
            return "claim" if name == "claim" else None
        return fn

    monkeypatch.setattr(session_worker, "_claim", blocking("claim"))
    monkeypatch.setattr(session_worker, "_verify", blocking("verify"))
    first = asyncio.ensure_future(_request(route, _connection(), "A"))
    second = asyncio.ensure_future(_request("persist", _connection(), "B"))
    await second
    other_answered.set()
    await first
    assert waited["A"], f"{route}: worker B was not answered while worker A's _{leg} was blocked"

@pytest.mark.asyncio
@pytest.mark.parametrize("leg", ["claim", "verify"])
@pytest.mark.parametrize("route", ROUTES)
async def test_retirement_during_off_loop_leg_refuses_before_writing(route, leg, store, monkeypatch):
    if route == "register" and leg == "verify":
        pytest.skip("a registration is never verified")
    connection = _connection()

    def retiring(name):
        def fn(*args):
            if name == leg:
                connection.authority.retired = True
            return "claim" if name == "claim" else None
        return fn

    monkeypatch.setattr(session_worker, "_claim", retiring("claim"))
    monkeypatch.setattr(session_worker, "_verify", retiring("verify"))
    with pytest.raises(AuthorityRefusal) as refused:
        await _request(route, connection, "A")
    assert refused.value.reason == "runtime_retired"
    assert store.writes == []
