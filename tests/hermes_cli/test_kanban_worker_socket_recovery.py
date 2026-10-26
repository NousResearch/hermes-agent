"""A Kanban worker whose gateway socket dies mid-turn redials and re-reads the same receipt.

A dead socket is a transport event, not a turn outcome: the owner keeps the admission and answers
``prompt.receipt`` over the next connection, so exiting 1 there loses work that was already running
(``_watch_submitter`` then ends the owner's in-flight turn). Measured before the fix: 93 runs booked
``crashed`` on a gateway disconnect, 80 of them already admitted and bound, 0 with a
``worker_result``. An owner's own refusal is still an answer, and a genuine refusal still fails.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest

from hermes_cli import gateway_client, kanban_worker_client


class _Owner:
    """Scripted owner: ``connects`` holds one entry per dial (an exception fails the dial), and
    ``receipts`` holds one entry per ``prompt.receipt`` read (an exception is a transport loss).
    ``submit_failures`` holds entries consumed by ``kanban.run``."""

    def __init__(self, connects=None, receipts=None, submit_failures=None):
        self.connects = list(connects or [])
        self.receipts = list(receipts or [])
        self.submit_failures = list(submit_failures or [])
        self.calls = []
        self.dials = 0

    def connect(self):
        owner = self

        @asynccontextmanager
        async def _connect():
            owner.dials += 1
            step = owner.connects.pop(0) if owner.connects else None
            if isinstance(step, Exception):
                raise step

            class _Client:
                def __init__(self):
                    self.events = asyncio.Queue()

                async def rpc(self, method, **params):
                    owner.calls.append(method)
                    if method == "kanban.run":
                        failure = owner.submit_failures.pop(0) if owner.submit_failures else None
                        if failure is not None:
                            raise failure
                        return {"session_id": "s1", "receipt": {"status": "pending", "admission_id": "a1"}}
                    status = owner.receipts.pop(0) if owner.receipts else "terminal"
                    if isinstance(status, Exception):
                        raise status
                    return {"status": status}

            yield _Client()

        return _connect


def _disconnect():
    return gateway_client.GatewayClientError(
        "Gateway disconnected; turn outcome is unknown. Resume the printed session ID")


@pytest.fixture(autouse=True)
def _worker_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, "worker_result",
                        lambda *_: {"exit_code": 0, "last_output": "card closed"})
    return tmp_path


async def _no_sleep(_seconds):
    return None


@pytest.mark.asyncio
async def test_a_socket_lost_mid_turn_redials_and_settles_the_same_attempt(monkeypatch, _worker_env):
    owner = _Owner(connects=[None, None],
                   receipts=["pending", _disconnect(), "terminal"])
    monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())

    assert await kanban_worker_client.run({"task_id": "t1"}, str(_worker_env / "kanban.db")) == 0
    # The claim is submitted once: the redial re-reads the receipt, it does not resubmit.
    assert owner.calls.count("kanban.run") == 1
    assert owner.dials == 2


@pytest.mark.asyncio
async def test_a_draining_owner_is_retried_before_anything_is_admitted(monkeypatch, _worker_env):
    owner = _Owner(connects=[None, None],
                   submit_failures=[gateway_client.GatewayRPCError("runtime_draining")],
                   receipts=["terminal"])
    monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())

    assert await kanban_worker_client.run({"task_id": "t1"}, str(_worker_env / "kanban.db")) == 0
    assert owner.calls.count("kanban.run") == 2


@pytest.mark.asyncio
async def test_an_owner_refusal_is_not_a_redial(monkeypatch, _worker_env):
    owner = _Owner(connects=[None, None],
                   submit_failures=[gateway_client.GatewayRPCError("claim_mismatch")])
    monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())

    with pytest.raises(gateway_client.GatewayRPCError):
        await kanban_worker_client.run({"task_id": "t1"}, str(_worker_env / "kanban.db"))
    assert owner.dials == 1


@pytest.mark.asyncio
async def test_a_gateway_that_never_comes_back_fails_honestly(monkeypatch, _worker_env):
    owner = _Owner(connects=[None, gateway_client.GatewayUnavailableError("no owner", "inaccessible")],
                   receipts=["pending", _disconnect()])
    monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())

    with pytest.raises(gateway_client.GatewayClientError):
        await kanban_worker_client.run({"task_id": "t1"}, str(_worker_env / "kanban.db"))


class _Clock:
    def __init__(self, values):
        self.values = iter(values)
        self.last = values[-1]

    def monotonic(self):
        try:
            self.last = next(self.values)
        except StopIteration:
            pass
        return self.last


@pytest.mark.asyncio
async def test_reconnect_starting_uses_redial_deadline_not_initial_deadline(monkeypatch, _worker_env):
    # Initial wait expires at t=180, but the admitted attempt's redial budget runs until t=361.
    monkeypatch.setattr(kanban_worker_client, "time", _Clock([0, 181, 181, 200]))
    owner = _Owner(
        connects=[None, gateway_client.GatewayUnavailableError("owner starting", "starting"), None],
        receipts=["pending", _disconnect(), "terminal"],
    )
    monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())

    assert await kanban_worker_client.run({"task_id": "t1"}, str(_worker_env / "kanban.db")) == 0
    assert owner.calls.count("kanban.run") == 1
    assert owner.dials == 3


@pytest.mark.asyncio
async def test_a_later_disconnect_gets_a_fresh_redial_budget(monkeypatch, _worker_env):
    # The second disconnect occurs at t=200, beyond the first episode's t=190 deadline.
    monkeypatch.setattr(kanban_worker_client, "time", _Clock([0, 10, 10, 200, 200]))
    owner = _Owner(
        connects=[None, None, None],
        receipts=["pending", _disconnect(), "pending", _disconnect(), "terminal"],
    )
    monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())

    assert await kanban_worker_client.run({"task_id": "t1"}, str(_worker_env / "kanban.db")) == 0
    assert owner.calls.count("kanban.run") == 1
    assert owner.dials == 3


@pytest.mark.asyncio
async def test_the_redial_budget_bounds_a_flapping_gateway(monkeypatch, _worker_env):
    monkeypatch.setattr(kanban_worker_client, "OWNER_WAIT_S", 0.0)
    owner = _Owner(connects=[None, None], receipts=["pending", _disconnect(), "terminal"])
    monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())

    with pytest.raises(gateway_client.GatewayClientError):
        await kanban_worker_client.run({"task_id": "t1"}, str(_worker_env / "kanban.db"))
    assert owner.calls.count("kanban.run") == 1
