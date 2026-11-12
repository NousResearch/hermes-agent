"""A saved gateway that stops answering never holds up this gateway's own traffic.

Reaching a saved SSH gateway blocks for as long as its dial waits (ssh's 15 s connect timeout
against a host that drops the connection). Relayed requests for it must not take the loop's default
executor, which this gateway's own ``/api/ws`` handshake and routes run on, and requests waiting
behind one dial must not each dial again, least of all after the connection is removed.
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from tests.hermes_cli.test_webapp_socket_tickets import _SESSION, serve_webapp
from tests.waiting import async_wait_until

_WORKERS = 4


@pytest.fixture
def unanswered(tmp_path, monkeypatch):
    """The loopback Webapp. Every SSH master it opens waits for ``answer``, then times out the way
    ``ssh`` does; ``dials`` lists the hosts dialed and ``asked`` counts the relay's resolutions."""
    from hermes_agent.hermes_cli import gateway_connections_ssh as ssh
    from hermes_agent.hermes_cli import webapp_relay_target as reach
    from hermes_agent.hermes_cli.gateway_connections_ssh_transport import SshReachError

    served = serve_webapp(tmp_path, monkeypatch)
    served.dials, served.asked, served.answer = [], [], threading.Event()

    class Unanswered:
        def __init__(self, endpoint, **_kwargs):
            self.endpoint = endpoint

        def open(self):
            served.dials.append(self.endpoint.host)
            served.answer.wait(30)
            raise SshReachError("timeout", f"SSH operation to {self.endpoint.target} timed out.")

    resolve = reach.resolve_target

    async def asked(connection_id, **kwargs):
        served.asked.append(connection_id)
        return await resolve(connection_id, **kwargs)

    monkeypatch.setattr(ssh, "SshConnection", Unanswered)
    monkeypatch.setattr(ssh, "resolve_ssh_binary", lambda: "ssh")
    monkeypatch.setattr(reach, "resolve_target", asked)
    yield served
    served.answer.set()


def _save_ssh(host: str) -> str:
    from hermes_agent.hermes_cli import gateway_connections as store

    saved, _registry = store.save_connection({"kind": "ssh", "label": f"Gone {host}", "host": host, "user": "operator"})
    return saved["id"]


def _relayed(served, connection_ids, each: int):
    transport = httpx.ASGITransport(app=served.server.app)
    tab = httpx.AsyncClient(transport=transport, base_url="http://localhost", headers=_SESSION)
    pending = [asyncio.create_task(tab.get(f"/api/relay/{cid}/api/status")) for cid in connection_ids
               for _ in range(each)]
    return tab, pending


def test_unanswered_saved_gateways_leave_the_default_executor_to_this_gateways_own_requests(unanswered):
    ids = [_save_ssh(f"192.0.2.{n}") for n in range(1, _WORKERS + 1)]

    async def scenario():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(_WORKERS))
        tab, pending = _relayed(unanswered, ids, each=3)
        try:
            await async_wait_until(lambda: len(unanswered.asked) == len(pending) and unanswered.dials,
                                   what="every relayed request waiting on a dial")
            await asyncio.wait_for(asyncio.to_thread(lambda: "local"), timeout=5)
        finally:
            unanswered.answer.set()
        statuses = [response.status_code for response in await asyncio.gather(*pending)]
        await tab.aclose()
        return statuses

    assert asyncio.run(scenario()) == [409] * (_WORKERS * 3)


def test_requests_waiting_on_one_dial_never_dial_a_removed_gateway_again(unanswered):
    from hermes_agent.hermes_cli import gateway_connections as store

    cid = _save_ssh("192.0.2.1")

    async def scenario():
        tab, pending = _relayed(unanswered, [cid], each=_WORKERS)
        try:
            await async_wait_until(lambda: len(unanswered.asked) == _WORKERS and unanswered.dials,
                                   what="every relayed request waiting on the dial")
            store.remove_connection(cid)
        finally:
            unanswered.answer.set()
        statuses = [response.status_code for response in await asyncio.gather(*pending)]
        after = (await tab.get(f"/api/relay/{cid}/api/status")).status_code
        await tab.aclose()
        return statuses, after

    statuses, after = asyncio.run(scenario())
    assert unanswered.dials == ["192.0.2.1"]
    assert statuses == [409] * _WORKERS and after == 404


def test_a_caller_that_gives_up_leaves_the_shared_resolution_to_the_others(unanswered, monkeypatch):
    from hermes_agent.hermes_cli import webapp_relay_target as reach

    busy, cid = _save_ssh("192.0.2.1"), _save_ssh("192.0.2.2")
    monkeypatch.setattr(reach, "_POOL", ThreadPoolExecutor(1))

    async def scenario():
        holding = asyncio.create_task(reach.resolve_target(busy))
        await async_wait_until(lambda: unanswered.dials, what="the only worker dialing another gateway")
        leaving, staying = (asyncio.create_task(reach.resolve_target(cid)) for _ in range(2))
        await async_wait_until(lambda: len(unanswered.asked) == 3, what="both callers waiting on one resolution")
        leaving.cancel()
        unanswered.answer.set()
        return await asyncio.gather(holding, staying, return_exceptions=True)

    outcomes = asyncio.run(scenario())
    assert [type(outcome) for outcome in outcomes] == [reach.Unreachable, reach.Unreachable]
    assert unanswered.dials == ["192.0.2.1", "192.0.2.2"]
