"""Control callbacks and shutdown must not depend on busy application workers."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from hermes_agent.gateway.control_socket import GatewayControlServer, query_gateway_control, resolve_client_socket_path
from tests.waiting import async_wait_until

pytestmark = pytest.mark.platforms('posix')


async def _request(home, verb, **params):
    raw = json.dumps({'protocol': 1, 'id': 1, 'verb': verb, 'params': params}).encode()
    return await _exchange(home, raw)


async def _exchange(home, raw):
    reader, writer = await asyncio.open_unix_connection(str(resolve_client_socket_path(home)))
    try:
        writer.write(raw + b'\n')
        await writer.drain()
        return json.loads(await asyncio.wait_for(reader.readline(), 10))
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_concurrent_producers_can_call_back_with_the_default_pool_busy(tmp_path, monkeypatch):
    from hermes_agent.gateway import control_dispatch

    # One worker per dependency layer: a single shared replacement pool would deadlock too.
    monkeypatch.setattr(control_dispatch, 'ThreadPoolExecutor',
        lambda **kwargs: ThreadPoolExecutor(max_workers=1, **kwargs))
    home = tmp_path / 'home'
    home.mkdir(mode=0o700)
    server = GatewayControlServer(home, verb_handlers={'identify': lambda: {'alive': True}})
    # Every producer synchronously asks this same listener for another control reply.
    server.private_handlers['hosted-producer'] = lambda params, peer: query_gateway_control(home, 'identify', timeout=5)
    assert await server.start()
    loop = asyncio.get_running_loop()
    await asyncio.to_thread(lambda: None)
    previous = getattr(loop, '_default_executor', None)
    assert isinstance(previous, ThreadPoolExecutor)
    pool = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(pool)
    entered, release = threading.Event(), threading.Event()

    def busy():
        entered.set()
        assert release.wait(15)

    blocker = loop.run_in_executor(None, busy)
    try:
        await async_wait_until(entered.is_set)
        replies = await asyncio.wait_for(asyncio.gather(
            *(_request(home, 'hosted-producer') for _ in range(8)), _request(home, 'identify')), 5)
        assert all(reply['ok'] and reply['result'] == {'alive': True} for reply in replies)
    finally:
        release.set()
        await blocker
        loop.set_default_executor(previous)
        pool.shutdown(wait=True)
        await server.stop()


@pytest.mark.asyncio
async def test_stop_drains_dispatch_workers_and_listener_can_restart(tmp_path):
    home = tmp_path / 'home'
    home.mkdir(mode=0o700)
    server = GatewayControlServer(home, verb_handlers={'identify': lambda: {'alive': True}})
    assert await server.start()
    assert (await _request(home, 'identify'))['result'] == {'alive': True}
    await server.stop()
    # A restarted listener needs fresh executors, not a shut-down pool or the loop's pool.
    assert await server.start()
    try:
        replies = await asyncio.gather(*(_request(home, 'identify') for _ in range(8)))
        assert all(reply['ok'] for reply in replies)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_dispatch_shutdown_cancels_queued_work_without_waiting_for_running_handler(monkeypatch):
    from hermes_agent.gateway import control_dispatch

    monkeypatch.setattr(control_dispatch, 'ThreadPoolExecutor',
        lambda **kwargs: ThreadPoolExecutor(max_workers=1, **kwargs))
    dispatcher = control_dispatch.ControlRequestDispatcher()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def handler(raw, peer):
        calls.append(raw)
        entered.set()
        assert release.wait(10)
        return b'done'

    running = asyncio.create_task(dispatcher.dispatch(handler, b'first', None))
    try:
        await async_wait_until(entered.is_set)
        queued = asyncio.create_task(dispatcher.dispatch(handler, b'second', None))
        # The second task submits before this callback runs, then awaits its queued future.
        submitted = asyncio.Event()
        asyncio.get_running_loop().call_soon(submitted.set)
        await submitted.wait()
        dispatcher.close()
        with pytest.raises(asyncio.CancelledError):
            await queued
    finally:
        release.set()
        assert await asyncio.wait_for(running, 5) == b'done'
        dispatcher.close()
    assert calls == [b'first']


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', [b'\xff', b'[' * 2000 + b']' * 2000])
async def test_dispatch_preserves_malformed_request_error_responses(tmp_path, raw):
    home = tmp_path / 'home'
    home.mkdir(mode=0o700)
    server = GatewayControlServer(home)
    assert await server.start()
    try:
        reply = await _exchange(home, raw)
        assert reply['ok'] is False
        assert reply['protocol'] == 1
        assert 'error' in reply
    finally:
        await server.stop()
