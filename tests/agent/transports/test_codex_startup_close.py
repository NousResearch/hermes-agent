"""Closing an owned Codex child must not depend on completing its startup I/O."""
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pytest

from agent.transports.codex_app_server_session import CodexAppServerSession


@pytest.fixture
def stalled_server(tmp_path):
    script = tmp_path / 'stalled-codex'
    script.write_text(f'#!{sys.executable}\n' + '''import json, sys, threading
for line in sys.stdin:
    obj = json.loads(line)
    if obj.get('method') == 'initialize':
        print(json.dumps({'id': obj['id'], 'result': {}}), flush=True)
    elif obj.get('method') == 'initialized':
        threading.Event().wait(60)
''')
    script.chmod(0o700)
    session = CodexAppServerSession(codex_bin=str(script), codex_home=str(tmp_path),
                                   developer_instructions='X' * (4 * 1024 * 1024))
    clients = []
    factory = session._client_factory
    def create(**kwargs):
        client = factory(**kwargs)
        clients.append(client)
        return client
    session._client_factory = create
    yield session, clients
    for client in clients:
        if client._proc.poll() is None:
            client._proc.kill()
        client._proc.wait(timeout=5)
    session.close()


@pytest.mark.platforms('posix')
def test_pipe_backpressure_cannot_hold_the_close_lock(stalled_server):
    session, clients = stalled_server
    entered = threading.Event()
    factory = session._client_factory
    def create(**kwargs):
        client = factory(**kwargs)
        request = client.request
        def observe(method, *args, **kw):
            if method == 'thread/start':
                entered.set()
            return request(method, *args, **kw)
        client.request = observe
        return client
    session._client_factory = create
    with ThreadPoolExecutor(max_workers=2) as pool:
        startup = pool.submit(session.ensure_started)
        try:
            assert entered.wait(5)
            pool.submit(session.close).result(timeout=5)
            assert clients[0]._proc.poll() is not None
            with pytest.raises(RuntimeError):
                startup.result(timeout=5)
            assert session._client is None and session._thread_id is None
            with pytest.raises(RuntimeError, match='closing'):
                session.ensure_started()
            assert len(clients) == 1
        finally:
            # Also free a regressed startup before the executor's blocking exit.
            for client in clients:
                if client._proc.poll() is None:
                    client._proc.kill()
                client._proc.wait(timeout=5)


@pytest.mark.platforms('posix')
def test_close_waits_for_unpublished_child_without_reopening(stalled_server):
    session, clients = stalled_server
    created, publish = threading.Event(), threading.Event()
    factory = session._client_factory
    def create(**kwargs):
        client = factory(**kwargs)
        created.set()
        assert publish.wait(10)
        return client
    session._client_factory = create
    with ThreadPoolExecutor(max_workers=2) as pool:
        startup = pool.submit(session.ensure_started)
        assert created.wait(5)
        closing = pool.submit(session.close)
        try:
            with pytest.raises(TimeoutError):
                closing.result(timeout=2)
            assert session._closed
            assert clients[0]._proc.poll() is None
        finally:
            publish.set()
        closing.result(timeout=5)
        with pytest.raises(RuntimeError, match='closing'):
            startup.result(timeout=5)
    assert clients[0]._proc.poll() is not None
    assert session._client is None
    with pytest.raises(RuntimeError, match='closing'):
        session.ensure_started()
    assert len(clients) == 1
