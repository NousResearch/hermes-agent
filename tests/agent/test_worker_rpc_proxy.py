"""Private loopback worker persistence does not use ambient HTTP proxies."""
from contextlib import contextmanager
import json
from types import SimpleNamespace


def test_worker_persistence_explicitly_bypasses_proxy(tmp_path, monkeypatch):
    from agent.runtime_session_store import WorkerRPC
    monkeypatch.setenv('HTTPS_PROXY', 'http://127.0.0.1:1')
    monkeypatch.delenv('NO_PROXY', raising=False)
    endpoint = SimpleNamespace(api_origin='http://127.0.0.1:1234')
    discoveries = iter([SimpleNamespace(state='ready', endpoint=endpoint),
                        SimpleNamespace(state='draining', endpoint=None)])
    monkeypatch.setattr('hermes_cli.gateway_runtime.discover_gateway_endpoint',
                        lambda *a, **k: next(discoveries))
    monkeypatch.setattr('hermes_cli.gateway_runtime.control_home_for', lambda home, endpoint: home)
    monkeypatch.setattr('hermes_cli.gateway_runtime_discovery.query_identify', lambda *a, **k: {'pid': -1})
    monkeypatch.setattr('hermes_cli.gateway_client._session_ticket', lambda *a, **k: 'ticket')
    calls = []
    @contextmanager
    def connect(url, **kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(send=lambda value: None, recv=lambda **k: json.dumps({'id': 1, 'result': {'ok': True}}))
    monkeypatch.setattr('websockets.sync.client.connect', connect)
    rpc = WorkerRPC(tmp_path)
    assert rpc('worker.persist') == {'ok': True}
    assert rpc('worker.persist') == {'ok': True}
    assert all(call['proxy'] is None for call in calls)
