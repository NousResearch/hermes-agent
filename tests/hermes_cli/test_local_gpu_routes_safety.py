"""GPU-setting routes must never acknowledge a failed apply or mutate another profile."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hermes_cli import config as config_mod
from hermes_cli.local_runtime import hardware
from hermes_cli.web_routers import local_models as routes

UUID = 'GPU-11111111-0000-0000-0000-000000000001'


@pytest.fixture
def client(monkeypatch):
    from hermes_cli import web_server
    rows = [dict(index=1, uuid=UUID, id=UUID, gpu_name='RTX 3070', total_bytes=8 << 30,
                 free_bytes=8 << 30, used_bytes=0, gpu_util_percent=0)]
    monkeypatch.setattr(hardware, 'gpu_cards', lambda: rows)
    monkeypatch.setattr(hardware, '_configured_engine', lambda: None)
    monkeypatch.setattr(hardware, 'nvidia_cards', lambda: rows)
    monkeypatch.setattr(routes, '_state_endpoint', lambda: None)
    monkeypatch.setattr(routes.bootstrap, 'get_supervisor', lambda: None)
    result = TestClient(web_server.app)
    result.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return result


def test_out_of_range_gpu_rejected_without_config_change(client):
    before = config_mod.load_config_readonly()['local_runtime']['gpu']
    r = client.post('/api/local-models/gpu', json={'gpu': '99'})
    assert r.status_code == 422
    assert config_mod.load_config_readonly()['local_runtime']['gpu'] == before


def test_uuid_is_persisted_and_stopped_server_is_not_started(client, monkeypatch):
    monkeypatch.setattr(routes.bootstrap, 'refresh_local_runtime', lambda: pytest.fail('server was stopped'))
    r = client.post('/api/local-models/gpu', json={'gpu': UUID})
    assert r.status_code == 200, r.text
    assert r.json()['server_restarted'] is False
    assert config_mod.load_config_readonly()['local_runtime']['gpu'] == UUID


def test_failed_restart_is_not_acknowledged_as_success(client, monkeypatch):
    cfg = config_mod.load_config()
    cfg['local_runtime']['gpu'] = 'auto'
    config_mod.save_config(cfg)
    monkeypatch.setattr(routes, '_state_endpoint', lambda: {'base_url': 'http://test/v1'})
    monkeypatch.setattr(routes.bootstrap, 'get_supervisor', lambda: SimpleNamespace(is_idle=lambda: True))
    monkeypatch.setattr(routes.bootstrap, 'ensure_local_runtime', lambda *a, **k: None)
    monkeypatch.setattr(routes, '_router_request', lambda *a, **k: [])
    monkeypatch.setattr(routes.bootstrap, 'refresh_local_runtime', lambda: False)
    r = client.post('/api/local-models/gpu', json={'gpu': '1'})
    assert r.status_code == 502
    assert config_mod.load_config_readonly()['local_runtime']['gpu'] == 'auto'


def test_active_inference_is_not_interrupted_by_picker(client, monkeypatch):
    monkeypatch.setattr(routes.bootstrap, 'get_supervisor', lambda: SimpleNamespace(is_idle=lambda: False))
    monkeypatch.setattr(routes.bootstrap, 'refresh_local_runtime', lambda: pytest.fail('must not restart busy server'))
    r = client.post('/api/local-models/gpu', json={'gpu': '1'})
    assert r.status_code == 409


def test_hardware_identity_and_utilisation_follow_selection(client, monkeypatch):
    monkeypatch.setattr(hardware, '_configured_gpu_selection', lambda: UUID)
    monkeypatch.setattr(hardware, '_device_pool_view', lambda: (8 << 30, False))
    r = client.get('/api/local-models/hardware')
    assert r.status_code == 200, r.text
    data = r.json()
    assert data['gpu_name'] == 'RTX 3070'
    assert data['gpu_selected'] == 1
    assert data['gpu_util_percent'] == 0
    assert data['vram_used_bytes'] == 0
    assert data['gpus'][0]['id'] == UUID

@pytest.mark.parametrize('state,deferred', [('loading', 0), ('loaded', 1)])
def test_real_supervisor_loading_and_queue_block_gpu_restart(client, monkeypatch, state, deferred):
    from io import BytesIO
    from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor

    # Exercise the real idle methods, not a replacement is_idle callback.
    sup = object.__new__(LlamaServerSupervisor)
    monkeypatch.setattr(sup, 'models', lambda: {'gemma': state})
    monkeypatch.setattr(sup, '_request', lambda *a, **k: [{'is_processing': False}])
    monkeypatch.setattr(sup, '_open', lambda *a, **k: BytesIO(
        f'llamacpp:requests_processing 0\nllamacpp:requests_deferred {deferred}\n'.encode()))
    monkeypatch.setattr(routes.bootstrap, 'get_supervisor', lambda: sup)
    monkeypatch.setattr(routes.bootstrap, 'refresh_local_runtime',
                        lambda: pytest.fail('must not interrupt loading or queued inference'))
    before = config_mod.load_config_readonly()['local_runtime']['gpu']
    response = client.post('/api/local-models/gpu', json={'gpu': '1'})
    assert response.status_code == 409, response.text
    assert config_mod.load_config_readonly()['local_runtime']['gpu'] == before



def test_refresh_serializes_non_picker_callers_too(monkeypatch):
    import threading
    from hermes_cli.local_runtime import bootstrap
    entered = threading.Event()
    release = threading.Event()
    duplicate_stop = threading.Event()
    second_started = threading.Event()
    calls = []
    def stop_old():
        calls.append('old-stop')
        if len(calls) > 1: duplicate_stop.set()
        entered.set()
        assert release.wait(5)
    monkeypatch.setattr(bootstrap, '_SUPERVISOR', SimpleNamespace(stop=stop_old))
    monkeypatch.setattr(config_mod, 'load_config', dict)
    replacements = []
    def ensure(*args, **kwargs):
        replacement = SimpleNamespace(stop=lambda: None)
        replacements.append(replacement)
        bootstrap._SUPERVISOR = replacement
        return replacement
    monkeypatch.setattr(bootstrap, 'ensure_local_runtime', ensure)
    first = threading.Thread(target=bootstrap.refresh_local_runtime)
    def second_call():
        second_started.set()
        bootstrap.refresh_local_runtime()
    second = threading.Thread(target=second_call)
    first.start()
    assert entered.wait(3)
    second.start()
    assert second_started.wait(3)
    try:
        assert not duplicate_stop.wait(1), 'a download/other refresh raced the GPU transaction'
    finally:
        release.set()
        first.join(5); second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert calls == ['old-stop']
    assert len(replacements) == 2
    assert bootstrap.get_supervisor() is replacements[-1]
