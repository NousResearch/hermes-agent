"""Physical telemetry must not change the first-GPU inference budget."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from hermes_cli.local_runtime import hardware


@pytest.fixture
def smi(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=(
        '32768, 24576, Same GPU, 0x123410DE, 8192, 10\n'
        '32768, 8192, Same GPU, 0x123410DE, 24576, 90\n')))
    monkeypatch.setattr(hardware, '_gpu_query_cache', None)
    monkeypatch.setattr(hardware, '_nvidia_smi_path', lambda: 'nvidia-smi')
    monkeypatch.setattr(hardware.subprocess, 'run', run)
    return run


def test_all_devices_share_query_but_budget_keeps_first(smi):
    assert hardware._nvidia_vram()[:3] == (32 << 30, 24 << 30, 'Same GPU')
    devices = hardware._cached_nvidia_gpu_queries()
    assert len(devices) == 2
    assert [gpu['gpu_util_percent'] for gpu in devices] == [10, 90]
    assert [gpu['used_bytes'] for gpu in devices] == [8 << 30, 24 << 30]
    assert hardware._cached_nvidia_gpu_query() == devices[0]
    assert smi.call_count == 1


def test_api_inventory_does_not_replace_legacy_budget(smi, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hermes_cli.web_routers import local_models
    monkeypatch.setattr(hardware, 'probe_budget', lambda: SimpleNamespace(
        uma=False, total_device_bytes=32 << 30, usable_vram_bytes=24 << 30))
    monkeypatch.setattr(hardware, '_ram_bytes', lambda: (128 << 30, 64 << 30))
    app = FastAPI()
    app.include_router(local_models.router)
    with TestClient(app) as client:
        response = client.get('/api/local-models/hardware')
    assert response.status_code == 200
    data = response.json()
    assert data['vram_total_bytes'] == 32 << 30
    assert data['vram_usable_bytes'] == 24 << 30
    assert data['gpu_util_percent'] == 10
    assert [gpu['index'] for gpu in data['gpus']] == [0, 1]
    assert sum(gpu['vram_total_bytes'] for gpu in data['gpus']) == 64 << 30
    assert smi.call_count == 1


@pytest.mark.parametrize('text, expected', [
    ('32768, 32768, A, N/A, 0, 0\n', [(0, 0)]),
    ('32768, 32768, A, N/A, N/A, [Not Supported]\n', [(None, None)]),
    ('', []),
])
def test_zero_unknown_and_no_devices(smi, text, expected):
    smi.return_value.stdout = text
    devices = hardware._cached_nvidia_gpu_queries() or []
    assert [(gpu['used_bytes'], gpu['gpu_util_percent']) for gpu in devices] == expected
    hardware._cached_nvidia_gpu_queries()
    assert smi.call_count == 1
