"""Unavailable NVIDIA metrics must not discard independently usable facts."""
import subprocess
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def hardware_route(monkeypatch):
    from hermes_cli.web_routers import local_models

    monkeypatch.setattr(local_models.hardware, "probe_budget", lambda: SimpleNamespace(
        uma=False, total_device_bytes=24 << 30, usable_vram_bytes=20 << 30))
    monkeypatch.setattr(local_models.hardware, "_ram_bytes", lambda: (32 << 30, 16 << 30))
    monkeypatch.setattr(local_models.hardware, "_nvidia_smi_path", lambda: "fixture-smi")
    app = FastAPI()
    app.include_router(local_models.router)
    with TestClient(app) as client:
        yield client, local_models


@pytest.mark.parametrize("util,used,expected_util,expected_used", [
    ("12", "1024", 12, 1024 << 20),
    ("N/A", "1024", None, 1024 << 20),
    ("12", "[N/A]", 12, None),
    ("N/A", "N/A", None, None),
    ("0", "0", 0, 0),
])
def test_hardware_keeps_independently_available_facts(
    hardware_route, monkeypatch, util, used, expected_util, expected_used,
):
    client, local_models = hardware_route
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=f"Fixture GPU, {util}, {used}\n")

    monkeypatch.setattr(local_models.subprocess, "run", run)
    response = client.get("/api/local-models/hardware")
    assert response.status_code == 200
    facts = response.json()
    assert facts["gpu_name"] == "Fixture GPU"
    assert facts["gpu_util_percent"] == expected_util
    assert facts["vram_used_bytes"] == expected_used
    assert facts["vram_total_bytes"] == 24 << 30
    assert calls == [([
        "fixture-smi", "--query-gpu=name,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ], {"capture_output": True, "text": True, "timeout": 5})]


@pytest.mark.parametrize("failure", ["empty", "exit", "missing", "oserror", "timeout", "malformed"])
def test_hardware_probe_failures_still_degrade_without_500(hardware_route, monkeypatch, failure):
    client, local_models = hardware_route
    if failure == "missing":
        monkeypatch.setattr(local_models.hardware, "_nvidia_smi_path", lambda: None)

    def run(*args, **kwargs):
        if failure == "missing":
            pytest.fail("No executable must not spawn a process")
        if failure == "oserror":
            raise OSError("fixture failure")
        if failure == "timeout":
            raise subprocess.TimeoutExpired("fixture-smi", 5)
        return SimpleNamespace(returncode=1 if failure == "exit" else 0,
                               stdout="malformed" if failure == "malformed" else "")

    monkeypatch.setattr(local_models.subprocess, "run", run)
    response = client.get("/api/local-models/hardware")
    assert response.status_code == 200
    facts = response.json()
    assert facts["gpu_name"] is None
    assert facts["gpu_util_percent"] is None
    assert facts["vram_used_bytes"] is None
    assert facts["vram_total_bytes"] == 24 << 30
