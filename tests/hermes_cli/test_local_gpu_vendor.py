import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import bootstrap, hardware
from hermes_cli.local_runtime.binaries import select_backend


@pytest.mark.windows_only
@pytest.mark.parametrize("device_ids,expected", [
    ("PCI\\VEN_1002&DEV_73BF\nPCI\\VEN_1002&DEV_164E", "vulkan"),
    ("PCI\\VEN_8086&DEV_56A0", "vulkan"),
    ("PCI\\VEN_1002&DEV_164E\nPCI\\VEN_10DE&DEV_2684", "cuda"),
    ("ROOT\\BasicDisplay\nROOT\\RDPIDD", "cpu"),
])
def test_auto_backend_uses_pci_vendors_without_runtime(monkeypatch, tmp_path, device_ids, expected):
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=device_ids))
    assert select_backend(bootstrap._detect_gpu_vendor()) == expected


@pytest.mark.windows_only
def test_auto_backend_probe_failure_preserves_cpu_fallback(monkeypatch):
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: None)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 10)
    monkeypatch.setattr(subprocess, "run", timeout)
    assert select_backend(bootstrap._detect_gpu_vendor()) == "cpu"
