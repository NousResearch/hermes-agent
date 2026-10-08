"""The shared nvidia-smi query keeps every cell the driver did answer.

Follow-up to b9b1328b7d (#120262): one ``N/A`` cell raised inside the shared query, the miss was
cached for the TTL, and the GPU name and VRAM budget went with it. Only nvidia-smi's output is
simulated; parsing, caching and budgeting run the production code.
"""

import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import bootstrap, hardware
from hermes_cli.web_routers import local_models

_SMI = "/usr/bin/nvidia-smi"
_WIDE = "memory.total,memory.free,name,pci.device_id,memory.used,utilization.gpu"
_NARROW = "name,memory.total,memory.free"


@pytest.fixture
def smi(monkeypatch):
    """nvidia-smi answers from ``replies`` (fields -> (rc, stdout) or an exception); ``queries``
    records each spawn's field list."""
    replies: dict = {}
    queries: list[str] = []
    real_run = subprocess.run

    def run(argv, **kwargs):
        if argv[0] != _SMI:
            return real_run(argv, **kwargs)
        fields = argv[1].removeprefix("--query-gpu=")
        queries.append(fields)
        reply = replies[fields]
        if isinstance(reply, BaseException):
            raise reply
        return SimpleNamespace(returncode=reply[0], stdout=reply[1])

    monkeypatch.setattr(hardware, "_gpu_query_cache", None)
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: _SMI)
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (64 << 30, 40 << 30))
    monkeypatch.setattr(hardware, "_device_pool_view", lambda: None)
    monkeypatch.setattr(hardware, "_accelerator_device", lambda **_kwargs: None)
    monkeypatch.setattr(hardware.subprocess, "run", run)
    return SimpleNamespace(replies=replies, queries=queries)


def _margin(total: int) -> int:
    return max(hardware._MARGIN_FLOOR, int(total * hardware._MARGIN_FRACTION))


def test_na_cells_keep_name_and_vram_budget(smi):
    smi.replies[_WIDE] = (0, "15360, [N/A], Tesla T4, 0x1EB810DE, 460, [Not Supported]\n")
    total, used = 15360 << 20, 460 << 20

    budget = hardware.probe_budget()
    assert budget.uma is False
    assert budget.gpu_name == "Tesla T4"
    assert budget.total_device_bytes == total
    assert budget.usable_vram_bytes == total - used - _margin(total)
    assert bootstrap._detect_gpu_vendor() == "nvidia Tesla T4"
    assert local_models._nvidia_smi_facts() == dict(
        gpu_name="Tesla T4", gpu_util_percent=None, vram_used_bytes=used)
    assert smi.queries == [_WIDE]  # an answered row is trusted: no narrow re-query


def test_unknown_free_keeps_the_capacity_plan(smi):
    smi.replies[_WIDE] = (0, "15360, [N/A], Tesla T4, 0x1EB810DE, [N/A], 0\n")
    total = 15360 << 20

    capacity = hardware.probe_budget(planning=True)
    assert hardware.probe_budget().usable_vram_bytes == total - _margin(total)
    assert capacity.usable_vram_bytes == total - _margin(total)
    assert hardware.launch_budget(capacity) is None
    assert smi.queries == [_WIDE]


def test_unknown_free_adds_nothing_to_the_unified_live_budget(smi, monkeypatch):
    pool = 120 << 30
    monkeypatch.setattr(hardware, "_device_pool_view", lambda: (pool, True))
    smi.replies[_WIDE] = (0, "32704, [N/A], NVIDIA RTX Spark N1X, 0x2E0310DE, [N/A], [N/A]\n")

    budget = hardware.probe_budget()
    assert budget.uma is True
    assert budget.gpu_name == "NVIDIA RTX Spark N1X"
    assert budget.usable_vram_bytes == int((40 << 30) * (1 - hardware._UMA_HEADROOM_FRACTION))


def test_rejected_wide_query_retries_once_with_name_and_memory(smi):
    smi.replies[_WIDE] = (2, "")
    smi.replies[_NARROW] = (0, "NVIDIA GeForce RTX 4090, 24564, 23010\n")
    total, free = 24564 << 20, 23010 << 20

    budget = hardware.probe_budget()
    assert budget.gpu_name == "NVIDIA GeForce RTX 4090"
    assert budget.total_device_bytes == total
    assert budget.usable_vram_bytes == free - _margin(total)
    assert local_models._nvidia_smi_facts() == dict(
        gpu_name="NVIDIA GeForce RTX 4090", gpu_util_percent=None, vram_used_bytes=None)
    assert smi.queries == [_WIDE, _NARROW]  # the partial reading is what the TTL caches


@pytest.mark.parametrize("replies, extra", [
    ({_WIDE: (0, "24564, 23010, NVIDIA GeForce RTX 4090, OEM, 0x268410DE, 1200, 7\n")},
     dict(gpu_pci_id=0x268410DE, used_bytes=1200 << 20, gpu_util_percent=7)),
    ({_WIDE: (2, ""), _NARROW: (0, "NVIDIA GeForce RTX 4090, OEM, 24564, 23010\n")},
     dict(gpu_pci_id=None, used_bytes=None, gpu_util_percent=None)),
], ids=["wide", "fallback"])
def test_comma_in_name_does_not_shift_numeric_columns(smi, replies, extra):
    """nvidia-smi does not quote the name, so a comma in it adds a cell."""
    smi.replies.update(replies)

    assert hardware._cached_nvidia_gpu_query() == dict(
        gpu_name="NVIDIA GeForce RTX 4090, OEM", total_bytes=24564 << 20, free_bytes=23010 << 20, **extra)


def test_timeout_is_not_retried(smi):
    smi.replies[_WIDE] = subprocess.TimeoutExpired(_SMI, 10)

    assert hardware._cached_nvidia_gpu_query() is None
    assert hardware._cached_nvidia_gpu_query() is None
    assert smi.queries == [_WIDE]
