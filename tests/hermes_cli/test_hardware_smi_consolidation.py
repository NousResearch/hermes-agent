"""Tests for consolidated and cached nvidia-smi hardware probes (Issue #120262)."""

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest
import pytest

from hermes_cli.local_runtime import bootstrap, hardware
from hermes_cli.web_routers import local_models


@pytest.fixture(autouse=True)
def reset_hardware_caches():
    """Reset module-level caches before each test."""
    hardware._gpu_query_cache = None
    hardware._smi_path_cache = None
    hardware._pool_probe_cache = None
    yield
    hardware._gpu_query_cache = None
    hardware._smi_path_cache = None
    hardware._pool_probe_cache = None


def test_cached_nvidia_gpu_query_parses_all_fields():
    mock_out = MagicMock()
    mock_out.returncode = 0
    # name, memory.total (MiB), memory.free (MiB), memory.used (MiB), utilization.gpu (%)
    mock_out.stdout = "NVIDIA GeForce RTX 4090, 24576, 20480, 4096, 15\n"

    with patch.object(hardware, "_nvidia_smi_path", return_value="/fake/nvidia-smi"), \
         patch("subprocess.run", return_value=mock_out) as mock_run:
        data = hardware._cached_nvidia_gpu_query()

        assert data is not None
        assert data["gpu_name"] == "NVIDIA GeForce RTX 4090"
        assert data["total_bytes"] == 24576 << 20
        assert data["free_bytes"] == 20480 << 20
        assert data["used_bytes"] == 4096 << 20
        assert data["gpu_util_percent"] == 15

        # Verify command line arguments
        called_args = mock_run.call_args[0][0]
        assert called_args[0] == "/fake/nvidia-smi"
        assert "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu" in called_args
        assert "--format=csv,noheader,nounits" in called_args


def test_cached_nvidia_gpu_query_reuses_cache_within_ttl():
    mock_out = MagicMock()
    mock_out.returncode = 0
    mock_out.stdout = "NVIDIA GeForce RTX 5090, 32768, 28672, 4096, 10\n"

    with patch.object(hardware, "_nvidia_smi_path", return_value="/fake/nvidia-smi"), \
         patch("subprocess.run", return_value=mock_out) as mock_run:
        # First call
        res1 = hardware._cached_nvidia_gpu_query(ttl_s=5.0)
        assert mock_run.call_count == 1
        assert res1 is not None

        # Second call within TTL - should hit cache without spawning subprocess
        res2 = hardware._cached_nvidia_gpu_query(ttl_s=5.0)
        assert mock_run.call_count == 1
        assert res2 == res1

        # Simulate TTL expiration
        hardware._gpu_query_cache = (time.monotonic() - 10.0, res1)
        res3 = hardware._cached_nvidia_gpu_query(ttl_s=5.0)
        assert mock_run.call_count == 2
        assert res3 == res1


def test_cached_nvidia_gpu_query_passes_windows_hide_flags():
    mock_out = MagicMock()
    mock_out.returncode = 0
    mock_out.stdout = "NVIDIA GeForce RTX 3080, 10240, 8192, 2048, 5\n"

    with patch.object(hardware, "_nvidia_smi_path", return_value="nvidia-smi.exe"), \
         patch("hermes_cli._subprocess_compat.windows_hide_flags", return_value=0x08000000), \
         patch("subprocess.run", return_value=mock_out) as mock_run:
        hardware._cached_nvidia_gpu_query()
        assert mock_run.call_args[1].get("creationflags") == 0x08000000


def test_cached_nvidia_gpu_query_handles_failures_gracefully():
    # 1. Non-zero returncode
    mock_fail = MagicMock(returncode=1, stdout="")
    with patch.object(hardware, "_nvidia_smi_path", return_value="/fake/nvidia-smi"), \
         patch("subprocess.run", return_value=mock_fail):
        assert hardware._cached_nvidia_gpu_query() is None

    # 2. TimeoutExpired
    hardware._gpu_query_cache = None
    with patch.object(hardware, "_nvidia_smi_path", return_value="/fake/nvidia-smi"), \
         patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)):
        assert hardware._cached_nvidia_gpu_query() is None

    # 3. nvidia-smi not found
    hardware._gpu_query_cache = None
    with patch.object(hardware, "_nvidia_smi_path", return_value=None):
        assert hardware._cached_nvidia_gpu_query() is None


def test_nvidia_vram_and_facts_and_bootstrap_share_single_query():
    """Verify probe_budget, _nvidia_smi_facts, and _detect_gpu_vendor share 1 subprocess call."""
    mock_out = MagicMock()
    mock_out.returncode = 0
    mock_out.stdout = "NVIDIA GeForce RTX 4090, 24576, 20480, 4096, 22\n"

    with patch.object(hardware, "_nvidia_smi_path", return_value="/fake/nvidia-smi"), \
         patch.object(hardware, "_cuda_driver_pool", return_value=None), \
         patch.object(hardware, "_engine_device_pool", return_value=None), \
         patch("subprocess.run", return_value=mock_out) as mock_run:

        # 1. _nvidia_vram (called by probe_budget)
        vram = hardware._nvidia_vram()
        assert vram == (24576 << 20, 20480 << 20)
        assert mock_run.call_count == 1

        # 2. _nvidia_smi_facts (called by /api/local-models/hardware)
        facts = local_models._nvidia_smi_facts()
        assert facts["gpu_name"] == "NVIDIA GeForce RTX 4090"
        assert facts["gpu_util_percent"] == 22
        assert facts["vram_used_bytes"] == 4096 << 20
        # Call count should STILL be 1 (shared cache)
        assert mock_run.call_count == 1

        # 3. _detect_gpu_vendor
        vendor = bootstrap._detect_gpu_vendor()
        assert vendor == "nvidia NVIDIA GeForce RTX 4090"
        # Call count should STILL be 1
        assert mock_run.call_count == 1


def test_hardware_endpoint_executes_single_smi_subprocess():
    """Calling local_models_hardware() should result in at most 1 subprocess spawn."""
    mock_out = MagicMock()
    mock_out.returncode = 0
    mock_out.stdout = "NVIDIA GeForce RTX 4080, 16384, 12288, 4096, 12\n"

    with patch.object(hardware, "_nvidia_smi_path", return_value="/fake/nvidia-smi"), \
         patch.object(hardware, "_cuda_driver_pool", return_value=None), \
         patch.object(hardware, "_engine_device_pool", return_value=None), \
         patch("subprocess.run", return_value=mock_out) as mock_run:

        data = local_models.local_models_hardware()

        assert data["gpu_name"] == "NVIDIA GeForce RTX 4080"
        assert data["gpu_util_percent"] == 12
        assert data["vram_used_bytes"] == 4096 << 20
        assert data["vram_total_bytes"] == 16384 << 20

        # EXACTLY 1 subprocess spawn for the entire endpoint request
        assert mock_run.call_count == 1
