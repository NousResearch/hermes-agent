"""
Tests for the fully-offline GPU agent (Victus / RTX 3050).
Hands are exercised directly (no model needed). The full loop test is skipped
unless ollama is reachable at :11434 (keeps CI green without a GPU/model).
"""
import json, subprocess, sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_agent import GPUAgent, OLLAMA_URL  # noqa: E402


def _ollama_up() -> bool:
    try:
        r = subprocess.run(["curl", "-s", "-m", "5", "-d", json.dumps({"model": "qwen2.5-coder:3b", "prompt": "x", "stream": False}), OLLAMA_URL],
                           capture_output=True, text=True, timeout=8)
        return r.returncode == 0 and "response" in r.stdout
    except Exception:
        return False


def test_probe_gpu_returns_real_telemetry():
    ag = GPUAgent()
    g = ag.probe_gpu()
    assert "RTX 3050" in g["name"]
    assert 0 < g["memory_total_mib"] <= 8192
    assert 0 <= g["utilization_gpu"] <= 100
    assert g["temperature_gpu"] > 0


def test_compile_kernel_succeeds():
    ag = GPUAgent()
    c = ag.compile_kernel()
    assert c["ok"] is True, c.get("output")
    assert Path(c["bin"]).exists()


def test_run_kernel_executes():
    ag = GPUAgent()
    r = ag.run_kernel()
    assert r["ok"] is True, r.get("stderr")
    assert isinstance(r["host_ms"], float) and r["host_ms"] >= 0


@pytest.mark.skipif(not _ollama_up(), reason="ollama not reachable at :11434 (offline model engine)")
@pytest.mark.skipif("PYTEST_XDIST_WORKER" in os.environ, reason="serial-only: model+nvcc contention under xdist")
def test_full_loop_runs_offline():
    ag = GPUAgent(max_steps=6)
    result = ag.run("Probe the GPU, then compile and run a CUDA matmul kernel.")
    assert result and "FINAL" not in result  # run() strips the FINAL: prefix
    # at least one real tool must have fired
    assert any(t.startswith("tool ") for t in ag.log)
    assert any("probe_gpu" in t for t in ag.log)
