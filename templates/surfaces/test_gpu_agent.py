"""
Tests for the +æ^glocal Python host loop primitive and its GPU agent instance.
Hands (CUDA) are exercised directly (no model needed). The full-loop test is
skipped unless ollama is reachable at :11434 (keeps CI green without a GPU/model).
"""
import json, subprocess, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environments.host_loop import LocalHostLoop
from gpu_agent import GPUAgent, _VCVARS  # noqa: E402


def _ollama_up() -> bool:
    try:
        r = subprocess.run(["curl", "-s", "-m", "5", "-d",
                            json.dumps({"model": "qwen2.5-coder:3b", "prompt": "x", "stream": False}),
                            "http://localhost:11434/api/generate"],
                           capture_output=True, text=True, timeout=8)
        return r.returncode == 0 and "response" in r.stdout
    except Exception:
        return False


def test_base_loop_runs_with_mock_planner():
    calls = []

    def fake_planner(prompt):
        # turn 0: call the echo tool; turn 1: final
        if "TOOL_RESULT" not in prompt:
            return "TOOL: echo hello"
        return "FINAL: done"

    loop = LocalHostLoop(planner=fake_planner, tools={"echo": lambda a: calls.append(a) or {"ok": True, "got": a}})
    res = loop.run("do a thing")
    assert res["answer"] == "done"
    assert calls == ["hello"]
    assert loop.manifest_path.exists()


def test_probe_gpu_returns_real_telemetry():
    ag = GPUAgent()
    g = ag.probe_gpu()
    assert "RTX 3050" in g["name"]
    assert 0 < g["memory_total_mib"] <= 8192
    assert 0 <= g["utilization_gpu"] <= 100


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


def test_vcvars_resolves():
    assert Path(_VCVARS).exists(), "MSVC BuildTools vcvars64.bat expected on Victus"


@pytest.mark.skipif(not _ollama_up(), reason="ollama not reachable at :11434 (offline model engine)")
@pytest.mark.skipif("PYTEST_XDIST_WORKER" in __import__("os").environ,
                    reason="serial-only: model+nvcc contention under xdist")
def test_full_gpu_loop_runs_offline():
    ag = GPUAgent(max_steps=6)
    res = ag.run("Probe the GPU, then compile and run a CUDA matmul kernel.")
    assert res["answer"]
    assert any(s.action == "probe_gpu" for s in ag.steps)
