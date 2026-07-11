"""
Tests for GPU-MCP: a stdlib-only Model Context Protocol server exposing the
local CUDA + Rust/WASM hands over stdio JSON-RPC.

We drive the server by piping initialize/tools/list/tools/call messages and
asserting the JSON-RPC responses. No network, no MCP SDK dependency.
"""
import json, subprocess, sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
# gpu-mcp is a submodule at <REPO>/gpu-mcp; its server runs as `python -m gpu_mcp`
SUBMODULE = REPO / "gpu-mcp"
CRATE = REPO / "environments" / "testdata" / "wasm_hello"


def _run_mcp(messages: list[dict]) -> list[dict]:
    inp = "\n".join(json.dumps(m) for m in messages) + "\n"
    p = subprocess.run([sys.executable, "-m", "gpu_mcp"],
                       input=inp, capture_output=True, text=True, timeout=120,
                       cwd=str(SUBMODULE),
                       env={**__import__("os").environ, "PYTHONPATH": str(SUBMODULE)})
    out = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
    return out


def test_initialize():
    out = _run_mcp([{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}])
    info = out[0]["result"]["serverInfo"]
    assert info["name"] == "gpu-mcp"


def test_tools_list_has_five():
    out = _run_mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ])
    names = [t["name"] for t in out[1]["result"]["tools"]]
    assert names == ["probe_gpu", "compile_kernel", "run_kernel", "rust_build_wasm", "rust_run_wasm"]


def test_probe_gpu_over_mcp():
    out = _run_mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "probe_gpu", "arguments": {}}},
    ])
    res = json.loads(out[1]["result"]["content"][0]["text"])
    assert "RTX 3050" in res["name"]


def test_compile_kernel_over_mcp():
    out = _run_mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "compile_kernel", "arguments": {"name": "matmul"}}},
    ])
    res = json.loads(out[1]["result"]["content"][0]["text"])
    assert res["ok"] is True


def test_rust_build_wasm_over_mcp():
    out = _run_mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
         "params": {"name": "rust_build_wasm", "arguments": {"crate_dir": str(CRATE)}}},
    ])
    res = json.loads(out[1]["result"]["content"][0]["text"])
    assert res["ok"] is True and res["wasm"].endswith(".wasm")
