"""
Tests for the Rust/WASM agent instance of the +æ^glocal host loop.
Build path is exercised for real (rustup env fix -> real .wasm). Run degrades
honestly when no wasm runtime is on PATH.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from templates.surfaces.wasm_agent import WasmAgent  # noqa: E402

FIXTURE = Path(__file__).resolve().parent.parent.parent / "environments" / "testdata" / "wasm_hello"


def test_wasm_agent_builds_offline():
    ag = WasmAgent(str(FIXTURE))
    res = ag._hand_build()
    assert res["ok"] is True, res.get("error")
    wasm = Path(res["wasm"])
    assert wasm.exists() and wasm.suffix == ".wasm"


def test_wasm_agent_run_degrades_without_runtime():
    ag = WasmAgent(str(FIXTURE))
    built = ag._hand_build()
    res = ag._hand_run(built["wasm"])
    assert res.get("reason") == "no_runtime"


def test_wasm_agent_loop_build_step():
    ag = WasmAgent(str(FIXTURE))
    res = ag.run("Build this Rust crate to wasm32-unknown-unknown and report the artifact.")
    assert res["answer"]
    assert any(s.action == "rust_build_wasm" for s in ag.steps)
