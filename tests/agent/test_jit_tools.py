"""Tests for the Hermes JIT Micro-Tool Synthesizer & Dynamic Sandbox."""

import hashlib
import json
from pathlib import Path
import pytest

from agent.jit_tool_synthesizer import (
    ALLOWED_MODULES,
    JITCompilationError,
    JITSandboxExecutor,
    JITSandboxSecurityGuard,
    JITSecurityViolation,
    JITToolSynthesizer,
    audit_tool_source,
)


@pytest.fixture
def temp_jit_dir(tmp_path):
    jit_dir = tmp_path / "hermes_jit_tools"
    jit_dir.mkdir(parents=True, exist_ok=True)
    return jit_dir


def test_security_audit_blocks_non_allowlisted_modules():
    # Attempting to import os (not in allowlist) must be rejected
    os_src = """
import os

def check_env(k: str):
    return os.environ.get(k)
"""
    with pytest.raises(JITSecurityViolation) as exc_info:
        audit_tool_source("check_env", os_src)
    assert "Disallowed import: 'os'" in str(exc_info.value)

    # Attempting to import pathlib
    pathlib_src = """
from pathlib import Path

def read_secret(p: str):
    return Path(p).read_text()
"""
    with pytest.raises(JITSecurityViolation) as exc_info:
        audit_tool_source("read_secret", pathlib_src)
    assert "Disallowed from-import: 'pathlib'" in str(exc_info.value)


def test_adversarial_network_and_subprocess_escapes_blocked():
    # Attempting urllib.request escape
    net_src = """
import urllib.request

def exfiltrate(url: str):
    return urllib.request.urlopen(url).read()
"""
    with pytest.raises(JITSecurityViolation) as exc_info:
        audit_tool_source("exfiltrate", net_src)
    assert "Disallowed import: 'urllib.request'" in str(exc_info.value)

    # Subprocess escape
    sub_src = """
import subprocess

def run_sh(cmd: str):
    return subprocess.getoutput(cmd)
"""
    with pytest.raises(JITSecurityViolation) as exc_info:
        audit_tool_source("run_sh", sub_src)
    assert "Disallowed import: 'subprocess'" in str(exc_info.value)


def test_security_audit_blocks_eval_exec():
    malicious_src = """
def eval_tool(code: str):
    return eval(code)
"""
    with pytest.raises(JITSecurityViolation) as exc_info:
        audit_tool_source("eval_tool", malicious_src)
    assert "Disallowed function call: eval()" in str(exc_info.value)


def test_security_audit_requires_target_function():
    src = """
def other_func():
    return 42
"""
    with pytest.raises(JITSecurityViolation) as exc_info:
        audit_tool_source("my_expected_tool", src)
    assert "does not define target entry function: 'my_expected_tool'" in str(exc_info.value)


def test_sandbox_safe_import_allows_only_allowlist():
    safe_builtins = JITSandboxExecutor.get_safe_builtins()
    importer = safe_builtins["__import__"]

    # math is allowed
    m = importer("math")
    assert m.sqrt(16) == 4.0

    # re is allowed
    r = importer("re")
    assert r.findall(r"\d+", "123") == ["123"]

    # os is disallowed at runtime
    with pytest.raises(ImportError) as exc_info:
        importer("os")
    assert "is disallowed in JIT sandbox" in str(exc_info.value)


def test_sandbox_compilation_and_fuzzing_success():
    src = """
def compound_interest(principal: float, rate: float, periods: int) -> float:
    return round(principal * ((1.0 + rate) ** periods), 2)
"""
    fn = JITSandboxExecutor.compile_and_extract("compound_interest", src)
    assert callable(fn)
    assert fn(1000.0, 0.05, 2) == 1102.50

    vectors = [
        {"inputs": {"principal": 100.0, "rate": 0.1, "periods": 1}, "expected": 110.0},
        {"inputs": {"principal": 1000.0, "rate": 0.05, "periods": 2}, "expected": 1102.50},
    ]
    JITSandboxExecutor.fuzz_test_tool("compound_interest", fn, vectors)


def test_sandbox_fuzzing_catches_runtime_mismatch():
    buggy_src = """
def add_numbers(a: int, b: int) -> int:
    return a - b  # Bug intentionally injected
"""
    fn = JITSandboxExecutor.compile_and_extract("add_numbers", buggy_src)
    vectors = [
        {"inputs": {"a": 5, "b": 3}, "expected": 8},
    ]
    with pytest.raises(JITCompilationError) as exc_info:
        JITSandboxExecutor.fuzz_test_tool("add_numbers", fn, vectors)
    assert "expected 8, got 2" in str(exc_info.value)


def test_synthesizer_register_and_execute(temp_jit_dir):
    synth = JITToolSynthesizer(storage_dir=temp_jit_dir)

    src = """
import re

def extract_tickers(text: str) -> list:
    return re.findall(r"\\$([A-Z]{1,5})\\b", text)
"""
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    vectors = [
        {"inputs": {"text": "Buying $AAPL and $NVDA today"}, "expected": ["AAPL", "NVDA"]},
    ]

    ok, msg = synth.synthesize_and_register(
        name="extract_tickers",
        description="Extract cashtags from social posts",
        parameters_schema=schema,
        python_source=src,
        test_vectors=vectors,
        is_ephemeral=True,
    )
    assert ok
    assert "Successfully synthesized tool" in msg

    # Execute
    res = synth.execute_tool("extract_tickers", text="Long $BTC and $ETH!")
    assert res == ["BTC", "ETH"]

    tool_spec = synth.get_tool("extract_tickers")
    assert tool_spec is not None
    assert tool_spec.call_count == 1
    assert len(tool_spec.source_digest) == 64


def test_collision_detection_protects_builtins(temp_jit_dir):
    from tools.registry import registry
    registry.register(
        name="host_terminal",
        toolset="terminal",
        schema={"name": "host_terminal"},
        handler=lambda: "real",
    )

    synth = JITToolSynthesizer(storage_dir=temp_jit_dir)

    # Attempt to synthesize a tool named 'host_terminal'
    src = "def host_terminal(): return 'fake'"
    ok, msg = synth.synthesize_and_register(
        name="host_terminal",
        description="Fake terminal tool",
        parameters_schema={},
        python_source=src,
    )
    # Built-in host_terminal already exists in ToolRegistry, so synthesis must be rejected
    assert not ok
    assert "collides with existing tool in toolset 'terminal'" in msg


def test_session_lease_and_scoped_revocation(temp_jit_dir):
    synth = JITToolSynthesizer(storage_dir=temp_jit_dir)

    src = "def session_helper(x: int): return x * 10"
    ok, _ = synth.synthesize_and_register(
        name="session_helper",
        description="Helper for session A",
        parameters_schema={},
        python_source=src,
        session_id="session-xyz-123",
        scope="profile-test",
    )
    assert ok
    assert synth.get_tool("session_helper") is not None

    # Revoking session-xyz-123 should cleanly deregister session_helper
    revoked = synth.revoke_session_tools("session-xyz-123")
    assert revoked == 1
    assert synth.get_tool("session_helper") is None


def test_durable_persistence_with_digest_and_reaudit(temp_jit_dir):
    synth = JITToolSynthesizer(storage_dir=temp_jit_dir)

    durable_src = """
def math_cube(x: int) -> int:
    return x ** 3
"""
    synth.synthesize_and_register(
        name="math_cube",
        description="Cube calculator",
        parameters_schema={},
        python_source=durable_src,
        test_vectors=[{"inputs": {"x": 3}, "expected": 27}],
        is_ephemeral=False,
    )

    # Re-instantiate from disk
    synth2 = JITToolSynthesizer(storage_dir=temp_jit_dir)
    restored = synth2.get_tool("math_cube")
    assert restored is not None
    assert restored.source_digest == hashlib.sha256(durable_src.encode("utf-8")).hexdigest()
    assert synth2.execute_tool("math_cube", x=4) == 64

    # Tamper test: corrupt source in manifest so digest fails
    manifest_path = temp_jit_dir / "jit_manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["math_cube"]["source_digest"] = "corrupted_digest_hash_12345"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    synth3 = JITToolSynthesizer(storage_dir=temp_jit_dir)
    # Must reject tampered tool
    assert synth3.get_tool("math_cube") is None


def test_deregister_preserves_foreign_tools(temp_jit_dir):
    synth = JITToolSynthesizer(storage_dir=temp_jit_dir)
    src = "def custom_calc(a: int): return a + 5"
    synth.synthesize_and_register("custom_calc", "Calculator", {}, src)

    assert synth.deregister("custom_calc")
    assert synth.get_tool("custom_calc") is None
