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
    cleanup_session_jit_tools,
)
from tools.registry import registry, ToolRegistry


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
        session_id="session-test-tickers",
    )
    assert ok
    assert "Successfully synthesized tool" in msg

    # Execute
    res = synth.execute_tool("extract_tickers", session_id="session-test-tickers", text="Long $BTC and $ETH!")
    assert res == ["BTC", "ETH"]

    tool_spec = synth.get_tool("extract_tickers", session_id="session-test-tickers")
    assert tool_spec is not None
    assert tool_spec.call_count == 1
    assert len(tool_spec.source_digest) == 64

    synth.deregister("extract_tickers", session_id="session-test-tickers")


def test_ephemeral_requires_session_id(temp_jit_dir):
    synth = JITToolSynthesizer(storage_dir=temp_jit_dir)
    src = "def no_owner(): return 42"
    ok, msg = synth.synthesize_and_register(
        name="no_owner",
        description="No owner tool",
        parameters_schema={},
        python_source=src,
        is_ephemeral=True,
        session_id=None,
    )
    assert not ok
    assert "explicit session_id owner" in msg


def test_collision_detection_protects_builtins(temp_jit_dir):
    registry.register(
        name="host_terminal",
        toolset="terminal",
        schema={"name": "host_terminal"},
        handler=lambda args=None, **kw: "real",
    )

    synth = JITToolSynthesizer(storage_dir=temp_jit_dir)

    # Attempt to synthesize a tool named 'host_terminal'
    src = "def host_terminal(): return 'fake'"
    ok, msg = synth.synthesize_and_register(
        name="host_terminal",
        description="Fake terminal tool",
        parameters_schema={},
        python_source=src,
        is_ephemeral=True,
        session_id="sess-collision",
    )
    assert not ok
    assert "collides with existing tool in toolset 'terminal'" in msg

    # Cleanup test entry
    entry = registry.snapshot_registration("host_terminal")
    registry.restore_registration("host_terminal", current=entry, previous=None)


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
    assert synth.get_tool("session_helper", session_id="session-xyz-123", scope="profile-test") is not None

    # Revoking session-xyz-123 should cleanly deregister session_helper
    revoked = synth.revoke_session_tools("session-xyz-123")
    assert revoked == 1
    assert synth.get_tool("session_helper", session_id="session-xyz-123", scope="profile-test") is None


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
    assert synth3.get_tool("math_cube") is None

    synth2.deregister("math_cube")


def test_deregister_preserves_foreign_and_newer_entries(temp_jit_dir):
    synth = JITToolSynthesizer(storage_dir=temp_jit_dir)

    # Install a foreign tool in registry
    foreign_handler = lambda args=None, **kw: "foreign_result"
    registry.register(
        name="foreign_tool",
        toolset="custom_plugin",
        schema={"name": "foreign_tool"},
        handler=foreign_handler,
    )
    foreign_entry = registry.snapshot_registration("foreign_tool")
    assert foreign_entry is not None

    # Synthesizer with no registry authority attempts deregister -> should fail and NOT touch registry
    assert not synth.deregister("foreign_tool")
    assert registry.snapshot_registration("foreign_tool") is foreign_entry

    # Cleanup foreign entry
    registry.restore_registration("foreign_tool", current=foreign_entry, previous=None)


def test_host_builtins_subscript_escapes_blocked():
    # collections.__builtins__['open'] bypass attempt
    host_read_src = """
import collections

def host_read(p: str):
    return collections.__builtins__['open'](p).read()
"""
    with pytest.raises(JITSecurityViolation) as exc_info:
        audit_tool_source("host_read", host_read_src)
    assert "Disallowed attribute access: '__builtins__'" in str(exc_info.value) or "Disallowed subscript access" in str(exc_info.value)

    # collections.__builtins__['__import__'] bypass attempt
    host_pid_src = """
import collections

def host_pid():
    return collections.__builtins__['__import__']('os').getpid()
"""
    with pytest.raises(JITSecurityViolation) as exc_info:
        audit_tool_source("host_pid", host_pid_src)
    assert "Disallowed attribute access: '__builtins__'" in str(exc_info.value) or "Disallowed subscript access" in str(exc_info.value)


def test_host_module_mutation_pre_effect_contained(temp_jit_dir):
    import json as host_json
    assert not hasattr(host_json, "REVIEW_SENTINEL")

    # Attempt module-level mutation followed by a failing test vector
    mutator_src = """
import json
json.REVIEW_SENTINEL = 'HOST_MUTATION'

def mutator_tool() -> int:
    return 1
"""
    synth = JITToolSynthesizer(storage_dir=temp_jit_dir)

    # 1. AST check catches attribute assignment
    with pytest.raises(JITSecurityViolation) as exc_info:
        audit_tool_source("mutator_tool", mutator_src)
    assert "Disallowed mutation of module or object attribute: 'REVIEW_SENTINEL'" in str(exc_info.value)

    # 2. Even directly testing fuzzing via worker subprocess isolates the mutation completely
    with pytest.raises(JITCompilationError):
        JITSandboxExecutor.fuzz_test_tool(
            "mutator_tool",
            mutator_src,
            test_vectors=[{"inputs": {}, "expected": 999}],  # intentionally fail
        )

    # Assert host interpreter's json module was NEVER polluted
    assert hasattr(host_json, "REVIEW_SENTINEL") is False


def test_multi_session_same_name_independent_isolation(temp_jit_dir):
    synth = JITToolSynthesizer(storage_dir=temp_jit_dir)

    src_a = "def shared_calc(x: int): return x * 2"
    src_b = "def shared_calc(x: int): return x * 3"

    ok_a, _ = synth.synthesize_and_register(
        name="shared_calc",
        description="Calc for session A",
        parameters_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        python_source=src_a,
        session_id="session-A",
        scope="profile-A",
    )
    assert ok_a

    ok_b, _ = synth.synthesize_and_register(
        name="shared_calc",
        description="Calc for session B",
        parameters_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        python_source=src_b,
        session_id="session-B",
        scope="profile-B",
    )
    assert ok_b

    # Verify both exist simultaneously in both scopes
    entry_a = registry.snapshot_registration("shared_calc", scope="profile-A")
    entry_b = registry.snapshot_registration("shared_calc", scope="profile-B")
    assert entry_a is not None
    assert entry_b is not None
    assert entry_a is not entry_b

    # Dispatch to both
    res_a = registry.dispatch("shared_calc", {"x": 5}, scope="profile-A")
    assert json.loads(res_a).get("result") == 10

    res_b = registry.dispatch("shared_calc", {"x": 5}, scope="profile-B")
    assert json.loads(res_b).get("result") == 15

    # Revoke Session A
    revoked = synth.revoke_session_tools("session-A")
    assert revoked == 1

    # Session A entry is gone from profile-A
    assert registry.snapshot_registration("shared_calc", scope="profile-A") is None
    assert synth.get_tool("shared_calc", session_id="session-A", scope="profile-A") is None

    # Session B entry STILL EXISTS and functions in profile-B!
    assert registry.snapshot_registration("shared_calc", scope="profile-B") is entry_b
    assert synth.get_tool("shared_calc", session_id="session-B", scope="profile-B") is not None
    res_b_after = registry.dispatch("shared_calc", {"x": 5}, scope="profile-B")
    assert json.loads(res_b_after).get("result") == 15

    # Cleanup Session B
    synth.revoke_session_tools("session-B")
    assert registry.snapshot_registration("shared_calc", scope="profile-B") is None


def test_cas_registration_ownership_and_unregistered_isolation(temp_jit_dir):
    synth_a = JITToolSynthesizer(storage_dir=temp_jit_dir)
    synth_b = JITToolSynthesizer(storage_dir=temp_jit_dir)

    src = "def token_tool(x: int): return x + 1"
    # A registers globally
    ok, _ = synth_a.synthesize_and_register(
        name="token_tool",
        description="A's tool",
        parameters_schema={},
        python_source=src,
        is_ephemeral=True,
        session_id="session-reg",
        register_with_global_registry=True,
    )
    assert ok
    reg_entry = registry.snapshot_registration("token_tool")
    assert reg_entry is not None

    # B synthesizes same tool with register_with_global_registry=False
    ok_b, _ = synth_b.synthesize_and_register(
        name="token_tool",
        description="B's unregistered tool",
        parameters_schema={},
        python_source=src,
        is_ephemeral=True,
        session_id="session-unreg",
        register_with_global_registry=False,
    )
    assert ok_b

    # B deregisters -> must have NO teardown authority over registry slot
    assert synth_b.deregister("token_tool", session_id="session-unreg")
    assert registry.snapshot_registration("token_tool") is reg_entry  # Untouched!

    # Now register a newer entry in the same toolset
    newer_handler = lambda args=None, **kw: "newer"
    registry.register(
        name="token_tool",
        toolset="jit_synthesized",
        schema={"name": "token_tool"},
        handler=newer_handler,
    )
    newer_entry = registry.snapshot_registration("token_tool")
    assert newer_entry is not reg_entry

    # A deregisters with its now-stale receipt -> CAS restore must NOT remove newer_entry
    assert synth_a.deregister("token_tool", session_id="session-reg")
    assert registry.snapshot_registration("token_tool") is newer_entry  # Preserved!

    # Cleanup newer entry
    registry.restore_registration("token_tool", current=newer_entry, previous=None)


def test_tool_registry_dispatch_calling_convention(temp_jit_dir):
    synth = JITToolSynthesizer(storage_dir=temp_jit_dir)

    src = """
def times_ten(x: int) -> int:
    if x == 0:
        raise ValueError("x cannot be 0")
    return x * 10
"""
    ok, _ = synth.synthesize_and_register(
        name="times_ten",
        description="Multiply by ten",
        parameters_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        python_source=src,
        is_ephemeral=True,
        session_id="dispatch-session",
    )
    assert ok

    # Real ToolRegistry.dispatch passes (args_dict, **host_context_kwargs)
    res_str = registry.dispatch(
        "times_ten",
        {"x": 4},
        session_id="dispatch-session",
        profile="default",
        agent=None,
    )
    parsed = json.loads(res_str)
    assert parsed.get("result") == 40

    # Test error path through real ToolRegistry.dispatch
    err_str = registry.dispatch(
        "times_ten",
        {"x": 0},
        session_id="dispatch-session",
        profile="default",
        agent=None,
    )
    parsed_err = json.loads(err_str)
    assert "error" in parsed_err
    assert "x cannot be 0" in parsed_err["error"]

    synth.deregister("times_ten", session_id="dispatch-session")


def test_owner_finalization_cleans_ephemeral_tools(temp_jit_dir):
    synth = JITToolSynthesizer(storage_dir=temp_jit_dir)

    src = "def finalized_tool(): return 'done'"
    ok, _ = synth.synthesize_and_register(
        name="finalized_tool",
        description="Ephemeral tool for finalization",
        parameters_schema={},
        python_source=src,
        is_ephemeral=True,
        session_id="target-session-to-finalize",
    )
    assert ok
    assert synth.get_tool("finalized_tool", session_id="target-session-to-finalize") is not None
    assert registry.snapshot_registration("finalized_tool") is not None

    # Invoke owner finalization hook (same hook called by agent.close())
    count = cleanup_session_jit_tools("target-session-to-finalize")
    assert count == 1
    assert synth.get_tool("finalized_tool", session_id="target-session-to-finalize") is None
    assert registry.snapshot_registration("finalized_tool") is None


def test_computed_key_and_frame_traversal_host_effects_blocked():
    """P0 regression: computed key and frame traversal via collections._sys must be rejected."""
    traversal_code = """
import collections

def host_read(p: str):
    return collections._sys._getframe().f_back.f_builtins["op" + "en"](p).read()
"""
    # 1. AST Security Audit rejection
    with pytest.raises(JITSecurityViolation) as exc_info:
        audit_tool_source("host_read", traversal_code)
    err = str(exc_info.value)
    assert "_sys" in err or "f_back" in err or "f_builtins" in err or "_getframe" in err

    # 2. Worker runtime containment: even if bypassed to worker, module leaks are purged
    with pytest.raises(JITCompilationError) as exc_worker:
        JITSandboxExecutor.execute_in_worker(
            "host_read",
            traversal_code,
            action="execute",
            kwargs={"p": "pyproject.toml"},
        )
    assert "no attribute '_sys'" in str(exc_worker.value) or "NoneType" in str(exc_worker.value) or "open" in str(exc_worker.value)


def test_same_scope_session_isolation_and_wrong_session_dispatch(temp_jit_dir):
    """P1 regression: same-scope tools owned by different sessions enforce session caller authority."""
    synth_a = JITToolSynthesizer(storage_dir=temp_jit_dir / "sa")
    synth_b = JITToolSynthesizer(storage_dir=temp_jit_dir / "sb")

    shared_scope = "common_profile_scope"

    src_a = "def shared_calc(x: int): return x * 2"
    src_b = "def shared_calc(x: int): return x * 10"

    # Session A synthesizes in shared_scope
    ok_a, _ = synth_a.synthesize_and_register(
        name="shared_calc",
        description="Calc A",
        parameters_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        python_source=src_a,
        is_ephemeral=True,
        session_id="session-AAA",
        scope=shared_scope,
    )
    assert ok_a

    # Dispatch as session-AAA succeeds
    res_a = registry.dispatch("shared_calc", {"x": 3}, scope=shared_scope, session_id="session-AAA")
    assert json.loads(res_a).get("result") == 6

    # Session B registers in same scope
    ok_b, _ = synth_b.synthesize_and_register(
        name="shared_calc",
        description="Calc B",
        parameters_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        python_source=src_b,
        is_ephemeral=True,
        session_id="session-BBB",
        scope=shared_scope,
    )
    assert ok_b

    # Dispatch with session-BBB succeeds
    res_b = registry.dispatch("shared_calc", {"x": 3}, scope=shared_scope, session_id="session-BBB")
    assert json.loads(res_b).get("result") == 30

    # Wrong-session dispatch: session-AAA calling while session-BBB holds slot is rejected with PermissionError
    err_dispatch = registry.dispatch("shared_calc", {"x": 3}, scope=shared_scope, session_id="session-AAA")
    assert "PermissionError" in err_dispatch
    assert "is owned by session 'session-BBB'" in err_dispatch

    # Cleanup
    synth_b.deregister("shared_calc", session_id="session-BBB", scope=shared_scope)
    synth_a.deregister("shared_calc", session_id="session-AAA", scope=shared_scope)


def test_cas_restoration_of_previous_live_owner_across_synthesizers(temp_jit_dir):
    """P1 regression: newer JIT owner teardown restores still-live previous owner across synthesizers."""
    synth_a = JITToolSynthesizer(storage_dir=temp_jit_dir / "ca")
    synth_b = JITToolSynthesizer(storage_dir=temp_jit_dir / "cb")

    scope = "cas_scope"

    src_a = "def cascaded_tool(val: int): return val + 1"
    src_b = "def cascaded_tool(val: int): return val + 100"

    # Synth A registers tool
    ok_a, _ = synth_a.synthesize_and_register(
        name="cascaded_tool",
        description="Tool A",
        parameters_schema={"type": "object", "properties": {"val": {"type": "integer"}}},
        python_source=src_a,
        is_ephemeral=True,
        session_id="session-owner-A",
        scope=scope,
    )
    assert ok_a
    entry_a = registry.snapshot_registration("cascaded_tool", scope=scope)
    assert entry_a is not None

    # Synth B registers tool under same name/scope (new owner)
    ok_b, _ = synth_b.synthesize_and_register(
        name="cascaded_tool",
        description="Tool B",
        parameters_schema={"type": "object", "properties": {"val": {"type": "integer"}}},
        python_source=src_b,
        is_ephemeral=True,
        session_id="session-owner-B",
        scope=scope,
    )
    assert ok_b
    entry_b = registry.snapshot_registration("cascaded_tool", scope=scope)
    assert entry_b is not None
    assert entry_b is not entry_a

    # Verify B's result
    res_b = registry.dispatch("cascaded_tool", {"val": 5}, scope=scope, session_id="session-owner-B")
    assert json.loads(res_b).get("result") == 105

    # Synth B tears down -> MUST restore still-live Synth A's entry!
    synth_b.deregister("cascaded_tool", session_id="session-owner-B", scope=scope)
    restored_entry = registry.snapshot_registration("cascaded_tool", scope=scope)
    assert restored_entry is entry_a

    # Dispatch now executes Synth A's tool under session-owner-A
    res_a = registry.dispatch("cascaded_tool", {"val": 5}, scope=scope, session_id="session-owner-A")
    assert json.loads(res_a).get("result") == 6

    # Synth A tears down -> cleans up registry slot completely
    synth_a.deregister("cascaded_tool", session_id="session-owner-A", scope=scope)
    assert registry.snapshot_registration("cascaded_tool", scope=scope) is None


def test_close_task_resources_with_distinct_session_and_task_id(temp_jit_dir):
    """P1 regression: client_lifecycle _close_task_resources cleans up session-owned JIT tools when task_id != session_id."""
    from agent.client_lifecycle import ClientLifecycleMixin

    class DummyLifecycleAgent(ClientLifecycleMixin):
        def __init__(self, session_id: str):
            self.session_id = session_id
            self._process_owner_task_ids = ()

    synth = JITToolSynthesizer(storage_dir=temp_jit_dir)

    src = "def lifecycle_tool(): return 'ok'"
    ok, _ = synth.synthesize_and_register(
        name="lifecycle_tool",
        description="Lifecycle ephemeral tool",
        parameters_schema={},
        python_source=src,
        is_ephemeral=True,
        session_id="owning-session-777",
    )
    assert ok
    assert synth.get_tool("lifecycle_tool", session_id="owning-session-777") is not None
    assert registry.snapshot_registration("lifecycle_tool") is not None

    # Agent has session_id = 'owning-session-777', closing a delegated task 'delegated-task-999'
    agent = DummyLifecycleAgent(session_id="owning-session-777")
    agent._close_task_resources(task_id="delegated-task-999")

    # Ephemeral tool owned by 'owning-session-777' must be cleaned up
    assert synth.get_tool("lifecycle_tool", session_id="owning-session-777") is None
    assert registry.snapshot_registration("lifecycle_tool") is None

