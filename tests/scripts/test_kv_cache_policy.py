"""Tests for scripts/kv_cache_policy.py (P12 KV-cache placement policy).

Covers the acceptance criteria in kanban task t_510d90f1:

- A policy module exists that accepts context length (and optional
  overrides) and returns the KV-cache tier: GPU-resident or host-RAM
  offload.
- Default threshold is 128K tokens: contexts <=128K use GPU-resident,
  contexts >128K up to 256K use host-RAM offload.
- All thresholds and modes are configurable via config/env with
  documented defaults.
- The policy is the single entry point used by both the GPU fast path and
  the offload path (integration check).
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "kv_cache_policy.py"
)

DEFAULT_THRESHOLD = 131072  # 128K
DEFAULT_MAX = 262144        # 256K


def load_module():
    spec = importlib.util.spec_from_file_location("kv_cache_policy", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Boundary conditions (default config)
# ---------------------------------------------------------------------------

def test_exactly_128k_is_gpu_resident():
    module = load_module()
    decision = module.decide_tier(DEFAULT_THRESHOLD)
    assert decision.tier is module.KVCacheTier.GPU_RESIDENT


def test_128k_plus_1_is_host_ram_offload():
    module = load_module()
    decision = module.decide_tier(DEFAULT_THRESHOLD + 1)
    assert decision.tier is module.KVCacheTier.HOST_RAM_OFFLOAD


def test_zero_length_context_is_gpu_resident():
    module = load_module()
    decision = module.decide_tier(0)
    assert decision.tier is module.KVCacheTier.GPU_RESIDENT


def test_256k_is_host_ram_offload():
    module = load_module()
    decision = module.decide_tier(DEFAULT_MAX)
    assert decision.tier is module.KVCacheTier.HOST_RAM_OFFLOAD


def test_just_below_threshold_is_gpu():
    module = load_module()
    assert module.decide_tier(DEFAULT_THRESHOLD - 1).tier is module.KVCacheTier.GPU_RESIDENT


def test_negative_context_is_config_error():
    module = load_module()
    with pytest.raises(module.PolicyConfigError):
        module.decide_tier(-1)


def test_context_above_max_raises_out_of_range():
    module = load_module()
    with pytest.raises(module.ContextOutOfRangeError):
        module.decide_tier(DEFAULT_MAX + 1)


# ---------------------------------------------------------------------------
# Env/config override behavior
# ---------------------------------------------------------------------------

def test_env_threshold_override_changes_boundary():
    module = load_module()
    env = {"KV_CACHE_OFFLOAD_THRESHOLD": "65536"}
    # 65536 is now the boundary: exactly at it -> GPU, +1 -> offload.
    assert module.decide_tier(65536, env=env).tier is module.KVCacheTier.GPU_RESIDENT
    assert module.decide_tier(65537, env=env).tier is module.KVCacheTier.HOST_RAM_OFFLOAD
    # And the default 128K is now well past the threshold -> offload.
    assert module.decide_tier(131072, env=env).tier is module.KVCacheTier.HOST_RAM_OFFLOAD


def test_env_mode_gpu_forces_gpu_even_above_threshold():
    module = load_module()
    env = {"KV_CACHE_MODE": "gpu"}
    decision = module.decide_tier(200000, env=env)
    assert decision.tier is module.KVCacheTier.GPU_RESIDENT
    assert decision.forced is True


def test_env_mode_offload_forces_offload_even_below_threshold():
    module = load_module()
    env = {"KV_CACHE_MODE": "offload"}
    decision = module.decide_tier(1000, env=env)
    assert decision.tier is module.KVCacheTier.HOST_RAM_OFFLOAD
    assert decision.forced is True


def test_env_max_context_override():
    module = load_module()
    env = {"KV_CACHE_MAX_CONTEXT": "200000"}
    # 200000 is now the max; 200001 overflows.
    assert module.decide_tier(200000, env=env).tier is module.KVCacheTier.HOST_RAM_OFFLOAD
    with pytest.raises(module.ContextOutOfRangeError):
        module.decide_tier(200001, env=env)


def test_config_file_loaded_and_env_wins():
    module = load_module()
    import tempfile
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False
    ) as handle:
        json.dump({"offload_threshold": 8192, "mode": "gpu"}, handle)
        path = handle.name
    try:
        # File alone: threshold 8192, mode gpu.
        env_file = {"KV_CACHE_CONFIG": path}
        config = module.load_config(env=env_file)
        assert config["offload_threshold"] == 8192
        assert config["mode"] == "gpu"
        # Env wins over file: mode override.
        env_both = {"KV_CACHE_CONFIG": path, "KV_CACHE_MODE": "offload"}
        config2 = module.load_config(env=env_both)
        assert config2["mode"] == "offload"
        assert config2["offload_threshold"] == 8192
    finally:
        os.unlink(path)


def test_config_file_missing_raises():
    module = load_module()
    env = {"KV_CACHE_CONFIG": "/nonexistent/path/kv.json"}
    with pytest.raises(module.PolicyConfigError):
        module.load_config(env=env)


def test_config_file_invalid_json_raises():
    module = load_module()
    import tempfile
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False
    ) as handle:
        handle.write("{not valid json")
        path = handle.name
    try:
        env = {"KV_CACHE_CONFIG": path}
        with pytest.raises(module.PolicyConfigError):
            module.load_config(env=env)
    finally:
        os.unlink(path)


def test_config_source_reflects_override_precedence():
    module = load_module()
    assert module.load_config(env={})["source"] == "defaults"
    assert module.load_config(env={"KV_CACHE_MODE": "gpu"})["source"] == "env"


# ---------------------------------------------------------------------------
# Validation errors (invalid values)
# ---------------------------------------------------------------------------

def test_invalid_mode_raises():
    module = load_module()
    with pytest.raises(module.PolicyConfigError):
        module.load_config(env={"KV_CACHE_MODE": "bogus"})


def test_non_integer_threshold_raises():
    module = load_module()
    with pytest.raises(module.PolicyConfigError):
        module.load_config(env={"KV_CACHE_OFFLOAD_THRESHOLD": "abc"})


def test_zero_threshold_raises():
    module = load_module()
    with pytest.raises(module.PolicyConfigError):
        module.load_config(env={"KV_CACHE_OFFLOAD_THRESHOLD": "0"})


def test_negative_threshold_raises():
    module = load_module()
    with pytest.raises(module.PolicyConfigError):
        module.load_config(env={"KV_CACHE_OFFLOAD_THRESHOLD": "-5"})


def test_max_context_below_threshold_raises():
    module = load_module()
    env = {"KV_CACHE_MAX_CONTEXT": "1000", "KV_CACHE_OFFLOAD_THRESHOLD": "2000"}
    with pytest.raises(module.PolicyConfigError):
        module.load_config(env=env)


def test_invalid_override_raises():
    module = load_module()
    with pytest.raises(module.PolicyConfigError):
        module.decide_tier(1000, override="bogus")


# ---------------------------------------------------------------------------
# Per-request override (testing/edge cases)
# ---------------------------------------------------------------------------

def test_override_gpu_on_long_context():
    module = load_module()
    decision = module.decide_tier(200000, override="gpu")
    assert decision.tier is module.KVCacheTier.GPU_RESIDENT
    assert decision.forced is True


def test_override_offload_on_short_context():
    module = load_module()
    decision = module.decide_tier(1000, override="offload")
    assert decision.tier is module.KVCacheTier.HOST_RAM_OFFLOAD
    assert decision.forced is True


def test_override_accepts_enum_member():
    module = load_module()
    decision = module.decide_tier(
        1000, override=module.KVCacheTier.HOST_RAM_OFFLOAD
    )
    assert decision.tier is module.KVCacheTier.HOST_RAM_OFFLOAD


def test_override_wins_over_env_mode():
    module = load_module()
    env = {"KV_CACHE_MODE": "offload"}
    decision = module.decide_tier(1000, override="gpu", env=env)
    assert decision.tier is module.KVCacheTier.GPU_RESIDENT


# ---------------------------------------------------------------------------
# Decision shape / reason
# ---------------------------------------------------------------------------

def test_decision_has_reason_and_dict_shape():
    module = load_module()
    decision = module.decide_tier(131073)
    assert decision.reason
    d = decision.to_dict()
    assert d["tier"] == "offload"
    assert d["context_len"] == 131073
    assert d["threshold"] == DEFAULT_THRESHOLD
    assert d["max_context"] == DEFAULT_MAX
    assert d["mode"] == "auto"
    assert d["forced"] is False


def test_decision_is_deterministic():
    module = load_module()
    first = module.decide_tier(150000).to_dict()
    second = module.decide_tier(150000).to_dict()
    assert first == second


# ---------------------------------------------------------------------------
# Serving-path integration check
# ---------------------------------------------------------------------------

def test_policy_is_single_entry_point_for_serving_path():
    """The serving path consults the policy before KV-cache allocation.

    This simulates the call site a serving layer would use: it asks the
    policy for the tier, then allocates the cache on that tier. It asserts
    the decision object carries everything the allocator needs and that no
    consumer-side threshold duplication is required.
    """
    module = load_module()

    def serving_allocate(context_len, override=None):
        decision = module.decide_tier(context_len, override=override)
        if decision.tier is module.KVCacheTier.GPU_RESIDENT:
            return "gpu-allocated", decision
        return "host-ram-allocated", decision

    # Ordinary context -> GPU fast path.
    tier, decision = serving_allocate(65536)
    assert tier == "gpu-allocated"
    assert decision.tier is module.KVCacheTier.GPU_RESIDENT

    # Long context -> host-RAM offload path.
    tier, decision = serving_allocate(200000)
    assert tier == "host-ram-allocated"
    assert decision.tier is module.KVCacheTier.HOST_RAM_OFFLOAD

    # The allocator never needs to re-implement the threshold rule: the
    # decision already carries it.
    assert decision.threshold == DEFAULT_THRESHOLD


def test_cli_returns_tier_json():
    module = load_module()
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--ctx-size", "131073"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["tier"] == "offload"
    assert payload["context_len"] == 131073


def test_cli_invalid_context_exits_nonzero():
    module = load_module()
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--ctx-size", "999999"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "exceeds supported maximum" in proc.stdout
