"""Tests for scripts/p12_offload_gate.py (P12 opt-in KV-cache offload gate).

Covers the acceptance criteria in kanban task t_14bd82aa:

- An explicit opt-in flag exists for CPU/system-RAM context/KV-cache offload.
- The flag defaults to off, preserving the normal fast path.
- Offload is limited to context/KV-cache data; model weights stay on GPU0.
- Enabled mode supports rare huge-context workloads without corrupting
  inference state (argv produced is the exact offload invocation).
- Disabled mode refuses or neutralizes offload attempts.
- GPU1 isolation is preserved: no weight placement and no offload target
  ever references GPU1.
"""

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "p12_offload_gate.py"
)

HUGE = 262144
NORMAL = 65536
# The placement policy's default boundary (single source of truth):
# <=128K GPU-resident, >128K offload. The gate must not duplicate it.
POLICY_THRESHOLD = 131072  # 128K

# A realistic main-lane fast-path argv (mirrors turbohaul-main.service).
FAST_ARGV = ["-m", "darwin.gguf", "--host", "127.0.0.1", "--port", "11500",
             "-ngl", "99", "--ctx-size", "65536", "--mlock", "--no-mmap"]

# A huge-context offload request.
OFFLOAD_ARGV = ["-m", "darwin.gguf", "-ngl", "99", "--no-kv-offload",
                "--ctx-size", "262144"]


def load_module():
    spec = importlib.util.spec_from_file_location("p12_offload_gate", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Opt-in flag exists and defaults to off
# ---------------------------------------------------------------------------

def test_flag_env_name_is_explicit():
    module = load_module()
    assert module.ENV_OFFLOAD_FLAG == "P12_ALLOW_HUGE_CONTEXT_OFFLOAD"


def test_offload_defaults_to_off_when_env_unset():
    module = load_module()
    assert module.offload_enabled(env={}) is False


def test_offload_off_for_arbitrary_values():
    module = load_module()
    for value in ("0", "", "false", "no", "off", "yesplease", "2"):
        assert module.offload_enabled(env={module.ENV_OFFLOAD_FLAG: value}) is False


def test_offload_on_only_for_true_values():
    module = load_module()
    for value in ("1", "true", "TRUE", "yes", "on"):
        assert module.offload_enabled(env={module.ENV_OFFLOAD_FLAG: value}) is True


# ---------------------------------------------------------------------------
# Disabled mode refuses / neutralizes offload attempts
# ---------------------------------------------------------------------------

def test_off_disabled_rejects_offload_request():
    module = load_module()
    decision = module.decide_offload(OFFLOAD_ARGV, HUGE, env={})
    assert decision.allowed is True       # request is recognized
    assert decision.enabled is False      # but not executed
    assert "--no-kv-offload" not in decision.argv  # neutralized


def test_off_disabled_preserves_fast_path():
    module = load_module()
    decision = module.decide_offload(FAST_ARGV, NORMAL, env={})
    assert decision.enabled is False
    # Fast path preserved: original flags intact, plus GPU0-only pins.
    for flag in ("-m", "-ngl", "--mlock", "--no-mmap"):
        assert flag in decision.argv
    assert "--no-kv-offload" not in decision.argv
    assert decision.argv[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


def test_off_disabled_strips_kv_offload_but_keeps_weights_gpu0():
    module = load_module()
    decision = module.decide_offload(OFFLOAD_ARGV, HUGE, env={})
    # Weights stay on GPU0 (main-gpu 0 or binary default), KV offload gone.
    assert "--no-kv-offload" not in decision.argv
    assert "-nkvo" not in decision.argv
    # No offload flag remains.
    for flag in ("--no-kv-offload", "-nkvo"):
        assert flag not in decision.argv
    # GPU0-only pins appended (canonical split-mode none + main-gpu 0).
    assert decision.argv[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


def test_off_disabled_short_context_never_offloads():
    module = load_module()
    decision = module.decide_offload(OFFLOAD_ARGV, NORMAL, env={})
    assert decision.enabled is False
    assert "--no-kv-offload" not in decision.argv


# ---------------------------------------------------------------------------
# Enabled mode: huge-context KV-cache offload works
# ---------------------------------------------------------------------------

def test_on_enabled_keeps_kv_offload_for_huge_context():
    module = load_module()
    env = {module.ENV_OFFLOAD_FLAG: "1"}
    decision = module.decide_offload(OFFLOAD_ARGV, HUGE, env=env)
    assert decision.allowed is True
    assert decision.enabled is True
    assert "--no-kv-offload" in decision.argv
    assert decision.kv_offload is True


def test_on_enabled_normal_context_still_rejected():
    module = load_module()
    env = {module.ENV_OFFLOAD_FLAG: "1"}
    decision = module.decide_offload(OFFLOAD_ARGV, NORMAL, env=env)
    assert decision.enabled is False
    assert "--no-kv-offload" not in decision.argv


def test_on_enabled_forces_weights_to_gpu0():
    module = load_module()
    env = {module.ENV_OFFLOAD_FLAG: "1"}
    # A manifest tries to pin to GPU1; enabled path must force GPU0.
    bad = ["-m", "darwin.gguf", "--no-kv-offload", "--main-gpu", "1",
           "--ctx-size", "262144"]
    decision = module.decide_offload(bad, HUGE, env=env)
    # Weights-pinned-to-GPU1 is a hard violation regardless of the flag.
    assert decision.allowed is False
    assert decision.enabled is False
    assert "--main-gpu" not in decision.argv or decision.argv[decision.argv.index("--main-gpu") + 1] == "0"


def test_on_enabled_rewrites_device_to_gpu0():
    module = load_module()
    env = {module.ENV_OFFLOAD_FLAG: "1"}
    argv = ["-m", "darwin.gguf", "--no-kv-offload", "--device", "0", "--ctx-size", "262144"]
    decision = module.decide_offload(argv, HUGE, env=env)
    assert decision.enabled is True
    # --device 0 is GPU0-only, so the gate canonicalizes it away and emits
    # the explicit GPU0-only pins (split-mode none + main-gpu 0). No device
    # reference other than the canonical GPU0 pins may survive.
    assert "--device" not in decision.argv
    # GPU0-only pins present.
    assert decision.argv[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


def test_on_enabled_produces_roundtrippable_argv():
    module = load_module()
    env = {module.ENV_OFFLOAD_FLAG: "1"}
    decision = module.decide_offload(OFFLOAD_ARGV, HUGE, env=env)
    # The sanitized argv is exactly what a launch wrapper would exec:
    # KV offload kept, weights pinned GPU0-only.
    assert decision.argv == ["-m", "darwin.gguf", "-ngl", "99",
                             "--no-kv-offload", "--ctx-size", "262144",
                             "--split-mode", "none", "--main-gpu", "0"]


# ---------------------------------------------------------------------------
# Weights GPU0-only — never CPU / system RAM / GPU1
# ---------------------------------------------------------------------------

def test_gpu1_main_gpu_is_hard_violation_off():
    module = load_module()
    argv = ["-m", "darwin.gguf", "-ngl", "99", "--main-gpu", "1",
            "--ctx-size", "262144"]
    decision = module.decide_offload(argv, HUGE, env={})
    assert decision.allowed is False
    assert "--no-kv-offload" not in decision.argv


def test_gpu1_main_gpu_is_hard_violation_even_when_enabled():
    module = load_module()
    env = {module.ENV_OFFLOAD_FLAG: "1"}
    argv = ["-m", "darwin.gguf", "--no-kv-offload", "--main-gpu", "1",
            "--ctx-size", "262144"]
    decision = module.decide_offload(argv, HUGE, env=env)
    assert decision.allowed is False
    assert decision.enabled is False
    assert "--no-kv-offload" not in decision.argv


def test_device_list_including_gpu1_is_violation():
    module = load_module()
    argv = ["-m", "darwin.gguf", "--device", "0,1", "--ctx-size", "262144"]
    decision = module.decide_offload(argv, HUGE, env={})
    assert decision.allowed is False
    assert "--device" not in decision.argv


def test_device_gpu0_alone_is_allowed():
    module = load_module()
    argv = ["-m", "darwin.gguf", "--device", "0", "--ctx-size", "262144"]
    decision = module.decide_offload(argv, HUGE, env={})
    assert decision.allowed is True
    assert decision.enabled is False


def test_no_weight_offload_flags_survive_disabled_path():
    module = load_module()
    # Every flag that could move weights off GPU0 must be neutralized.
    argv = ["-m", "darwin.gguf", "--main-gpu", "1", "--device", "1",
            "--no-kv-offload", "--ctx-size", "262144"]
    decision = module.decide_offload(argv, HUGE, env={})
    assert "--no-kv-offload" not in decision.argv
    # Only the canonical GPU0 pins may reference --main-gpu / --device.
    assert decision.argv[-4:] == ["--split-mode", "none", "--main-gpu", "0"]
    assert "--device" not in decision.argv[:-4]


# ---------------------------------------------------------------------------
# GPU1 isolation — offload target never GPU1
# ---------------------------------------------------------------------------

def test_offload_decision_never_reports_gpu1_as_target():
    module = load_module()
    env = {module.ENV_OFFLOAD_FLAG: "1"}
    decision = module.decide_offload(OFFLOAD_ARGV, HUGE, env=env)
    assert decision.main_gpu == 0
    assert "1" not in [str(decision.main_gpu)]


def test_enabled_argv_never_contains_gpu1_reference():
    module = load_module()
    env = {module.ENV_OFFLOAD_FLAG: "1"}
    argv = ["-m", "darwin.gguf", "--no-kv-offload", "--device", "0",
            "--ctx-size", "262144"]
    decision = module.decide_offload(argv, HUGE, env=env)
    joined = " ".join(decision.argv)
    # No bare '1' device reference; '--ctx-size 262144' etc must not confuse.
    assert "--device" not in joined or decision.argv[decision.argv.index("--device") + 1] == "0"
    assert "--main-gpu" not in joined or decision.argv[decision.argv.index("--main-gpu") + 1] == "0"


# ---------------------------------------------------------------------------
# GPU0-only weights — split-mode guard (production bug caught by live test)
# ---------------------------------------------------------------------------

def test_split_mode_layer_is_hard_violation():
    module = load_module()
    argv = ["-m", "darwin.gguf", "--split-mode", "layer", "--ctx-size", "262144"]
    decision = module.decide_offload(argv, HUGE, env={})
    assert decision.allowed is False
    # Neutralized: the dangerous split-mode layer is stripped, leaving only
    # the canonical GPU0-only pins (split-mode none + main-gpu 0).
    assert decision.argv[-4:] == ["--split-mode", "none", "--main-gpu", "0"]
    assert "--split-mode" in decision.argv  # only as the canonical "none" pin
    assert decision.argv[decision.argv.index("--split-mode") + 1] == "none"


def test_split_mode_tensor_is_hard_violation():
    module = load_module()
    argv = ["-m", "darwin.gguf", "--split-mode", "tensor", "--ctx-size", "262144"]
    decision = module.decide_offload(argv, HUGE, env={})
    assert decision.allowed is False


def test_split_mode_none_is_allowed():
    module = load_module()
    argv = ["-m", "darwin.gguf", "--split-mode", "none", "--ctx-size", "262144"]
    decision = module.decide_offload(argv, HUGE, env={})
    assert decision.allowed is True
    # GPU0-only pins kept.
    assert decision.argv[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


def test_fast_path_always_pins_gpu0_weights():
    module = load_module()
    # The production main.service argv (no split-mode / main-gpu / device).
    prod_argv = ["-m", "darwin.gguf", "--host", "127.0.0.1", "--port", "11500",
                 "-ngl", "99", "--ctx-size", "65536", "--mlock", "--no-mmap"]
    decision = module.decide_offload(prod_argv, NORMAL, env={})
    # Fast path must still emit GPU0-only pins so weights never pipeline-split
    # onto GPU1 on the dual-GPU rig.
    assert decision.argv[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


# ---------------------------------------------------------------------------
# Deterministic / pure-function guarantees
# ---------------------------------------------------------------------------

def test_decision_is_pure_and_repeatable():
    module = load_module()
    env = {module.ENV_OFFLOAD_FLAG: "1"}
    first = module.decide_offload(OFFLOAD_ARGV, HUGE, env=env).to_dict()
    second = module.decide_offload(OFFLOAD_ARGV, HUGE, env=env).to_dict()
    assert first == second


def test_argv_with_equals_forms_is_normalized():
    module = load_module()
    argv = ["-m", "darwin.gguf", "--no-kv-offload", "--main-gpu=1",
            "--ctx-size", "262144"]
    decision = load_module().decide_offload(argv, HUGE, env={})
    assert decision.allowed is False  # main-gpu=1 caught after normalization


def test_negative_ctx_or_zero_is_never_huge():
    module = load_module()
    env = {module.ENV_OFFLOAD_FLAG: "1"}
    for ctx in (0, -1, 4096, 65536, 131072):
        decision = module.decide_offload(OFFLOAD_ARGV, ctx, env=env)
        assert decision.enabled is False


def test_unknown_argv_passthrough_is_safe():
    module = load_module()
    argv = ["--weird-flag", "x", "-m", "model.gguf"]
    decision = module.decide_offload(argv, NORMAL, env={})
    # Unknown flags pass through untouched; the gate only appends the
    # canonical GPU0-only pins so weights can never leave GPU0.
    assert decision.argv[: len(argv)] == argv
    assert decision.argv[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


# ---------------------------------------------------------------------------
# Placement-policy coordination (single source of truth, kanban t_510d90f1)
# ---------------------------------------------------------------------------

def test_gate_uses_policy_threshold_for_offload_eligibility():
    """The gate must not duplicate the 128K threshold; it consults the
    placement policy (scripts/kv_cache_policy.py)."""
    module = load_module()
    env_on = {module.ENV_OFFLOAD_FLAG: "1"}

    # 128K exactly -> policy says GPU-resident -> gate refuses offload even
    # with the flag on.
    at = module.decide_offload(OFFLOAD_ARGV, POLICY_THRESHOLD, env=env_on)
    assert at.enabled is False
    assert "--no-kv-offload" not in at.argv

    # 128K+1 -> policy says offload -> gate allows when flag is on.
    above = module.decide_offload(OFFLOAD_ARGV, POLICY_THRESHOLD + 1, env=env_on)
    assert above.allowed is True
    assert above.enabled is True
    assert "--no-kv-offload" in above.argv

    # 128K+1 with flag off -> gate refuses (opt-in still required).
    above_off = module.decide_offload(OFFLOAD_ARGV, POLICY_THRESHOLD + 1, env={})
    assert above_off.enabled is False
    assert "--no-kv-offload" not in above_off.argv


def test_gate_threshold_alias_matches_policy_default():
    """HUGE_CONTEXT_MIN_TOKENS is a backward-compatible alias for the
    policy default; it must equal the policy's DEFAULT_OFFLOAD_THRESHOLD."""
    module = load_module()
    import importlib.util as _util
    from pathlib import Path as _Path
    policy_path = _Path(__file__).resolve().parents[2] / "scripts" / "kv_cache_policy.py"
    spec = _util.spec_from_file_location("kv_cache_policy", policy_path)
    assert spec and spec.loader
    policy = _util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    assert module.HUGE_CONTEXT_MIN_TOKENS == policy.DEFAULT_OFFLOAD_THRESHOLD


# ---------------------------------------------------------------------------
# Direct-CLI import regression (t_dd6c32b4 docs task)
# ---------------------------------------------------------------------------

def test_direct_script_invocation_imports_real_policy_not_fallback():
    """`python scripts/p12_offload_gate.py` must resolve the repo-root
    `scripts` package so it imports the real kv_cache_policy instead of the
    ImportError fallback (which hardcodes 262144 as the offload threshold and
    silently diverges from the central policy)."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--ctx-size", "131073",
         "--", "-m", "darwin.gguf", "-ngl", "99", "--no-kv-offload",
         "--ctx-size", "262144"],
        capture_output=True, text=True,
        env={"P12_ALLOW_HUGE_CONTEXT_OFFLOAD": "1", "PATH": "/usr/bin:/bin"},
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert proc.returncode == 0, proc.stderr
    import json
    decision = json.loads(proc.stdout)
    # 131073 (>128K) with the flag on must be allowed+enabled by the real
    # policy. Before the fix, the fallback threshold 262144 made the gate
    # refuse it (allowed=False) — the documented CLI example was wrong.
    assert decision["allowed"] is True
    assert decision["enabled"] is True
    assert "--no-kv-offload" in decision["argv"]

