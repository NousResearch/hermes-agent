"""Tests for the P12 normal-daily-work fast-path contract (CI-safe).

Covers the acceptance criteria of kanban task t_b3c101d0 that can be
asserted WITHOUT a GPU or a live llama-server. These run in normal CI on
every PR:

- Normal short-context workloads take the fast path: model weights on
  GPU0, no CPU/system-RAM weight offload, no KV-cache offload flags.
- GPU1 isolation is preserved at the argv level: neither the default fast
  path nor a huge-context offload request (flag OFF, the default) may
  reference GPU1, and the neutralized offload argv contains no offload flag.
- A huge-context offload job cannot unintentionally consume GPU1.

The live device-level checks (nvidia-smi placement, perf baseline) live in
scripts/p12_fast_path_verify.py and run on the rig / a self-hosted GPU
runner (see .github/workflows/p12-gpu-verify.yml, manual dispatch).
"""

import importlib.util
from pathlib import Path

import pytest

WEIGHT_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "p12_weight_placement.py"
)
GATE_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "p12_offload_gate.py"
)

# The production main-lane fast path (mirrors turbohaul-main.service).
FAST_ARGV = ["-m", "/home/sahil/ai/models/llm/Darwin-28B-REASON.Q5_K_M.gguf",
             "--host", "127.0.0.1", "--port", "11500",
             "-ngl", "99", "--ctx-size", "65536",
             "--threads", "8", "--threads-batch", "8",
             "--mlock", "--no-mmap"]

# A huge-context offload request (the P12 256K path).
HUGE = 262144
OFFLOAD_ARGV = ["-m", "darwin.gguf", "-ngl", "99", "--no-kv-offload",
                "--ctx-size", str(HUGE)]

# Flags that would place weights off GPU0 or KV cache off the GPU.
WEIGHT_OFFLOAD_FLAGS = {"--device", "--cpu-moe", "-cmoe", "--n-cpu-moe",
                        "--op-offload", "--split-mode"}
KV_OFFLOAD_FLAGS = {"--no-kv-offload", "-nkvo"}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def weight_module():
    return load_module(WEIGHT_MODULE_PATH, "p12_weight_placement")


@pytest.fixture(scope="module")
def gate_module():
    return load_module(GATE_MODULE_PATH, "p12_offload_gate")


# ---------------------------------------------------------------------------
# Fast path: weights GPU0-only, no offload
# ---------------------------------------------------------------------------

def test_default_fast_path_pins_gpu0(weight_module):
    result = weight_module.enforce_gpu0_weights(FAST_ARGV)
    # Canonical GPU0-only pins appended; original flags preserved.
    assert result[-4:] == ["--split-mode", "none", "--main-gpu", "0"]
    assert result[: len(FAST_ARGV)] == FAST_ARGV


def test_fast_path_has_no_offload_flags(weight_module):
    """Normal daily work must not carry any weight- or KV-offload flag."""
    result = weight_module.enforce_gpu0_weights(FAST_ARGV)
    joined = " ".join(result)
    for flag in WEIGHT_OFFLOAD_FLAGS - {"--split-mode"} | KV_OFFLOAD_FLAGS:
        assert flag not in joined, f"fast path must not contain {flag}"
    # --split-mode is allowed only as the canonical "none" pin.
    assert result[result.index("--split-mode") + 1] == "none"


def test_fast_path_never_references_gpu1(weight_module):
    result = weight_module.enforce_gpu0_weights(FAST_ARGV)
    joined = " ".join(result)
    assert "--main-gpu" in joined  # canonical pin present
    assert result[result.index("--main-gpu") + 1] == "0"
    assert "--device" not in joined


def test_ngl_all_layers_on_gpu_allowed(weight_module):
    # Darwin 28B has 64 layers; -ngl 99 covers them all (no CPU tail).
    result = weight_module.enforce_gpu0_weights(FAST_ARGV, model_layers=64)
    assert result[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


def test_ngl_zero_rejected_as_cpu_offload(weight_module):
    with pytest.raises(weight_module.WeightPlacementError, match="CPU"):
        weight_module.enforce_gpu0_weights(["-m", "m.gguf", "-ngl", "0"])


def test_partial_ngl_rejected_as_cpu_offload(weight_module):
    with pytest.raises(weight_module.WeightPlacementError, match="offload"):
        weight_module.enforce_gpu0_weights(
            ["-m", "m.gguf", "-ngl", "32"], model_layers=64
        )


def test_moe_cpu_offload_rejected(weight_module):
    for flag in ("--cpu-moe", "-cmoe", "--op-offload"):
        with pytest.raises(weight_module.WeightPlacementError, match="CPU"):
            weight_module.enforce_gpu0_weights(["-m", "m.gguf", flag])


def test_cuda_visible_devices_remap_rejected(weight_module):
    # CUDA_VISIBLE_DEVICES=1 renumbers physical GPUs: "GPU0" would be the
    # physical GPU1 card — a hard violation of the isolation contract.
    with pytest.raises(weight_module.WeightPlacementError, match="CUDA_VISIBLE_DEVICES"):
        weight_module.enforce_gpu0_weights(FAST_ARGV, env={"CUDA_VISIBLE_DEVICES": "1"})


# ---------------------------------------------------------------------------
# Offload gate: default OFF preserves fast path, GPU1 never consumed
# ---------------------------------------------------------------------------

def test_offload_flag_defaults_off(gate_module):
    assert gate_module.offload_enabled(env={}) is False


def test_normal_context_never_offloads(gate_module):
    decision = gate_module.decide_offload(FAST_ARGV, 65536, env={})
    assert decision.enabled is False
    for flag in KV_OFFLOAD_FLAGS:
        assert flag not in decision.argv


def test_huge_context_flag_off_neutralized_to_fast_path(gate_module):
    """A huge-context offload request with the flag OFF (default) must fall
    back to the fast path — no offload, GPU1 untouched."""
    decision = gate_module.decide_offload(OFFLOAD_ARGV, HUGE, env={})
    assert decision.enabled is False
    for flag in KV_OFFLOAD_FLAGS:
        assert flag not in decision.argv
    # GPU0-only pins present; no GPU1 reference.
    assert decision.argv[-4:] == ["--split-mode", "none", "--main-gpu", "0"]
    joined = " ".join(decision.argv)
    assert "--device" not in joined


def test_huge_context_flag_on_never_uses_gpu1(gate_module):
    """Even opted-in huge-context offload targets host RAM, never GPU1."""
    env = {gate_module.ENV_OFFLOAD_FLAG: "1"}
    decision = gate_module.decide_offload(OFFLOAD_ARGV, HUGE, env=env)
    assert decision.enabled is True
    assert decision.main_gpu == 0
    joined = " ".join(decision.argv)
    assert "--device" not in joined or decision.argv[decision.argv.index("--device") + 1] == "0"


def test_gpu1_main_gpu_refused_even_when_enabled(gate_module):
    env = {gate_module.ENV_OFFLOAD_FLAG: "1"}
    bad = ["-m", "darwin.gguf", "--no-kv-offload", "--main-gpu", "1",
           "--ctx-size", str(HUGE)]
    decision = gate_module.decide_offload(bad, HUGE, env=env)
    assert decision.allowed is False
    assert decision.enabled is False
    assert "--no-kv-offload" not in decision.argv


def test_split_mode_spread_refused(gate_module):
    for mode in ("layer", "row", "tensor"):
        argv = ["-m", "darwin.gguf", "--split-mode", mode, "--ctx-size", str(HUGE)]
        decision = gate_module.decide_offload(argv, HUGE, env={})
        assert decision.allowed is False
        # Neutralized to the canonical GPU0-only pins.
        assert decision.argv[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


# ---------------------------------------------------------------------------
# GPU1 isolation end-to-end at the argv level
# ---------------------------------------------------------------------------

def test_default_workflow_never_references_gpu1(weight_module, gate_module):
    """The whole default pipeline (enforce + decide) must never emit a GPU1
    device reference for normal work."""
    enforced = weight_module.enforce_gpu0_weights(FAST_ARGV)
    decision = gate_module.decide_offload(enforced, 65536, env={})
    assert decision.enabled is False
    joined = " ".join(decision.argv)
    assert "--device" not in joined
    assert "--main-gpu" not in joined or decision.argv[decision.argv.index("--main-gpu") + 1] == "0"
    assert decision.argv[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


def test_huge_context_offload_does_not_consume_gpu1(gate_module):
    """The huge-context path (flag OFF default) produces an argv with no GPU1
    reference at all — the offload job cannot unintentionally consume GPU1."""
    decision = gate_module.decide_offload(OFFLOAD_ARGV, HUGE, env={})
    joined = " ".join(decision.argv)
    assert "--device" not in joined
    assert "--main-gpu" not in joined or decision.argv[decision.argv.index("--main-gpu") + 1] == "0"
    # The only split-mode is the canonical "none" pin.
    assert decision.argv[decision.argv.index("--split-mode") + 1] == "none"
