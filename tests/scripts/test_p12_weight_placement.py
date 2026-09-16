"""Tests for scripts/p12_weight_placement.py (P12 GPU0-only weight enforcement).

Covers the acceptance criteria in kanban task t_048cef47:

- Model weights are explicitly pinned to GPU0 during load.
- Any attempt to load weights to a non-GPU0 device fails fast with an
  actionable error message.
- Default runtime configuration does not offload weights.
- Existing supported inference paths still load correctly.

The module under test is dependency-free (stdlib only) so these tests run
without torch or the heavy stack. A separate live integration check
(scripts/p12_offload_live_check.py, sibling task) proves real device
placement via nvidia-smi.
"""

import importlib.util
import re
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "p12_weight_placement.py"
)

# A realistic main-lane fast-path argv (mirrors turbohaul-main.service).
FAST_ARGV = ["-m", "darwin.gguf", "--host", "127.0.0.1", "--port", "11500",
             "-ngl", "99", "--ctx-size", "65536", "--mlock", "--no-mmap"]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "p12_weight_placement", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Weights explicitly pinned to GPU0 during load
# ---------------------------------------------------------------------------

def test_default_fast_path_pins_gpu0():
    module = load_module()
    result = module.enforce_gpu0_weights(FAST_ARGV)
    # Canonical GPU0-only pins appended.
    assert result[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


def test_no_weight_offload_flags_in_default_config():
    module = load_module()
    result = module.enforce_gpu0_weights(FAST_ARGV)
    joined = " ".join(result)
    # No flag that moves weights off GPU0 survives the enforcement.
    assert "--device" not in joined
    assert "--n-cpu-moe" not in joined
    assert "--cpu-moe" not in joined
    assert "--op-offload" not in joined


def test_ngl_minus1_all_on_gpu_allowed():
    module = load_module()
    result = module.enforce_gpu0_weights(["-m", "m.gguf", "-ngl", "-1"])
    assert result[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


def test_ngl_all_string_allowed():
    module = load_module()
    result = module.enforce_gpu0_weights(["-m", "m.gguf", "--n-gpu-layers", "all"])
    assert result[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


def test_positive_ngl_covering_all_layers_allowed():
    module = load_module()
    # Darwin 28B has 64 layers; -ngl 99 covers them all.
    result = module.enforce_gpu0_weights(FAST_ARGV, model_layers=64)
    assert result[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


def test_equals_form_normalized_and_allowed():
    module = load_module()
    result = module.enforce_gpu0_weights(
        ["-m", "m.gguf", "--n-gpu-layers=64", "--main-gpu=0"]
    )
    # '=' forms are normalized; canonical pins present.
    assert "--split-mode" in result
    assert result[result.index("--split-mode") + 1] == "none"
    assert result[result.index("--main-gpu") + 1] == "0"


# ---------------------------------------------------------------------------
# Fail-fast: any attempt to place weights off GPU0 raises
# ---------------------------------------------------------------------------

def test_ngl_0_all_weights_cpu_fails_fast():
    module = load_module()
    with pytest.raises(module.WeightPlacementError, match="CPU"):
        module.enforce_gpu0_weights(["-m", "m.gguf", "-ngl", "0"])


def test_ngl_below_model_layers_partial_cpu_fails_fast():
    module = load_module()
    # 32 of 64 layers on GPU -> tail 32 on CPU.
    with pytest.raises(module.WeightPlacementError, match="partial weight offload"):
        module.enforce_gpu0_weights(
            ["-m", "m.gguf", "-ngl", "32"], model_layers=64
        )


def test_ngl_invalid_negative_fails_fast():
    module = load_module()
    with pytest.raises(module.WeightPlacementError, match="invalid"):
        module.enforce_gpu0_weights(["-m", "m.gguf", "-ngl", "-2"])


def test_main_gpu_1_fails_fast():
    module = load_module()
    with pytest.raises(module.WeightPlacementError, match="GPU1"):
        module.enforce_gpu0_weights(["-m", "m.gguf", "--main-gpu", "1"])


def test_main_gpu_1_equals_form_fails_fast():
    module = load_module()
    with pytest.raises(module.WeightPlacementError, match="GPU1"):
        module.enforce_gpu0_weights(["-m", "m.gguf", "--main-gpu=1"])


def test_device_includes_gpu1_fails_fast():
    module = load_module()
    with pytest.raises(module.WeightPlacementError, match="non-GPU0"):
        module.enforce_gpu0_weights(["-m", "m.gguf", "--device", "0,1"])


def test_device_gpu1_only_fails_fast():
    module = load_module()
    with pytest.raises(module.WeightPlacementError, match="non-GPU0"):
        module.enforce_gpu0_weights(["-m", "m.gguf", "--device", "1"])


def test_split_mode_layer_fails_fast():
    module = load_module()
    with pytest.raises(module.WeightPlacementError, match="spread"):
        module.enforce_gpu0_weights(["-m", "m.gguf", "--split-mode", "layer"])


def test_split_mode_tensor_fails_fast():
    module = load_module()
    with pytest.raises(module.WeightPlacementError, match="spread"):
        module.enforce_gpu0_weights(["-m", "m.gguf", "--split-mode", "tensor"])


def test_cpu_moe_fails_fast():
    module = load_module()
    with pytest.raises(module.WeightPlacementError, match="CPU"):
        module.enforce_gpu0_weights(["-m", "m.gguf", "--cpu-moe"])


def test_n_cpu_moe_positive_fails_fast():
    module = load_module()
    with pytest.raises(module.WeightPlacementError, match="CPU"):
        module.enforce_gpu0_weights(["-m", "m.gguf", "--n-cpu-moe", "2"])


def test_op_offload_fails_fast():
    module = load_module()
    with pytest.raises(module.WeightPlacementError, match="CPU"):
        module.enforce_gpu0_weights(["-m", "m.gguf", "--op-offload"])


def test_cuda_visible_devices_1_fails_fast():
    module = load_module()
    with pytest.raises(module.WeightPlacementError, match="CUDA_VISIBLE_DEVICES"):
        module.enforce_gpu0_weights(FAST_ARGV, env={"CUDA_VISIBLE_DEVICES": "1"})


def test_cuda_visible_devices_0_allowed():
    module = load_module()
    result = module.enforce_gpu0_weights(
        FAST_ARGV, env={"CUDA_VISIBLE_DEVICES": "0"}
    )
    assert result[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


def test_cuda_visible_devices_unset_allowed():
    module = load_module()
    result = module.enforce_gpu0_weights(FAST_ARGV, env={})
    assert result[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


# ---------------------------------------------------------------------------
# Error messages are actionable (naming the fix)
# ---------------------------------------------------------------------------

def test_error_message_names_remediation():
    module = load_module()
    with pytest.raises(module.WeightPlacementError) as excinfo:
        module.enforce_gpu0_weights(["-m", "m.gguf", "--main-gpu", "1"])
    msg = str(excinfo.value)
    assert "GPU0" in msg
    assert "0" in msg  # names the fix (change to 0 / remove)


def test_error_message_names_ngl_fix():
    module = load_module()
    with pytest.raises(module.WeightPlacementError) as excinfo:
        module.enforce_gpu0_weights(["-m", "m.gguf", "-ngl", "0"])
    msg = str(excinfo.value)
    assert "-ngl -1" in msg or "layer count" in msg


# ---------------------------------------------------------------------------
# Existing supported inference paths still load correctly
# ---------------------------------------------------------------------------

def test_production_main_service_argv_passes():
    module = load_module()
    # The exact ExecStart from scripts/systemd/turbohaul-main.service.
    prod_argv = ["-m", "/home/sahil/ai/models/llm/Darwin-28B-REASON.Q5_K_M.gguf",
                 "--host", "127.0.0.1", "--port", "11500",
                 "-ngl", "99", "--ctx-size", "65536",
                 "--threads", "8", "--threads-batch", "8",
                 "--mlock", "--no-mmap"]
    result = module.enforce_gpu0_weights(prod_argv)
    assert result[-4:] == ["--split-mode", "none", "--main-gpu", "0"]
    # Original fast-path flags preserved.
    for flag in ("-m", "--host", "--port", "-ngl", "--ctx-size", "--mlock", "--no-mmap"):
        assert flag in result


def test_unknown_argv_passthrough_is_safe():
    module = load_module()
    argv = ["--weird-flag", "x", "-m", "model.gguf", "-ngl", "99"]
    result = module.enforce_gpu0_weights(argv)
    # Unknown flags pass through untouched; canonical GPU0 pins appended.
    assert result[: len(argv)] == argv
    assert result[-4:] == ["--split-mode", "none", "--main-gpu", "0"]


def test_empty_argv_gets_gpu0_pins():
    module = load_module()
    result = module.enforce_gpu0_weights([])
    assert result == ["--split-mode", "none", "--main-gpu", "0"]


# ---------------------------------------------------------------------------
# Deterministic / pure-function guarantees
# ---------------------------------------------------------------------------

def test_enforce_is_pure_and_repeatable():
    module = load_module()
    first = module.enforce_gpu0_weights(FAST_ARGV)
    second = module.enforce_gpu0_weights(FAST_ARGV)
    assert first == second


def test_aux_lane_gpu1_not_touched_by_enforcement():
    module = load_module()
    # The aux lane argv (GPU1) must be REJECTED for the main-lane gate:
    # its CUDA_VISIBLE_DEVICES=1 env remaps GPU0 -> physical GPU1.
    aux_argv = ["-m", "qwopus.gguf", "--host", "127.0.0.1", "--port", "8082",
                "-ngl", "99", "--ctx-size", "65536"]
    with pytest.raises(module.WeightPlacementError, match="CUDA_VISIBLE_DEVICES"):
        module.enforce_gpu0_weights(aux_argv, env={"CUDA_VISIBLE_DEVICES": "1"})


def test_cli_exits_2_on_violation():
    module = load_module()
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--", "-m", "m.gguf", "--main-gpu", "1"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 2
    assert "VIOLATION" in proc.stderr


def test_cli_ok_on_fast_path():
    module = load_module()
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--", "-m", "m.gguf", "-ngl", "99"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert "OK: weights GPU0-only" in proc.stdout
