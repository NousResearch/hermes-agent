#!/usr/bin/env python3
"""P12 GPU0-only weight-placement enforcement (fail-fast).

Implements the weight clause of docs/adr/0012-p12-256k-memory-policy.md:

  Model weights are GPU0-only, always. They must never be placed on CPU,
  system RAM, or GPU1. There is no opt-in that lifts this.

This module is the strict FAIL-FAST gate for the *weight load path*. Where
``p12_offload_gate.py`` *neutralizes* offload requests at the argv level
(removing flags so the server falls back to the fast path), this module
*raises* on any configuration that would place weight tensors off GPU0 —
because there is no safe fallback for weights: a model whose weights land
on CPU or GPU1 has silently destroyed the fast path and, in the GPU1 case,
violated the isolation contract.

Checked before spawn (any code path that turns a manifest into argv):

1. ``n_gpu_layers`` / ``-ngl`` — must be -1 (all) or >= 1. ``0`` means ALL
   weights on CPU; a small positive value means the tail layers stay on CPU
   (partial weight offload). Both violate GPU0-only.
2. ``--main-gpu`` / ``-mg`` — must reference GPU0 only.
3. ``--device`` / ``-dev`` — comma list must contain only GPU0.
4. ``--split-mode`` / ``-sm`` — must be ``none`` (layer/row/tensor spread
   weights across GPUs, placing some on GPU1).
5. MoE CPU offload flags (``-cmoe`` / ``--cpu-moe``, ``-ncmoe`` /
   ``--n-cpu-moe``, ``--op-offload``) — place expert/op weights on CPU.
6. ``CUDA_VISIBLE_DEVICES`` — if set, must be exactly ``0`` (or
   ``0,<others>`` ONLY when the first entry is 0 AND the variable cannot
   silently remap GPU0 -> another physical card; the strict policy for the
   main lane is ``0`` alone, because on this rig the variable renumbers
   physical GPUs and ``CUDA_VISIBLE_DEVICES=1`` would make "GPU0" the
   physical GPU1 card).

The module is deliberately dependency-free (stdlib only) so it can run as a
pure decision function in tests, CI, or any launch wrapper without importing
torch or the heavy stack.

Usage (launch wrapper):
    from scripts.p12_weight_placement import (
        enforce_gpu0_weights, WeightPlacementError,
    )

    # argv is the llama-server argv the spawn would exec.
    enforce_gpu0_weights(argv, env=os.environ)  # raises on violation

Return value is the (possibly normalized) argv — the canonical GPU0-only
pins (--split-mode none --main-gpu 0) are appended so the server cannot
pipeline-split weights onto GPU1 even when the manifest omitted them.
"""

from __future__ import annotations

import os
from typing import Sequence

# The only GPU the main lane is ever allowed to pin weights to.
ALLOWED_MAIN_GPU = 0

# The split mode that guarantees GPU0-only weights on the main lane.
ALLOWED_SPLIT_MODE = "none"

# llama.cpp flags that place model weights on a specific GPU.
MAIN_GPU_FLAGS = {"--main-gpu", "-mg"}

# llama.cpp flags that select devices for layer/weight offload.
DEVICE_FLAGS = {"--device", "-dev"}

# llama.cpp flag controlling how the model is split across GPUs.
SPLIT_MODE_FLAGS = {"--split-mode", "-sm"}

# llama.cpp flags controlling how many layers are placed on the GPU.
# -1 = all layers on GPU (llama.cpp convention), 0 = none (all on CPU),
# N >= 1 = first N layers on GPU, remainder on CPU.
N_GPU_LAYERS_FLAGS = {"--n-gpu-layers", "-ngl"}

# llama.cpp flags that place MoE expert / op weights on CPU.
MOE_CPU_OFFLOAD_FLAGS = {
    "--cpu-moe",      # -cmoe: all MoE experts on CPU
    "-cmoe",
    "--n-cpu-moe",    # -ncmoe N: N MoE layers on CPU
    "-ncmoe",
    "--op-offload",   # Tom's Fork: offload ops to CPU
}

# Environment variable that renumbers physical GPUs. The main lane must run
# with GPU0 = physical GPU0; any other value silently remaps the contract.
CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"


class WeightPlacementError(ValueError):
    """Raised when a load path would place weights off GPU0.

    Carries a human-actionable message naming the offending flag/value and
    the exact remediation (remove the flag or pin to GPU0).
    """


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """Normalize argv so flag+value pairs are detected even with '=' forms."""
    normalized: list[str] = []
    for token in argv:
        if token.startswith("--") and "=" in token:
            key, _, value = token.partition("=")
            if key in MAIN_GPU_FLAGS or key in DEVICE_FLAGS or key in SPLIT_MODE_FLAGS:
                normalized.extend([key, value])
            elif key in N_GPU_LAYERS_FLAGS:
                normalized.extend([key, value])
            elif key in MOE_CPU_OFFLOAD_FLAGS and key in {
                "--cpu-moe", "--n-cpu-moe", "--op-offload",
            }:
                normalized.extend([key, value])
            else:
                normalized.append(token)
        else:
            normalized.append(token)
    return normalized


def _flag_value(argv: Sequence[str], flags: set[str], i: int):
    """Return (found, value) for the flag at index i, handling '=' forms.

    A following token is treated as the flag's VALUE when it does not look
    like a flag — with one important exception: negative numbers (``-1``,
    ``-2``) ARE values (llama.cpp ``-ngl -1`` means all layers on GPU), so
    a bare ``-<digit>`` token is consumed as the value, not treated as a
    new flag.
    """
    token = argv[i]
    if "=" in token and token.startswith("--"):
        key, _, value = token.partition("=")
        if key in flags:
            return True, value
    if token in flags:
        if i + 1 < len(argv):
            nxt = argv[i + 1]
            looks_like_flag = nxt.startswith("-") and not _is_negative_number(nxt)
            if not looks_like_flag:
                return True, nxt
        # Flag with no value: treat as bare (e.g. -cmoe)
        return True, None
    return False, None


def _is_negative_number(token: str) -> bool:
    """True if token is a negative integer (e.g. '-1', '-2')."""
    if not token.startswith("-") or token == "-":
        return False
    return token[1:].lstrip("+").isdigit()


def _parse_ngl(value) -> int:
    """Parse an n_gpu_layers value (int or 'all'/'auto' strings)."""
    if isinstance(value, bool):
        raise WeightPlacementError(
            "n_gpu_layers must be an integer (-1 = all on GPU) or "
            f"'all', got bool {value!r}"
        )
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("all", "auto"):
            return -1  # all layers on GPU — allowed
        try:
            return int(v)
        except ValueError:
            raise WeightPlacementError(
                f"n_gpu_layers value {value!r} is not a valid integer "
                "(expected -1, >= 1, or 'all'/'auto')"
            )
    return int(value)


def _check_cuda_visible_devices(env: dict | None) -> None:
    """Fail fast if CUDA_VISIBLE_DEVICES would remap GPU0 off the physical card."""
    if env is None:
        return
    raw = env.get(CUDA_VISIBLE_DEVICES_ENV)
    if raw is None or str(raw).strip() == "":
        return
    value = str(raw).strip()
    # GPU0-first only. "0" is the canonical safe value; "0,1" would keep
    # GPU0 as physical GPU0 but expose GPU1 — tolerated only as an explicit
    # operator choice, never silently. Anything else (e.g. "1", "1,0")
    # renumbers physical cards so "GPU0" becomes GPU1 — hard violation.
    first = value.split(",")[0].strip()
    if first != str(ALLOWED_MAIN_GPU):
        raise WeightPlacementError(
            f"{CUDA_VISIBLE_DEVICES_ENV}={value!r} would remap GPU0 to a "
            f"different physical card (first entry {first!r} != 0); P12 "
            "policy requires the main lane to run with physical GPU0. "
            f"Unset it or set it to '0'."
        )


def _check_split_mode(argv: Sequence[str]) -> None:
    """Fail fast if split-mode would spread weights across GPUs."""
    for i, token in enumerate(argv):
        found, value = _flag_value(argv, SPLIT_MODE_FLAGS, i)
        if not found:
            continue
        if value is None:
            raise WeightPlacementError(
                f"{token} requires a value; P12 policy requires "
                f"--split-mode {ALLOWED_SPLIT_MODE}"
            )
        if value != ALLOWED_SPLIT_MODE:
            raise WeightPlacementError(
                f"--split-mode {value!r} would spread weight tensors across "
                f"GPUs (placing some on GPU1); P12 policy requires "
                f"{ALLOWED_SPLIT_MODE}. Remove --split-mode or set it to "
                f"{ALLOWED_SPLIT_MODE}."
            )


def _check_main_gpu(argv: Sequence[str]) -> None:
    """Fail fast if --main-gpu references anything other than GPU0."""
    for i, token in enumerate(argv):
        found, value = _flag_value(argv, MAIN_GPU_FLAGS, i)
        if not found:
            continue
        if value is None:
            raise WeightPlacementError(
                f"{token} requires a value; P12 policy requires --main-gpu "
                f"{ALLOWED_MAIN_GPU}"
            )
        try:
            gpu = int(value)
        except ValueError:
            raise WeightPlacementError(
                f"--main-gpu value {value!r} is not an integer; P12 policy "
                f"requires --main-gpu {ALLOWED_MAIN_GPU}"
            )
        if gpu != ALLOWED_MAIN_GPU:
            raise WeightPlacementError(
                f"--main-gpu {gpu} would pin weights to GPU{gpu}; P12 policy "
                f"requires GPU0-only weights (--main-gpu "
                f"{ALLOWED_MAIN_GPU}). Change it to 0 or remove it."
            )


def _check_device(argv: Sequence[str]) -> None:
    """Fail fast if --device references any GPU other than GPU0."""
    for i, token in enumerate(argv):
        found, value = _flag_value(argv, DEVICE_FLAGS, i)
        if not found:
            continue
        if value is None:
            raise WeightPlacementError(
                f"{token} requires a comma-separated device list; P12 policy "
                f"allows only GPU0 ({ALLOWED_MAIN_GPU})"
            )
        devices = [d.strip() for d in value.split(",") if d.strip()]
        if not devices:
            raise WeightPlacementError(
                f"--device value {value!r} is empty; P12 policy requires "
                f"--device {ALLOWED_MAIN_GPU}"
            )
        bad = [d for d in devices if d != str(ALLOWED_MAIN_GPU)]
        if bad:
            raise WeightPlacementError(
                f"--device {value!r} includes non-GPU0 device(s) {bad}; P12 "
                f"policy requires GPU0-only weights (--device "
                f"{ALLOWED_MAIN_GPU}). Remove the offending device(s)."
            )


def _check_ngl(argv: Sequence[str], model_layers: int | None = None) -> None:
    """Fail fast if n_gpu_layers would place any weight tensor on CPU.

    From argv alone the gate can only prove two hard violations:
      - ``0`` — ALL weight tensors on CPU (llama.cpp convention);
      - ``< -1`` — invalid.
    ``-1`` / ``'all'`` means all layers on GPU (allowed).

    For ``0 < ngl < 999`` the gate cannot prove full-GPU residency from argv
    alone: llama.cpp loads the first ``ngl`` layers to GPU and leaves the
    tail on CPU, and the tail is empty only when ``ngl >= model layer
    count``. When the caller knows the model's layer count (e.g. from GGUF
    metadata), pass ``model_layers`` and the gate verifies full residency;
    without it, ``ngl >= 1`` is accepted (the runtime nvidia-smi check in
    the live integration test proves actual placement).
    """
    for i, token in enumerate(argv):
        found, value = _flag_value(argv, N_GPU_LAYERS_FLAGS, i)
        if not found:
            continue
        if value is None:
            raise WeightPlacementError(
                f"{token} requires a value (-1 = all on GPU, N >= 1); P12 "
                "policy requires all weights on GPU0"
            )
        try:
            ngl = _parse_ngl(value)
        except WeightPlacementError:
            raise
        if ngl == 0:
            raise WeightPlacementError(
                f"n_gpu_layers 0 places ALL weight tensors on CPU; P12 "
                "policy requires GPU0-only weights. Use -ngl -1 (all on "
                "GPU) or a value >= the model layer count."
            )
        if ngl < -1:
            raise WeightPlacementError(
                f"n_gpu_layers {ngl} is invalid (must be -1 or >= 1); P12 "
                "policy requires GPU0-only weights."
            )
        if model_layers is not None and ngl > 0 and ngl < model_layers:
            raise WeightPlacementError(
                f"n_gpu_layers {ngl} is less than the model layer count "
                f"{model_layers}: layers {ngl}..{model_layers - 1} would "
                "stay on CPU (partial weight offload); P12 policy requires "
                "ALL weights on GPU0. Use -ngl -1 or -ngl >= "
                f"{model_layers}."
            )


def _check_moe_cpu(argv: Sequence[str]) -> None:
    """Fail fast if MoE/op offload would place weights on CPU."""
    for i, token in enumerate(argv):
        found, value = _flag_value(argv, MOE_CPU_OFFLOAD_FLAGS, i)
        if not found:
            continue
        flag_name = token.split("=")[0]
        if flag_name in ("--cpu-moe", "-cmoe", "--op-offload"):
            # Bare boolean flags: presence means MoE/op weights on CPU.
            raise WeightPlacementError(
                f"{flag_name} places MoE expert / op weights on CPU; P12 "
                "policy requires GPU0-only weights. Remove it."
            )
        if flag_name in ("--n-cpu-moe", "-ncmoe"):
            try:
                n = int(value) if value is not None else 0
            except ValueError:
                n = -1
            if n > 0:
                raise WeightPlacementError(
                    f"--n-cpu-moe {n} places {n} MoE layers on CPU; P12 "
                    "policy requires GPU0-only weights. Remove it or set it "
                    "to 0."
                )


def enforce_gpu0_weights(
    argv: Sequence[str], env: dict | None = None,
    model_layers: int | None = None,
) -> list[str]:
    """Validate a load-path argv strictly pins weights to GPU0; raise on violation.

    Checks (in order, all fail-fast with WeightPlacementError):
      1. CUDA_VISIBLE_DEVICES does not remap physical GPU0.
      2. --split-mode is 'none'.
      3. --main-gpu is 0.
      4. --device is only 0.
      5. n_gpu_layers is -1 / 'all' / >= the model layer count (all weights
         on GPU). ``0`` is a hard violation (all weights on CPU). Pass
         ``model_layers`` from GGUF metadata when known to also prove that
         a positive ngl leaves no tail on CPU.
      6. No MoE/op CPU offload flags.

    On success returns the argv with the canonical GPU0-only pins
    (--split-mode none --main-gpu 0) appended, so the spawn cannot
    pipeline-split weights onto GPU1 even if the manifest omitted them.
    """
    norm = _normalize_argv(list(argv))
    _check_cuda_visible_devices(env)
    _check_split_mode(norm)
    _check_main_gpu(norm)
    _check_device(norm)
    _check_ngl(norm, model_layers=model_layers)
    _check_moe_cpu(norm)
    return _ensure_gpu0_pins(norm)


def _ensure_gpu0_pins(argv: list[str]) -> list[str]:
    """Append the canonical GPU0-only pins (idempotent, no earlier overrides)."""
    stripped = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in SPLIT_MODE_FLAGS or token in MAIN_GPU_FLAGS or token in DEVICE_FLAGS:
            # Remove the flag and (if present) its value token.
            i += 1
            if i < len(argv) and not argv[i].startswith("-"):
                i += 1
            continue
        stripped.append(token)
        i += 1
    stripped.append("--split-mode")
    stripped.append(ALLOWED_SPLIT_MODE)
    stripped.append("--main-gpu")
    stripped.append(str(ALLOWED_MAIN_GPU))
    return stripped


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="P12 GPU0-only weight-placement enforcement gate"
    )
    parser.add_argument(
        "argv", nargs=argparse.REMAINDER,
        help="llama-server argv to validate (after --)",
    )
    args, _ = parser.parse_known_args()

    server_argv = list(args.argv or [])
    if server_argv and server_argv[0] == "--":
        server_argv = server_argv[1:]

    try:
        result = enforce_gpu0_weights(server_argv)
    except WeightPlacementError as exc:
        print(f"VIOLATION: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print("OK: weights GPU0-only")
    print("argv: " + " ".join(result))
