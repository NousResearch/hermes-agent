#!/usr/bin/env python3
"""P12 opt-in CPU/system-RAM KV-cache offload gate.

Implements the policy contract in docs/adr/0012-p12-256k-memory-policy.md:

- Model weights are GPU0-only, always. They must never be placed on CPU,
  system RAM, or GPU1. There is no opt-in that lifts this.
- CPU/system-RAM offload is permitted for context/KV-cache data ONLY, and
  only for rare, genuinely huge-context workloads (e.g. 256K-context jobs).
- Offload is explicitly opt-in via the environment variable
  P12_ALLOW_HUGE_CONTEXT_OFFLOAD=1. It defaults to OFF. When off, any
  request to offload context/KV-cache is rejected (and the argv passed to
  the inference server is neutralized so a huge-context job falls back to
  the fast path instead of silently offloading).
- GPU1 isolation is preserved at all times. Offload targets CPU/system RAM,
  never GPU1. Even the enabled path must not pin the main lane to GPU1.

The module is deliberately dependency-free (stdlib only) so it can run as a
pure decision function in tests, CI, or any launch wrapper without importing
torch or the heavy stack.

Usage (launch wrapper):
    from scripts.p12_offload_gate import offload_enabled, decide_offload

    # argv is the llama-server / turbohaul-manifest argv to sanitize.
    decision = decide_offload(argv, ctx_size=ctx)
    if decision.enabled:
        # pass decision.argv (--no-kv-offload + --main-gpu 0 preserved)
        ...
    else:
        # decision.argv has offload flags stripped; fast path preserved
        ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `python scripts/p12_offload_gate.py` resolve the repo-root `scripts`
# package the same way `python -m scripts.p12_offload_gate` does. Without
# this, the direct CLI form hits the ImportError fallback below and silently
# uses a different threshold than the central policy.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Opt-in switch. Anything other than "1" means OFF (normal fast path).
ENV_OFFLOAD_FLAG = "P12_ALLOW_HUGE_CONTEXT_OFFLOAD"

# Contexts at or above this are considered "huge" for offload eligibility.
# The policy module (scripts/kv_cache_policy.py) is the single source of
# truth for the context-length threshold; this constant is kept as a
# backward-compatible alias for older importers. The decision logic uses
# decide_tier() so both the GPU fast path and the offload path share one
# policy.
try:
    from scripts.kv_cache_policy import (
        DEFAULT_OFFLOAD_THRESHOLD as HUGE_CONTEXT_MIN_TOKENS,
        KVCacheTier,
        decide_tier,
    )
except ImportError:  # pragma: no cover - policy always present in-repo
    # Fall back to the P12 256K target if the policy module is unavailable.
    HUGE_CONTEXT_MIN_TOKENS = 262144

    class KVCacheTier:  # type: ignore[no-redef]
        GPU_RESIDENT = "gpu"
        HOST_RAM_OFFLOAD = "offload"

    def decide_tier(ctx_size):  # type: ignore[no-redef]
        class _Decision:
            tier = (
                KVCacheTier.HOST_RAM_OFFLOAD
                if int(ctx_size) > HUGE_CONTEXT_MIN_TOKENS
                else KVCacheTier.GPU_RESIDENT
            )
        return _Decision()

# llama.cpp flags that put the KV cache / context in host RAM.
KV_OFFLOAD_FLAGS = {
    "--no-kv-offload",
    "-nkvo",
    "--kv-offload=false",  # explicit negation form
}

# llama.cpp flags that place model weights on a specific GPU.
MAIN_GPU_FLAGS = {"--main-gpu", "-mg"}

# llama.cpp flags that select devices for layer/weight offload.
DEVICE_FLAGS = {"--device", "-dev"}

# llama.cpp flag controlling how the model is split across GPUs. The main
# lane must always run with split-mode none so weights stay on one GPU.
SPLIT_MODE_FLAGS = {"--split-mode", "-sm"}

# The only GPU the main lane is ever allowed to pin weights to.
ALLOWED_MAIN_GPU = 0

# The split mode that guarantees GPU0-only weights on the main lane.
ALLOWED_SPLIT_MODE = "none"

# GPU1 belongs to the aux lane + media/speech. Never reference it from the
# main lane's offload path.
FORBIDDEN_DEVICE_REF = "1"

# Value form accepted for the opt-in env var.
ENV_TRUE_VALUES = {"1", "true", "yes", "on"}


class OffloadDecision:
    """Result of evaluating an offload request against the P12 policy."""

    __slots__ = ("allowed", "enabled", "reason", "argv", "kv_offload", "main_gpu")

    def __init__(self, allowed, enabled, reason, argv, kv_offload=False, main_gpu=0):
        self.allowed = allowed      # True if the request may offload at all
        self.enabled = enabled      # True if offload is active for this run
        self.reason = reason        # human-readable explanation
        self.argv = argv            # sanitized argv to pass to the server
        self.kv_offload = kv_offload
        self.main_gpu = main_gpu

    def to_dict(self):
        return {
            "allowed": self.allowed,
            "enabled": self.enabled,
            "reason": self.reason,
            "argv": list(self.argv),
            "kv_offload": self.kv_offload,
            "main_gpu": self.main_gpu,
        }

    def __repr__(self):
        return (
            f"OffloadDecision(allowed={self.allowed}, enabled={self.enabled}, "
            f"reason={self.reason!r}, argv={self.argv!r})"
        )


def offload_enabled(env=None) -> bool:
    """Return True only when the opt-in flag is explicitly set to a true value.

    Defaults to OFF (fast path). Any value other than 1/true/yes/on is treated
    as OFF, so an unset or typo'd variable never silently enables offload.
    """
    raw = (env or os.environ).get(ENV_OFFLOAD_FLAG, "")
    return raw.strip().lower() in ENV_TRUE_VALUES


def _strip_value(argv, index, name):
    """Remove argv[index] and (if present) its value token."""
    del argv[index]
    if index < len(argv) and not argv[index].startswith("-"):
        del argv[index]
    return argv


def _normalize_argv(argv):
    """Normalize argv so flag+value pairs are detected even with '=' forms."""
    normalized = list(argv)
    for i, token in enumerate(normalized[:]):
        # --main-gpu=0 / --no-kv-offload=true / --kv-offload=false
        if token.startswith("--") and "=" in token:
            key, _, value = token.partition("=")
            if key in MAIN_GPU_FLAGS:
                normalized[i] = f"{key} {value}".split()
            elif key in KV_OFFLOAD_FLAGS:
                normalized[i] = [key]
    # Flatten any lists produced above back into the argv
    flat = []
    for token in normalized:
        if isinstance(token, list):
            flat.extend(token)
        else:
            flat.append(token)
    return flat


def _ensure_gpu0_only(result):
    """Force the argv to pin weights to GPU0 (split-mode none, main-gpu 0).

    Mutates and returns ``result``. Removes any conflicting split-mode /
    main-gpu / device overrides and appends the canonical GPU0-only pins so
    the server cannot pipeline-split weights onto GPU1.
    """
    stripped = []
    i = 0
    while i < len(result):
        token = result[i]
        if token in SPLIT_MODE_FLAGS:
            result = _strip_value(result, i, token)
            continue
        if token in MAIN_GPU_FLAGS:
            result = _strip_value(result, i, token)
            continue
        if token in DEVICE_FLAGS:
            result = _strip_value(result, i, token)
            continue
        i += 1
    # Canonical GPU0-only pins (idempotent: no earlier overrides remain).
    result.append("--split-mode")
    result.append("none")
    result.append("--main-gpu")
    result.append("0")
    return result


def sanitize_argv(argv, enabled, main_gpu_allowed=ALLOWED_MAIN_GPU):
    """Strip or neutralize offload flags per policy.

    When ``enabled`` is False (default), every KV-offload flag and every
    main-gpu override is removed so the server runs the fast path: weights on
    GPU0 (the binary's default), KV cache in VRAM, GPU1 untouched.

    When ``enabled`` is True, KV-offload flags are KEPT (they are the whole
    point) and main-gpu is forced to the allowed GPU (0) so weights never
    leave GPU0.

    Returns (argv, kv_offload_active, main_gpu).
    """
    argv = _normalize_argv(argv)
    result = list(argv)
    kv_offload_active = False
    main_gpu = main_gpu_allowed

    i = 0
    while i < len(result):
        token = result[i]
        if token in KV_OFFLOAD_FLAGS:
            if enabled:
                kv_offload_active = True
                i += 1
            else:
                result = _strip_value(result, i, token)
            continue
        if token in MAIN_GPU_FLAGS:
            # Read the value if present
            value = None
            if i + 1 < len(result) and not result[i + 1].startswith("-"):
                value = result[i + 1]
            if enabled:
                # Force the allowed GPU; never trust a manifest's override.
                if value is not None:
                    result[i + 1] = str(main_gpu_allowed)
                else:
                    result.insert(i + 1, str(main_gpu_allowed))
                main_gpu = main_gpu_allowed
                i += 2
            else:
                result = _strip_value(result, i, token)
            continue
        if token in DEVICE_FLAGS:
            # --device <dev1,dev2,...> selects which devices weights are
            # offloaded to. Only GPU0 (or "none"/CPU) is acceptable on the
            # main lane. When offload is disabled this is stripped entirely;
            # when enabled it is forced to GPU0.
            value = None
            if i + 1 < len(result) and not result[i + 1].startswith("-"):
                value = result[i + 1]
            if enabled:
                if value is not None:
                    result[i + 1] = str(main_gpu_allowed)
                else:
                    result.insert(i + 1, str(main_gpu_allowed))
                i += 2
            else:
                result = _strip_value(result, i, token)
            continue
        if token in SPLIT_MODE_FLAGS:
            # split-mode is stripped in both modes; _ensure_gpu0_only appends
            # the canonical "none" pin.
            result = _strip_value(result, i, token)
            continue
        i += 1

    # Weights must be pinned to GPU0 in every mode. This appends
    # --split-mode none --main-gpu 0 when absent (or after stripping any
    # override), guaranteeing no pipeline-split onto GPU1.
    result = _ensure_gpu0_only(result)

    return result, kv_offload_active, main_gpu


def decide_offload(argv, ctx_size, env=None):
    """Evaluate an offload request against the P12 policy.

    ``argv`` is the inference-server argv (e.g. llama-server flags or a
    turbohaul manifest argv). ``ctx_size`` is the requested context size in
    tokens. ``env`` may be a dict to substitute for os.environ in tests.

    Returns an OffloadDecision. The decision never throws; it reports policy
    violations via ``allowed``/``enabled`` and returns a neutralized argv.
    Invalid or out-of-range context values are treated as "not an offload
    request" (fast path preserved) rather than crashing the launch wrapper.
    """
    try:
        ctx_size = int(ctx_size or 0)
    except (TypeError, ValueError):
        ctx_size = 0
    if ctx_size < 0:
        ctx_size = 0
    enabled = offload_enabled(env)

    # 1. Weights must never move off GPU0. If the argv pins the main lane to
    #    GPU1 (or any non-GPU0 device), that is a hard policy violation
    #    regardless of the offload flag.
    argv_norm = _normalize_argv(argv)
    for i, token in enumerate(argv_norm):
        if token in MAIN_GPU_FLAGS and i + 1 < len(argv_norm):
            value = argv_norm[i + 1]
            if value != str(ALLOWED_MAIN_GPU):
                sanitized, _, _ = sanitize_argv(argv, enabled=False)
                return OffloadDecision(
                    allowed=False,
                    enabled=False,
                    reason=(
                        f"weights pinned to GPU {value} (main-gpu); P12 policy "
                        "requires GPU0-only weights; offload refused and argv "
                        "neutralized to the fast path"
                    ),
                    argv=sanitized,
                )
        if token in DEVICE_FLAGS and i + 1 < len(argv_norm):
            value = argv_norm[i + 1]
            # --device takes a comma-separated list. Any GPU1 (or higher)
            # reference is a policy violation.
            devices = [d.strip() for d in value.split(",") if d.strip()]
            if any(d != str(ALLOWED_MAIN_GPU) for d in devices):
                sanitized, _, _ = sanitize_argv(argv, enabled=False)
                return OffloadDecision(
                    allowed=False,
                    enabled=False,
                    reason=(
                        f"device list {value!r} includes a non-GPU0 device; "
                        "P12 policy requires GPU0-only weights; offload "
                        "refused and argv neutralized to the fast path"
                    ),
                    argv=sanitized,
                )
        if token in SPLIT_MODE_FLAGS and i + 1 < len(argv_norm):
            value = argv_norm[i + 1]
            # split-mode layer/tensor pipelines weights across GPUs and would
            # place weights on GPU1; only "none" is acceptable.
            if value != ALLOWED_SPLIT_MODE:
                sanitized, _, _ = sanitize_argv(argv, enabled=False)
                return OffloadDecision(
                    allowed=False,
                    enabled=False,
                    reason=(
                        f"split-mode {value!r} would spread weights across "
                        "GPUs; P12 policy requires GPU0-only weights "
                        f"({ALLOWED_SPLIT_MODE}); offload refused and argv "
                        "neutralized to the fast path"
                    ),
                    argv=sanitized,
                )

    # 2. Offload only applies to contexts the placement policy routes to
    #    host-RAM offload (default: >128K, up to 256K). The policy is the
    #    single source of truth for the boundary — the gate never
    #    duplicates the threshold.
    policy = decide_tier(ctx_size)
    if policy.tier is not KVCacheTier.HOST_RAM_OFFLOAD:
        sanitized, _, _ = sanitize_argv(argv, enabled=False)
        return OffloadDecision(
            allowed=False,
            enabled=False,
            reason=(
                f"ctx {ctx_size} not routed to offload by policy "
                f"(default threshold {HUGE_CONTEXT_MIN_TOKENS}); fast path "
                "preserved"
            ),
            argv=sanitized,
        )

    # 3. Even for huge contexts, offload is strictly opt-in.
    if not enabled:
        sanitized, _, _ = sanitize_argv(argv, enabled=False)
        return OffloadDecision(
            allowed=True,
            enabled=False,
            reason=(
                f"ctx {ctx_size} qualifies but "
                f"{ENV_OFFLOAD_FLAG} is not set; offload refused, fast path "
                "preserved"
            ),
            argv=sanitized,
        )

    # 4. Opted-in huge-context job: keep KV offload, force weights to GPU0.
    sanitized, kv_active, main_gpu = sanitize_argv(argv, enabled=True)
    return OffloadDecision(
        allowed=True,
        enabled=True,
        reason=(
            f"ctx {ctx_size} qualifies and {ENV_OFFLOAD_FLAG}=1; "
            "context/KV-cache offload to CPU/system RAM enabled, weights "
            "forced to GPU0"
        ),
        argv=sanitized,
        kv_offload=kv_active,
        main_gpu=main_gpu,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="P12 opt-in CPU/system-RAM KV-cache offload gate"
    )
    parser.add_argument("--ctx-size", type=int, required=True,
                        help="requested context size in tokens")
    parser.add_argument("argv", nargs=argparse.REMAINDER,
                        help="llama-server argv to sanitize (after --)")
    args, _ = parser.parse_known_args(argv)

    # parse_known_args leaves a leading '--' separator in the remainder;
    # drop it so a launch wrapper can pass argv straight through.
    server_argv = list(args.argv or [])
    if server_argv and server_argv[0] == "--":
        server_argv = server_argv[1:]

    decision = decide_offload(server_argv, args.ctx_size)
    import json
    print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
    return 0 if decision.allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
