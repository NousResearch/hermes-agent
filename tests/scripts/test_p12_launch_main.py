"""P12 launch-seam tests — prove fast path and opt-in long-context path.

These tests exercise ``scripts/p12_launch_main.py`` — the production launch
wrapper for the main-lane spawn — WITHOUT any live activation: they only
inspect the argv the wrapper produces, exactly as a spawner would consume
it. No llama-server is spawned, no service is touched, no GPU is used.

Contract proven:
1. Fast path (default): argv keeps the GPU0-only pins and any stray offload
   flags are neutralized; the server argv is GPU-KV resident.
2. Long-context route (``darwin-28b-256k``): with the opt-in flag set and a
   huge context, the argv enables host-RAM KV (``--no-kv-offload``) while
   weights stay GPU0-only.
3. Fail-closed: long-context WITHOUT the opt-in flag is rejected — the
   wrapper emits the neutralized fast-path argv and ``allowed=False``.
4. Weight-placement violations still fail fast via the enforce gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.p12_launch_main import (  # noqa: E402
    FAST_PATH_MAX_CONTEXT,
    LONG_CONTEXT_MODEL_TAG,
    build_launch_argv,
)
from scripts.benchmarks.p12_weight_placement import WeightPlacementError  # noqa: E402

# The production main-lane argv shape (mirrors turbohaul-main.service
# ExecStart, minus the binary).
BASE_ARGV = [
    "-m", "/home/sahil/ai/models/llm/Darwin-28B-REASON.Q5_K_M.gguf",
    "--host", "127.0.0.1", "--port", "11500",
    "-ngl", "99", "--ctx-size", "65536",
    "--threads", "8", "--threads-batch", "8",
    "--mlock", "--no-mmap",
]

HUGE_CTX = 262144  # 256K — above the fast-path ceiling
FAST_CTX = 65536


def _run_cli(*args, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "p12_launch_main.py"), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


class TestFastPath:
    def test_fast_path_keeps_gpu0_pins_and_weights(self):
        result = build_launch_argv(list(BASE_ARGV), ctx_size=FAST_CTX)
        assert result["mode"] == "fast"
        assert result["allowed"] is True
        assert result["kv_offload"] is False
        argv = result["argv"]
        # Canonical GPU0-only pins present.
        assert "--split-mode" in argv
        assert argv[argv.index("--split-mode") + 1] == "none"
        assert "--main-gpu" in argv
        assert argv[argv.index("--main-gpu") + 1] == "0"
        # Weights stay fully on GPU (-ngl 99 preserved).
        assert "-ngl" in argv
        assert argv[argv.index("-ngl") + 1] == "99"

    def test_fast_path_neutralizes_stray_offload_flags(self):
        poisoned = list(BASE_ARGV) + ["--no-kv-offload"]
        result = build_launch_argv(poisoned, ctx_size=FAST_CTX)
        assert result["mode"] == "fast"
        assert result["kv_offload"] is False
        assert "--no-kv-offload" not in result["argv"]

    def test_fast_path_cli_emits_argv_one_token_per_line(self):
        proc = _run_cli("--ctx-size", str(FAST_CTX), "--", *BASE_ARGV)
        assert proc.returncode == 0, proc.stderr
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        assert lines == result_argv_for(BASE_ARGV, FAST_CTX)


class TestLongContextRoute:
    def test_long_context_enabled_with_optin_and_huge_ctx(self):
        env = {"P12_ALLOW_HUGE_CONTEXT_OFFLOAD": "1"}
        result = build_launch_argv(
            list(BASE_ARGV),
            ctx_size=HUGE_CTX,
            model_tag=LONG_CONTEXT_MODEL_TAG,
            env=env,
        )
        assert result["mode"] == "long-context"
        assert result["allowed"] is True
        assert result["kv_offload"] is True
        assert "--no-kv-offload" in result["argv"]
        # Weights STILL GPU0-only in long-context mode.
        argv = result["argv"]
        assert "--split-mode" in argv
        assert argv[argv.index("--split-mode") + 1] == "none"
        assert "--main-gpu" in argv
        assert argv[argv.index("--main-gpu") + 1] == "0"

    def test_long_context_fails_closed_without_optin(self):
        # No env flag -> offload disallowed -> fast path emitted, rejected.
        result = build_launch_argv(
            list(BASE_ARGV),
            ctx_size=HUGE_CTX,
            model_tag=LONG_CONTEXT_MODEL_TAG,
            env={},
        )
        assert result["allowed"] is False
        assert result["mode"] == "fast"
        assert result["kv_offload"] is False
        assert "--no-kv-offload" not in result["argv"]
        # The neutralized argv still carries the GPU0-only pins.
        argv = result["argv"]
        assert "--main-gpu" in argv

    def test_long_context_small_ctx_denied_even_with_optin(self):
        # The route tag was explicitly requested but the tier policy routes
        # a small ctx to GPU-KV — the long-context request is DENIED and the
        # fast-path argv is emitted (no silent partial offload).
        env = {"P12_ALLOW_HUGE_CONTEXT_OFFLOAD": "1"}
        result = build_launch_argv(
            list(BASE_ARGV),
            ctx_size=FAST_CTX,
            model_tag=LONG_CONTEXT_MODEL_TAG,
            env=env,
        )
        assert result["mode"] == "fast"
        assert result["allowed"] is False
        assert result["kv_offload"] is False
        assert "--no-kv-offload" not in result["argv"]

    def test_long_context_cli_json_reports_mode_and_exit(self):
        proc = _run_cli(
            "--json", "--ctx-size", str(HUGE_CTX),
            "--model-tag", LONG_CONTEXT_MODEL_TAG,
            "--", *BASE_ARGV,
            env_extra={"P12_ALLOW_HUGE_CONTEXT_OFFLOAD": "1"},
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["mode"] == "long-context"
        assert payload["allowed"] is True
        assert payload["kv_offload"] is True
        assert "--no-kv-offload" in payload["argv"]


class TestFailClosedCLI:
    def test_cli_exits_1_when_offload_rejected(self):
        proc = _run_cli(
            "--ctx-size", str(HUGE_CTX),
            "--model-tag", LONG_CONTEXT_MODEL_TAG,
            "--", *BASE_ARGV,
        )
        assert proc.returncode == 1
        # Non-JSON denial still emits the neutralized fast-path argv on
        # stdout (a spawner may choose to proceed on the fast path) but the
        # exit code is the authoritative rejection signal.
        out = proc.stdout
        assert "--no-kv-offload" not in out
        assert "--main-gpu" in out

    def test_weight_placement_violation_fails_fast(self):
        bad = ["-ngl", "0"] + BASE_ARGV[1:]  # ngl=0 -> all weights on CPU
        with pytest.raises(WeightPlacementError):
            build_launch_argv(bad, ctx_size=FAST_CTX)

    def test_cli_weight_violation_exits_1(self):
        bad = ["-ngl", "0"] + BASE_ARGV[1:]
        proc = _run_cli("--ctx-size", str(FAST_CTX), "--", *bad)
        assert proc.returncode == 1
        assert "WeightPlacementError" in proc.stderr or "p12-launch-error" in proc.stderr


def result_argv_for(argv, ctx_size, model_tag=None, env=None):
    """Recompute the expected argv via the same builder (CLI parity check)."""
    return build_launch_argv(
        list(argv), ctx_size=ctx_size, model_tag=model_tag, env=env
    )["argv"]
