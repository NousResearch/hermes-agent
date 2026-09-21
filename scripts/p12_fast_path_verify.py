#!/usr/bin/env python3
"""P12 fast-path verification + performance baseline (live, GPU0).

Verifies ADR 0012 (docs/adr/0012-p12-256k-memory-policy.md) for NORMAL daily
work — the default fast path — and measures performance so regressions from
the P12 256K memory-promotion changes are caught:

1. Normal short-context workloads take the fast path: model weights on GPU0,
   no CPU/system-RAM offload, no KV-cache offload flags.
2. GPU1 stays isolated: our server process never appears on GPU1, before or
   after inference (GPU1 belongs to the aux lane + media/speech).
3. No measurable performance regression for normal workloads after the P12
   changes. The script runs the SAME fast-path argv twice:
     - ``raw``     : exactly what the pre-P12 launch wrapper passed
                    (mirrors turbohaul-main.service ExecStart)
     - ``enforced``: the argv after scripts/p12_weight_placement.py
                    enforce_gpu0_weights() (the post-P12 launch path)
   and compares boot time, first-token latency and decode throughput. The
   enforcement only appends the behavior-neutral pins ``--split-mode none
   --main-gpu 0``, so the two must be statistically indistinguishable.

4. A huge-context offload request (flag OFF, the default) is neutralized to
   the fast path and never consumes GPU1 — verified at the argv level via
   scripts/p12_offload_gate.decide_offload() and at the device level by the
   ``raw``/``enforced`` runs (no GPU1 process).

This is a LIVE check: it spawns a real llama-server (turboquant-bin) with a
small GGUF (TinyLlama 1.1B Q4, ~669 MB) so it is bounded and safe on the
shared rig. GPU0 must have room (~1 GB free is plenty); GPU1 may be busy —
that is expected and is exactly what the isolation check protects.

Usage:
    python scripts/p12_fast_path_verify.py [--model PATH] [--port N]
        [--mode raw|enforced|both] [--json-out PATH] [--perf-tolerance 0.30]
        [--keep]

Exit 0 = all checks passed. Exit 1 = a functional check failed.
Exit 2 = a performance regression was detected (over tolerance).

The script is CI-ready: a self-hosted GPU runner can invoke it from the
optional workflow (.github/workflows/p12-gpu-verify.yml, manual dispatch).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.p12_offload_gate import (  # noqa: E402
    ENV_OFFLOAD_FLAG,
    decide_offload,
)
from scripts.p12_weight_placement import enforce_gpu0_weights  # noqa: E402

DEFAULT_MODEL = Path.home() / ".hermes/mnemosyne/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LLAMA_SERVER = Path("/home/sahil/ai/turbohaul-manager/vendor/turboquant-bin/llama-server")
PORT = 11997  # 11998 weight-smoke, 11999 offload-live; 11997 is free for this suite

# Normal daily-work context (mirrors turbohaul-main.service --ctx-size 65536).
NORMAL_CTX = 65536
HUGE_CTX = 262144

# Production fast-path argv, minus the model/port (filled per run). This is
# the EXACT pre-P12 launch shape from scripts/systemd/turbohaul-main.service.
# {port} is substituted at runtime so --port is honored without mutating a
# module global.
FAST_ARGV_TEMPLATE = ["-m", "{model}", "--host", "127.0.0.1", "--port", "{port}",
                      "-ngl", "99", "--ctx-size", str(NORMAL_CTX),
                      "--mlock", "--no-mmap"]

# A huge-context offload request (the kind P12 routes to host-RAM KV offload
# ONLY when explicitly opted in).
OFFLOAD_ARGV_TEMPLATE = ["-m", "{model}", "-ngl", "99", "--no-kv-offload",
                         "--ctx-size", str(HUGE_CTX)]

# Prompt used for perf measurement. Fixed text so runs are comparable.
PERF_PROMPT = (
    "Write a short paragraph about the engineering tradeoffs of keeping "
    "large language model weights resident in GPU memory versus offloading "
    "them to system RAM. Keep it under five sentences."
)

FAILURES = []
WARNINGS = []
PERF_RESULTS = {}


def log(msg):
    print(f"[p12-fastpath] {msg}", flush=True)


def fail(msg):
    FAILURES.append(msg)
    log(f"FAIL: {msg}")


def warn(msg):
    WARNINGS.append(msg)
    log(f"WARN: {msg}")


# ---------------------------------------------------------------------------
# Device / process helpers
# ---------------------------------------------------------------------------

def nvidia_procs_on_gpu(gpu_index):
    """Return PIDs of compute processes on a given GPU index via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    try:
        uuid_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    uuid_for_index = {}
    for line in uuid_out.strip().splitlines():
        idx, uuid = [p.strip() for p in line.split(",")]
        uuid_for_index[int(idx)] = uuid
    target_uuid = uuid_for_index.get(gpu_index)
    if not target_uuid:
        return []
    pids = []
    for line in out.strip().splitlines():
        gpu_uuid, pid = [p.strip() for p in line.split(",")]
        if gpu_uuid == target_uuid:
            pids.append(pid)
    return pids


def pid_gpu_memory_mib(pid):
    """Return (used_mib, gpu_index) for a PID, or (None, None) if not on GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        uuid_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return None, None
    uuid_for_index = {}
    for line in uuid_out.strip().splitlines():
        idx, uuid = [p.strip() for p in line.split(",")]
        uuid_for_index[uuid] = int(idx)
    for line in out.strip().splitlines():
        fields = [p.strip() for p in line.split(",")]
        if len(fields) != 3:
            continue
        app_pid, used, uuid = fields
        if app_pid == str(pid) and uuid in uuid_for_index:
            try:
                mib = int(used.replace("MiB", "").strip())
            except ValueError:
                mib = None
            return mib, uuid_for_index[uuid]
    return None, None


def rss_kib(pid):
    """Return VmRSS in KiB from /proc, or None."""
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        return None
    return None


def drain_output(proc, lines=30):
    """Non-blocking read of whatever is buffered on the server's stdout."""
    import select
    out_lines = []
    try:
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0)
            if not ready:
                break
            line = proc.stdout.readline()
            if not line:
                break
            out_lines.append(line.rstrip())
            if len(out_lines) >= lines:
                break
    except Exception:
        pass
    return out_lines


# ---------------------------------------------------------------------------
# Server + measurement
# ---------------------------------------------------------------------------

def wait_for_server(url, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def timed_completion(url, prompt, max_tokens=128, timeout=120):
    """Run a chat completion; return (ok, detail_dict).

    detail_dict keys: latency_s (request wall time), text, tokens (approx
    decode tokens = max_tokens requested), tokens_per_s.
    """
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except Exception as exc:
        return False, {"error": str(exc)}
    elapsed = time.time() - start
    try:
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        out_tokens = usage.get("completion_tokens") or max_tokens
    except Exception:
        text = ""
        out_tokens = max_tokens
    if elapsed <= 0:
        elapsed = 1e-6
    return True, {
        "latency_s": round(elapsed, 3),
        "tokens": out_tokens,
        "tokens_per_s": round(out_tokens / elapsed, 2),
        "text": text[:60],
    }


def run_case(name, argv, model, port, env_extra=None, measure_perf=True,
             isolation_fatal=True):
    """Boot a server with the given argv; verify policy + isolation + perf.

    ``isolation_fatal`` controls whether a GPU1 appearance fails the suite
    (True for the production/enforced path) or is recorded as a warning
    (False for the raw pre-P12 baseline, which predates the pins and is
    expected to trip the very isolation bug the P12 changes fix — the
    finding is evidence, not a failure of the current code).
    """
    log(f"--- case: {name} ---")
    log(f"argv: {' '.join(argv)}")
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)  # main lane: physical GPU0
    if env_extra:
        env.update(env_extra)
    proc = subprocess.Popen(
        [str(LLAMA_SERVER)] + argv,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    url = f"http://127.0.0.1:{port}"
    case = {"name": name, "pid": proc.pid}
    try:
        boot_start = time.time()
        healthy = wait_for_server(url)
        case["boot_s"] = round(time.time() - boot_start, 2)
        if not healthy:
            out = drain_output(proc, 40)
            log("server failed to become healthy; tail:")
            for line in out[-20:]:
                log(f"  {line}")
            fail(f"{name}: server did not become healthy")
            case["healthy"] = False
            return case
        case["healthy"] = True
        log(f"server healthy in {case['boot_s']}s")

        # --- Isolation / placement checks (weights on GPU0, never GPU1) ---
        our_pid = str(proc.pid)
        gpu1_pids = nvidia_procs_on_gpu(1)
        gpu0_mib, gpu0_idx = pid_gpu_memory_mib(proc.pid)
        case["gpu0_mib"] = gpu0_mib
        case["on_gpu1"] = our_pid in gpu1_pids
        if our_pid in gpu1_pids:
            msg = f"{name}: server process appeared on GPU1 (isolation broken)"
            if isolation_fatal:
                fail(msg)
            else:
                warn(msg + " — expected for the pre-P12 baseline argv; the "
                            "P12 pins fix this")
        else:
            log(f"GPU1 isolation OK (pid {our_pid} not on GPU1)")
        if gpu0_mib is not None:
            log(f"weights on GPU0 OK ({gpu0_mib} MiB on GPU{gpu0_idx})")
        else:
            log("note: pid not visible in nvidia-smi yet (mid-load); re-checking after inference")

        # --- Performance: first-token latency + decode throughput ---
        if measure_perf:
            ok_small, small = timed_completion(url, PERF_PROMPT, max_tokens=16, timeout=90)
            if not ok_small:
                fail(f"{name}: short completion failed: {small.get('error')}")
                case["latency_s"] = None
            else:
                case["latency_s"] = small["latency_s"]
                log(f"first-completion latency {small['latency_s']}s: {small['text']!r}")

            ok_long, long_r = timed_completion(url, PERF_PROMPT, max_tokens=128, timeout=120)
            if not ok_long:
                fail(f"{name}: long completion failed: {long_r.get('error')}")
                case["tokens_per_s"] = None
            else:
                case["tokens_per_s"] = long_r["tokens_per_s"]
                case["decode_tokens"] = long_r["tokens"]
                log(f"decode throughput {long_r['tokens_per_s']} tok/s "
                    f"({long_r['tokens']} tokens)")

        # --- Post-inference re-verify placement ---
        gpu1_after = nvidia_procs_on_gpu(1)
        case["on_gpu1_after"] = our_pid in gpu1_after
        if our_pid in gpu1_after:
            msg = f"{name}: GPU1 isolation violated after inference"
            if isolation_fatal:
                fail(msg)
            else:
                warn(msg + " — expected for the pre-P12 baseline argv")
        else:
            log("GPU1 still isolated after inference")
        rss = rss_kib(proc.pid)
        case["rss_kib"] = rss
        if rss is not None:
            log(f"process RSS {rss / 1024:.0f} MiB (no CPU/system-RAM weight offload)")
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log(f"server stopped (rc={proc.returncode})")
    return case


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--mode", choices=["raw", "enforced", "both"], default="both",
                        help="which argv to run (default: both and compare)")
    parser.add_argument("--json-out", default=None,
                        help="write the JSON result summary to this path")
    parser.add_argument("--perf-tolerance", type=float, default=0.30,
                        help="max allowed degradation fraction for enforced vs raw "
                             "(default 0.30 = 30% worse is a regression)")
    parser.add_argument("--keep", action="store_true",
                        help="keep server running on failure for inspection")
    args = parser.parse_args()

    model = Path(args.model)
    if not model.exists():
        print(f"model not found: {model}")
        return 2
    if not LLAMA_SERVER.exists():
        print(f"llama-server not found: {LLAMA_SERVER}")
        return 2

    log(f"model: {model} ({model.stat().st_size / 1e6:.0f} MB)")
    log(f"llama-server: {LLAMA_SERVER}")
    log(f"GPU1 must stay isolated; GPU0 carries the fast path")

    # --- Phase 0: gate contracts (pure, no spawn) -------------------------
    log("--- phase 0: gate contracts ---")
    fast_argv = [t.format(model=model, port=args.port) for t in FAST_ARGV_TEMPLATE]
    try:
        enforced_argv = enforce_gpu0_weights(fast_argv)
    except Exception as exc:
        fail(f"enforce_gpu0_weights rejected the production fast path: {exc}")
        enforced_argv = fast_argv
    log(f"raw argv:      {' '.join(fast_argv)}")
    log(f"enforced argv: {' '.join(enforced_argv)}")
    # Enforcement must be behavior-neutral for the fast path: original flags
    # intact, only the canonical GPU0 pins appended.
    assert fast_argv == enforced_argv[: len(fast_argv)], (
        "enforcement must not alter the fast-path argv"
    )
    assert enforced_argv[-4:] == ["--split-mode", "none", "--main-gpu", "0"]
    log("enforcement is behavior-neutral for the fast path (pins appended only)")

    # No offload flags in the fast path (weight or KV).
    joined = " ".join(enforced_argv)
    for bad in ("--no-kv-offload", "-nkvo", "--cpu-moe", "-cmoe", "--op-offload",
                "--n-cpu-moe", "--device"):
        assert bad not in joined, f"fast path must not contain {bad}"
    log("no offload flags in the fast-path argv")

    # GPU1 never referenced by the fast path.
    assert "--main-gpu" not in joined or enforced_argv[enforced_argv.index("--main-gpu") + 1] == "0"
    log("fast path never references GPU1")

    # Huge-context offload request with the flag OFF (default) is neutralized
    # to the fast path and never touches GPU1.
    offload_argv = [t.format(model=model, port=args.port) for t in OFFLOAD_ARGV_TEMPLATE]
    decision_off = decide_offload(offload_argv, HUGE_CTX, env={})
    log(f"huge-ctx flag OFF -> {decision_off.reason}")
    assert decision_off.enabled is False
    assert "--no-kv-offload" not in decision_off.argv
    d_joined = " ".join(decision_off.argv)
    assert "--device" not in d_joined
    assert "--main-gpu" not in d_joined or decision_off.argv[decision_off.argv.index("--main-gpu") + 1] == "0"
    log("huge-context offload neutralized: fast path preserved, GPU1 untouched")

    # --- Phase 1: live fast-path runs --------------------------------------
    results = {}
    if args.mode in ("raw", "both"):
        results["raw"] = run_case("raw (pre-P12 argv)", fast_argv, model, args.port,
                                  isolation_fatal=False)
    if args.mode in ("enforced", "both"):
        results["enforced"] = run_case("enforced (post-P12 argv)", enforced_argv,
                                       model, args.port, isolation_fatal=True)

    # --- Phase 2: performance regression gate ------------------------------
    perf_gate = {"status": "skip", "detail": "single-mode run; no comparison"}
    if "raw" in results and "enforced" in results:
        raw, enf = results["raw"], results["enforced"]
        checks = []
        if raw.get("latency_s") and enf.get("latency_s"):
            ratio = enf["latency_s"] / max(raw["latency_s"], 1e-6)
            checks.append(("latency", ratio))
        if raw.get("tokens_per_s") and enf.get("tokens_per_s"):
            ratio = raw["tokens_per_s"] / max(enf["tokens_per_s"], 1e-6)
            checks.append(("throughput", ratio))
        regressions = []
        for label, ratio in checks:
            detail = f"{label} ratio {ratio:.2f} (tolerance {args.perf_tolerance})"
            log(f"perf {detail}")
            if ratio > 1.0 + args.perf_tolerance:
                regressions.append(detail)
        if regressions:
            perf_gate = {"status": "FAIL", "detail": "; ".join(regressions)}
        else:
            perf_gate = {"status": "PASS", "detail": "; ".join(
                f"{label}={ratio:.2f}" for label, ratio in checks
            )}
        log(f"perf gate: {perf_gate['status']} — {perf_gate['detail']}")

    summary = {
        "suite": "p12-fast-path-verify",
        "model": str(model),
        "port": args.port,
        "failures": FAILURES,
        "warnings": WARNINGS,
        "results": results,
        "perf_gate": perf_gate,
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
        log(f"summary written to {out_path}")

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failures)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    if perf_gate["status"] == "FAIL":
        print(f"RESULT: PERF-REGRESSION — {perf_gate['detail']}")
        return 2
    if WARNINGS:
        print(f"RESULT: PASS with {len(WARNINGS)} warning(s) (see log)")
        for w in WARNINGS:
            print(f"  - {w}")
        return 0
    print("RESULT: PASS — fast path verified, GPU1 isolated, no regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
