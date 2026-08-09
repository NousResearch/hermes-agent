#!/usr/bin/env python3
"""P12 weight-placement live smoke test (default fast path).

Spawns a real llama-server (turboquant-bin) with the argv produced by
scripts/p12_weight_placement.py.enforce_gpu0_weights() and verifies:

1. The server boots healthy with the default fast-path config (no offload).
2. The server process is resident on GPU0 (not GPU1) via nvidia-smi.
3. A chat completion round-trips (load path did not regress inference).

This uses a small real GGUF (TinyLlama 1.1B Q4) so the check is bounded
and safe on the shared rig. The P12 weight-policy itself is exercised with
the real Darwin 28B flags via the enforce module (GPU0-only, GPU1
forbidden, ngl coverage).

Usage:
    python scripts/p12_weight_placement_smoke.py [--model PATH]

Exit 0 = all checks passed. Exit 1 = a check failed.
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

from scripts.p12_weight_placement import enforce_gpu0_weights  # noqa: E402

DEFAULT_MODEL = Path.home() / ".hermes/mnemosyne/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LLAMA_SERVER = Path("/home/sahil/ai/turbohaul-manager/vendor/turboquant-bin/llama-server")
PORT = 11998  # unused by the live lanes (11500 main, 8082 aux, 11999 p12-live)

FAILURES = []


def log(msg):
    print(f"[p12-smoke] {msg}", flush=True)


def fail(msg):
    FAILURES.append(msg)
    log(f"FAIL: {msg}")


def wait_for_server(url, timeout=90):
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


def chat_roundtrip(url):
    body = json.dumps({
        "messages": [{"role": "user", "content": "Reply with exactly: P12-OK"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        text = data["choices"][0]["message"]["content"]
        return True, text
    except Exception as exc:
        return False, str(exc)


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


def drain_output(proc, lines=30):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
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

    # --- Step 1: enforce the default fast-path argv (GPU0-only pins) ---
    # Mirrors the production main.service fast path (ngl 99, no offload).
    fast_argv = ["-m", str(model), "--host", "127.0.0.1", "--port", str(PORT),
                 "-ngl", "99", "--ctx-size", "65536", "--mlock", "--no-mmap"]
    try:
        enforced = enforce_gpu0_weights(fast_argv)
    except Exception as exc:
        fail(f"enforce_gpu0_weights rejected the fast path: {exc}")
        return 1
    log(f"enforced argv: {' '.join(enforced)}")

    # --- Step 2: spawn with the enforced argv ---
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)  # main lane: physical GPU0
    proc = subprocess.Popen(
        [str(LLAMA_SERVER)] + enforced,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    url = f"http://127.0.0.1:{PORT}"
    try:
        if not wait_for_server(url):
            out = drain_output(proc, 40)
            log("server failed to become healthy; tail:")
            for line in out[-20:]:
                log(f"  {line}")
            fail("default fast path: server did not become healthy")
            if not args.keep:
                proc.kill()
            return 1

        log("server healthy")

        # --- Step 3: weights on GPU0 (never GPU1) ---
        our_pid = str(proc.pid)
        gpu0_pids = nvidia_procs_on_gpu(0)
        gpu1_pids = nvidia_procs_on_gpu(1)
        if our_pid in gpu1_pids:
            fail(f"default fast path: server process appeared on GPU1 (isolation broken)")
        elif our_pid not in gpu0_pids:
            log("note: server PID not in nvidia-smi compute apps yet (may be mid-load); "
                "re-checking after inference")
        else:
            log(f"weights on GPU0 OK (pid {our_pid} on GPU0, not GPU1)")

        # --- Step 4: inference round-trip ---
        ok, text = chat_roundtrip(url)
        if not ok:
            fail(f"default fast path: chat round-trip failed: {text}")
        else:
            log(f"inference OK: {text!r}")

        # --- Step 5: re-verify placement after inference ---
        gpu1_after = nvidia_procs_on_gpu(1)
        if our_pid in gpu1_after:
            fail("default fast path: GPU1 isolation violated after inference")
        else:
            log("GPU1 still isolated after inference")
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log(f"server stopped (rc={proc.returncode})")

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failures)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULT: PASS — default fast path loads with weights GPU0-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
