"""Measure remote tool-RPC overhead on a real Docker backend, without an LLM/API.

Run: .venv/bin/python scripts/bench_remote_code_rpc.py --calls 200 --repeats 3
Uses a fresh temporary Hermes home and disposable Docker container. The tool
handler returns a synthetic row so provider latency does not hide transport cost.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--image", default="python:3.14-slim-bookworm")
    args = parser.parse_args()
    if args.calls < 1 or args.repeats < 1:
        parser.error("calls and repeats must be positive")
    with tempfile.TemporaryDirectory(prefix="hermes-rpc-bench-") as home:
        os.environ["HERMES_HOME"] = home
        os.environ["HERMES_RUNTIME_DIR"] = str(Path(home) / "runtime")
        from tools.code_execution_tool import _run_remote_per_call
        from tools.code_kernel_remote import execute_in_remote_kernel, shutdown_all_remote_kernels
        from tools.environments.docker import DockerEnvironment
        env = DockerEnvironment(image=args.image, cwd="/workspace", network=False,
                                task_id="rpc-bench-" + uuid.uuid4().hex[:8],
                                persist_across_processes=False, auto_mount_cwd=False)
        code = f'''
from hermes_tools import read_file
import json, time
started = time.monotonic()
rows = [read_file(str(i)) for i in range({args.calls})]
assert [row["value"] for row in rows] == list(range({args.calls}))
print(json.dumps({{"script_seconds": time.monotonic() - started,
                  "sum": sum(row["value"] for row in rows)}}))
'''
        results = []
        opener = getattr(env, "open_code_rpc", None)
        try:
            for mode in ("per-call", "kernel"):
                for transport in (("file", "stream") if opener else ("file",)):
                    env.open_code_rpc = opener if transport == "stream" else None
                    for repetition in range(args.repeats):
                        started = time.monotonic()
                        with patch("model_tools.handle_function_call", side_effect=lambda name, a, **kw:
                                   json.dumps({"value": int(a["path"])})) as handler:
                            if mode == "kernel":
                                result = execute_in_remote_kernel(
                                    code, env=env, env_type="docker", task_env_id=mode + transport,
                                    sandbox_tools=frozenset({"read_file"}), timeout=300,
                                    max_tool_calls=args.calls, reset=False)
                                output = result["stdout"]
                            else:
                                result = json.loads(_run_remote_per_call(
                                    env, "docker", code, "bench", frozenset({"read_file"}),
                                    timeout=300, max_tool_calls=args.calls, exec_start=started))
                                output = result["output"]
                        assert result["status"] == "success", result
                        assert result["tool_calls_made"] == handler.call_count == args.calls, result
                        row = {"mode": mode, "transport": transport, "repetition": repetition,
                               "wall_seconds": time.monotonic() - started, "calls": handler.call_count,
                               **json.loads(output)}
                        results.append(row)
                        print(json.dumps(row), flush=True)
                    shutdown_all_remote_kernels()
            for mode in ("per-call", "kernel"):
                medians = {t: statistics.median(r["script_seconds"] for r in results
                           if r["mode"] == mode and r["transport"] == t)
                           for t in (("file", "stream") if opener else ("file",))}
                wall = {t: statistics.median(r["wall_seconds"] for r in results
                        if r["mode"] == mode and r["transport"] == t) for t in medians}
                print(json.dumps({"mode": mode, "median_wall_seconds": wall,
                                  "wall_speedup": wall["file"] / wall["stream"] if opener else None,
                                  "median_script_seconds": medians,
                                  "speedup": medians["file"] / medians["stream"] if opener else None}), flush=True)
        finally:
            shutdown_all_remote_kernels()
            env.cleanup()
            assert DockerEnvironment.wait_for_all_teardowns(timeout=30), "container cleanup timed out"


if __name__ == "__main__":
    main()
