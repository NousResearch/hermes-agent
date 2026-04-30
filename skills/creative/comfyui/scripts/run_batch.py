#!/usr/bin/env python3
"""
run_batch.py — Run a workflow many times, varying parameters per run.

Two modes:
  1. --count N --randomize-seed
       Submit N runs, each with a fresh random seed. Use for quick variations.
  2. --sweep '{"seed": [1,2,3], "steps": [20,30]}'
       Cartesian product of values. With cloud subscription, runs in parallel
       up to your tier's concurrent-job limit.

Both modes write each run's outputs into output-dir/run_NNN/.

Examples:
    python3 run_batch.py --workflow flux_dev.json \
        --args '{"prompt": "a cat"}' \
        --count 8 --randomize-seed \
        --output-dir ./outputs/cat-batch

    python3 run_batch.py --workflow sdxl.json \
        --args '{"prompt": "abstract"}' \
        --sweep '{"seed": [1,2,3], "steps": [20, 40]}' \
        --output-dir ./outputs/sweep
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DEFAULT_LOCAL_HOST, ENV_API_KEY, coerce_seed, emit_json, log,
    looks_like_video_workflow, resolve_api_key, unwrap_workflow,
)
from run_workflow import (  # noqa: E402
    ComfyRunner, download_outputs, inject_params,
)
from extract_schema import extract_schema  # noqa: E402


def expand_sweep(sweep: dict, base_args: dict, count: int, randomize_seed: bool) -> list[dict]:
    """Generate a list of args dicts for each run."""
    if sweep:
        # Cartesian product
        keys = list(sweep.keys())
        values = [sweep[k] if isinstance(sweep[k], list) else [sweep[k]] for k in keys]
        runs = []
        for combo in itertools.product(*values):
            ar = dict(base_args)
            for k, v in zip(keys, combo):
                ar[k] = v
            runs.append(ar)
        return runs
    # Count mode
    runs = []
    for _ in range(count):
        ar = dict(base_args)
        if randomize_seed:
            ar["seed"] = coerce_seed(None)
        runs.append(ar)
    return runs


def execute_one(
    runner: ComfyRunner, workflow: dict, schema: dict, args: dict,
    *, output_dir: Path, timeout: int, ws: bool,
) -> dict:
    wf, warnings = inject_params(workflow, schema, args)
    sub = runner.submit(wf)
    if "_http_error" in sub:
        return {"status": "error", "error": "submission HTTP error",
                "details": sub.get("body"), "args": args}
    pid = sub.get("prompt_id")
    if not pid:
        return {"status": "error", "error": "no prompt_id", "response": sub, "args": args}
    if sub.get("node_errors"):
        return {"status": "error", "error": "validation failed",
                "node_errors": sub["node_errors"], "args": args}

    if ws:
        result = runner.monitor_ws(pid, timeout=timeout)
    else:
        result = runner.poll_status(pid, timeout=timeout)

    if result["status"] != "success":
        return {
            "status": result["status"],
            "prompt_id": pid,
            "details": result.get("data"),
            "args": args,
        }

    outputs = result.get("outputs") or runner.get_outputs(pid)
    downloaded = download_outputs(runner, outputs, output_dir, preserve_subfolder=False)
    return {
        "status": "success",
        "prompt_id": pid,
        "args": args,
        "outputs": downloaded,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Submit a workflow many times with varying parameters.",
    )
    p.add_argument("--workflow", required=True)
    p.add_argument("--args", default="{}", help="Base parameters JSON")
    p.add_argument("--count", type=int, default=0,
                   help="Number of runs (use with --randomize-seed)")
    p.add_argument("--sweep", default="",
                   help='JSON dict of param→list of values. Cartesian product. '
                        'e.g. \'{"seed":[1,2,3],"cfg":[5,8]}\'')
    p.add_argument("--randomize-seed", action="store_true",
                   help="In --count mode, vary seed per run")
    p.add_argument("--host", default=DEFAULT_LOCAL_HOST)
    p.add_argument("--api-key", help=f"or set ${ENV_API_KEY}")
    p.add_argument("--partner-key")
    p.add_argument("--parallel", type=int, default=1,
                   help="Concurrent submissions (cloud: up to your tier limit). "
                        "Default 1 (sequential)")
    p.add_argument("--output-dir", default="./outputs/batch")
    p.add_argument("--timeout", type=int, default=0)
    p.add_argument("--ws", action="store_true")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Don't stop the batch when a run fails")
    args = p.parse_args(argv)

    if args.count <= 0 and not args.sweep:
        emit_json({"error": "Specify --count N or --sweep '{...}'"})
        return 1

    base_args = json.loads(args.args) if args.args.strip() else {}
    sweep = json.loads(args.sweep) if args.sweep.strip() else {}

    # Validate sweep shape
    if sweep:
        if not isinstance(sweep, dict):
            emit_json({"error": "--sweep must be a JSON object {param: [values]}"})
            return 1
        empty = [k for k, v in sweep.items() if isinstance(v, list) and len(v) == 0]
        if empty:
            emit_json({"error": f"--sweep parameters have empty value lists: {empty}"})
            return 1
        # If user passed BOTH --sweep and --count/--randomize-seed, --sweep wins
        if args.count or args.randomize_seed:
            log("--sweep set; ignoring --count / --randomize-seed (sweep defines the runs)")

    wf_path = Path(args.workflow).expanduser()
    if not wf_path.exists():
        emit_json({"error": f"Workflow not found: {args.workflow}"})
        return 1
    try:
        with wf_path.open() as f:
            workflow = unwrap_workflow(json.load(f))
    except (ValueError, json.JSONDecodeError) as e:
        emit_json({"error": str(e)})
        return 1

    schema = extract_schema(workflow)
    runs = expand_sweep(sweep, base_args, args.count, args.randomize_seed)
    log(f"Planned {len(runs)} run(s)")

    api_key = resolve_api_key(args.api_key)
    runner = ComfyRunner(host=args.host, api_key=api_key, partner_key=args.partner_key)

    ok, info = runner.check_server()
    if not ok:
        emit_json({"error": "Cannot reach server", "details": info, "host": args.host})
        return 1

    timeout = args.timeout
    if timeout <= 0:
        timeout = 900 if looks_like_video_workflow(workflow) else 300

    base_dir = Path(args.output_dir).expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    failures = 0

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            future_to_idx = {}
            for i, ar in enumerate(runs):
                run_dir = base_dir / f"run_{i:04d}"
                fut = ex.submit(
                    execute_one, runner, workflow, schema, ar,
                    output_dir=run_dir, timeout=timeout, ws=args.ws,
                )
                future_to_idx[fut] = i
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"status": "error", "error": str(e), "args": runs[i]}
                r["index"] = i
                results.append(r)
                if r["status"] != "success":
                    failures += 1
                    log(f"  run {i} → {r['status']}: {r.get('error','?')}")
                    if not args.continue_on_error:
                        log("  --continue-on-error not set; aborting batch")
                        break
                else:
                    log(f"  run {i} → success: {len(r.get('outputs', []))} files")
    else:
        for i, ar in enumerate(runs):
            run_dir = base_dir / f"run_{i:04d}"
            r = execute_one(runner, workflow, schema, ar,
                           output_dir=run_dir, timeout=timeout, ws=args.ws)
            r["index"] = i
            results.append(r)
            if r["status"] != "success":
                failures += 1
                log(f"  run {i} → {r['status']}: {r.get('error','?')}")
                if not args.continue_on_error:
                    log("  --continue-on-error not set; aborting batch")
                    break
            else:
                log(f"  run {i} → success: {len(r.get('outputs', []))} files")

    results.sort(key=lambda x: x.get("index", 0))
    emit_json({
        "status": "success" if failures == 0 else "partial",
        "total": len(runs),
        "completed": sum(1 for r in results if r["status"] == "success"),
        "failed": failures,
        "output_dir": str(base_dir),
        "results": results,
    })
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
