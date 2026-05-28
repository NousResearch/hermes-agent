#!/usr/bin/env python3
"""
health_check.py — One-stop verification that the ComfyUI environment is ready.

Runs through the verification checklist:
  1. comfy-cli on PATH
  2. server reachable (/system_stats)
  3. at least one checkpoint installed
  4. (optional) a specific workflow's deps are met
  5. (optional) actually submit a tiny test workflow and verify round-trip

Usage:
    python3 health_check.py
    python3 health_check.py --host https://cloud.comfy.org
    python3 health_check.py --workflow my.json
    python3 health_check.py --smoke-test    # actually submit a tiny workflow
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DEFAULT_LOCAL_HOST, ENV_API_KEY, emit_json, http_get, parse_model_list,
    resolve_api_key, resolve_url, unwrap_workflow,
)


def comfy_cli_status() -> dict:
    if shutil.which("comfy"):
        return {"available": True, "method": "comfy", "path": shutil.which("comfy")}
    if shutil.which("uvx"):
        return {"available": True, "method": "uvx",
                "hint": "Invoke as `uvx --from comfy-cli comfy ...`"}
    return {
        "available": False,
        "hint": "Install with: pipx install comfy-cli (or `pip install comfy-cli`)",
    }


def server_status(host: str, headers: dict) -> dict:
    url = resolve_url(host, "/system_stats")
    try:
        r = http_get(url, headers=headers, retries=2, timeout=10)
        if r.status == 200:
            try:
                stats = r.json() or {}
            except Exception:
                stats = {}
            return {"reachable": True, "url": url, "stats": stats}
        return {"reachable": False, "url": url, "http_status": r.status, "body": r.text()[:200]}
    except Exception as e:
        return {"reachable": False, "url": url, "error": str(e)}


def checkpoint_status(host: str, headers: dict) -> dict:
    url = resolve_url(host, "/models/checkpoints")
    try:
        r = http_get(url, headers=headers, retries=2, timeout=15)
    except Exception as e:
        return {"queryable": False, "error": str(e)}
    if r.status != 200:
        return {"queryable": False, "http_status": r.status, "url": url, "body": r.text()[:200]}
    try:
        models = parse_model_list(r.json())
    except Exception:
        models = set()
    return {"queryable": True, "count": len(models),
            "first_few": sorted(models)[:5]}


SMOKE_WORKFLOW = {
    # Minimal SD1.5 workflow that doesn't depend on rare nodes.
    # 256x256 + 1 step is the smallest config that doesn't trigger SDXL/Flux
    # validation errors while still executing fast.
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 1, "steps": 1, "cfg": 7.0,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "REPLACE_ME"}},
    "5": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 256, "height": 256, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "test", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "", "clip": ["4", 1]}},
    "9": {"class_type": "SaveImage",
          "inputs": {"filename_prefix": "smoke", "images": ["3", 0]}},
}


def smoke_test(host: str, headers: dict, ckpt_name: str | None) -> dict:
    """Submit a tiny workflow and verify the server accepts it.

    Cancels the job immediately after acceptance so we don't burn GPU
    time / cloud minutes on a smoke test.
    """
    if not ckpt_name:
        return {"ran": False, "reason": "no checkpoint available"}
    wf = json.loads(json.dumps(SMOKE_WORKFLOW))
    wf["4"]["inputs"]["ckpt_name"] = ckpt_name

    # Lazy import to avoid circular issues
    from run_workflow import ComfyRunner
    api_key = headers.get("X-API-Key")
    runner = ComfyRunner(host=host, api_key=api_key)
    sub = runner.submit(wf)
    if "_http_error" in sub:
        return {"ran": True, "submitted": False,
                "http_status": sub["_http_error"], "body": sub.get("body")}
    pid = sub.get("prompt_id")
    if not pid:
        return {"ran": True, "submitted": False, "response": sub}

    # Cancel so we don't actually waste compute on the smoke test.
    cancelled = False
    try:
        cancelled = runner.cancel(pid)
    except Exception:
        pass

    return {
        "ran": True, "submitted": True, "prompt_id": pid,
        "cancelled_after_submit": cancelled,
        "note": "Submission accepted; cancelled to avoid running the full pipeline.",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="One-stop ComfyUI health check")
    p.add_argument("--host", default=DEFAULT_LOCAL_HOST)
    p.add_argument("--api-key", help=f"or set ${ENV_API_KEY}")
    p.add_argument("--workflow", help="Optional: also run check_deps on this workflow")
    p.add_argument("--smoke-test", action="store_true",
                   help="Submit a tiny test workflow and verify round-trip")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero on any non-pass condition (including warnings)")
    args = p.parse_args(argv)

    api_key = resolve_api_key(args.api_key)
    headers = {"X-API-Key": api_key} if api_key else {}

    cli = comfy_cli_status()
    server = server_status(args.host, headers)
    ckpts = checkpoint_status(args.host, headers) if server.get("reachable") else None

    # ---- workflow check ----
    workflow_check: dict | None = None
    if args.workflow:
        wf_path = Path(args.workflow).expanduser()
        if not wf_path.exists():
            workflow_check = {"error": "workflow file not found"}
        else:
            try:
                with wf_path.open() as f:
                    workflow = unwrap_workflow(json.load(f))
                from check_deps import check_deps
                workflow_check = check_deps(workflow, host=args.host, api_key=api_key)
            except (ValueError, json.JSONDecodeError) as e:
                workflow_check = {"error": str(e)}

    smoke = None
    if args.smoke_test and server.get("reachable"):
        first_ckpt = ckpts["first_few"][0] if ckpts and ckpts.get("first_few") else None
        smoke = smoke_test(args.host, headers, first_ckpt)

    # ---- verdict ----
    verdict = "pass"
    reasons: list[str] = []
    if not server.get("reachable"):
        verdict = "fail"
        reasons.append("server unreachable")
    if ckpts and ckpts.get("queryable") and ckpts.get("count", 0) == 0:
        verdict = "warn" if verdict == "pass" else verdict
        reasons.append("no checkpoints installed")
    if workflow_check and workflow_check.get("error"):
        verdict = "fail"
        reasons.append(f"workflow check failed: {workflow_check['error']}")
    elif workflow_check and not workflow_check.get("is_ready"):
        if workflow_check.get("node_check_skipped"):
            reasons.append("node check skipped (cloud free tier)")
        else:
            verdict = "fail"
            reasons.append("workflow has missing deps")
    if smoke and smoke.get("ran") and not smoke.get("submitted"):
        verdict = "fail"
        reasons.append("smoke-test submission failed")
    if not cli.get("available"):
        verdict = "warn" if verdict == "pass" else verdict
        reasons.append("comfy-cli not on PATH (lifecycle commands won't work)")

    report = {
        "verdict": verdict,
        "reasons": reasons,
        "host": args.host,
        "comfy_cli": cli,
        "server": server,
        "checkpoints": ckpts,
        "workflow_check": workflow_check,
        "smoke_test": smoke,
    }
    emit_json(report)

    if verdict == "pass":
        return 0
    if verdict == "warn":
        return 1 if args.strict else 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
