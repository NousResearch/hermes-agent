#!/usr/bin/env python3
"""
fetch_logs.py — Retrieve workflow execution diagnostics from a ComfyUI server.

When a workflow errors, the server's /history (local) or /jobs (cloud) entry
contains the full Python traceback. This script makes it easy to fetch by
prompt_id, with sensible formatting.

Usage:
    python3 fetch_logs.py <prompt_id>
    python3 fetch_logs.py <prompt_id> --host https://cloud.comfy.org
    python3 fetch_logs.py --tail-queue            # show currently queued/running jobs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DEFAULT_LOCAL_HOST, ENV_API_KEY, emit_json, http_get, is_cloud_host,
    resolve_api_key, resolve_url,
)


def fetch_history_entry(host: str, headers: dict, prompt_id: str, *, is_cloud: bool) -> dict:
    if is_cloud:
        # Try /jobs/{id} first
        url = resolve_url(host, f"/jobs/{prompt_id}", is_cloud=True)
        r = http_get(url, headers=headers, retries=2, timeout=30)
        if r.status == 200:
            try:
                return {"ok": True, "entry": r.json(), "source": "/api/jobs"}
            except Exception:
                pass
        # Fallback to history_v2
        url = resolve_url(host, f"/history/{prompt_id}", is_cloud=True)
        r = http_get(url, headers=headers, retries=2, timeout=30)
        try:
            data = r.json()
        except Exception:
            data = None
        if r.status == 200 and data:
            return {"ok": True, "entry": data, "source": "/api/history_v2"}
        return {"ok": False, "http_status": r.status, "body": r.text()[:500]}

    url = resolve_url(host, f"/history/{prompt_id}", is_cloud=False)
    r = http_get(url, headers=headers, retries=2, timeout=30)
    if r.status != 200:
        return {"ok": False, "http_status": r.status, "body": r.text()[:500]}
    try:
        data = r.json()
    except Exception:
        return {"ok": False, "reason": "non-JSON response"}
    if not isinstance(data, dict) or prompt_id not in data:
        return {"ok": False, "reason": "prompt_id not found in history",
                "history_keys": list(data.keys())[:5] if isinstance(data, dict) else []}
    return {"ok": True, "entry": data[prompt_id], "source": "/history"}


def fetch_queue(host: str, headers: dict) -> dict:
    url = resolve_url(host, "/queue")
    r = http_get(url, headers=headers, retries=2, timeout=15)
    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text()[:500]}
    return {"http_status": r.status, "data": data}


def extract_diagnostics(entry: dict) -> dict:
    """Pull out the parts a human cares about: status, errors, traceback, timing."""
    diag: dict = {}
    status = entry.get("status") or {}
    diag["status_str"] = status.get("status_str")
    diag["completed"] = status.get("completed")

    messages = status.get("messages") or []
    diag["execution_log"] = []
    for msg in messages:
        if isinstance(msg, list) and len(msg) >= 2:
            mtype, mdata = msg[0], msg[1]
            diag["execution_log"].append({"type": mtype, "data": mdata})
        else:
            diag["execution_log"].append(msg)

    # Look for execution_error inside messages
    errors = []
    for msg in messages:
        if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "execution_error":
            errors.append(msg[1])
    if errors:
        diag["errors"] = errors

    # Cloud's /jobs response shape: top-level outputs / status / etc.
    if "outputs" in entry:
        out = entry["outputs"] or {}
        if isinstance(out, dict):
            diag["output_node_ids"] = list(out.keys())
            # Count file refs across all output buckets (images / video / etc.)
            total = 0
            for node_output in out.values():
                if not isinstance(node_output, dict):
                    continue
                for v in node_output.values():
                    if isinstance(v, list):
                        total += len(v)
            diag["output_count"] = total
        else:
            diag["output_node_ids"] = []
            diag["output_count"] = 0
    return diag


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch workflow execution diagnostics")
    p.add_argument("prompt_id", nargs="?", help="prompt_id to look up")
    p.add_argument("--host", default=DEFAULT_LOCAL_HOST)
    p.add_argument("--api-key", help=f"or set ${ENV_API_KEY}")
    p.add_argument("--raw", action="store_true",
                   help="Print the full history entry instead of the digest")
    p.add_argument("--tail-queue", action="store_true",
                   help="Show currently running/pending jobs instead")
    args = p.parse_args(argv)

    api_key = resolve_api_key(args.api_key)
    headers = {"X-API-Key": api_key} if api_key else {}
    is_cloud = is_cloud_host(args.host)

    if args.tail_queue:
        emit_json(fetch_queue(args.host, headers))
        return 0

    if not args.prompt_id:
        print("Error: prompt_id is required (or use --tail-queue)", file=sys.stderr)
        return 1

    res = fetch_history_entry(args.host, headers, args.prompt_id, is_cloud=is_cloud)
    if not res.get("ok"):
        emit_json(res)
        return 1

    if args.raw:
        emit_json(res)
        return 0

    diag = extract_diagnostics(res["entry"])
    diag["source"] = res.get("source")
    diag["prompt_id"] = args.prompt_id
    emit_json(diag)
    return 0 if diag.get("status_str") not in ("error",) else 1


if __name__ == "__main__":
    sys.exit(main())
