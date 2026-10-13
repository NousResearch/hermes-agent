#!/usr/bin/env python3
"""LivePair API client — models, balance, generate, poll, download.

Stdlib only (urllib); no third-party deps. Reads LIVEPAIR_API_KEY (lp_…)
from the environment. All output is JSON so an agent can parse it.

Usage:
  python livepair.py models [--kind image|video]
  python livepair.py balance
  python livepair.py generate --modelId seedream-5.0 --prompt "..."
      [--image PATH_OR_URL] [--aspectRatio 16:9] [--resolution 1080p]
      [--duration 5] [--wait] [--out FILE]
  python livepair.py status JOB_ID [--out FILE]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://livepairai.com"
POLL_INTERVAL_S = 3.0
POLL_TIMEOUT_S = 600.0


def _key() -> str:
    key = os.environ.get("LIVEPAIR_API_KEY", "").strip()
    if not key:
        print(
            "LIVEPAIR_API_KEY is not set — create an lp_ key at "
            "https://livepairai.com/settings",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return key


def _request(method: str, path: str, body: dict | None = None,
             auth: bool = True) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("content-type", "application/json")
    req.add_header("user-agent", "hermes-livepair-skill/1.0")
    if auth:
        req.add_header("x-api-key", _key())
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        print(
            json.dumps({"error": exc.code, "detail": detail, "path": path}),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def _image_param(value: str) -> str:
    """Accept an https(s) URL or a local file; files become data URIs."""
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value)
    raw = path.read_bytes()
    if len(raw) > 8 * 1024 * 1024:
        print("image file exceeds the 8MB data-URI limit", file=sys.stderr)
        raise SystemExit(1)
    mime = {".png": "image/png", ".webp": "image/webp",
            ".gif": "image/gif"}.get(path.suffix.lower(), "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _download(url: str, out_path: str) -> dict:
    out = Path(out_path)
    with urllib.request.urlopen(url, timeout=120) as resp:
        out.write_bytes(resp.read())
    return {"saved": str(out), "bytes": out.stat().st_size}


def cmd_models(args: argparse.Namespace) -> None:
    cat = _request("GET", "/v1/agent/models", auth=False)
    models = cat.get("models", [])
    if args.kind:
        models = [m for m in models if m.get("kind") == args.kind]
    print(json.dumps(models, indent=2))


def cmd_balance(_args: argparse.Namespace) -> None:
    print(json.dumps(_request("GET", "/v1/agent/balance"), indent=2))


def _poll(job_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while True:
        job = _request("GET", f"/v1/agent/jobs/{job_id}")
        if job.get("status") in ("done", "failed") \
                or time.monotonic() > deadline:
            return job
        time.sleep(POLL_INTERVAL_S)


def _emit(job: dict, out: str | None) -> None:
    if job.get("status") == "done" and out and job.get("url"):
        job["download"] = _download(job["url"], out)
    print(json.dumps(job, indent=2))
    if job.get("status") not in (None, "done"):
        raise SystemExit(1)


def cmd_generate(args: argparse.Namespace) -> None:
    body: dict = {"modelId": args.modelId, "prompt": args.prompt}
    for field in ("aspectRatio", "resolution", "imageSize"):
        value = getattr(args, field)
        if value is not None:
            body[field] = value
    if args.duration is not None:
        body["duration"] = args.duration
    if args.image:
        body["image"] = _image_param(args.image)
    resp = _request("POST", "/v1/agent/generate", body)
    if args.wait and resp.get("jobId") and not resp.get("url"):
        resp = _poll(resp["jobId"])
    _emit(resp, args.out)


def cmd_status(args: argparse.Namespace) -> None:
    _emit(_poll(args.job_id), args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("models")
    p.add_argument("--kind", choices=["image", "video"])
    p.set_defaults(fn=cmd_models)
    p = sub.add_parser("balance")
    p.set_defaults(fn=cmd_balance)

    p = sub.add_parser("generate")
    p.add_argument("--modelId", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", help="https URL or local file for i2i/tool modes")
    p.add_argument("--aspectRatio")
    p.add_argument("--resolution")
    p.add_argument("--imageSize")
    p.add_argument("--duration", type=int, help="seconds, video models only")
    p.add_argument("--wait", action="store_true",
                   help="poll the returned jobId until done/failed")
    p.add_argument("--out", help="download the result to this path")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("status")
    p.add_argument("job_id")
    p.add_argument("--out", help="download the result to this path")
    p.set_defaults(fn=cmd_status)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
