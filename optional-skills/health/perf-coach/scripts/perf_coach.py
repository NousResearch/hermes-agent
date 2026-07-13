#!/usr/bin/env python3
"""Read-only client for a local perf-coach instance.

Every request is an explicit HTTP GET against http://localhost:<port>.
There is no write path in this file by design — the perf-coach skill is
read-only, and no flag here can turn a call into a POST/PUT/PATCH/DELETE.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

ENDPOINTS = {
    "training_load": "/api/training/load",
    "scores": "/api/scores",
    "today": "/api/plan/today",
    "weight": "/api/weight/recent",
}

# Keep tool results small enough to stay well within the model's context
# budget — a cron/chat turn should never choke on a large JSON blob.
MAX_CHARS = 1500
MAX_LIST_ITEMS = 10


def _get_json(base_url: str, path: str, timeout: float):
    req = urllib.request.Request(base_url.rstrip("/") + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def _cap_lists(value, max_items=MAX_LIST_ITEMS):
    """Trim any list fields to their last N entries (most recent first data)."""
    if isinstance(value, list) and len(value) > max_items:
        return value[-max_items:]
    if isinstance(value, dict):
        return {k: _cap_lists(v, max_items) for k, v in value.items()}
    return value


def _shrink(value, max_chars=MAX_CHARS):
    value = _cap_lists(value)
    text = json.dumps(value, separators=(",", ":"))
    if len(text) <= max_chars:
        return value
    return {"truncated": True, "preview": text[:max_chars]}


def fetch(endpoint_key: str, base_url: str, timeout: float):
    data = _get_json(base_url, ENDPOINTS[endpoint_key], timeout)
    return _shrink(data)


def bedtime_snapshot(base_url: str, timeout: float):
    """One combined read for the nightly cron job: training_load + today +
    weight in a single process, so the cron agent needs exactly one tool call
    to gather everything it narrates."""
    out = {}
    for key in ("training_load", "today", "weight"):
        try:
            out[key] = fetch(key, base_url, timeout)
        except Exception as exc:  # unreachable / bad response — report, don't crash
            out[key] = {"error": str(exc)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "endpoint",
        choices=[*ENDPOINTS.keys(), "bedtime"],
        help="Which read to perform ('bedtime' combines training_load+today+weight)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="perf-coach local API port (default: 8000)"
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    try:
        if args.endpoint == "bedtime":
            result = bedtime_snapshot(base_url, args.timeout)
        else:
            result = fetch(args.endpoint, base_url, args.timeout)
        print(json.dumps(result, separators=(",", ":")))
    except urllib.error.URLError as exc:
        print(json.dumps({"error": f"perf-coach unreachable at {base_url}: {exc}"}))
        sys.exit(1)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
