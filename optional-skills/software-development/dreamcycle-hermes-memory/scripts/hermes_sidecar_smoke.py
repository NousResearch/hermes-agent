#!/usr/bin/env python3
"""Run a tiny end-to-end sidecar smoke test for Hermes compatibility."""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class Runtime:
    base_url: str
    api_key: str
    namespace: str
    user_id: str
    timeout: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight DreamCycle sidecar smoke check for Hermes"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DREAMCYCLE_BASE_URL", "http://127.0.0.1:8765"),
        help="DreamCycle sidecar base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DREAMCYCLE_API_KEY", ""),
        help="Bearer token for DreamCycle API",
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("DREAMCYCLE_NAMESPACE", ""),
        help="Identity namespace",
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("DREAMCYCLE_USER_ID", ""),
        help="Identity user ID",
    )
    parser.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout seconds")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate settings and endpoints without writing memory",
    )
    return parser.parse_args()


def require_non_empty(value: str, name: str) -> str:
    if not value.strip():
        raise SystemExit(f"missing required input: {name}")
    return value.strip()


def request(runtime: Runtime, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = runtime.base_url.rstrip("/") + path
    data: bytes | None = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "authorization": f"Bearer {runtime.api_key}",
            "content-type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=runtime.timeout) as response:
            body = response.read()
            if not body:
                return {}
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"request failed: {method} {path} -> HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"request failed: {method} {path} -> {exc}") from exc


def healthcheck(runtime: Runtime) -> None:
    req = urllib.request.Request(runtime.base_url.rstrip("/") + "/healthz")
    try:
        with urllib.request.urlopen(req, timeout=runtime.timeout) as response:
            payload = json.loads(response.read() or b"{}")
            if payload.get("status") != "ok":
                raise SystemExit("health check failed: unexpected payload")
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"health check failed: HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"health check failed: {exc}") from exc


def main() -> None:
    args = parse_args()
    runtime = Runtime(
        base_url=args.base_url.rstrip("/"),
        api_key=require_non_empty(args.api_key, "DREAMCYCLE_API_KEY"),
        namespace=require_non_empty(args.namespace, "DREAMCYCLE_NAMESPACE"),
        user_id=require_non_empty(args.user_id, "DREAMCYCLE_USER_ID"),
        timeout=args.timeout,
    )

    healthcheck(runtime)

    if args.dry_run:
        print("health ok")
        return

    marker = "".join(random.choice(string.ascii_letters) for _ in range(12))
    query = f"hermes sidecar smoke marker {marker}"
    payload = {
        "user_content": f"Remember this phrase for smoke test: {query}",
        "assistant_content": f"Absolutely, I have stored {query}.",
        "source": "hermes-sidecar-smoke",
        "conversation_id": "hermes-sidecar-smoke",
        "trace_id": "hermes-sidecar-smoke",
        "metadata": {
            "namespace": runtime.namespace,
            "user_id": runtime.user_id,
            "marker": marker,
        },
    }
    response = request(runtime, "POST", "/v1/memory/turns", payload)
    if not response.get("user", {}).get("content"):
        raise SystemExit("write failed: /v1/memory/turns returned unexpected response")

    search = request(
        runtime,
        "POST",
        "/v1/memory/search",
        {
            "query": marker,
            "limit": 5,
        },
    )
    memories = search.get("memories", [])
    if not any(
        isinstance(item, dict) and marker in str(item.get("content", "")) for item in memories
    ):
        raise SystemExit("smoke failed: wrote turn, but marker was not found in search")

    summary = {
        "status": "ok",
        "namespace": runtime.namespace,
        "user_id": runtime.user_id,
        "memory_count": len(memories),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
