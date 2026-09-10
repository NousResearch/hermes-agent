#!/usr/bin/env python3
"""Bounded credentialed measurements for llm-benchmark-weekly.

Credentials are named by ``key_env`` in the input configuration, read only from
the process environment, and never serialised. Run this as a separate no-agent
job before the report collector.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.llm_benchmark_weekly import sanitise_measurements

PROMPTS = {
    "coding": "Write a Python function that returns the two integers summing to a target. Include edge cases.",
    "reasoning": "Two trains 300 miles apart travel towards each other at 60 and 80 mph. When and where do they meet?",
    "summarisation": "Summarise the key differences between TCP and UDP in three bullet points.",
}


def _measure(provider: dict[str, Any], task: str, *, timeout_seconds: float) -> dict[str, Any]:
    measured_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    base = {"provider": provider.get("name"), "model": provider.get("model"), "task": task, "measured_at": measured_at}
    key = os.environ.get(str(provider.get("key_env", "")), "")
    if not key:
        return {**base, "status": "credential_unavailable", "latency_ms": None, "char_count": 0, "content_preview": ""}
    payload = json.dumps({"model": provider["model"], "messages": [{"role": "user", "content": PROMPTS[task]}], "max_tokens": 256}).encode()
    request = urllib.request.Request(
        str(provider["endpoint"]).rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.load(response)
        content = str(result.get("choices", [{}])[0].get("message", {}).get("content", ""))
        status = "success"
    except (OSError, ValueError, KeyError, IndexError, urllib.error.URLError):
        content = ""
        status = "request_failed"
    return {
        **base,
        "status": status,
        "latency_ms": round((time.monotonic() - started) * 1000),
        "char_count": len(content),
        "content_preview": content,
    }


def collect(config: dict[str, Any], *, timeout_seconds: float = 30, max_providers: int = 10) -> list[dict[str, Any]]:
    records = [
        _measure(provider, task, timeout_seconds=timeout_seconds)
        for provider in config.get("providers", [])[:max_providers]
        for task in PROMPTS
    ]
    return sanitise_measurements(records, max_records=max_providers * len(PROMPTS))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Non-secret provider route configuration")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--max-providers", type=int, default=10)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = collect(config, timeout_seconds=min(max(args.timeout_seconds, 1), 60), max_providers=min(max(args.max_providers, 1), 20))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(args.output, 0o600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
