#!/usr/bin/env python3
"""Smoke-test a profile-local LiteLLM proxy without exposing secrets."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import dotenv_values


def profile_home() -> Path:
    configured = os.environ.get("HERMES_HOME")
    if configured:
        return Path(configured).expanduser()
    result = subprocess.run(
        ["hermes", "config", "path"], check=True, capture_output=True, text=True
    )
    return Path(result.stdout.strip()).expanduser().parent


def main() -> None:
    home = profile_home()
    env = dotenv_values(home / ".env")
    key = env.get("LITELLM_MASTER_KEY")
    if not key:
        raise SystemExit("LITELLM_MASTER_KEY is not configured")

    base = os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4000")
    parsed = urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("LITELLM_BASE_URL must be an HTTP loopback endpoint")
    model = os.environ.get("LITELLM_MODEL", "")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    session = requests.Session()
    session.trust_env = False  # never send the local gateway key through an ambient HTTP proxy
    catalog = session.get(
        f"{base}/v1/models", headers=headers, timeout=30, allow_redirects=False
    )
    catalog.raise_for_status()
    models = [item.get("id") for item in catalog.json().get("data", [])]
    if not model and len(models) == 1:
        model = models[0]
    if not model:
        raise SystemExit("Set LITELLM_MODEL when the proxy exposes multiple models")
    if model not in models:
        raise SystemExit(f"Configured model {model!r} missing from catalog")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: LITELLM_OK"}],
        "max_tokens": 16,
    }
    results = []
    for _ in range(2):
        started = time.perf_counter()
        response = session.post(
            f"{base}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
            allow_redirects=False,
        )
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        if not text:
            raise SystemExit("Model response was empty")
        if not data.get("usage"):
            raise SystemExit("LiteLLM usage metadata missing")
        results.append((elapsed, response.headers, text))

    first, second = results
    cache_key = second[1].get("x-litellm-cache-key")
    cost = first[1].get("x-litellm-response-cost")
    if not cache_key:
        raise SystemExit("Second response did not include LiteLLM's cache-hit key")
    if second[2] != first[2]:
        raise SystemExit("Cached response content differs from the original response")
    if cost is None:
        raise SystemExit("LiteLLM cost header missing")
    print(
        "LiteLLM smoke test OK "
        f"(first={first[0]:.3f}s, second={second[0]:.3f}s, cache=yes, cost={cost})"
    )


if __name__ == "__main__":
    main()
