#!/usr/bin/env python3
"""Local Hermes gateway/dashboard smoke check.

This script is intentionally machine-local: it reads the API key from
~/.hermes/config.yaml but never prints it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_GATEWAY_URL = "http://127.0.0.1:8642"
DEFAULT_DASHBOARD_URL = "http://127.0.0.1:9119"
CLAUDE_MARKER = "HERMES_SMOKE_CLAUDE_OK"


class SmokeError(RuntimeError):
    pass


def _load_api_key(config_path: Path) -> str:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        key = (
            data.get("platforms", {})
            .get("api_server", {})
            .get("extra", {})
            .get("key", "")
        )
        return str(key or "")
    except ImportError:
        text = config_path.read_text(encoding="utf-8")
        match = re.search(r"(?m)^\s*key:\s*['\"]?([^'\"\s#]+)", text)
        return match.group(1) if match else ""


def _request(
    method: str,
    url: str,
    *,
    api_key: str = "",
    body: dict[str, Any] | None = None,
    timeout: float = 10,
    expect_status: int | None = None,
) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read().decode("utf-8", errors="replace")
    except OSError as exc:
        raise SmokeError(f"{method} {url} failed: {exc}") from exc

    if expect_status is not None and status != expect_status:
        raise SmokeError(f"{method} {url} returned {status}, expected {expect_status}")

    if not raw:
        return status, None
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def _check_gateway(gateway_url: str, api_key: str) -> None:
    status, health = _request("GET", f"{gateway_url}/health/detailed", expect_status=200)
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise SmokeError(f"Gateway health payload was not ok: {health!r}")
    platforms = health.get("platforms") or {}
    platform_states = {
        name: info.get("state")
        for name, info in platforms.items()
        if isinstance(info, dict)
    }
    print(f"gateway: ok pid={health.get('pid')} platforms={platform_states}")

    _request("GET", f"{gateway_url}/v1/models", expect_status=401)
    status, models = _request("GET", f"{gateway_url}/v1/models", api_key=api_key, expect_status=200)
    if not isinstance(models, dict) or not models.get("data"):
        raise SmokeError(f"Authenticated models response was unexpected: {models!r}")
    model_ids = [m.get("id") for m in models.get("data", []) if isinstance(m, dict)]
    print(f"api-auth: ok models={model_ids}")


def _check_dashboard(dashboard_url: str) -> None:
    status, dashboard = _request("GET", f"{dashboard_url}/api/status", expect_status=200)
    if not isinstance(dashboard, dict) or not dashboard.get("gateway_running"):
        raise SmokeError(f"Dashboard status was not running: {dashboard!r}")
    print(
        "dashboard: ok "
        f"pid={dashboard.get('gateway_pid')} "
        f"state={dashboard.get('gateway_state')}"
    )


def _check_claude_run(gateway_url: str, api_key: str, timeout: float) -> None:
    prompt = (
        "Run this exact local command and return its stdout only: "
        f"/Users/imnfst/.ason/bin/claude -p 'Return exactly {CLAUDE_MARKER}' "
        "--model sonnet --permission-mode plan --allowedTools 'Read' "
        "--disallowedTools 'Edit,Write,MultiEdit,Bash'"
    )
    _, started = _request(
        "POST",
        f"{gateway_url}/v1/runs",
        api_key=api_key,
        body={"input": prompt, "session_id": f"local-smoke-{int(time.time())}"},
        expect_status=202,
    )
    if not isinstance(started, dict) or not started.get("run_id"):
        raise SmokeError(f"Run start response was unexpected: {started!r}")

    run_id = started["run_id"]
    deadline = time.monotonic() + timeout
    last_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        _, status = _request("GET", f"{gateway_url}/v1/runs/{run_id}", api_key=api_key, expect_status=200)
        if not isinstance(status, dict):
            raise SmokeError(f"Run status response was unexpected: {status!r}")
        last_status = status
        state = status.get("status")
        if state in {"completed", "failed", "cancelled"}:
            break
        time.sleep(2)
    else:
        raise SmokeError(f"Claude smoke timed out waiting for {run_id}; last status={last_status!r}")

    if not last_status or last_status.get("status") != "completed":
        raise SmokeError(f"Claude smoke did not complete: {last_status!r}")
    output = str(last_status.get("output") or "")
    if CLAUDE_MARKER not in output:
        raise SmokeError(f"Claude smoke output did not contain marker; run_id={run_id}")
    print(f"claude-run: ok run_id={run_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default=os.getenv("HERMES_GATEWAY_URL", DEFAULT_GATEWAY_URL))
    parser.add_argument("--dashboard-url", default=os.getenv("HERMES_DASHBOARD_URL", DEFAULT_DASHBOARD_URL))
    parser.add_argument("--config", default=os.path.expanduser("~/.hermes/config.yaml"))
    parser.add_argument("--include-claude", action="store_true", help="Run a live /v1/runs Claude CLI smoke.")
    parser.add_argument("--run-timeout", type=float, default=180.0)
    args = parser.parse_args()

    config_path = Path(args.config)
    api_key = _load_api_key(config_path)
    if not api_key:
        raise SmokeError(f"No API key found in {config_path}")

    _check_gateway(args.gateway_url.rstrip("/"), api_key)
    _check_dashboard(args.dashboard_url.rstrip("/"))
    if args.include_claude:
        _check_claude_run(args.gateway_url.rstrip("/"), api_key, args.run_timeout)
    print("hermes-local-smoke: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeError as exc:
        print(f"hermes-local-smoke: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
