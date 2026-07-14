#!/usr/bin/env python3
"""Exercise Gateway Health & Diagnostics Export against a local OTLP capture collector."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:4318/v1/traces")
    parser.add_argument("--log", required=True, help="JSONL file written by otel_capture_collector.py")
    parser.add_argument("--wait", type=float, default=7.0)
    args = parser.parse_args()

    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-otel-smoke-"))
    os.environ["HERMES_HOME"] = str(hermes_home)

    from gateway.status import write_runtime_status
    from agent.telemetry.gateway_health_export import start_gateway_health_export
    from agent.telemetry import emitter

    config = {
        "telemetry": {
            "local": True,
            "gateway_health_export": {
                "enabled": True,
                "metrics_enabled": True,
                "diagnostic_events_enabled": True,
                "warning_error_events_enabled": True,
                "export_interval_seconds": 5,
                "logs_export_interval_seconds": 5,
                "resource_attributes": {
                    "service.name": "hermes-gateway-smoke",
                    "deployment.environment": "local-smoke",
                },
                "redaction": {"enabled": True, "include_raw_stack": False},
            },
            "export": {
                "otlp": {
                    "enabled": True,
                    "endpoint": args.endpoint,
                    "headers_env": {},
                }
            },
        }
    }

    runtime = start_gateway_health_export(config)
    if not runtime.enabled:
        raise SystemExit(f"gateway health exporter did not enable: {runtime.reason}")

    write_runtime_status(gateway_state="starting", active_agents=0)
    write_runtime_status(gateway_state="running", active_agents=2)
    write_runtime_status(platform="slack", platform_state="running")
    write_runtime_status(
        platform="slack",
        platform_state="fatal",
        error_code="auth_failed",
        error_message="Bearer *** rejected for smoke@example.com",
    )
    logging.getLogger("gateway.platforms.slack").warning("Slack token *** rejected for smoke@example.com")
    emitter.get_emitter().flush(timeout=2.0)
    time.sleep(args.wait)
    runtime.shutdown()
    emitter.get_emitter().flush(timeout=2.0)

    log_path = Path(args.log)
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = {row["path"] for row in rows}
    print(json.dumps({"hermes_home": str(hermes_home), "requests": len(rows), "paths": sorted(paths)}, indent=2))
    if "/v1/traces" not in paths:
        raise SystemExit("missing /v1/traces request")
    if "/v1/logs" not in paths:
        raise SystemExit("missing /v1/logs request")
    if "/v1/metrics" not in paths:
        raise SystemExit("missing /v1/metrics request")


if __name__ == "__main__":
    main()
