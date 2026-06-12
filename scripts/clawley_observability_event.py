#!/usr/bin/env python3
"""Redacted local observability event records for Clawley/Hermes runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = {"prompt", "messages", "api_key", "apikey", "token", "authorization", "password", "secret"}


def build_observability_event(
    *,
    run_id: str,
    tool_name: str,
    status: str,
    duration_ms: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "clawley_observability_event.v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": str(run_id),
        "tool_name": str(tool_name),
        "status": str(status),
        "duration_ms": float(duration_ms),
        "payload_redacted": _redact(payload or {}),
        "safety_flags": {
            "raw_prompt_logged": False,
            "secrets_logged": False,
            "raw_telegram_text_logged": False,
            "pii_logged": False,
            "local_event_only": True,
        },
    }


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                result[key_text] = "[redacted]"
            else:
                result[key_text] = _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in SENSITIVE_KEYS or any(part in lowered for part in ("secret", "token", "password", "api_key"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tool-name", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--duration-ms", type=float, required=True)
    parser.add_argument("--payload-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.payload_json.read_text(encoding="utf-8")) if args.payload_json else {}
    event = build_observability_event(run_id=args.run_id, tool_name=args.tool_name, status=args.status, duration_ms=args.duration_ms, payload=payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "local_event_only": True}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
