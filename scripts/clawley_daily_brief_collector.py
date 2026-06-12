#!/usr/bin/env python3
"""Collect redacted local status for the Clawley daily brief."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def build_daily_brief_snapshot(
    *,
    quantos_channel_performance: str | Path | None = None,
    gateway_log_lines: Iterable[str] = (),
    kanban: dict[str, Any] | None = None,
    cron: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quantos: dict[str, Any] = {"telegram_channel_performance": _load_json_object(quantos_channel_performance)}
    gateway_errors = sum(1 for line in gateway_log_lines if "error" in line.lower() or "exception" in line.lower())
    return {
        "schema": "clawley_daily_brief_snapshot.v1",
        "quantos": quantos,
        "kanban": kanban or {"blocked": 0, "ready": 0},
        "cron": cron or {"failed_last_24h": 0},
        "gateway": {"errors_last_24h": gateway_errors},
        "safety_flags": {
            "read_only": True,
            "paths_redacted": True,
            "secrets_redacted": True,
            "write_performed": False,
        },
    }


def _load_json_object(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_supplied"}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "missing"}
    if not isinstance(payload, dict):
        return {"status": "invalid_non_object"}
    payload = dict(payload)
    payload.pop("artifact_path", None)
    payload.pop("path", None)
    return payload


def _tail_lines(path: Path, limit: int = 500) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []
    return lines[-limit:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantos-channel-performance", type=Path)
    parser.add_argument("--gateway-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    snapshot = build_daily_brief_snapshot(
        quantos_channel_performance=args.quantos_channel_performance,
        gateway_log_lines=_tail_lines(args.gateway_log) if args.gateway_log else (),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "read_only": True, "write_performed": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
