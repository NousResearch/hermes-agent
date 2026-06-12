#!/usr/bin/env python3
"""Build a read-only Clawley daily brief payload.

This module is intentionally side-effect free: callers feed it already-redacted
status snapshots from QuantOS, kanban, cron, gateway, or smart-home probes. It
returns a compact JSON/Markdown brief suitable for a Telegram cron delivery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_daily_brief(sections: dict[str, Any]) -> dict[str, Any]:
    recommendations = _recommendations(sections)
    payload = {
        "schema": "clawley_daily_brief.v1",
        "write_performed": False,
        "safety_flags": {
            "read_only": True,
            "github_mutation_allowed": False,
            "broker_order_submitted": False,
            "live_trading": False,
            "secrets_redacted": True,
        },
        "sections": sections,
        "recommendations": recommendations,
    }
    payload["markdown"] = render_markdown(payload)
    return payload


def _recommendations(sections: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    kanban_raw = sections.get("kanban")
    cron_raw = sections.get("cron")
    gateway_raw = sections.get("gateway")
    kanban: dict[str, Any] = kanban_raw if isinstance(kanban_raw, dict) else {}
    cron: dict[str, Any] = cron_raw if isinstance(cron_raw, dict) else {}
    gateway: dict[str, Any] = gateway_raw if isinstance(gateway_raw, dict) else {}
    if int(kanban.get("blocked", 0) or 0) > 0:
        recs.append("review_blocked_kanban_items")
    if int(cron.get("failed_last_24h", 0) or 0) > 0:
        recs.append("inspect_failed_cron_jobs")
    if int(gateway.get("errors_last_24h", 0) or 0) > 0:
        recs.append("inspect_gateway_errors")
    if not recs:
        recs.append("no_operator_action_required")
    return recs


def render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Clawley daily brief", "", "Geen automatische acties uitgevoerd.", ""]
    for key, value in payload["sections"].items():
        lines.append(f"## {key}")
        lines.append(json.dumps(value, sort_keys=True, ensure_ascii=False))
        lines.append("")
    lines.append("## Aanbevolen volgende stap")
    for rec in payload["recommendations"]:
        lines.append(f"- {rec}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Redacted JSON status snapshot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_daily_brief(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "write_performed": False, "read_only": True}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
