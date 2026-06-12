#!/usr/bin/env python3
"""Convert read-only maintainer-sweep summaries to Kanban proposals.

No board writes happen here. This is the deterministic bridge from durable
ClawSweeper/Clownfish-style reports to proposed work items a human/operator can
review before importing into Hermes Kanban.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HUMAN_GATE_LABELS = {"security", "privacy", "secrets", "auth", "infra", "broker", "trading", "live"}


def build_kanban_proposals(summary: dict[str, Any]) -> dict[str, Any]:
    repo = str(summary.get("repo") or "unknown/repo")
    raw_items = summary.get("items")
    items: list[Any] = raw_items if isinstance(raw_items, list) else []
    proposals = [_proposal(repo, item) for item in items if isinstance(item, dict)]
    return {
        "schema": "maintainer_sweep_kanban_proposals.v1",
        "repo": repo,
        "source_action_state": summary.get("action_state", "proposal"),
        "mutation_allowed": False,
        "write_performed": False,
        "proposals": proposals,
        "totals": {
            "source_records": int(summary.get("records", len(items)) or 0),
            "proposal_count": len(proposals),
            "human_gated": sum(1 for item in proposals if item["requires_human_gate"]),
        },
        "next_safe_action": "review_proposals_before_kanban_import",
    }


def _proposal(repo: str, item: dict[str, Any]) -> dict[str, Any]:
    labels = _labels(item.get("labels"))
    requires_human = bool(HUMAN_GATE_LABELS.intersection(labels)) or str(item.get("recommendation")) == "needs_human"
    kind = str(item.get("kind") or item.get("type") or "item")
    number = item.get("number", "unknown")
    title = str(item.get("title") or "Untitled")
    return {
        "idempotency_key": f"maintainer-sweep:{repo}:{kind}:{number}",
        "title": f"{repo} {kind} #{number}: {title}",
        "status": "proposal",
        "suggested_priority": "human_gate" if requires_human else "normal",
        "requires_human_gate": requires_human,
        "source": {"repo": repo, "kind": kind, "number": number, "labels": sorted(labels)},
        "mutation_allowed": False,
        "github_write_allowed": False,
    }


def _labels(value: Any) -> set[str]:
    result: set[str] = set()
    for label in value or []:
        if isinstance(label, dict):
            result.add(str(label.get("name") or "").lower())
        else:
            result.add(str(label).lower())
    return {label for label in result if label}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="maintainer_sweep summary.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    proposals = build_kanban_proposals(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proposals, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "proposal_count": proposals["totals"]["proposal_count"], "write_performed": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
