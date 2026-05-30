#!/usr/bin/env python3.11
"""Seed deterministic Hermes Lab data for self-improvement smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.dev_control.lab_environment import lab_paths_from_env, validate_lab_or_raise  # noqa: E402
from gateway.dev_control.product_events import DevProductEventStore  # noqa: E402
from gateway.dev_control.production_signals import DevProductionSignalStore  # noqa: E402
from gateway.dev_control.reliability import DevReliabilityStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed lab-only reliability and product signals.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--gateway-port", default="8662")
    args = parser.parse_args()
    paths = lab_paths_from_env()
    db_path = Path(args.db_path or paths["db_path"]).expanduser()
    repos_dir = Path(paths["repos_dir"])
    validate_lab_or_raise(
        hermes_home=db_path.parent,
        gateway_port=args.gateway_port,
        repo_roots=[
            repos_dir / "hermes-agent",
            repos_dir / "Oryn",
        ],
    )
    result = seed_lab_data(db_path)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def seed_lab_data(db_path: Path) -> dict:
    now = time.time()
    reliability_store = DevReliabilityStore(db_path)
    product_store = DevProductEventStore(db_path)
    signal_store = DevProductionSignalStore(db_path)

    outcomes = [
        _outcome("lab-plan-implement-1", "lab-task-implement-1", "workspace.implement", "high", now - 3600, failed=True),
        _outcome("lab-plan-implement-2", "lab-task-implement-2", "workspace.implement", "high", now - 1800, failed=False),
        _outcome("lab-plan-verify-1", "lab-task-verify-1", "workspace.test", "high", now - 1200, failed=True, escaped=True),
    ]
    for outcome in outcomes:
        reliability_store.upsert_outcome(outcome)

    product = product_store.ingest_batch({
        "events": [
            {
                "event_id": "lab-product-api-1",
                "type": "product.api_failure",
                "client_ts": now - 900,
                "app_version": "lab",
                "session_id": "lab-session",
                "message_redacted": "Synthetic lab API failure",
                "context": {"endpoint": "/v1/dev/reliability", "status": "500", "screen": "Lab"},
            },
            {
                "event_id": "lab-product-api-2",
                "type": "product.api_failure",
                "client_ts": now - 850,
                "app_version": "lab",
                "session_id": "lab-session",
                "message_redacted": "Synthetic lab API failure repeat",
                "context": {"endpoint": "/v1/dev/reliability", "status": "500", "screen": "Lab"},
            },
        ]
    })

    return {
        "ok": True,
        "object": "hermes.dev_lab_seed",
        "db_path": str(db_path),
        "outcome_count": len(outcomes),
        "product_events_accepted": product["accepted"],
        "signal_report_count": len(signal_store.list_reports()),
    }


def _outcome(plan_id: str, task_id: str, profile_id: str, risk_level: str, when: float, *, failed: bool, escaped: bool = False) -> dict:
    return {
        "plan_id": plan_id,
        "task_id": task_id,
        "category": f"{profile_id}/{risk_level}",
        "profile_id": profile_id,
        "risk_level": risk_level,
        "terminal_status": "completed",
        "merged": True,
        "verification_verdict": "failed" if failed else "verified",
        "ci_state": "failure" if failed else "success",
        "code_review_verdict": "changes_requested" if failed else "approved",
        "output_contract_score": 0.42 if failed else 0.96,
        "rework_count": 2 if failed else 0,
        "escaped": escaped,
        "escape_refs": [{"type": "product_event", "event_id": "lab-product-api-1"}] if escaped else [],
        "source_refs": {"source": "seed_dev_lab_data", "seeded": True},
        "completed_at": when,
        "merged_at": when,
    }


if __name__ == "__main__":
    raise SystemExit(main())
