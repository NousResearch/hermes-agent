"""Construction of bounded test-discovery receipts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .receipts import validate_receipt


def build_test_discovery_receipt(
    *,
    repository: str,
    head_sha: str,
    target: str,
    cases: list[dict[str, Any]],
    commands: list[dict[str, Any]],
    task_id: str | None = None,
) -> dict[str, Any]:
    """Record test ideas and observed execution without executing commands."""

    incomplete_reasons = [
        str(command.get("reason"))
        for command in commands
        if command.get("status") in {"unavailable", "skipped", "timeout"} and command.get("reason")
    ]
    statuses = {str(command.get("status") or "unknown") for command in commands}
    if not commands:
        status = "incomplete"
    elif "failed" in statuses:
        status = "failed"
    elif incomplete_reasons:
        status = "degraded"
    elif statuses <= {"passed"}:
        status = "passed"
    else:
        status = "incomplete"
    stable = json.dumps(
        {"repository": repository, "head_sha": head_sha, "target": target, "task_id": task_id},
        sort_keys=True,
    ).encode()
    return validate_receipt(
        {
            "schema_name": "engineering_evidence_v1",
            "receipt_type": "test_discovery",
            "receipt_id": f"tests-{hashlib.sha256(stable).hexdigest()[:24]}",
            "repository": repository,
            "head_sha": head_sha,
            "authority": "diagnostic-only",
            "task_id": task_id,
            "target": target,
            "cases": cases,
            "commands": commands,
            "status": status,
            "incomplete_reasons": incomplete_reasons,
        }
    )
