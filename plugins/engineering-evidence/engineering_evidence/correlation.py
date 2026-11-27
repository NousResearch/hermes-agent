"""Diagnostic-only correlation receipts for DevOps evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .receipts import validate_receipt


def build_deployment_correlation_receipt(
    *,
    repository: str,
    head_sha: str,
    identity: dict[str, Any],
    evidence: dict[str, list[dict[str, Any]]],
    confidence: str,
    uncertainty: list[str],
    rollback_recommendation: str | None = None,
    operator_decision: str | None = None,
    post_deploy_status: str = "unavailable",
) -> dict[str, Any]:
    """Correlate supplied references without deciding operational action."""

    stable = json.dumps(
        {"repository": repository, "head_sha": head_sha, "identity": identity, "evidence": evidence},
        sort_keys=True,
    ).encode()
    missing = [name for name in ("ci", "dependency_security", "runtime", "post_deploy") if not evidence.get(name)]
    status = "incomplete" if missing else "diagnostic"
    return validate_receipt(
        {
            "schema_name": "engineering_evidence_v1",
            "receipt_type": "deployment_correlation",
            "receipt_id": f"correlation-{hashlib.sha256(stable).hexdigest()[:24]}",
            "repository": repository,
            "head_sha": head_sha,
            "authority": "diagnostic-only",
            "identity": identity,
            "evidence": evidence,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "rollback_recommendation": rollback_recommendation,
            "operator_decision": operator_decision,
            "post_deploy_status": post_deploy_status,
            "status": status,
            "incomplete_reasons": [f"missing_evidence:{item}" for item in missing],
        }
    )
