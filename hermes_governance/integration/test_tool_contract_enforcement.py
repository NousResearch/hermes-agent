from __future__ import annotations

import json

import pytest

from agent.governance_runtime import GovernanceRuntime, GovernanceBlocked


def _runtime() -> GovernanceRuntime:
    return GovernanceRuntime(pack_root="hermes_governance")


def test_validate_tool_arguments_rejects_missing_privacy_fields():
    runtime = _runtime()

    with pytest.raises(GovernanceBlocked, match="required property"):
        runtime.validate_tool_arguments(
            "get_client_records",
            {
                "client_id": "client-42",
                "record_types": ["ledger_entries"],
                "purpose": "Analyse comptable",
            },
        )


def test_validate_tool_result_rejects_incomplete_envelope():
    runtime = _runtime()

    with pytest.raises(GovernanceBlocked, match="required property"):
        runtime.validate_tool_result(
            "get_client_records",
            json.dumps(
                {
                    "ok": True,
                    "tool_name": "get_client_records",
                    "result": {
                        "records": [],
                        "fields_disclosed": [],
                        "privacy_flags": [],
                        "retention_class": "standard",
                    },
                },
                ensure_ascii=False,
            ),
        )


def test_validate_tool_result_rejects_untyped_audit_metadata():
    runtime = _runtime()

    with pytest.raises(GovernanceBlocked, match="unexpected|Additional properties"):
        runtime.validate_tool_arguments(
            "log_audit_event",
            {
                "event_type": "policy_block",
                "severity": "high",
                "summary": "Blocage policy",
                "metadata": {
                    "unexpected": "value"
                },
            },
        )
