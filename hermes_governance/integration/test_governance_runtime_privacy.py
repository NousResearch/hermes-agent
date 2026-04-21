from __future__ import annotations

import json

from agent.governance_runtime import GovernanceRuntime, GovernanceState, GovernanceBlocked


def _runtime() -> GovernanceRuntime:
    return GovernanceRuntime(pack_root="hermes_governance")


def test_data_access_with_pii_triggers_sensitive_data_escalation():
    runtime = _runtime()
    state = GovernanceState()

    runtime.update_state_from_tool_result(
        state,
        "get_client_records",
        json.dumps(
            {
                "ok": True,
                "tool_name": "get_client_records",
                "trace_id": "trace-1",
                "timestamp": "2026-04-19T10:00:00Z",
                "result": {
                    "records": [{"amount": 12.5, "counterparty_label": "ACME"}],
                    "fields_disclosed": ["amount", "counterparty_label"],
                    "privacy_flags": ["contains_pii"],
                    "retention_class": "standard",
                    "requires_supervisor_approval": False
                }
            },
            ensure_ascii=False,
        ),
        {
            "client_id": "client-42",
            "record_types": ["bank_transactions"],
            "purpose": "Réconciliation bancaire de fin de mois",
            "purpose_code": "bank_reconciliation",
            "lawful_basis": "legal_obligation",
            "requested_fields": ["amount"],
            "minimize_fields": True,
        },
    )

    outcome = runtime.evaluate_policy(state)

    assert outcome["matched"] is True
    assert outcome["rule_id"] == "sensitive-data-escalation"
    assert state.audit_logged_for_data_access is False
    assert state.requested_fields_exceed_policy is True


def test_validate_final_response_rejects_raw_identifier():
    runtime = _runtime()
    with_email = {
        "status": "BLOQUE",
        "certainty": "FAIBLE_VERIFICATION_REQUISE",
        "scope": {
            "domain": "privacy",
            "jurisdiction": "FR",
            "fact_date": None,
            "source_checked_at": "2026-04-19T10:00:00Z",
            "risk_level": "high",
        },
        "facts": {
            "verified_facts": ["Client contact: compta@example.com"],
            "assumptions": [],
            "missing_facts": [],
        },
        "sources": [
            {
                "source_type": "other",
                "reference": "governance_runtime",
                "effective_date": None,
                "checked_at": "2026-04-19T10:00:00Z",
                "verified": False,
                "url": None,
            }
        ],
        "analysis": {
            "mode": "blocked",
            "major": "Blocage",
            "minor": "Fuite d'identifiant",
            "conclusion": "Refus",
        },
        "risks": [],
        "next_action": {
            "type": "blocked",
            "message": "Masquage requis",
            "escalation_id": None,
        },
        "audit_trail": {
            "tools_called": [],
            "policy_rule_hits": [],
            "audit_summary": "tool_calls=0;audit_events=0;redactions=0;violations=0",
            "redactions_applied": False,
        },
    }

    try:
        runtime.validate_final_response_text(json.dumps(with_email, ensure_ascii=False))
    except GovernanceBlocked as exc:
        assert "raw_identifier" in str(exc)
    else:
        raise AssertionError("expected final response validation to reject raw identifier")


def test_hmac_fingerprint_changes_per_runtime_instance():
    runtime_a = _runtime()
    runtime_b = _runtime()
    arguments = {
        "client_id": "client-42",
        "record_types": ["ledger_entries"],
        "purpose": "Analyse du grand livre comptable",
        "purpose_code": "ledger_analysis",
        "lawful_basis": "legal_obligation",
        "requested_fields": ["amount"],
    }

    fp_a = runtime_a.build_tool_call_fingerprint("get_client_records", arguments)
    fp_b = runtime_b.build_tool_call_fingerprint("get_client_records", arguments)

    assert fp_a != fp_b
