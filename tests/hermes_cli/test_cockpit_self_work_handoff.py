from __future__ import annotations

import json

from hermes_cli.self_work_handoff import (
    cockpit_self_work_handoff_payload,
    latest_handoff_path,
    normalise_self_work_handoff,
    read_latest_self_work_handoff,
    write_self_work_handoff,
)


def test_normalise_self_work_handoff_reuses_operator_checklist_and_redacts_sensitive_values():
    record = normalise_self_work_handoff(
        {
            "issue_identifier": "BIF-547",
            "goal": "Restart-safe self-work handoff",
            "last_action": "Ran curl with api_key=raw-secret-value and touched gateway/session code",
            "next_safe_step": "Run scripts/run_tests.sh tests/hermes_cli/test_cockpit_self_work_handoff.py",
            "touched_files": ["hermes_cli/self_work_handoff.py", "~/.local/bin/get_credential.sh"],
            "operator_checklist": {
                "title": "BIF-547",
                "current_index": 1,
                "steps": [
                    {"label": "Capture exact work position", "status": "done"},
                    {"label": "Expose Cockpit resume brief", "status": "pending"},
                ],
            },
        }
    )

    assert record["issue_identifier"] == "BIF-547"
    assert "✓ done  Capture exact work position" in record["rendered_checklist"]
    assert "▶ now   Expose Cockpit resume brief" in record["rendered_checklist"]
    serialized = json.dumps(record)
    assert "raw-secret-value" not in serialized
    assert "get_credential.sh" not in serialized
    assert "[redacted credential path]" in serialized


def test_write_self_work_handoff_persists_latest_and_history(_isolate_hermes_home):
    record = write_self_work_handoff(
        {
            "issue_identifier": "BIF-547",
            "current_phase": "verification",
            "last_action": "Added durable handoff helper",
            "pending_verification": ["Run focused pytest"],
            "operator_checklist": {
                "title": "BIF-547",
                "steps": [
                    {"label": "Implement helper", "status": "done"},
                    {"label": "Run tests", "status": "current"},
                ],
            },
        }
    )

    assert latest_handoff_path().exists()
    assert latest_handoff_path().stat().st_mode & 0o777 == 0o600
    loaded = read_latest_self_work_handoff()
    assert loaded is not None
    assert loaded["issue_identifier"] == "BIF-547"
    assert loaded["current_phase"] == "verification"
    assert loaded["operator_checklist"]["steps"][1]["status"] == "current"
    payload = cockpit_self_work_handoff_payload()
    assert payload["read_only"] is True
    assert payload["actions_enabled"] is False
    assert payload["mutation_enabled"] is False
    assert payload["has_handoff"] is True
    assert payload["handoff"]["last_action"] == record["last_action"]
