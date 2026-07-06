from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from torben_skill_appointments import build_appointment_packet, packet_to_decision_options


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(
    os.environ.get(
        "HERMES_REPO_ROOT",
        "/Users/eric/hermes-agent-torben-work" if Path("/Users/eric/hermes-agent-torben-work").exists() else "/Users/ericfreeman/.hermes/hermes-agent",
    )
)


def _uv_python(code: str, *args: str) -> dict:
    env = dict(os.environ)
    env["HERMES_REPO_ROOT"] = str(REPO_ROOT)
    result = subprocess.run(
        ["uv", "run", "python", "-c", code, str(SCRIPTS_DIR), *args],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(result.stdout)


def test_appointment_packet_has_best_backup_needed_list_and_no_action() -> None:
    packet = build_appointment_packet(
        goal="Book HVAC service",
        constraints={"who": "Eric", "where": "home", "earliest": "2026-07-08", "latest": "2026-07-12", "duration": "2h"},
        candidates=[
            {"label": "Friday 3 PM", "score": 0.6, "provider_link": "https://provider.example/book"},
            {"label": "Thursday 10 AM", "score": 0.9, "phone": "+15551234567"},
        ],
        provider={"name": "Northside HVAC", "phone": "+15551234567"},
        forms=["docs/forms/hvac-intake.pdf"],
        questions=["Confirm ladder access"],
    )

    assert packet["schema"] == "torben.skill-appointments.v1"
    assert packet["category"] == "booking"
    assert packet["status"] == "packet_only"
    assert packet["best_option"]["label"] == "Thursday 10 AM"
    assert packet["backup_option"]["label"] == "Friday 3 PM"
    assert packet["needed_before_booking"] == []
    assert packet["external_actions_taken"] == []
    assert "no appointment booked" in packet["blocked_actions"]
    assert packet["forms_doc_pointers"] == ["docs/forms/hvac-intake.pdf"]


def test_appointment_packet_redacts_secret_like_fields() -> None:
    packet = build_appointment_packet(
        goal="Book specialist appointment",
        constraints={"who": "Eric", "member id": "ABC-123", "where": "clinic"},
        candidates=[{"label": "Monday 9 AM", "score": 1.0}],
        provider={"portal password": "secret", "name": "Clinic"},
    )
    serialized = json.dumps(packet).lower()

    assert "abc-123" not in serialized
    assert '"secret"' not in serialized
    assert "member id" in packet["needed_before_booking"]
    assert "portal password" in packet["needed_before_booking"]


def test_appointment_decision_options_are_decision_packet_ready() -> None:
    packet = build_appointment_packet(
        goal="Book dentist",
        constraints={"who": "Eric", "where": "Dentist", "earliest": "2026-07-08", "latest": "2026-07-12", "duration": "30m"},
        candidates=[{"label": "Wednesday 11 AM", "score": 0.8}, {"label": "Friday 2 PM", "score": 0.7}],
    )

    options = packet_to_decision_options(packet)

    assert [option["label"] for option in options] == [
        "Book best option: Wednesday 11 AM",
        "Use backup option: Friday 2 PM",
    ]
    assert all(set(option) == {"label", "upside", "downside", "cost_time", "risk"} for option in options)


def test_booking_execute_refuses_until_approval_then_records_mutation(tmp_path: Path) -> None:
    ledger = tmp_path / "torben-action-ledger.jsonl"
    code = r'''
import json
import sys
from pathlib import Path
scripts_dir = Path(sys.argv[1])
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
from torben_decision_packet import resolve_decision_reply
from torben_skill_appointments import build_appointment_packet, execute_approved_booking, stage_appointment_decision

ledger = Path(sys.argv[2])
packet = build_appointment_packet(
    goal="Book HVAC service",
    constraints={"who": "Eric", "where": "home", "earliest": "2026-07-08", "latest": "2026-07-12", "duration": "2h"},
    candidates=[{"label": "Thursday 10 AM", "score": 0.9}],
)
stage = stage_appointment_decision(ledger_path=ledger, loop_id=3, packet=packet)
handle = stage["record"]["handle"]
before = execute_approved_booking(ledger_path=ledger, handle=handle, confirmation_pointer="provider:confirmation:pending")
approved = resolve_decision_reply(ledger_path=ledger, reply_text=f"approve option 1 {handle}")
after = execute_approved_booking(ledger_path=ledger, handle=handle, confirmation_pointer="provider:confirmation:abc123")
print(json.dumps({"before": before, "approved": approved, "after": after}, sort_keys=True))
'''
    payload = _uv_python(code, str(ledger))

    assert payload["before"]["status"] == "refused"
    assert payload["before"]["reason"] == "explicit_approval_required"
    assert payload["approved"]["status"] == "approved"
    assert payload["after"]["status"] == "executed"
    assert payload["after"]["dispatch"]["category"] == "booking"
    assert payload["after"]["mutation"]["record"]["executor_state"]["schema"] == "torben.mutation-spine.v1"
    assert payload["after"]["mutation"]["record"]["executor_state"]["category"] == "booking"
    assert payload["after"]["mutation"]["record"]["executor_state"]["metadata"]["confirmation_pointer"] == "provider:confirmation:abc123"
