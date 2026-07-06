from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from torben_skill_forms import build_form_packet, verify_source


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


def test_form_packet_shape_prepares_answers_and_no_submission() -> None:
    packet = build_form_packet(
        action={
            "what": "Submit dependent-care reimbursement",
            "deadline": "2026-07-31",
            "consequence": "missed reimbursement window",
            "cost": "$0",
            "refundability": "reimbursable",
            "required_docs": ["receipt.pdf"],
            "channel": "benefits portal",
        },
        source={"sender": "benefits@example.com", "url": "https://benefits.example.com/forms"},
        answers={"employee name": "Eric", "amount": "$42"},
        required_fields=["employee name", "amount"],
        docs_to_attach=["state/receipts/daycare.pdf"],
        trusted_domains=["example.com"],
    )

    assert packet["schema"] == "torben.skill-forms.v1"
    assert packet["status"] == "packet_only"
    assert packet["source_check"]["status"] == "trusted"
    assert packet["prepared_answers"] == {"employee name": "Eric", "amount": "$42"}
    assert packet["missing_info"] == []
    assert packet["docs_to_attach"] == ["state/receipts/daycare.pdf"]
    assert packet["approval_request"].startswith("Approve submission")
    assert packet["external_actions_taken"] == []
    assert "no form submitted" in packet["blocked_actions"]


def test_secret_fields_are_handed_to_eric_and_not_stored() -> None:
    packet = build_form_packet(
        action={"what": "Insurance form", "deadline": "2026-08-01", "channel": "portal"},
        source={"sender": "insurance@example.com", "url": "https://insurance.example.com"},
        answers={"member id": "MID-123", "card number": "4111111111111111", "name": "Eric"},
        required_fields=["member id", "card number", "name"],
        trusted_domains=["example.com"],
        category="payment_adjacent",
    )
    serialized = json.dumps(packet)

    assert "MID-123" not in serialized
    assert "4111111111111111" not in serialized
    assert packet["prepared_answers"] == {"name": "Eric"}
    assert packet["hand_to_eric"] == ["card number", "member id"]
    assert packet["category"] == "payment_adjacent"


def test_untrusted_source_flags_and_does_not_fill() -> None:
    packet = build_form_packet(
        action={"what": "Fill payroll form", "deadline": "2026-08-01", "channel": "web"},
        source={"sender": "payroll@unknown.example", "url": "https://unknown.example/form"},
        answers={"name": "Eric", "address": "Home"},
        required_fields=["name", "address"],
        trusted_domains=["company.example"],
    )

    assert packet["status"] == "flagged_unknown_source"
    assert packet["source_check"]["status"] == "flagged"
    assert packet["prepared_answers"] == {}
    assert "source verification" in packet["missing_info"]
    assert set(packet["missing_info"]) >= {"name", "address"}
    assert "unknown source routes to inbox-safety semantics" in packet["blocked_actions"]


def test_payment_link_subdomain_is_trusted_under_trusted_domain() -> None:
    check = verify_source(
        sender="benefits@example.com",
        source_url="https://benefits.example.com",
        payment_links=["https://pay.example.com/invoice/1"],
        trusted_domains=["example.com"],
    )

    assert check["trusted"] is True
    assert check["status"] == "trusted"


def test_submit_refuses_until_approval_then_records_spine_mutation(tmp_path: Path) -> None:
    ledger = tmp_path / "torben-action-ledger.jsonl"
    code = r'''
import json
import sys
from pathlib import Path
scripts_dir = Path(sys.argv[1])
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
from torben_decision_packet import resolve_decision_reply
from torben_skill_forms import build_form_packet, stage_form_decision, submit_approved_form

ledger = Path(sys.argv[2])
packet = build_form_packet(
    action={"what": "Submit reimbursement", "deadline": "2026-07-31", "channel": "benefits portal"},
    source={"sender": "benefits@example.com", "url": "https://benefits.example.com/forms"},
    answers={"name": "Eric", "amount": "$42"},
    required_fields=["name", "amount"],
    trusted_domains=["example.com"],
)
stage = stage_form_decision(ledger_path=ledger, loop_id=4, packet=packet)
handle = stage["record"]["handle"]
before = submit_approved_form(
    ledger_path=ledger,
    handle=handle,
    confirmation_pointer="portal:pending",
    approved_channel="benefits portal",
)
approved = resolve_decision_reply(ledger_path=ledger, reply_text=f"approve option 1 {handle}")
after = submit_approved_form(
    ledger_path=ledger,
    handle=handle,
    confirmation_pointer="portal:confirmation:42",
    approved_channel="benefits portal",
)
print(json.dumps({"before": before, "approved": approved, "after": after}, sort_keys=True))
'''
    payload = _uv_python(code, str(ledger))

    assert payload["before"]["status"] == "refused"
    assert payload["before"]["reason"] == "explicit_approval_required"
    assert payload["approved"]["status"] == "approved"
    assert payload["after"]["status"] == "submitted"
    assert payload["after"]["mutation"]["record"]["executor_state"]["schema"] == "torben.mutation-spine.v1"
    assert payload["after"]["mutation"]["record"]["executor_state"]["category"] == "form_filing"
    assert payload["after"]["mutation"]["record"]["executor_state"]["metadata"]["confirmation_pointer"] == "portal:confirmation:42"
