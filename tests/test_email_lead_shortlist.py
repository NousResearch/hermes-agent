import csv
import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "email_lead_shortlist.py"
)
spec = importlib.util.spec_from_file_location("email_lead_shortlist", MODULE_PATH)
assert spec is not None
email_lead_shortlist = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = email_lead_shortlist
assert spec.loader is not None
spec.loader.exec_module(email_lead_shortlist)


def record(sender, subject, body, *, tags=None, date="Tue, 19 May 2026 05:00:00 +0000"):
    return {
        "schema": "hermes.email_inbound_lead.v1",
        "no_send": True,
        "human_review_required": True,
        "source": "email_gateway",
        "sender_email": sender,
        "sender_name": sender.split("@")[0].title(),
        "subject": subject,
        "date": date,
        "message_id": f"<{sender}-{subject}@example.test>",
        "intent_tags": tags or ["inbound_email"],
        "contact_paths": {
            "emails": ["ops@example.test"],
            "phones": ["+1 555 010 9999"],
            "urls": ["https://example.test/demo"],
        },
        "body_excerpt": body,
        "attachment_count": 0,
    }


def test_build_shortlist_dedupes_by_sender_and_keeps_stronger_record():
    low = record(
        "Buyer@Example.com",
        "Question",
        "Could you send more info?",
        tags=["inbound_email"],
        date="Tue, 19 May 2026 04:00:00 +0000",
    )
    high = record(
        "buyer@example.com",
        "Need demo and pricing this week",
        "We have budget for a CRM automation pilot and want a proposal.",
        tags=["demo_request", "pricing"],
        date="Tue, 19 May 2026 05:00:00 +0000",
    )

    rows, skipped = email_lead_shortlist.build_shortlist([low, high])

    assert skipped == []
    assert len(rows) == 1
    assert rows[0].normalized_sender == "buyer@example.com"
    assert rows[0].confidence == "high"
    assert rows[0].record["subject"] == "Need demo and pricing this week"


def test_build_shortlist_skips_records_without_no_send_guardrails():
    unsafe = record("prospect@example.com", "demo", "call me", tags=["demo_request"])
    unsafe["no_send"] = False

    rows, skipped = email_lead_shortlist.build_shortlist([unsafe])

    assert rows == []
    assert len(skipped) == 1
    assert "no_send flag is not true" in skipped[0]


def test_cli_writes_csv_and_markdown_summary(tmp_path):
    input_path = tmp_path / "leads.jsonl"
    csv_path = tmp_path / "shortlist.csv"
    summary_path = tmp_path / "summary.md"
    records = [
        record(
            "founder@example.com",
            "Demo request for AI receptionist",
            "Can we book a call this week? We need pricing for appointment automation.",
            tags=["demo_request", "appointment", "pricing"],
        ),
        {"sender_email": "bad@example.com", "no_send": True},
    ]
    input_path.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8"
    )

    exit_code = email_lead_shortlist.main([
        str(input_path),
        "--csv",
        str(csv_path),
        "--summary",
        str(summary_path),
        "--min-confidence",
        "medium",
    ])

    assert exit_code == 0
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["sender_email"] == "founder@example.com"
    assert rows[0]["no_send"] == "true"
    assert rows[0]["human_review_required"] == "true"
    summary = summary_path.read_text(encoding="utf-8")
    assert "No-send email lead CRM shortlist" in summary
    assert "Shortlisted unique senders: 1" in summary
    assert "Skipped/flagged records: 1" in summary
