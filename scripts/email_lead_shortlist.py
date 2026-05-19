#!/usr/bin/env python3
"""Convert Hermes email inbound-lead JSONL into a no-send CRM shortlist.

This utility is intentionally local-only: it reads records captured by the
email gateway's EMAIL_LEAD_CAPTURE_PATH option and writes review artifacts for
humans to inspect before any CRM import or outbound action happens.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "hermes.email_inbound_lead.v1"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

INTENT_WEIGHTS = {
    "demo_request": 35,
    "sales_inquiry": 30,
    "consulting_interest": 30,
    "pricing": 25,
    "appointment": 25,
    "support": 10,
    "partnership": 10,
    "inbound_email": 5,
}

KEYWORD_WEIGHTS = {
    "urgent": 10,
    "asap": 10,
    "this week": 8,
    "budget": 8,
    "decision": 8,
    "proposal": 8,
    "quote": 8,
    "contract": 8,
    "pilot": 8,
    "integration": 6,
    "crm": 6,
    "automation": 6,
    "workflow": 6,
    "call": 5,
    "meeting": 5,
}

CSV_FIELDS = [
    "rank",
    "confidence",
    "sender_email",
    "sender_name",
    "subject",
    "date",
    "intent_tags",
    "score",
    "confidence_reasons",
    "contact_emails",
    "phones",
    "urls",
    "message_id",
    "human_review_required",
    "no_send",
    "recommended_next_step",
    "body_excerpt",
]


@dataclass(frozen=True)
class ShortlistRow:
    record: dict[str, Any]
    score: int
    confidence: str
    reasons: list[str]
    normalized_sender: str
    parsed_date: datetime


def compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_date(value: Any) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    text = str(value)
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(item, dict):
                errors.append(
                    f"line {line_no}: expected object, got {type(item).__name__}"
                )
                continue
            records.append(item)
    return records, errors


def validate_record(record: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if record.get("schema") not in {SCHEMA, None}:
        issues.append(f"unexpected schema {record.get('schema')!r}")
    sender = compact(record.get("sender_email")).lower()
    if not sender or not EMAIL_RE.match(sender):
        issues.append("missing or invalid sender_email")
    if record.get("no_send") is not True:
        issues.append("no_send flag is not true")
    if record.get("human_review_required") is not True:
        issues.append("human_review_required flag is not true")
    return issues


def _list_from_contact_paths(record: dict[str, Any], key: str) -> list[str]:
    paths = record.get("contact_paths") or {}
    values = paths.get(key) or []
    if not isinstance(values, list):
        return []
    return [compact(v) for v in values if compact(v)]


def score_record(record: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    tags = record.get("intent_tags") or []
    if not isinstance(tags, list):
        tags = []
    for tag in sorted({compact(t) for t in tags if compact(t)}):
        weight = INTENT_WEIGHTS.get(tag, 5)
        score += weight
        reasons.append(f"intent:{tag}+{weight}")

    text = f"{record.get('subject', '')}\n{record.get('body_excerpt', '')}".lower()
    for keyword, weight in KEYWORD_WEIGHTS.items():
        if keyword in text:
            score += weight
            reasons.append(f"keyword:{keyword}+{weight}")

    contact_bonus = 0
    for key, weight in (("emails", 5), ("phones", 8), ("urls", 4)):
        count = len(_list_from_contact_paths(record, key))
        if count:
            bonus = min(count * weight, 12)
            contact_bonus += bonus
            reasons.append(f"contact_{key}:{count}+{bonus}")
    score += contact_bonus

    if record.get("attachment_count"):
        score += 3
        reasons.append("attachment+3")
    return min(score, 100), reasons or ["base inbound email"]


def confidence_label(score: int, record: dict[str, Any]) -> str:
    has_reply_path = bool(
        _list_from_contact_paths(record, "emails")
        or compact(record.get("sender_email"))
    )
    if score >= 65 and has_reply_path:
        return "high"
    if score >= 35 and has_reply_path:
        return "medium"
    return "low"


def build_shortlist(
    records: Iterable[dict[str, Any]],
) -> tuple[list[ShortlistRow], list[str]]:
    skipped: list[str] = []
    best_by_sender: dict[str, ShortlistRow] = {}
    for idx, record in enumerate(records, 1):
        issues = validate_record(record)
        sender = compact(record.get("sender_email")).lower()
        if issues:
            skipped.append(
                f"record {idx} ({sender or 'unknown'}): " + "; ".join(issues)
            )
            continue
        score, reasons = score_record(record)
        row = ShortlistRow(
            record=record,
            score=score,
            confidence=confidence_label(score, record),
            reasons=reasons,
            normalized_sender=sender,
            parsed_date=parse_date(record.get("date")),
        )
        current = best_by_sender.get(sender)
        if current is None or (row.score, row.parsed_date) > (
            current.score,
            current.parsed_date,
        ):
            best_by_sender[sender] = row

    rows = sorted(
        best_by_sender.values(),
        key=lambda row: (row.score, row.parsed_date, row.normalized_sender),
        reverse=True,
    )
    return rows, skipped


def recommended_next_step(row: ShortlistRow) -> str:
    if row.confidence == "high":
        return "Review thread and approve CRM import / human follow-up"
    if row.confidence == "medium":
        return "Review contact details and qualify before CRM import"
    return "Keep in research queue until a stronger buying signal appears"


def write_csv(path: Path, rows: list[ShortlistRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            record = row.record
            writer.writerow({
                "rank": rank,
                "confidence": row.confidence,
                "sender_email": row.normalized_sender,
                "sender_name": compact(record.get("sender_name")),
                "subject": compact(record.get("subject")),
                "date": compact(record.get("date")),
                "intent_tags": ";".join(
                    compact(t) for t in record.get("intent_tags", []) if compact(t)
                ),
                "score": row.score,
                "confidence_reasons": ";".join(row.reasons),
                "contact_emails": ";".join(_list_from_contact_paths(record, "emails")),
                "phones": ";".join(_list_from_contact_paths(record, "phones")),
                "urls": ";".join(_list_from_contact_paths(record, "urls")),
                "message_id": compact(record.get("message_id")),
                "human_review_required": "true",
                "no_send": "true",
                "recommended_next_step": recommended_next_step(row),
                "body_excerpt": compact(record.get("body_excerpt"))[:500],
            })


def write_summary(
    path: Path,
    rows: list[ShortlistRow],
    skipped: list[str],
    input_path: Path,
    csv_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    high = sum(1 for row in rows if row.confidence == "high")
    medium = sum(1 for row in rows if row.confidence == "medium")
    low = sum(1 for row in rows if row.confidence == "low")
    lines = [
        "# No-send email lead CRM shortlist",
        "",
        f"Input: `{input_path}`",
        f"CSV: `{csv_path}`",
        "",
        "Guardrails: no outbound messages sent; all rows require human review before CRM import or follow-up.",
        "",
        "## Verification counters",
        "",
        f"- Shortlisted unique senders: {len(rows)}",
        f"- Confidence: high={high}, medium={medium}, low={low}",
        f"- Skipped/flagged records: {len(skipped)}",
        "",
        "## Top rows",
        "",
    ]
    if rows:
        for rank, row in enumerate(rows[:10], 1):
            record = row.record
            lines.extend([
                f"{rank}. **{row.confidence}** ({row.score}) — {row.normalized_sender} — {compact(record.get('subject'))}",
                f"   - Intent: {', '.join(compact(t) for t in record.get('intent_tags', []) if compact(t)) or 'inbound_email'}",
                f"   - Next step: {recommended_next_step(row)}",
            ])
    else:
        lines.append("No valid records were eligible for shortlist output.")

    if skipped:
        lines.extend(["", "## Skipped / flagged", ""])
        lines.extend(f"- {item}" for item in skipped[:25])
        if len(skipped) > 25:
            lines.append(f"- ... {len(skipped) - 25} more")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Inbound lead JSONL from EMAIL_LEAD_CAPTURE_PATH"
    )
    parser.add_argument(
        "--csv", type=Path, required=True, help="Output CSV shortlist path"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Output Markdown review summary path",
    )
    parser.add_argument(
        "--min-confidence", choices=("low", "medium", "high"), default="low"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    records, parse_errors = load_jsonl(args.input)
    rows, skipped = build_shortlist(records)
    skipped = parse_errors + skipped
    order = {"low": 0, "medium": 1, "high": 2}
    rows = [row for row in rows if order[row.confidence] >= order[args.min_confidence]]
    write_csv(args.csv, rows)
    write_summary(args.summary, rows, skipped, args.input, args.csv)
    print(
        json.dumps(
            {
                "input_records": len(records),
                "shortlisted": len(rows),
                "skipped_or_flagged": len(skipped),
                "csv": str(args.csv),
                "summary": str(args.summary),
                "no_send": True,
                "human_review_required": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
