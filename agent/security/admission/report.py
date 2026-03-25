from __future__ import annotations

from .models import AdmissionRecord, InspectionReport


def render_report(record: AdmissionRecord, report: InspectionReport) -> str:
    lines = [
        f"# Admission Report: {record.source.display_name}",
        "",
        f"- Record ID: `{record.record_id}`",
        f"- Kind: `{record.kind}`",
        f"- Status: `{record.status}`",
        f"- Source: `{record.source.uri}`",
        f"- Decision: `{report.decision}`",
        "",
        "## Summary",
        report.summary,
        "",
    ]
    if report.capabilities:
        lines.extend(["## Capabilities", *[f"- {item}" for item in report.capabilities], ""])
    if report.reasons:
        lines.extend(["## Reasons", *[f"- {item}" for item in report.reasons], ""])
    if report.warnings:
        lines.extend(["## Warnings", *[f"- {item}" for item in report.warnings], ""])
    return "\n".join(lines).rstrip() + "\n"
