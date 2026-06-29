#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Render the analysis report deterministically from findings JSON.

Injects a validated findings JSON object into the report shell's
report-data island and writes the self-contained HTML atomically.
With --md, also writes a markdown rendering of the same data as the
archival artifact.

Refuses (non-zero exit, message on stderr) when the JSON does not
parse, fails shape validation, or still carries the shell's
placeholder subject — a refused render means fix the findings file
and re-run, never hand-edit the HTML.

Usage:
  uv run render_report.py <findings.json> --shell <report-shell.html> \
      -o <out.html> [--md <out.md>]

On success prints one JSON line: output paths, grade, and severity
counts derived from the findings array.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

SEVERITIES = ("critical", "high", "medium", "low")
GRADES = ("excellent", "good", "fair", "poor")
PLACEHOLDER_SUBJECT = "__PLACEHOLDER__"
ISLAND_RE = re.compile(
    r'(<script[^>]*\bid="report-data"[^>]*>)(.*?)(</script>)', re.DOTALL
)


def fail(message: str) -> None:
    print(f"render_report: {message}", file=sys.stderr)
    sys.exit(1)


def validate(data: object) -> list[str]:
    """Return a list of shape errors; empty list means valid."""
    if not isinstance(data, dict):
        return ["top level must be a JSON object"]
    errors: list[str] = []

    subject = data.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        errors.append('"subject" must be a non-empty string')
    elif PLACEHOLDER_SUBJECT in subject:
        errors.append(
            f'"subject" still carries the placeholder {PLACEHOLDER_SUBJECT}; '
            "this is the unfilled shell sample, not real findings"
        )

    findings = data.get("findings")
    if not isinstance(findings, list):
        errors.append('"findings" must be an array (use [] for a clean pass)')
    else:
        for i, finding in enumerate(findings):
            if not isinstance(finding, dict):
                errors.append(f"findings[{i}] must be an object")

    grade = data.get("grade")
    if grade is not None and grade not in GRADES:
        errors.append(f'"grade" must be one of: {", ".join(GRADES)}')

    for key in ("themes", "recommendations"):
        value = data.get(key)
        if value is not None and (
            not isinstance(value, list)
            or any(not isinstance(item, dict) for item in value)
        ):
            errors.append(f'"{key}" must be an array of objects')

    strengths = data.get("strengths")
    if strengths is not None and (
        not isinstance(strengths, list)
        or any(not isinstance(item, str) for item in strengths)
    ):
        errors.append('"strengths" must be an array of strings')

    return errors


def severity_counts(findings: list[dict]) -> dict[str, int]:
    counts = {sev: 0 for sev in SEVERITIES}
    for finding in findings:
        sev = finding.get("severity")
        counts[sev if sev in counts else "low"] += 1
    return counts


def inject(shell_html: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    # A "</" sequence inside a JSON string would close the script tag
    # early in the browser; "<\/" is the same string to JSON.parse.
    payload = payload.replace("</", "<\\/")

    def replace(match: re.Match) -> str:
        return match.group(1) + "\n" + payload + "\n" + match.group(3)

    new_html, count = ISLAND_RE.subn(replace, shell_html, count=1)
    if count != 1:
        fail('shell has no <script id="report-data"> island to fill')
    return new_html


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _finding_lines(finding: dict, heading_level: str) -> list[str]:
    fid = str(finding.get("id", ""))
    title = str(finding.get("title", "(untitled finding)"))
    lines = [f"{heading_level} {fid} — {title}" if fid else f"{heading_level} {title}", ""]
    for key, label in (
        ("lens", "Lens"),
        ("location", "Location"),
        ("evidence", "Evidence"),
        ("recommendation", "Recommendation"),
        ("proposed_smallest", "Proposed smallest"),
        ("predicted_delta", "Predicted delta"),
    ):
        value = finding.get(key)
        if value:
            value = f"`{value}`" if key == "location" else str(value)
            lines.append(f"- {label}: {value}")
    lines.append("")
    return lines


def render_md(data: dict) -> str:
    findings = [f for f in data.get("findings", []) if isinstance(f, dict)]
    by_id = {str(f.get("id")): f for f in findings if f.get("id") is not None}
    counts = severity_counts(findings)
    lines: list[str] = []

    lines.append(f"# Analysis Report: {data.get('subject', '')}")
    lines.append("")
    meta = []
    if data.get("generated"):
        meta.append(f"Generated: {data['generated']}")
    if data.get("schema_version") is not None:
        meta.append(f"Schema: {data['schema_version']}")
    if meta:
        lines.append(" · ".join(meta))
        lines.append("")

    if data.get("grade"):
        lines.append(f"**Grade: {str(data['grade']).capitalize()}**")
        lines.append("")
    if data.get("verdict"):
        lines.append(f"> {data['verdict']}")
        lines.append("")
    summary = data.get("summary")
    if isinstance(summary, str) and summary:
        lines.append(summary)
        lines.append("")

    lines.append("| Severity | Count |")
    lines.append("| --- | --- |")
    for sev in SEVERITIES:
        lines.append(f"| {sev.capitalize()} | {counts[sev]} |")
    lines.append("")

    themes = data.get("themes") or []
    if themes:
        lines.append("## Themes")
        lines.append("")
        for i, theme in enumerate(themes, 1):
            lines.append(f"### {i}. {theme.get('title', '(untitled theme)')}")
            lines.append("")
            if theme.get("root_cause"):
                lines.append(f"- Root cause: {theme['root_cause']}")
            if theme.get("action"):
                lines.append(f"- Fix: {theme['action']}")
            ids = theme.get("finding_ids") or []
            if ids:
                lines.append("- Findings:")
                for fid in ids:
                    finding = by_id.get(str(fid))
                    if finding:
                        loc = finding.get("location")
                        suffix = f" — `{loc}`" if loc else ""
                        lines.append(
                            f"  - `{fid}` {finding.get('title', '')}{suffix}"
                        )
                    else:
                        lines.append(f"  - `{fid}`")
            lines.append("")

    strengths = data.get("strengths") or []
    if strengths:
        lines.append("## Strengths")
        lines.append("")
        for strength in strengths:
            lines.append(f"- {strength}")
        lines.append("")

    recommendations = data.get("recommendations") or []
    if recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(recommendations, 1):
            rank = rec.get("rank", i)
            resolves = rec.get("resolves")
            if isinstance(resolves, list) and resolves:
                suffix = " (resolves: " + ", ".join(map(str, resolves)) + ")"
            elif isinstance(resolves, (int, float)):
                suffix = f" (resolves {int(resolves)} findings)"
            else:
                suffix = ""
            lines.append(f"{rank}. {rec.get('action', '')}{suffix}")
        lines.append("")

    # Optional agent blocks: rendered only when present so the same
    # renderer serves both the workflow and agent schemas.
    profile = data.get("agent_profile")
    if isinstance(profile, dict) and any(profile.values()):
        lines.append("## Agent Profile")
        lines.append("")
        for key, label in (
            ("name", "Name"),
            ("title", "Title"),
            ("agent_type", "Type"),
            ("mission", "Mission"),
        ):
            if profile.get(key):
                lines.append(f"- {label}: {profile[key]}")
        lines.append("")

    capabilities = data.get("capabilities")
    if isinstance(capabilities, list) and capabilities:
        lines.append("## Capabilities")
        lines.append("")
        for cap in capabilities:
            if not isinstance(cap, dict) or not cap.get("name"):
                continue
            kind = f" ({cap['kind']})" if cap.get("kind") else ""
            note = f" — {cap['note']}" if cap.get("note") else ""
            lines.append(f"- **{cap['name']}**{kind}{note}")
        lines.append("")

    detailed = data.get("detailed_analysis")
    if isinstance(detailed, dict) and detailed:
        lines.append("## Per-Lens Verdicts")
        lines.append("")
        for lens, verdict in detailed.items():
            if verdict:
                lines.append(f"- **{lens}**: {verdict}")
        lines.append("")

    sanctum = data.get("sanctum")
    if isinstance(sanctum, dict) and sanctum.get("present") is not False:
        rows = []
        if sanctum.get("location"):
            rows.append(f"- Location: `{sanctum['location']}`")
        files = sanctum.get("files") or []
        if files:
            rows.append("- Files: " + ", ".join(f"`{f}`" for f in files))
        if sanctum.get("note"):
            rows.append(f"- Note: {sanctum['note']}")
        if rows:
            lines.append("## Sanctum (runtime memory)")
            lines.append("")
            lines.extend(rows)
            lines.append("")

    experience = data.get("experience")
    if isinstance(experience, dict):
        journeys = [
            j for j in experience.get("journeys") or [] if isinstance(j, dict)
        ]
        headless = experience.get("headless")
        if journeys or headless:
            lines.append("## Experience")
            lines.append("")
            for journey in journeys:
                steps = f" — {journey['steps']}" if journey.get("steps") else ""
                lines.append(f"- **{journey.get('name', '(unnamed journey)')}**{steps}")
            if headless:
                lines.append(f"- Headless: {headless}")
            lines.append("")

    lines.append("## Findings")
    lines.append("")
    if not findings:
        lines.append("No findings: the scanners returned a clean pass.")
        lines.append("")
    else:
        for sev in SEVERITIES:
            group = [
                f
                for f in findings
                if (f.get("severity") if f.get("severity") in SEVERITIES else "low")
                == sev
            ]
            if not group:
                continue
            lines.append(f"### {sev.capitalize()} ({len(group)})")
            lines.append("")
            for finding in group:
                lines.extend(_finding_lines(finding, "####"))

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inject findings JSON into the report shell and render HTML (+ optional markdown)."
    )
    parser.add_argument("findings", type=Path, help="path to findings.json")
    parser.add_argument(
        "--shell", type=Path, required=True, help="path to report-shell.html"
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="output HTML path"
    )
    parser.add_argument(
        "--md", type=Path, help="also write a markdown rendering to this path"
    )
    args = parser.parse_args()

    try:
        raw = args.findings.read_text(encoding="utf-8")
    except OSError as err:
        fail(f"cannot read {args.findings}: {err}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        fail(f"{args.findings} is not valid JSON: {err}")

    errors = validate(data)
    if errors:
        fail(
            f"{args.findings} failed shape validation:\n  - "
            + "\n  - ".join(errors)
        )

    try:
        shell_html = args.shell.read_text(encoding="utf-8")
    except OSError as err:
        fail(f"cannot read shell {args.shell}: {err}")

    atomic_write(args.output, inject(shell_html, data))
    if args.md:
        atomic_write(args.md, render_md(data))

    findings = [f for f in data.get("findings", []) if isinstance(f, dict)]
    print(
        json.dumps(
            {
                "html_report": str(args.output),
                "md_report": str(args.md) if args.md else None,
                "grade": data.get("grade"),
                "counts": severity_counts(findings),
                "findings": len(findings),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
