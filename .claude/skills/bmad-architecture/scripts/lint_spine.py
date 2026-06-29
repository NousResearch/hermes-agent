#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""lint-spine — the mechanical half of spine decision-integrity, done deterministically.

LLMs miscount IDs and miss literal placeholders; a grep does not. This linter owns the
checks a script does better than a prompt, and leaves the semantic half (is each Rule
actually enforceable? does the boundary make sense?) to the rubric walker.

It reads ARCHITECTURE-SPINE.md from a workspace and reports, as compact JSON on stdout:

  - placeholder    literal TBD / TODO / "similar to AD-n" / unfilled {template-token}
  - ad_id          duplicate or non-monotonic AD-n identifiers
  - ad_fields      an AD-n block missing Binds / Prevents / Rule
  - version_pin    a ## Stack table row with no version

Fenced code blocks are blanked (replaced with equal-count blank lines) before scanning, so
mermaid and source trees don't trip false positives AND reported line numbers still line up
with the real file. Reported lines are absolute file lines (frontmatter offset added). Exit
code is always 0 — findings travel in the JSON; the caller (Reviewer Gate / rubric walker)
decides what to do with them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SPINE = "ARCHITECTURE-SPINE.md"

AD_HEADING = re.compile(r"^#{2,4}\s*AD-(\d+)\b(.*)$", re.MULTILINE)
HEADING = re.compile(r"^#{1,6}\s", re.MULTILINE)
FENCE = re.compile(r"```.*?```", re.DOTALL)
PLACEHOLDER_WORD = re.compile(r"\b(TBD|TODO|FIXME|XXX)\b")
SIMILAR_TO = re.compile(r"similar to AD-\d+", re.IGNORECASE)
TEMPLATE_TOKEN = re.compile(r"\{[a-z_][a-z0-9_ /.-]*\}")


def split_frontmatter(text: str) -> tuple[str, str, int]:
    """Return (frontmatter, body, body_line_offset).

    Frontmatter is the content between the first two lines that are *exactly* `---`
    (line-exact, like memlog.split — a `---` inside a value or a body thematic break never
    truncates it). body_line_offset is the number of file lines before the body begins, so a
    body-relative line number plus the offset gives the absolute file line. Absent frontmatter
    → ('', text, 0)."""
    lines = text.split("\n")
    if lines and lines[0] == "---":
        for i in range(1, len(lines)):
            if lines[i] == "---":
                fm = "\n".join(lines[1:i])
                body = "\n".join(lines[i + 1:])
                return fm, body, i + 1
    return "", text, 0


def blank_fences(text: str) -> str:
    """Replace each fenced block with the same number of newlines, so scanning skips fenced
    content while every line number outside the fence stays put."""
    return FENCE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def find_placeholders(body: str, offset: int) -> list[dict]:
    findings: list[dict] = []
    scan = blank_fences(body)
    # (regex, label, severity) — TBD/TODO and dangling cross-refs are unambiguous; a bare
    # {template-token} can be legitimate brace prose, so it is flagged low ("possible") to keep
    # the mechanical pass near-zero false-positive rather than train reviewers to ignore it.
    for rx, label, severity in (
        (PLACEHOLDER_WORD, "placeholder marker", "high"),
        (SIMILAR_TO, "unresolved cross-reference", "high"),
        (TEMPLATE_TOKEN, "possible unfilled template token (verify)", "low"),
    ):
        for m in rx.finditer(scan):
            findings.append({
                "category": "placeholder",
                "severity": severity,
                "detail": f"{label}: {m.group(0)!r}",
                "location": f"{SPINE} (line {offset + line_of(scan, m.start())})",
            })
    return findings


def find_frontmatter_placeholders(frontmatter: str) -> list[dict]:
    """Catch unfilled tokens left in frontmatter (e.g. paradigm/scope/date) — part of the
    spine contract, but outside the body that find_placeholders scans."""
    findings: list[dict] = []
    for rx, label, severity in (
        (PLACEHOLDER_WORD, "placeholder marker", "high"),
        (TEMPLATE_TOKEN, "possible unfilled template token (verify)", "low"),
    ):
        for m in rx.finditer(frontmatter):
            findings.append({
                "category": "placeholder",
                "severity": severity,
                "detail": f"frontmatter {label}: {m.group(0)!r}",
                "location": f"{SPINE} frontmatter (line {1 + line_of(frontmatter, m.start())})",
            })
    return findings


def find_ad_issues(body: str, offset: int) -> list[dict]:
    findings: list[dict] = []
    scan = blank_fences(body)  # AD headings shown inside a code fence are not live ADs
    matches = list(AD_HEADING.finditer(scan))
    seen: dict[int, int] = {}
    prev: int | None = None
    for m in matches:
        num = int(m.group(1))
        file_line = offset + line_of(scan, m.start())
        loc = f"{SPINE} AD-{num} (line {file_line})"
        if num in seen:
            findings.append({
                "category": "ad_id",
                "severity": "high",
                "detail": f"AD-{num} id reused (also at line {seen[num]})",
                "location": loc,
            })
        else:
            seen[num] = file_line
        if prev is not None and num <= prev:
            findings.append({
                "category": "ad_id",
                "severity": "high",
                "detail": f"AD-{num} is non-monotonic (follows AD-{prev}); ids must ascend and never renumber",
                "location": loc,
            })
        prev = num if prev is None else max(prev, num)

        # block text = from this heading to the next heading of any level
        start = m.end()
        nxt = HEADING.search(scan, start)
        block = scan[start:nxt.start()] if nxt else scan[start:]
        low = block.lower()
        missing = [f for f in ("binds", "prevents", "rule") if f not in low]
        if missing:
            findings.append({
                "category": "ad_fields",
                "severity": "high",
                "detail": f"AD-{num} missing required field(s): {', '.join(missing)}",
                "location": loc,
            })
    return findings


def find_unpinned_stack(body: str, offset: int) -> list[dict]:
    """Flag a `## Stack` table row that names something but leaves its version blank or a
    placeholder. Pinning lives in the body table now, not frontmatter. A row whose name is
    still a `{token}` skeleton is left to the placeholder pass, not double-reported here.

    Fences are blanked first (like find_placeholders / find_ad_issues), so a pipe-row or
    heading inside a code block is never read as live Stack content. The heading match is
    `## Stack` with a word boundary, so a renamed heading (`## Stack & Versions`) still
    counts. Name and Version columns are located from the header row, so a reordered table
    pairs name to version correctly; both default to the canonical positions (0, 1)."""
    findings: list[dict] = []
    in_stack = False
    header_seen = False
    name_idx, ver_idx = 0, 1
    scan = blank_fences(body)
    for i, raw in enumerate(scan.splitlines()):
        if HEADING.match(raw):
            in_stack = re.match(r"^##\s+Stack\b", raw) is not None
            header_seen = False
            name_idx, ver_idx = 0, 1
            continue
        if not in_stack or not raw.lstrip().startswith("|"):
            continue
        if set(raw.strip()) <= set("|-: "):
            continue  # separator row
        cells = _table_cells(raw)
        if not header_seen:
            header_seen = True
            for j, c in enumerate(cells):
                if c.lower() == "name":
                    name_idx = j
                elif c.lower() == "version":
                    ver_idx = j
            continue
        name = cells[name_idx] if len(cells) > name_idx else ""
        version = cells[ver_idx] if len(cells) > ver_idx else ""
        if not name or TEMPLATE_TOKEN.search(name):
            continue
        if not version or TEMPLATE_TOKEN.search(version):
            findings.append({
                "category": "version_pin",
                "severity": "medium",
                "detail": f"Stack entry {name!r} has no version",
                "location": f"{SPINE} (line {offset + i + 1})",
            })
    return findings


def _table_cells(row: str) -> list[str]:
    """Split a markdown table row into trimmed cells, dropping the leading/trailing pipe."""
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def lint(text: str) -> dict:
    frontmatter, body, offset = split_frontmatter(text)
    findings: list[dict] = []
    findings += find_frontmatter_placeholders(frontmatter)
    findings += find_placeholders(body, offset)
    findings += find_ad_issues(body, offset)
    findings += find_unpinned_stack(body, offset)
    counts: dict[str, int] = {}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    return {
        "ok": len(findings) == 0,
        "spine": SPINE,
        "total_findings": len(findings),
        "by_severity": counts,
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lint an architecture spine for mechanical integrity.")
    ap.add_argument("--workspace", required=True, help="run folder containing ARCHITECTURE-SPINE.md")
    ap.add_argument("-o", "--output", help="write JSON here instead of stdout")
    args = ap.parse_args(argv)

    spine_path = Path(args.workspace) / SPINE
    if not spine_path.exists():
        result = {"ok": False, "error": f"{spine_path} not found", "findings": [], "total_findings": 0}
    else:
        try:
            text = spine_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            # honor the "exit code is always 0" contract: a read/decode failure travels in JSON
            result = {"ok": False, "error": f"could not read {spine_path}: {e}", "findings": [], "total_findings": 0}
        else:
            result = lint(text)

    out = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(out + "\n", encoding="utf-8")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
