#!/usr/bin/env python3
"""Check an agent identity file (``SOUL.md``) for a well-formed proactivity contract.

The contract a proactive agent stands on is four tiers (silent self-maintenance,
record-but-don't-send, bounded self-heal, and may-send-only-on-a-rule) plus the operator's
rules file and observations-log paths, a quiet-hours window, a send-suppression window, and
the delivery silence marker. Prose in ``SOUL.md`` is model-followed, not enforced — this
checker is the mechanical half: it catches the contract that was templated and never filled
in, the tier copied twice, the tier that quietly lost its body, and the identity file grown
past the truncation cap.

Usage:
    # $HERMES_HOME/SOUL.md (or ~/.hermes/SOUL.md when HERMES_HOME is unset)
    python3 scripts/check_proactivity_contract.py

    # an explicit file, a profile home, or a directory holding SOUL.md
    python3 scripts/check_proactivity_contract.py ~/.hermes/profiles/work/SOUL.md
    python3 scripts/check_proactivity_contract.py ~/.hermes/profiles/work

    # several targets, machine-readable output, warnings treated as failure
    python3 scripts/check_proactivity_contract.py work/SOUL.md ops/SOUL.md --json --strict

Exit status:
    0 — contract intact (warnings do not fail unless ``--strict``)
    1 — at least one finding, or a warning with ``--strict``
    2 — usage error (no readable target resolved)

Findings print as ``<path>:<line>: <code>: <message>``; line 0 means file-level. Every code
this script emits is asserted by ``tests/skills/test_proactive_agent_souls_skill.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ERROR = "error"
WARNING = "warning"

# The contract lives under a second-level heading; its tiers are subsections of it.
SECTION_HEADING_RE = re.compile(r"^(#{2,4})\s*proactivity\b", re.IGNORECASE)
TIER_HEADING_RE = re.compile(r"^(#{2,6})\s*tier\s*(\d+)\b", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
FENCE_RE = re.compile(r"^\s*(?:```|~~~)")

QUIET_HOURS_RE = re.compile(r"quiet\s+hours?", re.IGNORECASE)
CLOCK_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
SUPPRESSION_RE = re.compile(r"suppress\w*", re.IGNORECASE)
DURATION_RE = re.compile(r"\b\d+\s*(?:m|min|mins|minute|minutes|h|hr|hrs|hour|hours)\b", re.IGNORECASE)
TRIGGER_RE = re.compile(r"\btriggers?\b", re.IGNORECASE)
SILENCE_MARKER = "[SILENT]"
OBSERVATIONS_RE = re.compile(r"observations?\b", re.IGNORECASE)
RULES_RE = re.compile(r"\brules?\b", re.IGNORECASE)
MESSAGE_QUALITY_RE = re.compile(r"message\s+quality", re.IGNORECASE)
FILE_PATH_RE = re.compile(r"[~$\w][\w.$~/-]*\.(?:md|ya?ml|json|toml|txt|ini|csv)\b")
PLACEHOLDER_RE = re.compile(r"\b(?:TODO|TBD|FIXME)\b|<[a-z][a-z0-9 _-]*>", re.IGNORECASE)

# Same context-file cap class as ``agent.context_file_max_chars`` (floor 20K when unset);
# past it ``SOUL.md`` is head+tail truncated and the contract silently loses its middle.
DEFAULT_MAX_CHARS = 20_000

TIER_NAMES = {
    1: "silent self-maintenance",
    2: "record but do not send",
    3: "bounded self-heal",
    4: "may send on a rule the operator owns",
}


@dataclass
class Finding:
    """One contract violation or advisory note at a line (0 = file level)."""

    code: str
    message: str
    line: int = 0
    severity: str = ERROR

    def render(self, path: Path) -> str:
        return f"{path}:{self.line}: {self.code}: {self.message}"


def _active_lines(text: str) -> list[tuple[int, str]]:
    """(line number, line) outside fenced code blocks.

    A fenced sample of the contract pasted into ``SOUL.md`` is documentation, not the
    contract itself: tiers inside it must not satisfy the structural checks.
    """
    active: list[tuple[int, str]] = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            active.append((number, line))
    return active


def _section_range(lines: list[tuple[int, str]]) -> tuple[int, int] | None:
    """First ``## Proactivity`` heading and the line where that section ends."""
    start = level = None
    for number, line in lines:
        match = SECTION_HEADING_RE.match(line)
        if match:
            start, level = number, len(match.group(1))
            break
    if start is None or level is None:
        return None
    end = lines[-1][0] if lines else start
    for number, line in lines:
        if number <= start:
            continue
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) <= level:
            end = number - 1
            break
    return start, end


def _tier_bodies(section_lines: list[tuple[int, str]]) -> list[tuple[int, int, list[str]]]:
    """(tier number, heading line, body lines) for every ``Tier N`` heading, in order."""
    tiers: list[tuple[int, int, list[str]]] = []
    for index, (number, line) in enumerate(section_lines):
        match = TIER_HEADING_RE.match(line)
        if not match:
            continue
        heading_level = len(match.group(1))
        body: list[str] = []
        for _, following in section_lines[index + 1:]:
            next_heading = HEADING_RE.match(following)
            if next_heading and len(next_heading.group(1)) <= heading_level:
                break
            body.append(following)
        tiers.append((int(match.group(2)), number, body))
    return tiers


def check_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[Finding]:
    """Findings for one identity file's contents (no filesystem access)."""
    if not text.strip():
        return [Finding("empty", "identity file is empty — no identity, no contract")]

    findings: list[Finding] = []
    lines = _active_lines(text)
    section = _section_range(lines)
    if section is None:
        findings.append(
            Finding("section-missing", "no '## Proactivity …' heading — the contract is not in this file")
        )
        scoped = lines
    else:
        scoped = [entry for entry in lines if section[0] <= entry[0] <= section[1]]

    tiers = _tier_bodies(scoped)
    seen: dict[int, int] = {}
    order: list[int] = []
    for tier, line, body in tiers:
        if tier in seen:
            findings.append(
                Finding("duplicate-tier", f"Tier {tier} is defined twice (first at line {seen[tier]})", line)
            )
            continue
        seen[tier] = line
        order.append(tier)
        if not any(entry.strip() for entry in body):
            findings.append(
                Finding("tier-empty", f"Tier {tier} ({TIER_NAMES.get(tier, 'unnamed')}) has no body", line)
            )
    for tier in sorted(TIER_NAMES):
        if tier not in seen:
            findings.append(
                Finding(
                    "tier-missing",
                    f"Tier {tier} ({TIER_NAMES[tier]}) heading is missing",
                )
            )
    if order != sorted(order):
        findings.append(
            Finding("tier-order", f"tiers appear as {order} — order them Tier 1, 2, 3", 0)
        )

    scope_text = "\n".join(line for _, line in scoped)
    if not TRIGGER_RE.search(scope_text):
        findings.append(
            Finding("triggers-missing", "no trigger register — name the events that may interrupt someone")
        )
    if SILENCE_MARKER not in scope_text:
        findings.append(
            Finding(
                "silence-marker-missing",
                f"no {SILENCE_MARKER} — the quiet path needs the marker cron and heartbeat suppress on",
            )
        )
    if not any(
        OBSERVATIONS_RE.search(line) and FILE_PATH_RE.search(line) for _, line in scoped
    ):
        findings.append(
            Finding(
                "observations-log-missing",
                "no observations log path — Tier 2 needs a file to record into (name it on a line "
                "that mentions observations)",
            )
        )
    if not any(RULES_RE.search(line) and FILE_PATH_RE.search(line) for _, line in scoped):
        findings.append(
            Finding(
                "rules-file-missing",
                "no operator rules-file path — name the file whose rules decide what may be sent",
            )
        )
    if not MESSAGE_QUALITY_RE.search(scope_text):
        findings.append(
            Finding(
                "message-quality-missing",
                "no message-quality bar for sent messages (title, why it matters, source)",
                severity=WARNING,
            )
        )

    quiet_lines = [line for _, line in scoped if QUIET_HOURS_RE.search(line)]
    if not quiet_lines:
        findings.append(
            Finding(
                "quiet-hours-missing",
                "no quiet-hours window — say when nothing is allowed to send",
                severity=WARNING,
            )
        )
    elif not any(len(CLOCK_TIME_RE.findall(line)) >= 2 for line in quiet_lines):
        findings.append(
            Finding(
                "quiet-hours-incomplete",
                "quiet hours without two clock times (e.g. 23:00–07:30)",
                severity=WARNING,
            )
        )

    suppression_lines = [line for _, line in scoped if SUPPRESSION_RE.search(line)]
    if not suppression_lines:
        findings.append(
            Finding(
                "suppression-missing",
                "no send-suppression window — one message must hold off the next",
                severity=WARNING,
            )
        )
    elif not any(DURATION_RE.search(line) for line in suppression_lines):
        findings.append(
            Finding(
                "suppression-incomplete",
                "suppression window without a duration (e.g. 60 minutes)",
                severity=WARNING,
            )
        )

    for number, line in lines:
        match = PLACEHOLDER_RE.search(line)
        if match:
            findings.append(
                Finding("placeholder", f"unfilled placeholder {match.group(0)!r} — the template is not a contract", number)
            )

    if len(text) > max_chars:
        findings.append(
            Finding(
                "soul-too-long",
                f"{len(text)} chars exceeds {max_chars} — head+tail truncation will drop the middle",
                severity=WARNING,
            )
        )
    return findings


def resolve_targets(raw_paths: list[str], env: Mapping[str, str] | None = None) -> list[Path]:
    """Identity files to check: the arguments, or ``$HERMES_HOME/SOUL.md`` when none given."""
    environ: Mapping[str, str] = os.environ if env is None else env
    if not raw_paths:
        home = environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        return [Path(os.path.expanduser(home)) / "SOUL.md"]
    targets: list[Path] = []
    for raw in raw_paths:
        path = Path(os.path.expanduser(raw))
        targets.append(path / "SOUL.md" if path.is_dir() else path)
    return targets


def check_path(path: Path, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[int, list[Finding]]:
    """Read *path* and return (char count, findings); a missing file is a finding, not a crash."""
    try:
        if not path.is_file():
            return 0, [Finding("file-missing", "not a file — nothing to check")]
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:  # unreadable (permissions, dangling symlink) is not a traceback
        return 0, [Finding("file-missing", f"unreadable: {exc}")]
    return len(text), check_text(text, max_chars=max_chars)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_proactivity_contract.py",
        description="Validate the proactivity contract in an agent identity file (SOUL.md).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="SOUL.md files or the directories holding them (default: $HERMES_HOME/SOUL.md)",
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON document instead of text lines")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"advisory size cap for one identity file (default: {DEFAULT_MAX_CHARS})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    targets = resolve_targets(args.paths)
    if not targets:
        print("check_proactivity_contract: no target resolved", file=sys.stderr)
        return 2

    results = []
    errors = warnings = 0
    for target in targets:
        chars, findings = check_path(target, max_chars=args.max_chars)
        errors += sum(1 for f in findings if f.severity == ERROR)
        warnings += sum(1 for f in findings if f.severity == WARNING)
        results.append({"path": str(target), "chars": chars, "findings": findings})

    failed = bool(errors) or (bool(warnings) and args.strict)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": not failed,
                    "strict": args.strict,
                    "errors": errors,
                    "warnings": warnings,
                    "files": [
                        {
                            "path": item["path"],
                            "chars": item["chars"],
                            "findings": [
                                {
                                    "code": f.code,
                                    "severity": f.severity,
                                    "line": f.line,
                                    "message": f.message,
                                }
                                for f in item["findings"]
                            ],
                        }
                        for item in results
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for item in results:
            if not item["findings"]:
                print(f"{item['path']}: OK — proactivity contract intact ({item['chars']} chars)")
                continue
            for finding in item["findings"]:
                print(finding.render(Path(item["path"])))
        print(f"summary: {errors} error(s), {warnings} warning(s) in {len(results)} file(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
