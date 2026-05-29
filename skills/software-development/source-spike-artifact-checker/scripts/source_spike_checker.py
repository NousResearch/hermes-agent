#!/usr/bin/env python3
"""Validate Spearhead source-spike closure artifacts.

Usage:
  source_spike_checker.py /path/to/artifact-dir [more dirs or closure-summary.md files]

The checker validates artifact hygiene only. It does not prove the source analysis is correct.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


VERDICT_RE = re.compile(
    r"^\s*(?:[-*]\s*)?Verdict\s*[:\-]?.*$|^\s*##\s*Verdict\b",
    re.IGNORECASE | re.MULTILINE,
)
URL_RE = re.compile(r"https?://|\bgithub\.com/|\bgit@github\.com:", re.IGNORECASE)
REVISION_RE = re.compile(
    r"\b(commit|revision|rev(?:ision)? inspected|tag|release|default branch|paper date|not available|n/a)\b|\b[0-9a-f]{12,40}\b",
    re.IGNORECASE,
)
RETRIEVAL_RE = re.compile(
    r"\b(retriev(?:al|ed)|git ls-remote|clone|shallow clone|github api|raw readme|browser|curl|download|inspected via|local file)\b",
    re.IGNORECASE,
)
LICENSE_RE = re.compile(r"\b(license|licence|MIT|Apache|GPL|BSD|public domain|mixed licenses?|unknown license)\b", re.IGNORECASE)
ACCESS_RE = re.compile(r"\b(public|no login|auth(?:entication)? required|login gate|paywall|private|access)\b", re.IGNORECASE)
CLOSE_READY_RE = re.compile(r"\bCLOSE_READY\s*[:=]?\s*(yes|no)\b", re.IGNORECASE)
ROUTING_RE = re.compile(
    r"\b(NO_HANDOFF|Specialist handoff|Specialist routing|Handoff\s*:|Routing\s*:|Route\s*:|specialist lane|Gond implementation|Notion/admin)\b",
    re.IGNORECASE,
)
DOWNSTREAM_RE = re.compile(
    r"^\s*(?:##\s*)?(?:Next(?: recommended)? action|Recommended next action|Downstream decision)\b|\b(Follow-up scope|create .*follow-up|create .*card|needs EMA|needs approval|No immediate implementation|can be closed|monitor\b|backlog\b)\b",
    re.IGNORECASE | re.MULTILINE,
)
NOTION_RE = re.compile(r"\b(Notion|no Notion writes|Notion update|bulk edit|profile writes|config/profile/Notion writes)\b", re.IGNORECASE)
SOURCE_SPIKE_RE = re.compile(r"\bsource-spike\.md\b", re.IGNORECASE)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ArtifactResult:
    target: Path
    closure_path: Path | None
    source_spike_path: Path | None
    checks: list[Check]
    access_error: str | None = None

    @property
    def passed(self) -> bool:
        return self.access_error is None and all(c.passed for c in self.checks)


def find_closure_path(target: Path) -> tuple[Path | None, Path | None, str | None]:
    if target.is_file():
        closure = target
        source_spike = target.with_name("source-spike.md")
        return closure, source_spike if source_spike.exists() else None, None
    if target.is_dir():
        closure = target / "closure-summary.md"
        source_spike = target / "source-spike.md"
        if not closure.exists():
            return None, source_spike if source_spike.exists() else None, "missing closure-summary.md"
        return closure, source_spike if source_spike.exists() else None, None
    return None, None, "target is neither a file nor a directory"


def present(pattern: re.Pattern[str], text: str) -> bool:
    return bool(pattern.search(text))


def line_for(pattern: re.Pattern[str], text: str) -> str:
    for line in text.splitlines():
        if pattern.search(line):
            return line.strip()[:220]
    return ""


def validate_text(text: str, source_spike_path: Path | None) -> list[Check]:
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append(Check(name, ok, detail))

    source_ref_ok = bool(source_spike_path) or present(SOURCE_SPIKE_RE, text)
    source_heading_re = re.compile(r"\bSource(?: inspected| resolved)?\s*:", re.IGNORECASE)
    source_identifier_ok = present(URL_RE, text) or present(source_heading_re, text)
    add(
        "source-spike artifact reference",
        source_ref_ok,
        "source-spike.md exists or is referenced" if source_ref_ok else "missing source-spike.md reference/file",
    )
    add(
        "provenance: source URL/identifier",
        source_identifier_ok,
        line_for(URL_RE, text) or line_for(source_heading_re, text) or "missing source URL/source identifier",
    )
    add(
        "provenance: retrieval method",
        present(RETRIEVAL_RE, text),
        line_for(RETRIEVAL_RE, text) or "missing retrieval method",
    )
    add(
        "provenance: revision/commit/rationale",
        present(REVISION_RE, text),
        line_for(REVISION_RE, text) or "missing immutable revision or explicit rationale",
    )
    add(
        "license/access: license",
        present(LICENSE_RE, text),
        line_for(LICENSE_RE, text) or "missing license conclusion",
    )
    add(
        "license/access: access",
        present(ACCESS_RE, text),
        line_for(ACCESS_RE, text) or "missing access status",
    )
    add(
        "verdict",
        present(VERDICT_RE, text),
        line_for(VERDICT_RE, text) or "missing explicit verdict",
    )
    add(
        "specialist routing",
        present(ROUTING_RE, text),
        line_for(ROUTING_RE, text) or "missing NO_HANDOFF or specialist routing",
    )
    add(
        "downstream decision",
        present(DOWNSTREAM_RE, text),
        line_for(DOWNSTREAM_RE, text) or "missing close/follow-up/approval/monitor/backlog decision",
    )
    close_match = CLOSE_READY_RE.search(text)
    add(
        "CLOSE_READY explicit",
        bool(close_match),
        close_match.group(0) if close_match else "missing CLOSE_READY: yes/no",
    )
    add(
        "Notion closure hygiene",
        present(NOTION_RE, text),
        line_for(NOTION_RE, text) or "missing Notion closure/no-write status",
    )
    return checks


def validate_target(target: Path) -> ArtifactResult:
    closure_path, source_spike_path, access_error = find_closure_path(target)
    if access_error:
        return ArtifactResult(target, closure_path, source_spike_path, [], access_error)
    assert closure_path is not None
    try:
        text = closure_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = closure_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ArtifactResult(target, closure_path, source_spike_path, [], f"cannot read closure summary: {exc}")
    return ArtifactResult(target, closure_path, source_spike_path, validate_text(text, source_spike_path))


def print_result(result: ArtifactResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"{status} {result.target}")
    if result.closure_path:
        print(f"  closure: {result.closure_path}")
    if result.source_spike_path:
        print(f"  source_spike: {result.source_spike_path}")
    if result.access_error:
        print(f"  ERROR: {result.access_error}")
        return
    for check in result.checks:
        mark = "OK" if check.passed else "MISS"
        print(f"  [{mark}] {check.name}: {check.detail}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate Spearhead source-spike closure artifacts.")
    parser.add_argument("targets", nargs="+", help="artifact directories or closure-summary.md paths")
    args = parser.parse_args(argv)

    results = [validate_target(Path(t).expanduser().resolve()) for t in args.targets]
    for idx, result in enumerate(results):
        if idx:
            print()
        print_result(result)

    if any(r.access_error for r in results):
        return 2
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
