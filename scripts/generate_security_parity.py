#!/usr/bin/env python3
"""Generate the source-grounded Hermes security assurance matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "website/docs/developer-guide/security-assurance-matrix.md"


@dataclass(frozen=True)
class Guarantee:
    id: str
    claim: str
    status: str
    proofs: tuple[str, ...]
    limit: str


def catalog() -> tuple[Guarantee, ...]:
    """Return the reviewed assurance catalogue in stable display order."""
    return (
        Guarantee(
            "catastrophic-command-floor",
            "Catastrophic shell commands are blocked before approval or YOLO policy.",
            "proven",
            ("tests/tools/test_approval.py",),
            "This is command-shape enforcement, not an OS capability sandbox.",
        ),
        Guarantee(
            "dangerous-command-approval",
            "Dangerous terminal actions require the configured human or smart-approval decision.",
            "proven",
            (
                "tests/tools/test_approval.py",
                "tests/tools/test_approval_outcome_parity.py",
            ),
            "Only actions classified as dangerous enter this approval path.",
        ),
        Guarantee(
            "user-deny-floor",
            "User-defined command deny rules outrank approval and YOLO policy.",
            "proven",
            ("tests/tools/test_approval_deny_rules.py",),
            "Deny rules parse command text; renamed binaries and runtime-built commands need containment.",
        ),
        Guarantee(
            "file-write-protection",
            "File mutation tools enforce protected paths and configured safe roots.",
            "proven",
            ("tests/tools/test_file_write_safety.py",),
            "The guarantee covers Hermes file tools, not arbitrary writes by an allowed subprocess.",
        ),
        Guarantee(
            "execute-code-approval",
            "execute_code carries the calling session's approval identity and unattended policy.",
            "proven",
            ("tests/tools/test_execute_code_approval_cluster.py",),
            "A documented headless-local compatibility path remains present but not yet proven uniformly safe.",
        ),
        Guarantee(
            "cross-surface-approval-parity",
            "Equivalent approval outcomes are consistent across interactive and gateway surfaces.",
            "partial",
            (
                "tests/tools/test_approval_mode_parity.py",
                "tests/tools/test_approval_outcome_parity.py",
            ),
            "Not every tool family, backend, delegated child, cron path, and messaging surface is in one matrix.",
        ),
        Guarantee(
            "structured-security-reasons",
            "Security decisions expose a closed, machine-readable reason-code vocabulary.",
            "partial",
            ("tests/tools/test_approval_reason_codes.py",),
            "Approval denials, pending decisions, and hard floors are covered; other tool families still need migration.",
        ),
        Guarantee(
            "raise-only-policy-composition",
            "Independent security rules can raise a verdict but cannot weaken another rule.",
            "gap",
            (),
            "Hermes has hard floors, but no general PASS/AUTH/BLOCK reducer contract for all tools.",
        ),
        Guarantee(
            "cross-call-taint-correlation",
            "Sensitive values read in one call are correlated with later outbound calls.",
            "gap",
            (),
            "Redaction and credential isolation do not establish cross-call information-flow tracking.",
        ),
        Guarantee(
            "tool-output-secret-gate",
            "Credential-bearing tool output is withheld before it reaches the model.",
            "gap",
            (),
            "Output redaction is not yet catalogued as an execution-path blocking guarantee.",
        ),
        Guarantee(
            "raise-only-policy-change",
            "An agent cannot silently weaken previously accepted security policy.",
            "gap",
            (),
            "Hermes protects secret paths, but policy weakening is not represented by a pinned monotonic contract.",
        ),
    )


def _evidence(proofs: tuple[str, ...]) -> str:
    if not proofs:
        return "—"
    return "<br>".join(f"[`{Path(proof).name}`](../../../{proof})" for proof in proofs)


def render(guarantees: Iterable[Guarantee]) -> str:
    """Render a stable Markdown report from the catalogue."""
    lines = [
        "---",
        'title: "Security assurance matrix"',
        'description: "Test-linked security guarantees, partial coverage, and known gaps"',
        "---",
        "",
        "# Hermes security assurance matrix",
        "",
        "> This document is generated by `scripts/generate_security_parity.py`. Do not edit it by hand.",
        "",
        "This matrix distinguishes tested guarantees from partial coverage and explicit gaps. "
        "A linked test is evidence for the stated behavior only; it is not proof against every bypass.",
        "",
        "| Guarantee | Status | Evidence | Honest limit |",
        "|---|---|---|---|",
    ]
    for guarantee in guarantees:
        status = {"proven": "✅ proven", "partial": "◐ partial", "gap": "○ gap"}[
            guarantee.status
        ]
        lines.append(
            f'| <a id="{guarantee.id}"></a>{guarantee.claim} | {status} | '
            f"{_evidence(guarantee.proofs)} | {guarantee.limit} |"
        )
    lines.extend((
        "",
        "## Interpretation",
        "",
        "`partial` means the behavior is present but not yet proven uniformly across every relevant path. "
        "`gap` means no Hermes-wide enforcement contract was verified during this review.",
        "",
        "No security or compliance guarantee should be inferred beyond the exact behavior and limits listed here.",
        "",
    ))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail when the committed matrix is stale"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    expected = render(catalog())
    if args.check:
        if (
            not args.output.is_file()
            or args.output.read_text(encoding="utf-8") != expected
        ):
            print(f"stale security assurance matrix: {args.output}")
            return 1
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
