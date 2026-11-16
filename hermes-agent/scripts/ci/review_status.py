#!/usr/bin/env python3
"""Publish a repository guard's review_status for the live PR comment.

A review_status is a JSON list of ``{source, results: [...]}`` objects. A guard step publishes it
as a ``review_status=<json>`` line in ``$GITHUB_OUTPUT`` and, for scripts/ci/live_comment.py, in
``$RUNNER_TEMP/review-status/<artifact>/review-status.json``, which repo-guards.yml uploads as
the ``review-status-<artifact>`` artifact.

    python3 scripts/ci/review_status.py emit <check>              # see FINDINGS
    python3 scripts/ci/review_status.py merge <artifact> ENV...   # concatenate env-held lists

``emit`` publishes one ``action_required`` result, carrying the check's report file as its
detail, when the check's flag variable is ``true``, and ``[]`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def status_line(status: list[dict], *, compact: bool = True, ascii_only: bool = True) -> str:
    """The ``review_status=`` line; ``compact=False`` keeps json.dumps' default ``", "`` spacing."""
    separators = (",", ":") if compact else None
    return f"review_status={json.dumps(status, separators=separators, ensure_ascii=ascii_only)}\n"


def publish(status: list[dict], artifact: str | None = None, *, output: bool = True,
            compact: bool = True, ascii_only: bool = True) -> None:
    """Append the status to ``$GITHUB_OUTPUT`` and/or write it as the artifact's file."""
    line = status_line(status, compact=compact, ascii_only=ascii_only)
    if output:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            fh.write(line)
    if artifact:
        directory = Path(os.environ["RUNNER_TEMP"], "review-status", artifact)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "review-status.json").write_text(line, encoding="utf-8")


def action_required(source: str, title: str, summary: str, how_to_fix: str,
                    detail: str | None = None) -> list[dict]:
    result = {"kind": "action_required", "title": title, "summary": summary}
    if detail is not None:
        result["detail"] = detail
    result["how_to_fix"] = how_to_fix
    return [{"source": source, "results": [result]}]


@dataclass(frozen=True)
class Finding:
    flag_env: str  # set to "true" by the step that produced the report
    report: str  # file under $RUNNER_TEMP; its text is the result's detail
    artifact: str | None
    source: str
    title: str
    summary: str
    how_to_fix: str
    compact: bool = False


FINDINGS = {
    "prompt-surface": Finding(
        "CHANGED", "prompt-surface.md", "prompt-surface", "prompt-surface",
        "Prompt surface changed",
        "This PR changes the system prompt or tool schemas every session sends.",
        "Read the rendered change below (full text under tests/fixtures/prompt_surface/ "
        "in the Files tab). Merging needs a hermes-agent-core approval."),
    "lockfile-diff": Finding(
        "CHANGED", "lockfile-diff.md", "lockfile-diff", "lockfile-diff",
        "JS lockfiles",
        "Locked JS dependency versions changed.",
        "Verify the version changes are expected. Lockfile changes need a hermes-agent-core review.",
        compact=True),
    # No artifact: the merge subcommand combines these two into the supply-chain artifact.
    "supply-chain-scan": Finding(
        "FOUND", "supply-chain-findings.md", None, "supply chain",
        "Critical supply chain risk",
        "Critical supply chain risk patterns detected in this PR.",
        "Review the flagged code carefully. If it is a false positive, narrow the pattern in "
        "scripts/ci/scan_supply_chain.py."),
    "dependency-bounds": Finding(
        "FOUND", "supply-chain-unbounded.txt", None, "supply chain",
        "Unbounded PyPI dependencies",
        "This PR adds PyPI dependencies without upper bounds.",
        'Add a `<next_major` upper bound, e.g. `"package>=1.2.0,<2"`. '
        "See CONTRIBUTING.md dependency pinning policy."),
}


def emit(check: str) -> None:
    finding = FINDINGS[check]
    status: list[dict] = []
    if os.environ.get(finding.flag_env, "") == "true":
        detail = Path(os.environ["RUNNER_TEMP"], finding.report).read_text(encoding="utf-8-sig")
        status = action_required(finding.source, finding.title, finding.summary,
                                 finding.how_to_fix, detail)
    publish(status, finding.artifact, compact=finding.compact)


def merge(artifact: str, env_names: list[str]) -> None:
    """Concatenate the review_status lists held in env vars into the artifact file only."""
    merged: list[dict] = []
    for name in env_names:
        raw = os.environ.get(name, "")
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            merged.extend(data)
    publish(merged, artifact, output=False, compact=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish a repository guard's review_status.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("emit").add_argument("check", choices=sorted(FINDINGS))
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("artifact")
    merge_parser.add_argument("env_names", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "emit":
        emit(args.check)
    else:
        merge(args.artifact, args.env_names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
