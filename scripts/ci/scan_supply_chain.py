#!/usr/bin/env python3
"""Scan a PR's added lines for critical supply-chain indicators and unbounded PyPI deps.

Run by two steps of ``.github/workflows/repo-guards.yml`` on pull requests, with the PR's base
and head commits in ``_HERMES_PR_BASE_SHA`` and ``_HERMES_PR_HEAD_SHA``. Each subcommand appends ``found=true|false`` to
``$GITHUB_OUTPUT`` and, on a finding, writes the report that
``scripts/ci/review_status.py emit`` publishes:

    python3 scripts/ci/scan_supply_chain.py patterns            # $RUNNER_TEMP/supply-chain-findings.md
    python3 scripts/ci/scan_supply_chain.py dependency-bounds   # $RUNNER_TEMP/supply-chain-unbounded.txt

``patterns`` is a narrow, high-signal scanner: only critical indicators of supply chain attacks
(the litellm-style payloads). Low-signal heuristics were removed on purpose because they fired
on nearly every PR and trained reviewers to ignore the scanner; keep it ruthlessly narrow.
Both diff base...head, from the merge base, so only changes this PR introduces are judged, not
changes that landed on main after the PR branched off. An added line that the same diff also
removes from a file of the same name is a moved line and is not judged; see ``numbered_added_lines``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LOCKFILE_EXCLUDES = (":!uv.lock", ":!*.lock", ":!pnpm-lock.yaml", ":!yarn.lock")
MAX_MATCHES = 10


@dataclass(frozen=True)
class Indicator:
    heading: str
    explanation: str
    patterns: tuple[re.Pattern[str], ...]  # every pattern must match the added line


INDICATORS = (
    Indicator(
        "### 🚨 CRITICAL: base64 decode + exec/eval combo",
        "Base64-decoded strings passed directly to exec/eval — the signature of hidden "
        "credential-stealing payloads.",
        (re.compile(r"base64\.(b64decode|decodebytes|urlsafe_b64decode)", re.IGNORECASE),
         re.compile(r"exec\(|eval\(", re.IGNORECASE)),
    ),
    Indicator(
        "### 🚨 CRITICAL: subprocess with encoded/obfuscated command",
        "Subprocess calls whose command strings are base64- or hex-encoded are a strong "
        "indicator of payload execution.",
        (re.compile(r"subprocess\.(Popen|call|run)\s*\("),
         re.compile(r"base64|\\x[0-9a-f]{2}|chr\(", re.IGNORECASE)),
    ),
)

# A dep spec with ``>=`` and no ``,<`` ceiling (the pattern cannot match one); git+ URLs (commit
# SHAs) and comments never match.
# The workflow's ERE spelled the extras group ``\[[^\]]*\]``; in a POSIX bracket the backslash is
# literal, so that is one non-backslash character then ``]*``. Kept as is: an extras group matches
# only when it holds one character (``"a[b]>=2"``, not ``"pkg[extra]>=1"``).
UNBOUNDED_DEP = re.compile(r'"[a-zA-Z0-9_-]+(?:\[[^\\]\]*\])?>=[ 0-9.]+"')


def pr_diff(*pathspec: str) -> str:
    """Three-dot diff of the PR; a git failure yields an empty diff, as ``|| true`` did."""
    commits = f"{os.environ['_HERMES_PR_BASE_SHA']}...{os.environ['_HERMES_PR_HEAD_SHA']}"
    result = subprocess.run(("git", "diff", "--no-renames", commits, "--", *pathspec), stdout=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, timeout=300, check=False)
    return result.stdout.decode("utf-8", "surrogateescape")


def numbered_added_lines(diff: str) -> list[str]:
    """``grep -n '^+'`` over the diff, less the lines the PR only moved.

    The diff has no rename detection, so a moved file is all removed lines at its old path and all
    added lines at its new one. An added line is left out when the diff also removes a line with
    the same text from a file of the same name, once per removed line. A copy whose source keeps
    the line, a second copy of a moved file, and text taken from a file of another name are still
    judged.
    """
    lines = diff.split("\n")
    keys, name = [], ""
    for line in lines:
        if line.startswith("diff --git "):
            name = line.rsplit("/", 1)[-1].rstrip('"')
        keys.append((name, line[1:]))
    removed = Counter(key for key, line in zip(keys, lines) if line.startswith("-"))
    added = []
    for number, (key, line) in enumerate(zip(keys, lines), 1):
        if not line.startswith("+"):
            continue
        if removed[key]:
            removed[key] -= 1
        else:
            added.append(f"{number}:{line}")
    return added


def critical_findings(diff: str) -> str:
    added = numbered_added_lines(diff)
    findings = ""
    for indicator in INDICATORS:
        hits = [line for line in added if all(p.search(line) for p in indicator.patterns)]
        if hits:
            matches = "\n".join(hits[:MAX_MATCHES])
            findings += (f"\n{indicator.heading}\n{indicator.explanation}\n\n**Matches:**\n"
                         f"```\n{matches}\n```\n")
    return findings


def unbounded_dependencies(diff: str) -> list[str]:
    added = [line for line in diff.split("\n") if line.startswith("+") and not line.startswith("+++")]
    return [match for line in added for match in UNBOUNDED_DEP.findall(line)]


def _set_found(found: bool) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
        fh.write(f"found={'true' if found else 'false'}\n")


def _write_report(name: str, text: str) -> None:
    Path(os.environ["RUNNER_TEMP"], name).write_text(text + "\n", encoding="utf-8",
                                                     errors="surrogateescape")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan a PR diff for supply-chain risks.")
    parser.add_argument("scan", choices=("patterns", "dependency-bounds"))
    args = parser.parse_args(argv)
    if args.scan == "patterns":
        report = critical_findings(pr_diff(".", *LOCKFILE_EXCLUDES))
        report_name = "supply-chain-findings.md"
    else:
        report = "\n".join(unbounded_dependencies(pr_diff("pyproject.toml")))
        report_name = "supply-chain-unbounded.txt"
    _set_found(bool(report))
    if report:
        _write_report(report_name, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
