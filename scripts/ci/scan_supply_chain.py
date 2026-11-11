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
changes that landed on main after the PR branched off.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
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


def _diff_args(*pathspec: str) -> tuple[str, ...]:
    # A file moved to a new path while its old path keeps a forwarding file is a copy, not an
    # addition: --find-copies diffs it against its old self. -l0: a move of thousands of files
    # would otherwise skip copy detection.
    commits = f"{os.environ['_HERMES_PR_BASE_SHA']}...{os.environ['_HERMES_PR_HEAD_SHA']}"
    return ("--find-copies", "-l0", commits, "--", *pathspec)


def _git(*args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(("git", *args), input=stdin if stdin is not None else b"",
                          stdout=subprocess.PIPE, timeout=300, check=False).stdout


def pr_diff(*pathspec: str) -> str:
    """Three-dot diff of the PR; a git failure yields an empty diff, as ``|| true`` did."""
    return _git("diff", *_diff_args(*pathspec)).decode("utf-8", "surrogateescape")


def _read_at_head(paths: list[str]) -> dict[str, str]:
    """The text of each path in the PR head, from one ``git cat-file --batch``."""
    specs = "".join(f"{os.environ['_HERMES_PR_HEAD_SHA']}:{path}\0" for path in paths)
    out = _git("cat-file", "--batch", "-z", stdin=specs.encode("utf-8", "surrogateescape"))
    texts, pos = {}, 0
    for path in paths:
        end = out.index(b"\n", pos)
        header, pos = out[pos:end], end + 1
        if header.endswith(b" missing"):
            continue
        size = int(header.split()[2])
        texts[path] = out[pos:pos + size].decode("utf-8", "surrogateescape")
        pos += size + 1
    return texts


def copied_lines(*pathspec: str) -> list[str]:
    """Lines of each copied file that its source still holds in the PR head, as added lines.

    The copy's diff shows them as unchanged context, but the PR does add them in a new place. A
    line counts as moved, and is not judged, only when it left its source.
    """
    fields = _git("diff", "--name-status", "-z", *_diff_args(*pathspec)).decode(
        "utf-8", "surrogateescape").split("\0")
    pairs, i = [], 0
    while i + 1 < len(fields):
        status = fields[i][:1]
        if status == "C":
            pairs.append((fields[i + 1], fields[i + 2]))
        i += 3 if status in "RC" else 2
    texts = _read_at_head(sorted({path for pair in pairs for path in pair}))
    copied = []
    for source, copy in pairs:
        kept = set(texts.get(source, "").splitlines())
        copied += [f"{copy}:+{line}" for line in texts.get(copy, "").splitlines() if line in kept]
    return copied


def numbered_added_lines(diff: str) -> list[str]:
    """``grep -n '^+'`` over the diff."""
    return [f"{number}:{line}" for number, line in enumerate(diff.split("\n"), 1)
            if line.startswith("+")]


def critical_findings(diff: str, copied: list[str]) -> str:
    added = numbered_added_lines(diff) + copied
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
        pathspec = (".", *LOCKFILE_EXCLUDES)
        report = critical_findings(pr_diff(*pathspec), copied_lines(*pathspec))
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
