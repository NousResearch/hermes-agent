#!/usr/bin/env python3
"""Scan git diffs and owner-readable files for credential-shaped secrets.

Never prints the candidate value — only input label, line number, and detector.
Exit 1 on findings; exit 2 on malformed arguments/input.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

FIXTURE_PREFIXES = (
    "fixture-",
    "test-",
    "<secret>",
    "fixture-oauth-",
)

ANTHROPIC_PREFIX = re.compile(
    r"(?i)sk-ant-[A-Za-z0-9._~+/=-]{16,}"
)
JWT = re.compile(
    r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{16,}\b"
)
NAMED_SECRET = re.compile(
    r"(?i)\b(access_token|refresh_token|authorization|api_key|api-key)\b"
    r".{0,32}?(?:Bearer\s+)?([A-Za-z0-9._~+/=-]{24,})"
)
GENERIC_CANDIDATE = re.compile(r"[A-Za-z0-9._~+/=-]{40,4096}")
HEX_HASH = re.compile(r"^[0-9a-fA-F]{40,64}$")


def _shannon(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter

    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _classes(s: str) -> int:
    return sum(
        [
            any(c.islower() for c in s),
            any(c.isupper() for c in s),
            any(c.isdigit() for c in s),
            any(not c.isalnum() for c in s),
        ]
    )


def _allowlisted(value: str, *, under_tests: bool) -> bool:
    v = value.strip()
    if under_tests:
        return any(v.startswith(p) or p in v for p in FIXTURE_PREFIXES)
    return any(v.startswith(p) for p in FIXTURE_PREFIXES)


def detect_line(line: str, *, under_tests: bool, capture_output: bool) -> List[str]:
    findings: List[str] = []

    def consider(detector: str, candidate: str) -> None:
        if capture_output or under_tests:
            if _allowlisted(candidate, under_tests=under_tests):
                return
            # Whole-line fixture hint (split concatenations / keyword args).
            if under_tests and "fixture-" in line:
                return
        findings.append(detector)

    for m in ANTHROPIC_PREFIX.finditer(line):
        consider("anthropic_prefix", m.group(0))
    for m in JWT.finditer(line):
        consider("jwt", m.group(0))
    for m in NAMED_SECRET.finditer(line):
        consider("named_secret", m.group(2))
    for m in GENERIC_CANDIDATE.finditer(line):
        cand = m.group(0)
        if HEX_HASH.match(cand):
            continue
        if _classes(cand) < 3:
            continue
        if _shannon(cand) < 4.2:
            continue
        consider("generic_entropy", cand)
    return findings


def scan_text(
    label: str,
    text: str,
    *,
    under_tests: bool = False,
    capture_output: bool = False,
    added_lines_only: bool = False,
) -> List[Tuple[str, int, str]]:
    out: List[Tuple[str, int, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        content = line
        if added_lines_only:
            if not line.startswith("+") or line.startswith("+++"):
                continue
            content = line[1:]
        for det in detect_line(
            content, under_tests=under_tests, capture_output=capture_output
        ):
            out.append((label, i, det))
    return out


def _read_safe_file(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"refusing symlink: {path}")
    if not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"not a regular file: {path}")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        # Still allow if owner-readable by us via group? Spec: owner-readable regular.
        if not os.access(path, os.R_OK):
            raise ValueError(f"not readable: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def scan_git_diff(range_spec: str) -> List[Tuple[str, int, str]]:
    try:
        proc = subprocess.run(
            ["git", "diff", "-U0", range_spec],
            check=False,
            capture_output=True,
        )
    except Exception as exc:
        print(f"error: git diff failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if proc.returncode not in (0, 1):
        print("error: git diff failed", file=sys.stderr)
        raise SystemExit(2)
    data = proc.stdout.decode("utf-8", errors="replace")
    findings: List[Tuple[str, int, str]] = []
    current_file = "diff"
    under_tests = False
    buf_lines: List[str] = []
    file_start_line = 1

    def flush():
        nonlocal buf_lines
        if not buf_lines:
            return
        text = "\n".join(buf_lines)
        findings.extend(
            scan_text(
                current_file,
                text,
                under_tests=under_tests,
                added_lines_only=True,
            )
        )
        buf_lines = []

    for line in data.splitlines():
        if line.startswith("diff --git "):
            flush()
            # diff --git a/path b/path
            parts = line.split(" b/", 1)
            current_file = parts[1] if len(parts) == 2 else "diff"
            under_tests = current_file.startswith("tests/") or "/tests/" in current_file
            continue
        buf_lines.append(line)
    flush()
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-diff", action="append", default=[], dest="git_diffs")
    parser.add_argument("--input", action="append", default=[], dest="inputs")
    args = parser.parse_args(argv)
    if not args.git_diffs and not args.inputs:
        print("error: provide --git-diff and/or --input", file=sys.stderr)
        return 2

    findings: List[Tuple[str, int, str]] = []
    try:
        for gd in args.git_diffs:
            findings.extend(scan_git_diff(gd))
        for inp in args.inputs:
            path = Path(inp)
            text = _read_safe_file(path)
            findings.extend(
                scan_text(str(path), text, capture_output=True, under_tests=False)
            )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for label, lineno, det in findings:
        print(f"{label}:{lineno}: {det}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
