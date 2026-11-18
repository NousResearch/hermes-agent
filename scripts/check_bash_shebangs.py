#!/usr/bin/env python3
"""Require PATH-resolved Bash in scripts, generated scripts, and examples.

Nix and other non-FHS environments need not provide Bash at a fixed bin path.
Scan tracked text, including extensionless scripts and embedded shell payloads.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


FIXED_BASH = re.compile(rb"#![ \t]*/(?:usr/)?bin/bash\b")


def check(root: Path) -> list[str]:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True,
    ).stdout
    findings = []
    for name in tracked.split(b"\0"):
        if not name:
            continue
        relative = name.decode("utf-8", errors="surrogateescape")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        for number, line in enumerate(data.splitlines(), 1):
            if FIXED_BASH.search(line):
                findings.append(f"{relative}:{number}: use #!/usr/bin/env bash")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    findings = check(args.root)
    for finding in findings:
        print(finding)
    print(f"Bash shebang check: {len(findings)} violation(s)")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
