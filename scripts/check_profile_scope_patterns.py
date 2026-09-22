#!/usr/bin/env python3
"""Advisory lint: profile-scope hazard patterns on the lines a change adds.

One Hermes process may serve many profiles (multiplex gateway, Desktop/dashboard ``serve``), and
``os.environ`` / module globals hold only the LAUNCH profile's values. Every pattern in
``scripts/ci/profile_scope_patterns.json`` is a call-site shape that turned out to be
profile-sensitive at least once — a child env built from ``os.environ``, a raw ``os.getenv`` of a
platform credential, an RPC decorator that binds the home but not the secret scope, a bare PID
liveness check. The invariant itself is in the root ``AGENTS.md`` (§ Code Shape Rules).

Advisory by construction: it prints ``file:line  <id>/<class>  why`` for every hit and ALWAYS
exits 0, because most patterns have legitimate sites (a standalone ``hermes -p x`` process where
environ IS the profile). The reviewer reads each finding against its ``scope_hint``.

Usage:
    python scripts/check_profile_scope_patterns.py [--base origin/main] [--head HEAD]
    python scripts/check_profile_scope_patterns.py --files tools/bot_relay.py ...   # whole files
    python scripts/check_profile_scope_patterns.py --base origin/main --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ROOT / "scripts" / "ci" / "profile_scope_patterns.json"
SUFFIXES = (".py", ".ts", ".tsx")
SKIP_PREFIXES = ("tests/", "website/", "skills/", "optional-skills/", "evals/", "scripts/", ".worktrees/")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    pattern_id: str
    pattern_class: str
    why: str
    scope_hint: str
    text: str


def load_patterns(path: Path = PATTERNS) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for p in data["patterns"]:
        out.append({**p, "_rx": re.compile(p["pattern_regex"], re.M)})
    return out


def _lint_path(rel: str) -> bool:
    return rel.endswith(SUFFIXES) and not rel.startswith(SKIP_PREFIXES) and "/tests/" not in rel and "node_modules" not in rel


def scan_text(rel: str, text: str, patterns: list[dict], lines: set[int] | None = None) -> list[Finding]:
    """Findings for *text*; ``lines`` restricts to those 1-based line numbers (None = whole file).
    Multi-line patterns are anchored on the line where the match starts."""
    findings: list[Finding] = []
    src_lines = text.split("\n")
    for p in patterns:
        for m in p["_rx"].finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            if lines is not None and line_no not in lines:
                continue
            findings.append(Finding(rel, line_no, p["id"], p["class"], p["why"], p["scope_hint"],
                                    src_lines[line_no - 1].strip()[:160]))
    return findings


def _git(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.stdout


def added_lines_vs_base(base: str, head: str | None) -> dict[str, set[int]]:
    """``{path: {added line numbers in head}}`` for lint-able files changed between *base* and *head*
    (working tree when *head* is None)."""
    rev = [base, head] if head else [base]
    diff = _git("diff", "--no-color", "-U0", "--diff-filter=AM", *rev, "--")
    out: dict[str, set[int]] = {}
    current: str | None = None
    for raw in diff.split("\n"):
        if raw.startswith("+++ "):
            name = raw[4:]
            current = name[2:] if name.startswith("b/") else None
            if current is not None and not _lint_path(current):
                current = None
            continue
        if current is None:
            continue
        m = _HUNK_RE.match(raw)
        if m:
            start, count = int(m.group(1)), int(m.group(2) or "1")
            out.setdefault(current, set()).update(range(start, start + count))
    return out


def _read(rel: str, head: str | None) -> str | None:
    if head:
        cmd = ["git", "show", f"{head}:{rel}"]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return r.stdout if r.returncode == 0 else None
    path = ROOT / rel
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None


def run(base: str | None, head: str | None, files: list[str], patterns: list[dict]) -> list[Finding]:
    findings: list[Finding] = []
    if files:
        for f in files:
            path = Path(f)
            text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
            if text is None:
                continue
            resolved = path.resolve()
            rel = str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(path)
            findings.extend(scan_text(rel, text, patterns))
        return findings
    for rel, lines in sorted(added_lines_vs_base(base or "origin/main", head).items()):
        text = _read(rel, head)
        if text is not None:
            findings.extend(scan_text(rel, text, patterns, lines))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="base ref for the added-lines diff")
    ap.add_argument("--head", default=None, help="head ref (default: working tree)")
    ap.add_argument("--files", nargs="*", default=[], help="scan these whole files instead of a diff")
    ap.add_argument("--json", default=None, help="also write findings as JSON to this path")
    args = ap.parse_args(argv)

    patterns = load_patterns()
    findings = run(args.base, args.head, args.files, patterns)
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(f) for f in findings], indent=2) + "\n", encoding="utf-8")
    if not findings:
        print("profile-scope patterns: 0 findings")
        return 0
    print(f"profile-scope patterns: {len(findings)} finding(s) — ADVISORY, read each against its scope hint "
          f"(root AGENTS.md § Code Shape Rules; scripts/ci/profile_scope_patterns.json)")
    for f in findings:
        print(f"{f.path}:{f.line}  {f.pattern_id}/{f.pattern_class}  {f.why}")
        print(f"    | {f.text}")
        print(f"    hint: {f.scope_hint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
