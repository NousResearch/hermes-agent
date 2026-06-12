#!/usr/bin/env python3
"""High-signal GitHub Actions security drift check.

Fails on patterns that materially widen CI supply-chain blast radius:
- pull_request_target
- permissions: write-all
- unpinned external actions
- newly introduced id-token: write outside the explicit allowlist
- explicit install-script enabling

This is intentionally regex/text based to avoid runtime dependencies in CI.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
SHA_RE = re.compile(r"^[a-f0-9]{40}$")
USES_RE = re.compile(r"^\s*uses\s*:\s*([^\s#]+)", re.IGNORECASE)

# id-token is intentionally allowed only where GitHub Pages deployment needs OIDC.
ID_TOKEN_ALLOWLIST = {
    ".github/workflows/deploy-site.yml": 1,
    ".github/workflows/skills-index.yml": 1,
}

INSTALL_SCRIPT_ENABLE_PATTERNS = [
    re.compile(r"ignore-scripts\s*[=:]\s*false", re.IGNORECASE),
    re.compile(r"npm\s+config\s+set\s+ignore-scripts\s+false", re.IGNORECASE),
    re.compile(r"--ignore-scripts=false", re.IGNORECASE),
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_unpinned_action(ref: str) -> bool:
    # Local actions and docker images are outside this check.
    if ref.startswith("./") or ref.startswith("docker://"):
        return False
    if "@" not in ref:
        return True
    action, version = ref.rsplit("@", 1)
    if "/" not in action:
        return False
    return SHA_RE.fullmatch(version.strip('"\'')) is None


def main() -> int:
    findings: list[str] = []
    workflow_files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))

    for path in workflow_files:
        text = path.read_text(errors="ignore")
        r = rel(path)
        lines = text.splitlines()

        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            if re.search(r"\bpull_request_target\b", line):
                findings.append(f"{r}:{idx}: pull_request_target is not allowed without explicit security review")
            if re.search(r"permissions\s*:\s*write-all\b", line):
                findings.append(f"{r}:{idx}: permissions: write-all is not allowed")
            for pat in INSTALL_SCRIPT_ENABLE_PATTERNS:
                if pat.search(line):
                    findings.append(f"{r}:{idx}: install-script enabling detected: {stripped}")

            m = USES_RE.match(line)
            if m and is_unpinned_action(m.group(1).strip('"\'')):
                findings.append(f"{r}:{idx}: external action is not pinned to a full SHA: {m.group(1)}")

        id_token_count = len(re.findall(r"id-token\s*:\s*write", text, flags=re.IGNORECASE))
        allowed = ID_TOKEN_ALLOWLIST.get(r, 0)
        if id_token_count > allowed:
            findings.append(f"{r}: id-token: write count {id_token_count} exceeds allowlist {allowed}")
        if id_token_count and r not in ID_TOKEN_ALLOWLIST:
            findings.append(f"{r}: id-token: write is not allowlisted")

    if findings:
        print("Workflow security drift check failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print(f"Workflow security drift check passed ({len(workflow_files)} workflow files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
