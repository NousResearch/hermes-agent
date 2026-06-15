#!/usr/bin/env python3
"""
Architectural Guardrails — Enforce separation of concerns across hermes-agent.

Rules:
  A: agent/ files must NOT contain "import re" or "from re import"
     → Text processing belongs in utils/
  B: gateway/ or agent/ files must NOT contain hardcoded retry logic
     ("time.sleep" inside retry loops, raw "retry_count" comparisons)
     → Retry strategy belongs in conflict/policies/api_retry_policy.py

Exit codes: 0 = pass, 1 = violation found

Usage:
  python scripts/arch_guard.py [threshold]

  threshold  — Optional. Violation count above this fails the check (default: 69).
              Lower this number over time as technical debt is paid down.
              Example: python scripts/arch_guard.py 30
"""

from __future__ import annotations

import sys
import re
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
GATEWAY_DIR = PROJECT_ROOT / "gateway"
UTILS_DIR = PROJECT_ROOT / "utils"
CONFLICT_DIR = PROJECT_ROOT / "conflict"

RULES = [
    {
        "id": "A",
        "description": "agent/ files must NOT contain 'import re' or 'from re import'",
        "target_dirs": [AGENT_DIR],
        "file_pattern": "*.py",
        "pattern": re.compile(r'^\s*(import re|from re import)', re.MULTILINE),
        "message": "Rule A violated: 'import re' found in agent/ — text processing must go to utils/",
    },
    {
        "id": "A",
        "description": "agent/ files must NOT contain re.compile at module level (inline regex)",
        "target_dirs": [AGENT_DIR],
        "file_pattern": "*.py",
        "pattern": re.compile(r'^\s*re\.compile\s*\(', re.MULTILINE),
        "message": "Rule A violated: 're.compile(...)' found in agent/ — use utils/text_processor.py",
    },
    {
        "id": "B",
        "description": "gateway/ or agent/ files must NOT contain hardcoded 'time.sleep' in retry loops",
        "target_dirs": [GATEWAY_DIR, AGENT_DIR],
        "file_pattern": "*.py",
        # Match time.sleep with a numeric first argument (retry backoff, not general utility)
        "pattern": re.compile(r'time\.sleep\s*\(\s*[0-9]'),
        "message": "Rule B violated: 'time.sleep(<number>)' found — retry logic must use conflict/policies/api_retry_policy.py",
    },
    {
        "id": "B",
        "description": "agent/ files must NOT contain raw 'retry_count >= ' comparisons (inline retry logic)",
        "target_dirs": [AGENT_DIR],
        "file_pattern": "*.py",
        "pattern": re.compile(r'retry_count\s*>=\s*max_retries'),
        "message": "Rule B violated: 'retry_count >= max_retries' found in agent/ — use APIRetryPolicy from conflict/",
    },
    {
        "id": "B",
        "description": "agent/ files must NOT contain raw 'max_retries = ' assignment (hardcoded retry config)",
        "target_dirs": [AGENT_DIR],
        "file_pattern": "*.py",
        "pattern": re.compile(r'max_retries\s*=\s*(?!.*#.*override)'),
        "message": "Rule B violated: hardcoded 'max_retries =' found in agent/ — use APIRetryPolicy max_retries",
    },
]


# ─── Scanner ──────────────────────────────────────────────────────────────────

def scan_file(file_path: Path, rules: list[dict]) -> list[tuple[str, str, int]]:
    """
    Scan a single file against all applicable rules.

    Returns list of (rule_id, message, line_number) violations.
    """
    violations = []
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return violations

    lines = content.split('\n')
    for rule in rules:
        if file_path.parent not in rule["target_dirs"]:
            continue
        if not file_path.match(rule["file_pattern"]):
            continue
        for lineno, line in enumerate(lines, 1):
            if rule["pattern"].search(line):
                violations.append((rule["id"], rule["message"], lineno))
    return violations


def run_guardrails(threshold: int = 69) -> int:
    """
    Run all architectural guardrails.

    Returns 0 if all checks pass, 1 if any violation found.
    """
    all_violations = []

    for rule in RULES:
        for target_dir in rule["target_dirs"]:
            if not target_dir.exists():
                continue
            for file_path in target_dir.rglob(rule["file_pattern"]):
                if file_path.is_dir():
                    continue
                violations = scan_file(file_path, [rule])
                for rule_id, message, lineno in violations:
                    all_violations.append((str(file_path.relative_to(PROJECT_ROOT)), rule_id, message, lineno))

    violation_count = len(all_violations)
    if violation_count == 0:
        print("✅ All architectural guardrails passed.")
        print(f"   Scanned: agent/, gateway/")
        print(f"   Rules: A (no regex in agent/), B (no hardcoded retry in agent/gateway)")
        return 0

    # Sort and display
    all_violations.sort(key=lambda x: (x[0], x[3]))
    print("=" * 70)
    print(f"🚨 {violation_count} architectural violation(s) detected (baseline: {threshold})")
    print("=" * 70)
    for file_rel, rule_id, message, lineno in all_violations:
        print(f"  [{rule_id}] {file_rel}:{lineno}")
        print(f"         {message}")
    print()
    print(f"Total: {violation_count} violation(s) found.")
    print("=" * 70)

    if violation_count > threshold:
        print(f"❌ FAILED: {violation_count} > threshold {threshold}")
        print("   To allow this run, raise the threshold or pay down technical debt.")
        return 1
    else:
        print(f"⚠️  WARNING: {violation_count} violation(s) — within baseline (≤ {threshold})")
        print("   Target: reduce threshold over time as technical debt is paid down.")
        return 0


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    threshold = 69
    if len(sys.argv) > 1:
        try:
            threshold = int(sys.argv[1])
        except ValueError:
            print(f"Invalid threshold '{sys.argv[1]}' — using default 69")
    sys.exit(run_guardrails(threshold))