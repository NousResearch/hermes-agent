#!/usr/bin/env python3
"""audit_skills.py — Audit Hermes skills for quality and structure compliance.

Usage:
    python3 audit_skills.py                          # Audit all skills in ~/.hermes/skills/
    python3 audit_skills.py --path /path/to/skill     # Audit a specific skill
    python3 audit_skills.py --json                    # Output as JSON
    python3 audit_skills.py --skills-dir /other/dir   # Custom skills directory

Exit codes: 0 = all pass, 1 = some issues found, 2 = error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
DEFAULT_SKILLS_DIR = HERMES_HOME / "skills"

REQUIRED_SECTIONS = [
    "When to Use",
    "Prerequisites",
    "How to Run",
    "Quick Reference",
    "Procedure",
    "Pitfalls",
    "Verification",
]

BANNED_TOOL_REFS = {
    "grep": "search_files",
    "rg": "search_files",
    "cat": "read_file",
    "head": "read_file",
    "tail": "read_file",
    "sed": "patch",
    "awk": "patch",
    "find": "search_files (with target='files')",
    "ls": "search_files (with target='files')",
    "echo > file": "write_file",
}

DESCRIPTION_MAX_LEN = 60
MIN_LINES = 30
MAX_LINES = 500
IDEAL_MIN = 50
IDEAL_MAX = 300


# ── Data ───────────────────────────────────────────────────────────────────

@dataclass
class Issue:
    severity: str  # "error", "warning", "info"
    message: str


@dataclass
class AuditResult:
    skill_path: str
    skill_name: str
    score: int = 0
    issues: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ── Checks ─────────────────────────────────────────────────────────────────

def check_frontmatter(content: str) -> list[Issue]:
    """Check YAML frontmatter for required fields."""
    issues = []
    if not content.startswith("---"):
        issues.append(Issue("error", "Missing YAML frontmatter (must start with ---)"))
        return issues

    end = content.find("---", 3)
    if end == -1:
        issues.append(Issue("error", "Unclosed frontmatter"))
        return issues

    fm = content[3:end]

    for field_name in ["name", "description", "version", "author"]:
        if f"{field_name}:" not in fm:
            issues.append(Issue("warning", f"Missing frontmatter field: {field_name}"))

    # Check description length
    desc_match = re.search(r'^description:\s*"?(.+?)"?\s*$', fm, re.MULTILINE)
    if desc_match:
        desc = desc_match.group(1).strip('"').strip("'")
        if len(desc) > DESCRIPTION_MAX_LEN:
            issues.append(Issue(
                "warning",
                f"Description too long ({len(desc)} chars, max {DESCRIPTION_MAX_LEN}): \"{desc[:50]}...\""
            ))
        if not desc.endswith("."):
            issues.append(Issue("info", "Description should end with a period"))

    return issues


def check_sections(content: str) -> list[Issue]:
    """Check for required sections in SKILL.md body."""
    issues = []
    # Strip frontmatter
    body = content
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            body = content[end + 3:]

    headings = re.findall(r'^##\s+(.+)$', body, re.MULTILINE)
    heading_set = {h.strip().lower() for h in headings}

    for section in REQUIRED_SECTIONS:
        if section.lower() not in heading_set:
            issues.append(Issue("warning", f"Missing section: {section}"))

    # Check Pitfalls is non-empty
    pitfalls_match = re.search(r'##\s+Pitfalls\s*\n(.*?)(?=##|\Z)', body, re.DOTALL | re.IGNORECASE)
    if pitfalls_match:
        pitfalls_content = pitfalls_match.group(1).strip()
        if len(pitfalls_content) < 20:
            issues.append(Issue("info", "Pitfalls section is very short — add known failure modes"))
    elif "pitfalls" in heading_set:
        pass  # exists but maybe empty
    else:
        pass  # already flagged as missing

    return issues


def check_tool_references(content: str) -> list[Issue]:
    """Check for banned tool references (grep, cat, sed, etc.)."""
    issues = []
    body = content
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            body = content[end + 3:]

    lines = body.split("\n")
    in_code_block = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue  # Skip code blocks — shell commands are fine there

        # Check prose for banned references
        lower = line.lower()
        for banned, replacement in BANNED_TOOL_REFS.items():
            # Match word boundary, avoid matching inside backticks
            pattern = rf'(?<!`)\\b{re.escape(banned)}\\b(?!`)'
            if re.search(pattern, lower):
                issues.append(Issue(
                    "warning",
                    f"Prose references `{banned}` — use `{replacement}` instead"
                ))

    return issues


def check_size(content: str) -> list[Issue]:
    """Check line count is within acceptable range."""
    issues = []
    lines = content.count("\n") + 1

    if lines < MIN_LINES:
        issues.append(Issue("warning", f"Too short ({lines} lines, minimum {MIN_LINES})"))
    elif lines > MAX_LINES:
        issues.append(Issue("warning", f"Too long ({lines} lines, maximum {MAX_LINES})"))
    elif lines < IDEAL_MIN:
        issues.append(Issue("info", f"Could be more detailed ({lines} lines, ideal {IDEAL_MIN}-{IDEAL_MAX})"))
    elif lines > IDEAL_MAX:
        issues.append(Issue("info", f"Consider splitting ({lines} lines, ideal {IDEAL_MIN}-{IDEAL_MAX})"))

    return issues


def check_scripts(skill_dir: Path, content: str) -> list[Issue]:
    """Check that referenced scripts exist."""
    issues = []
    # Find script references in the skill
    script_refs = re.findall(r'(?:scripts/|`scripts/)([\w._-]+)', content)
    for ref in script_refs:
        ref = ref.rstrip('`),.')
        script_path = skill_dir / "scripts" / ref
        if not script_path.exists():
            issues.append(Issue("warning", f"Referenced script not found: scripts/{ref}"))

    return issues


def check_references_dir(skill_dir: Path, content: str) -> list[Issue]:
    """Check that referenced files exist."""
    issues = []
    ref_refs = re.findall(r'(?:references/|`references/)([\w._/-]+)', content)
    for ref in ref_refs:
        ref = ref.rstrip('`),.')
        ref_path = skill_dir / "references" / ref
        if not ref_path.exists():
            issues.append(Issue("info", f"Referenced file not found: references/{ref}"))

    return issues


# ── Scoring ────────────────────────────────────────────────────────────────

def compute_score(issues: list[Issue]) -> int:
    """Compute a 0-100 score from issues."""
    score = 100
    for issue in issues:
        if issue.severity == "error":
            score -= 15
        elif issue.severity == "warning":
            score -= 5
        elif issue.severity == "info":
            score -= 1
    return max(0, min(100, score))


# ── Main ───────────────────────────────────────────────────────────────────

def audit_skill(skill_dir: Path) -> Optional[AuditResult]:
    """Audit a single skill directory."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None

    content = skill_md.read_text(encoding="utf-8")
    lines = content.count("\n") + 1

    issues = []
    issues.extend(check_frontmatter(content))
    issues.extend(check_sections(content))
    issues.extend(check_tool_references(content))
    issues.extend(check_size(content))
    issues.extend(check_scripts(skill_dir, content))
    issues.extend(check_references_dir(skill_dir, content))

    score = compute_score(issues)

    # Extract name from frontmatter
    name_match = re.search(r'^name:\s*(.+)$', content, re.MULTILINE)
    skill_name = name_match.group(1).strip() if name_match else skill_dir.name

    return AuditResult(
        skill_path=str(skill_dir),
        skill_name=skill_name,
        score=score,
        issues=[asdict(i) for i in issues],
        stats={
            "lines": lines,
            "has_scripts": (skill_dir / "scripts").exists(),
            "has_references": (skill_dir / "references").exists(),
        },
    )


def find_skill_dirs(base: Path) -> list[Path]:
    """Find all skill directories (containing SKILL.md) under base."""
    dirs = []
    for skill_md in base.rglob("SKILL.md"):
        dirs.append(skill_md.parent)
    return sorted(dirs)


def print_report(results: list[AuditResult], as_json: bool = False) -> int:
    """Print audit report. Returns exit code."""
    if as_json:
        print(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
        return 1 if any(r.score < 90 for r in results) else 0

    # Table output
    print(f"\n{'Skill':<35} {'Score':>6} {'Lines':>6} {'Issues':>7}  Status")
    print("─" * 75)

    for r in sorted(results, key=lambda x: x.score):
        errors = sum(1 for i in r.issues if i["severity"] == "error")
        warnings = sum(1 for i in r.issues if i["severity"] == "warning")
        infos = sum(1 for i in r.issues if i["severity"] == "info")
        issue_str = f"{errors}E {warnings}W {infos}I"

        if r.score >= 90:
            status = "✅"
        elif r.score >= 70:
            status = "🟡"
        elif r.score >= 50:
            status = "🟠"
        else:
            status = "🔴"

        print(f"{r.skill_name:<35} {r.score:>5} {r.stats.get('lines', 0):>6} {issue_str:>7}  {status}")

    # Detailed issues for non-passing skills
    failing = [r for r in results if r.score < 90]
    if failing:
        print(f"\n{'=' * 75}")
        print("Issues to fix:\n")
        for r in sorted(failing, key=lambda x: x.score):
            print(f"  {r.skill_name} (score {r.score}):")
            for issue in r.issues:
                icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[issue["severity"]]
                print(f"    {icon} {issue['message']}")
            print()

    # Summary
    total = len(results)
    passing = sum(1 for r in results if r.score >= 90)
    print(f"\n{passing}/{total} skills pass (score ≥ 90)")

    return 1 if failing else 0


def main():
    parser = argparse.ArgumentParser(description="Audit Hermes skills for quality")
    parser.add_argument("--path", help="Audit a specific skill directory")
    parser.add_argument("--skills-dir", help="Skills directory (default: ~/.hermes/skills/)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    skills_dir = Path(args.skills_dir) if args.skills_dir else DEFAULT_SKILLS_DIR

    if args.path:
        skill_dir = Path(args.path)
        if not skill_dir.exists():
            print(f"Error: {skill_dir} does not exist", file=sys.stderr)
            sys.exit(2)
        result = audit_skill(skill_dir)
        if result is None:
            print(f"Error: No SKILL.md found in {skill_dir}", file=sys.stderr)
            sys.exit(2)
        results = [result]
    else:
        if not skills_dir.exists():
            print(f"Error: Skills directory not found: {skills_dir}", file=sys.stderr)
            sys.exit(2)
        skill_dirs = find_skill_dirs(skills_dir)
        if not skill_dirs:
            print(f"No skills found in {skills_dir}", file=sys.stderr)
            sys.exit(2)
        results = []
        for d in skill_dirs:
            r = audit_skill(d)
            if r:
                results.append(r)

    exit_code = print_report(results, as_json=args.json)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
