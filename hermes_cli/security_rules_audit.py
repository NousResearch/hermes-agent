"""Security-rule coverage audit — map natural-language rules to deterministic controls.

Based on arXiv:2608.23550 ("When 'Do Not' Is Not Deny: Security Rules in CLAUDE.md vs
Built-In Controls").

Scans memory entries, context files (AGENTS.md, CLAUDE.md, .cursorrules, SOUL.md),
and installed skills for imperative security rules ("never", "do not", "must not"),
and audits them against the deterministic enforcement layer:
1. HARDLINE unconditional blocks (tools.approval_detection.HARDLINE_PATTERNS)
2. Dangerous patterns / approval gates (tools.approval_detection.DANGEROUS_PATTERNS)
3. User-defined approvals.deny globs (tools.approval_floors._match_user_deny_rule)
4. Built-in sensitive write targets (_SENSITIVE_WRITE_TARGET)

Classifies rules into three actionable buckets:
- ENFORCED: active deterministic control exists (hardline, deny glob, or sensitive target).
- ENFORCEABLE: concrete deterministic command/path pattern can be added to approvals.deny.
- ADVISORY-ONLY: model-directed behavioral guidelines without a deterministic analogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import json
import logging
from pathlib import Path
import re
from typing import List, Optional, Sequence, Tuple

from hermes_cli.colors import Colors, color
from hermes_cli.config import get_hermes_home, get_project_root

logger = logging.getLogger(__name__)

# Imperative trigger pattern for negative security rules
_NEGATIVE_RULE_RE = re.compile(
    r"(?i)\b(?P<trigger>never|do\s+not|don't|must\s+not|shall\s+not|cannot|can't|should\s+not|shouldn't|strictly\s+forbidden|strictly\s+prohibited|prohibited\s+to|disallowed\s+to|forbidden\s+to)\b\s+(?P<action>[^\n.;`]+)",
)

_TRIGGER_STRIP_RE = re.compile(
    r"(?i)^(?:never|do\s+not|don't|must\s+not|shall\s+not|cannot|can't|should\s+not|shouldn't|strictly\s+forbidden\s+(?:to\s+)?|strictly\s+prohibited\s+(?:to\s+)?|prohibited\s+to\s+|disallowed\s+to\s+|forbidden\s+to\s+|disallow\s+|prohibit\s+)\s*",
)

# Common command pattern triggers for enforceable mapping
_COMMAND_MAPPINGS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"(?i)\bpush(?:\s+(?:directly|changes)?)?\s+to\s+(?:origin\s+)?(?:main|master|prod(?:uction)?)\b"), "git push *main*", "Git branch protection"),
    (re.compile(r"(?i)\bforce\s+push\b|\bpush\s+--force\b"), "git push *--force*", "Git force push"),
    (re.compile(r"(?i)\b(?:hard\s+reset|reset\s+--hard)\b"), "git reset --hard*", "Git hard reset"),
    (re.compile(r"(?i)\bterraform\s+apply\b"), "terraform apply*", "Terraform apply"),
    (re.compile(r"(?i)\bterraform\s+destroy\b"), "terraform destroy*", "Terraform destroy"),
    (re.compile(r"(?i)\bkubectl\s+delete\b"), "kubectl delete*", "Kubernetes resource deletion"),
    (re.compile(r"(?i)\bnpm\s+publish\b"), "npm publish*", "Package publish (npm)"),
    (re.compile(r"(?i)\bpip\s+install\s+--upgrade\b"), "pip install --upgrade*", "Unpinned package upgrade"),
    (re.compile(r"(?i)\b(?:docker|podman)\s+(?:rm|system\s+prune)\b"), "docker rm*", "Container deletion"),
    (re.compile(r"(?i)\bchmod\s+(?:-R\s+)?777\b"), "chmod *777*", "Insecure file permissions"),
    (re.compile(r"(?i)\b(?:curl|wget)\s+[^|\n]+\|\s*(?:ba)?sh\b"), "curl*|*sh*", "Piped remote execution"),
]

# Sensitive file keywords that map to file protection
_SENSITIVE_TARGET_KEYWORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\.env\b"), ".env / environment credentials"),
    (re.compile(r"(?i)\.ssh\b|\bid_rsa\b|\bid_ed25519\b"), "SSH private keys and configuration"),
    (re.compile(r"(?i)config\.yaml\b"), "Hermes security configuration"),
    (re.compile(r"(?i)\.(?:bashrc|zshrc|profile)\b"), "Shell configuration files"),
    (re.compile(r"(?i)\.(?:netrc|pgpass|npmrc|pypirc)\b"), "Credential store files"),
    (re.compile(r"(?i)/etc/|/private/etc/"), "System configuration files"),
]


@dataclass
class SecurityRule:
    """A single extracted natural-language security rule and its audit status."""

    source_file: str
    line_number: int
    raw_text: str
    imperative_phrase: str
    category: str  # "enforced" | "enforceable" | "advisory"
    enforcement_mechanism: Optional[str] = None
    suggested_deny_glob: Optional[str] = None
    suggested_command: Optional[str] = None


@dataclass
class SecurityRulesAuditReport:
    """Aggregated report of all audited security rules."""

    rules: list[SecurityRule] = field(default_factory=list)
    scanned_files: list[str] = field(default_factory=list)

    @property
    def enforced(self) -> list[SecurityRule]:
        return [r for r in self.rules if r.category == "enforced"]

    @property
    def enforceable(self) -> list[SecurityRule]:
        return [r for r in self.rules if r.category == "enforceable"]

    @property
    def advisory(self) -> list[SecurityRule]:
        return [r for r in self.rules if r.category == "advisory"]

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_rules": len(self.rules),
                "enforced_count": len(self.enforced),
                "enforceable_count": len(self.enforceable),
                "advisory_count": len(self.advisory),
                "scanned_files_count": len(self.scanned_files),
            },
            "scanned_files": self.scanned_files,
            "rules": [
                {
                    "source_file": r.source_file,
                    "line_number": r.line_number,
                    "raw_text": r.raw_text,
                    "category": r.category,
                    "enforcement_mechanism": r.enforcement_mechanism,
                    "suggested_deny_glob": r.suggested_deny_glob,
                    "suggested_command": r.suggested_command,
                }
                for r in self.rules
            ],
        }


def _discover_scan_files(hermes_home: Optional[Path] = None, cwd: Optional[Path] = None) -> list[Path]:
    """Find all candidate memory, context, and skill files to scan."""
    files: list[Path] = []
    seen: set[Path] = set()

    h_home = Path(hermes_home or get_hermes_home()).resolve()
    current_cwd = Path(cwd or Path.cwd()).resolve()

    # 1. Project context files
    context_filenames = [
        "AGENTS.md", "CLAUDE.md", ".hermes.md", ".cursorrules", "SOUL.md",
        "SECURITY.md", "CONVENTIONS.md", "agents.md", "claude.md",
    ]
    for search_dir in [current_cwd, h_home]:
        if search_dir.exists() and search_dir.is_dir():
            for fname in context_filenames:
                p = search_dir / fname
                if p.is_file() and p not in seen:
                    files.append(p)
                    seen.add(p)

    # 2. Subdirectory context files in current cwd (up to 2 levels)
    if current_cwd.exists() and current_cwd.is_dir():
        for pattern in ["*/AGENTS.md", "*/*/AGENTS.md", "*/CLAUDE.md", "*/*/CLAUDE.md"]:
            for p in current_cwd.glob(pattern):
                if p.is_file() and p not in seen:
                    files.append(p)
                    seen.add(p)

    # 3. Memories store
    memories_dir = h_home / "memories"
    if memories_dir.exists() and memories_dir.is_dir():
        for p in memories_dir.glob("*.md"):
            if p.is_file() and p not in seen:
                files.append(p)
                seen.add(p)

    # 4. Skills directory
    for skills_root in [h_home / "skills", current_cwd / "skills"]:
        if skills_root.exists() and skills_root.is_dir():
            for p in skills_root.glob("**/SKILL.md"):
                if p.is_file() and p not in seen:
                    files.append(p)
                    seen.add(p)
            for p in skills_root.glob("**/*.md"):
                if p.is_file() and p not in seen:
                    files.append(p)
                    seen.add(p)

    return sorted(files)


def _get_active_deny_patterns() -> list[str]:
    """Retrieve user configured approvals.deny globs from config."""
    try:
        from tools import approval_context as _ctx
        deny_patterns = _ctx._get_approval_config().get("deny") or []
        return [p.strip() for p in deny_patterns if isinstance(p, str) and p.strip()]
    except Exception:
        return []


def _extract_rules_from_file(path: Path) -> list[Tuple[int, str, str]]:
    """Extract line numbers, raw text lines, and imperative phrases from a file."""
    extracted: list[Tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.debug("Failed to read %s: %s", path, exc)
        return extracted

    in_code_block = False
    for line_idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not stripped or stripped.startswith("#"):
            continue

        match = _NEGATIVE_RULE_RE.search(stripped)
        if match:
            trigger = match.group("trigger")
            action = match.group("action").strip()
            # Clean markdown bullets, quotes, formatting
            cleaned_line = re.sub(r"^[-*+>]\s+", "", stripped)
            extracted.append((line_idx, cleaned_line, f"{trigger} {action}"))

    return extracted


def _classify_rule(raw_text: str, imperative_phrase: str, active_deny_globs: Sequence[str]) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Classify a rule into (category, enforcement_mechanism, suggested_deny_glob, suggested_command)."""
    text_lower = raw_text.lower()
    phrase_lower = imperative_phrase.lower()
    action_candidate = _TRIGGER_STRIP_RE.sub("", phrase_lower).strip()

    # 1. Check if matches active user approvals.deny globs
    for glob in active_deny_globs:
        glob_clean = glob.lower().strip()
        if fnmatch.fnmatchcase(phrase_lower, f"*{glob_clean}*") or fnmatch.fnmatchcase(text_lower, f"*{glob_clean}*") or fnmatch.fnmatchcase(action_candidate, f"*{glob_clean}*"):
            return "enforced", f"approvals.deny ('{glob}')", None, None

    # 2. Check HARDLINE blocklist
    try:
        from tools.approval_detection import detect_hardline_command
        is_hardline, hl_desc = detect_hardline_command(action_candidate)
        if not is_hardline:
            is_hardline, hl_desc = detect_hardline_command(text_lower)
        if is_hardline:
            return "enforced", f"HARDLINE floor ({hl_desc})", None, None
    except Exception:
        pass

    # 3. Check sensitive file write protections
    for pattern, desc in _SENSITIVE_TARGET_KEYWORDS:
        if pattern.search(phrase_lower) or pattern.search(text_lower) or pattern.search(action_candidate):
            if any(w in phrase_lower for w in ["touch", "edit", "modify", "write", "change", "overwrite", "delete", "leak", "commit", "expose"]):
                return "enforced", f"Built-in file/env protection ({desc})", None, None

    # 4. Check DANGEROUS_PATTERNS approval gates
    try:
        from tools.approval_detection import detect_dangerous_command
        is_dangerous, _, dg_desc = detect_dangerous_command(action_candidate)
        if not is_dangerous:
            is_dangerous, _, dg_desc = detect_dangerous_command(text_lower)
        if is_dangerous:
            return "enforced", f"DANGEROUS_PATTERNS gate ({dg_desc})", None, None
    except Exception:
        pass

    # 5. Check if it matches an enforceable deterministic command shape
    for cmd_re, suggested_glob, desc in _COMMAND_MAPPINGS:
        if cmd_re.search(action_candidate) or cmd_re.search(phrase_lower) or cmd_re.search(text_lower):
            suggested_cmd = f"hermes config set approvals.deny '{json.dumps(list(active_deny_globs) + [suggested_glob])}'"
            return "enforceable", None, suggested_glob, suggested_cmd

    # 6. Fallback: Advisory-only
    return "advisory", "Advisory model instruction only (no deterministic command analogue)", None, None


def audit_security_rules(hermes_home: Optional[Path] = None, cwd: Optional[Path] = None) -> SecurityRulesAuditReport:
    """Run security rules coverage audit across discovered context and memory files."""
    report = SecurityRulesAuditReport()
    files = _discover_scan_files(hermes_home=hermes_home, cwd=cwd)
    report.scanned_files = [str(p) for p in files]
    active_deny = _get_active_deny_patterns()

    for file_path in files:
        extracted = _extract_rules_from_file(file_path)
        for line_no, raw_text, phrase in extracted:
            cat, mech, glob, cmd = _classify_rule(raw_text, phrase, active_deny)
            rule = SecurityRule(
                source_file=str(file_path),
                line_number=line_no,
                raw_text=raw_text,
                imperative_phrase=phrase,
                category=cat,
                enforcement_mechanism=mech,
                suggested_deny_glob=glob,
                suggested_command=cmd,
            )
            report.rules.append(rule)

    return report


def run_security_rules_audit_cli(hermes_home: Optional[Path] = None, cwd: Optional[Path] = None) -> None:
    """Execute and render the full ``hermes doctor --security-rules`` audit report."""
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│       🛡️  Hermes Security-Rule Coverage Audit            │", Colors.CYAN))
    print(color("│         arXiv:2608.23550 'When Do Not Is Not Deny'      │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
    print()

    report = audit_security_rules(hermes_home=hermes_home, cwd=cwd)

    print(f"  Scanned {len(report.scanned_files)} files (context, memories, skills).")
    print(f"  Found {len(report.rules)} natural-language security rules:")
    print(f"    • {color(str(len(report.enforced)), Colors.GREEN, Colors.BOLD)} enforced by deterministic controls")
    print(f"    • {color(str(len(report.enforceable)), Colors.YELLOW, Colors.BOLD)} enforceable (can add to approvals.deny)")
    print(f"    • {color(str(len(report.advisory)), Colors.DIM)} advisory-only (guidelines for the model)")
    print()

    # 1. Enforced rules
    if report.enforced:
        print(color("◆ Enforced Rules (Backed by Built-In Controls)", Colors.GREEN, Colors.BOLD))
        for r in report.enforced:
            rel_file = Path(r.source_file).name
            print(f"  {color('✓', Colors.GREEN)} {color(rel_file + ':' + str(r.line_number), Colors.BOLD)}: \"{r.raw_text}\"")
            if r.enforcement_mechanism:
                print(f"    {color('↳ Enforced by:', Colors.DIM)} {color(r.enforcement_mechanism, Colors.GREEN)}")
        print()

    # 2. Enforceable rules
    if report.enforceable:
        print(color("◆ Enforceable Rules (Action Required: Add Deterministic Deny Glob)", Colors.YELLOW, Colors.BOLD))
        for r in report.enforceable:
            rel_file = Path(r.source_file).name
            print(f"  {color('⚠', Colors.YELLOW)} {color(rel_file + ':' + str(r.line_number), Colors.BOLD)}: \"{r.raw_text}\"")
            if r.suggested_deny_glob:
                print(f"    {color('↳ Suggested approvals.deny glob:', Colors.DIM)} {color(r.suggested_deny_glob, Colors.CYAN, Colors.BOLD)}")
            if r.suggested_command:
                print(f"    {color('↳ Run to enforce:', Colors.DIM)} {color(r.suggested_command, Colors.YELLOW)}")
        print()

    # 3. Advisory-only rules
    if report.advisory:
        print(color("◆ Advisory-Only Rules (Model Prompts Without Deterministic Analogue)", Colors.DIM, Colors.BOLD))
        for r in report.advisory:
            rel_file = Path(r.source_file).name
            print(f"  {color('ℹ', Colors.DIM)} {color(rel_file + ':' + str(r.line_number), Colors.DIM)}: \"{r.raw_text}\"")
        print()

    print(color("─" * 60, Colors.CYAN))
    if report.enforceable:
        print(color(f"  ⚡ {len(report.enforceable)} rule(s) can be converted to hard deterministic controls.", Colors.YELLOW, Colors.BOLD))
    else:
        print(color("  ✓ All actionable natural-language rules have deterministic controls.", Colors.GREEN, Colors.BOLD))
    print()
