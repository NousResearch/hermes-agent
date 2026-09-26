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
- ENFORCED: an explicit terminal command is denied by the active runtime policy.
- ENFORCEABLE: a deny glob can cover an identified terminal example, not all prose meanings.
- ADVISORY-ONLY: model-directed behavioral guidelines without a deterministic analogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import json
import logging
import os
from pathlib import Path
import re
import shlex
from typing import List, Optional, Sequence, Tuple

from hermes_cli.colors import Colors, color
from hermes_cli.config import apply_terminal_config_to_env, get_hermes_home

logger = logging.getLogger(__name__)

# Imperative trigger pattern for negative security rules
_NEGATIVE_RULE_RE = re.compile(
    r"(?i)\b(?P<trigger>never|do\s+not|don't|must\s+not|shall\s+not|cannot|can't|should\s+not|shouldn't|strictly\s+forbidden|strictly\s+prohibited|prohibited\s+to|disallowed\s+to|forbidden\s+to)\b\s+(?P<action>[^\n]+)",
)

_TRIGGER_STRIP_RE = re.compile(
    r"(?i)^(?:never|do\s+not|don't|must\s+not|shall\s+not|cannot|can't|should\s+not|shouldn't|strictly\s+forbidden\s+(?:to\s+)?|strictly\s+prohibited\s+(?:to\s+)?|prohibited\s+to\s+|disallowed\s+to\s+|forbidden\s+to\s+|disallow\s+|prohibit\s+)\s*",
)

# These examples propose command-level controls; they do not prove coverage of prose.
_COMMAND_MAPPINGS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"(?i)\bpush(?:\s+(?:directly|changes)?)?\s+to\s+(?:origin\s+)?(?P<branch>main|master|prod(?:uction)?)\b"), "git push *{branch}*", "git push origin {branch}"),
    (re.compile(r"(?i)\bforce\s+push\b|\bpush\s+--force\b"), "git push *--force*", "git push --force origin audit-example"),
    (re.compile(r"(?i)\b(?:hard\s+reset|reset\s+--hard)\b"), "git reset --hard*", "git reset --hard"),
    (re.compile(r"(?i)\bterraform\s+apply\b"), "terraform apply*", "terraform apply"),
    (re.compile(r"(?i)\bterraform\s+destroy\b"), "terraform destroy*", "terraform destroy"),
    (re.compile(r"(?i)\bkubectl\s+delete\b"), "kubectl delete*", "kubectl delete deployment audit-example"),
    (re.compile(r"(?i)\bnpm\s+publish\b"), "npm publish*", "npm publish"),
    (re.compile(r"(?i)\bpip\s+install\s+--upgrade\b"), "pip install --upgrade*", "pip install --upgrade audit-example"),
    (re.compile(r"(?i)\b(?P<engine>docker|podman)\s+(?P<operation>rm|system\s+prune)\b"), "{engine} {operation}*", "{engine} {operation}"),
    (re.compile(r"(?i)\bchmod\s+(?:-R\s+)?777\b"), "chmod *777*", "chmod 777 audit-example"),
    (re.compile(r"(?i)\b(?P<fetcher>curl|wget)\s+[^|\n]+\|\s*(?:ba)?sh\b"), "{fetcher}*|*sh*", "{fetcher} https://example.invalid/script | sh"),
]

# A file-write guard says nothing about reads or committing an existing file.
_FILE_OPERATION_RE = re.compile(
    r"(?i)^(?P<operation>read|write|commit)\s+(?P<path>\S+)$"
)
_FILE_COMMANDS = {
    "read": "cat {path}",
    "write": "printf '' > {path}",
    "commit": "git add {path} && git commit -m audit-example",
}


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
    from hermes_cli.approvals_test import evaluate_command

    action = _TRIGGER_STRIP_RE.sub("", imperative_phrase).strip().rstrip(".")
    explicit = re.fullmatch(r"(?:run\s+|execute\s+)?`([^`]+)`", action, re.IGNORECASE)
    # Do not reinterpret the profile's globs or natural-language words as runtime policy.
    # Resolving into a copy also leaves the caller's environment unchanged.
    env_type = apply_terminal_config_to_env(env=dict(os.environ)).get("TERMINAL_ENV", "local")
    command = explicit.group(1) if explicit else None
    suggested_glob = None
    for command_re, glob_template, command_template in _COMMAND_MAPPINGS:
        match = command_re.search(action)
        if match:
            values = match.groupdict()
            suggested_glob = glob_template.format(**values)
            command = command or command_template.format(**values)
            break
    if command is None:
        file_operation = _FILE_OPERATION_RE.fullmatch(action)
        if file_operation:
            command = _FILE_COMMANDS[file_operation.group("operation").lower()].format(
                path=shlex.quote(file_operation.group("path")),
            )

    if command is None:
        return ("advisory", f"Advisory instruction: active profile {get_hermes_home()}; "
                "no explicit terminal command; coverage is unverified.", None, None)

    verdict = evaluate_command(command, env_type=env_type)
    denied = verdict["verdict"] in {"hardline-deny", "user-deny"}
    description = {
        "hardline-deny": "hard denial",
        "user-deny": "hard denial",
        "ask-approval": "conditional approval, not a denial",
        "allow": "allowed by command policy",
    }[verdict["verdict"]]
    mechanism = (
        f"Active profile {get_hermes_home()}: terminal example {command!r} "
        f"on {env_type}: {verdict['verdict']} "
        f"({description}); {verdict['detail']}"
    )
    if explicit and denied:
        return "enforced", mechanism + ". This command spelling only; other tools are not audited.", None, None

    mechanism += ". Whole instruction coverage is unverified."
    if suggested_glob and not denied:
        # The suggested glob must match its canonical example verbatim. Runtime
        # normalization and configured-policy evaluation remain owned by evaluate_command.
        if fnmatch.fnmatchcase(command.lower().strip(), suggested_glob.lower()):
            suggested_cmd = "hermes config set approvals.deny " + shlex.quote(
                json.dumps(list(active_deny_globs) + [suggested_glob])
            )
            return "enforceable", mechanism, suggested_glob, suggested_cmd
    return "advisory", mechanism, None, None


def audit_security_rules(hermes_home: Optional[Path] = None, cwd: Optional[Path] = None) -> SecurityRulesAuditReport:
    """Scan the supplied locations against the active profile's command policy.

    ``hermes_home`` selects files to scan; it does not switch the active profile.
    The shared config loader may initialize standard directories in that profile.
    """
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

    print(f"  Policy: active profile {get_hermes_home()}.")
    print(f"  Scanned {len(report.scanned_files)} files (context, memories, skills).")
    print(f"  Found {len(report.rules)} natural-language security rules:")
    print(f"    • {color(str(len(report.enforced)), Colors.GREEN, Colors.BOLD)} explicit terminal commands denied by active policy")
    print(f"    • {color(str(len(report.enforceable)), Colors.YELLOW, Colors.BOLD)} terminal examples with deny suggestions (review coverage)")
    print(f"    • {color(str(len(report.advisory)), Colors.DIM)} advisory or unverified instructions")
    print()

    # 1. Enforced rules
    if report.enforced:
        print(color("◆ Enforced Commands (Exact Terminal Spelling Only)", Colors.GREEN, Colors.BOLD))
        for r in report.enforced:
            rel_file = Path(r.source_file).name
            print(f"  {color('✓', Colors.GREEN)} {color(rel_file + ':' + str(r.line_number), Colors.BOLD)}: \"{r.raw_text}\"")
            if r.enforcement_mechanism:
                print(f"    {color('↳ Enforced by:', Colors.DIM)} {color(r.enforcement_mechanism, Colors.GREEN)}")
        print()

    # 2. Enforceable rules
    if report.enforceable:
        print(color("◆ Suggested Command Denials (Partial Coverage; Review Before Applying)", Colors.YELLOW, Colors.BOLD))
        for r in report.enforceable:
            rel_file = Path(r.source_file).name
            print(f"  {color('⚠', Colors.YELLOW)} {color(rel_file + ':' + str(r.line_number), Colors.BOLD)}: \"{r.raw_text}\"")
            if r.enforcement_mechanism:
                print(f"    {r.enforcement_mechanism}")
            if r.suggested_deny_glob:
                print(f"    {color('↳ Suggested approvals.deny glob:', Colors.DIM)} {color(r.suggested_deny_glob, Colors.CYAN, Colors.BOLD)}")
            if r.suggested_command:
                print(f"    {color('↳ Add this command-pattern denial:', Colors.DIM)} {color(r.suggested_command, Colors.YELLOW)}")
        print()

    # 3. Advisory-only rules
    if report.advisory:
        print(color("◆ Advisory or Unverified Instructions", Colors.DIM, Colors.BOLD))
        for r in report.advisory:
            rel_file = Path(r.source_file).name
            print(f"  {color('ℹ', Colors.DIM)} {color(rel_file + ':' + str(r.line_number), Colors.DIM)}: \"{r.raw_text}\"")
            if r.enforcement_mechanism:
                print(f"    {r.enforcement_mechanism}")
        print()

    print(color("─" * 60, Colors.CYAN))
    if report.enforceable:
        print(color(f"  ⚡ {len(report.enforceable)} terminal example(s) have candidate deny globs; whole instructions remain unverified.", Colors.YELLOW, Colors.BOLD))
    else:
        print(color("  No additional deny suggestions. This does not prove all instructions are enforced.", Colors.GREEN, Colors.BOLD))
    print()
