#!/usr/bin/env python3
"""
Deterministic audit script for 'argus' → 'agathos' reference checks.

Scans the agathos codebase for any remaining references to 'argus' that
may need updating to 'agathos'. Does NOT auto-fix - only reports for review.

Usage:
    python argus_audit.py              # Full report
    python argus_audit.py --strict   # Exit error if issues found
    python argus_audit.py --json     # Machine-readable output
"""

import ast
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple


@dataclass
class Finding:
    file: str
    line: int
    column: int
    context: str
    category: str  # 'import', 'function_name', 'string_literal', 'path', 'comment', 'docstring', 'variable'
    severity: str  # 'critical', 'warning', 'info'
    suggestion: str


class ArgusAuditor:
    """Audits agathos codebase for stale 'argus' references."""

    # Files to skip (tests that intentionally check for stale refs, etc.)
    SKIP_FILES = {
        'argus_audit.py',  # This script
        'test_environment_portability.py',  # Intentionally checks for stale refs
    }

    # Safe patterns that are OK (brand names, CLI commands users type, etc.)
    SAFE_PATTERNS = [
        r'ARGUS',  # Product name in docs/comments is OK
        r'com\.hermes\.agathos',  # Already correct
        r'agathos',  # Already correct
        r'argus-watcher',  # Internal label constant
        r'_ARGUS_',  # Internal constants
    ]

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.findings: List[Finding] = []
        self.files_checked = 0
        self.lines_checked = 0

    def should_skip_file(self, path: Path) -> bool:
        """Check if file should be skipped."""
        if path.name in self.SKIP_FILES:
            return True
        if path.suffix != '.py':
            return True
        # Skip __pycache__, .git, etc.
        for part in path.parts:
            if part.startswith('.') or part == '__pycache__':
                return True
        return False

    def is_safe_pattern(self, text: str) -> bool:
        """Check if text matches a safe pattern."""
        for pattern in self.SAFE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def categorize_and_score(self, match_text: str, line_text: str, file_path: Path) -> Tuple[str, str, str]:
        """
        Categorize the finding and determine severity.
        Returns: (category, severity, suggestion)
        """
        # Critical: Import statements - will cause ImportError
        if re.search(r'from\s+argus\s+import|import\s+argus', line_text):
            return (
                'import',
                'critical',
                f"Change 'from argus import' to 'from agathos import' or use relative imports"
            )

        # Critical: Function calls to wrong module
        if re.search(r'argus_\w+\s*\(', line_text):
            func_name = re.search(r'(argus_\w+)\s*\(', line_text)
            if func_name:
                old_name = func_name.group(1)
                new_name = old_name.replace('argus_', 'agathos_')
                return (
                    'function_call',
                    'critical',
                    f"Change '{old_name}()' to '{new_name}()'"
                )

        # Critical: File paths containing .hermes/argus
        if '.hermes/argus' in line_text or '.hermes" / "argus' in line_text:
            return (
                'path',
                'critical',
                "Change path from '.hermes/argus' to '.hermes/agathos'"
            )

        # Critical: Module references like python -m argus
        if re.search(r'python\s+-m\s+argus', line_text):
            return (
                'module_reference',
                'critical',
                "Change 'python -m argus' to 'python -m agathos'"
            )

        # Warning: Variable names
        if re.search(r'\bargus_\w+\b', line_text) and not re.search(r'"[^"]*argus_\w+[^"]*"', line_text):
            var_match = re.search(r'\b(argus_\w+)\b', line_text)
            if var_match:
                old_var = var_match.group(1)
                new_var = old_var.replace('argus_', 'agathos_')
                return (
                    'variable',
                    'warning',
                    f"Consider renaming variable '{old_var}' to '{new_var}'"
                )

        # Warning: Launchd label
        if 'com.hermes.argus' in line_text:
            return (
                'launchd_label',
                'warning',
                "Change 'com.hermes.argus' to 'com.hermes.agathos'"
            )

        # Info: Comments/docstrings mentioning argus
        if line_text.strip().startswith('#') or line_text.strip().startswith('"""') or line_text.strip().startswith("'''"):
            return (
                'comment',
                'info',
                "Review if comment/docstring should reference 'agathos' instead of 'argus'"
            )

        # Info: String literals
        if re.search(r'["\'][^"\']*argus[^"\']*["\']', line_text):
            return (
                'string_literal',
                'info',
                "Review if string should use 'agathos' instead of 'argus'"
            )

        # Default
        return (
            'unknown',
            'info',
            "Review this reference"
        )

    def check_line(self, line: str, line_num: int, file_path: Path) -> List[Finding]:
        """Check a single line for argus references."""
        findings = []

        # Skip if no argus reference
        if 'argus' not in line.lower():
            return findings

        # Skip safe patterns
        if self.is_safe_pattern(line):
            return findings

        # Find all occurrences of 'argus' (case insensitive, word boundaries)
        for match in re.finditer(r'\b[aA][rR][gG][uU][sS]\w*\b', line):
            match_text = match.group()
            context = line.strip()
            category, severity, suggestion = self.categorize_and_score(match_text, line, file_path)

            findings.append(Finding(
                file=str(file_path.relative_to(self.root_dir.parent)),
                line=line_num,
                column=match.start(),
                context=context,
                category=category,
                severity=severity,
                suggestion=suggestion
            ))

        return findings

    def check_file_ast(self, file_path: Path) -> List[Finding]:
        """Use AST to find string literals with argus references."""
        findings = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                # Check string literals
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if 'argus' in node.value.lower() and not self.is_safe_pattern(node.value):
                        # Get line number
                        line_num = getattr(node, 'lineno', 0)
                        col_num = getattr(node, 'col_offset', 0)

                        # Get the actual line content
                        lines = content.split('\n')
                        context = lines[line_num - 1] if 0 < line_num <= len(lines) else node.value[:50]

                        # Determine if this is a user-facing string
                        category = 'user_string'
                        severity = 'warning'
                        suggestion = "Review if user-facing message should reference 'agathos'"

                        findings.append(Finding(
                            file=str(file_path.relative_to(self.root_dir.parent)),
                            line=line_num,
                            column=col_num,
                            context=context.strip(),
                            category=category,
                            severity=severity,
                            suggestion=suggestion
                        ))

        except SyntaxError as e:
            # File has syntax errors, skip AST check
            pass
        except Exception as e:
            # Other errors, log but don't crash
            print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)

        return findings

    def audit(self) -> List[Finding]:
        """Run full audit."""
        py_files = list(self.root_dir.rglob('*.py'))

        for file_path in py_files:
            if self.should_skip_file(file_path):
                continue

            self.files_checked += 1

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Error reading {file_path}: {e}", file=sys.stderr)
                continue

            # Line-by-line check
            for line_num, line in enumerate(lines, 1):
                self.lines_checked += 1
                line_findings = self.check_line(line, line_num, file_path)
                self.findings.extend(line_findings)

            # AST-based string literal check
            ast_findings = self.check_file_ast(file_path)
            # Only add AST findings for lines not already covered
            covered_lines = {(f.file, f.line) for f in self.findings}
            for finding in ast_findings:
                if (finding.file, finding.line) not in covered_lines:
                    self.findings.append(finding)

        return self.findings

    def report(self, output_format: str = 'text') -> str:
        """Generate report."""
        if output_format == 'json':
            return json.dumps([asdict(f) for f in self.findings], indent=2)

        # Text format
        lines = []
        lines.append("=" * 80)
        lines.append("ARGUS → AGATHOS AUDIT REPORT")
        lines.append("=" * 80)
        lines.append(f"Files checked: {self.files_checked}")
        lines.append(f"Lines checked: {self.lines_checked}")
        lines.append(f"Findings: {len(self.findings)}")
        lines.append("")

        # Group by severity
        critical = [f for f in self.findings if f.severity == 'critical']
        warning = [f for f in self.findings if f.severity == 'warning']
        info = [f for f in self.findings if f.severity == 'info']

        if critical:
            lines.append("\n" + "-" * 80)
            lines.append(f"CRITICAL ISSUES ({len(critical)}) - Must fix")
            lines.append("-" * 80)
            for finding in critical:
                lines.append(f"\n{finding.file}:{finding.line}")
                lines.append(f"  Category: {finding.category}")
                lines.append(f"  Context:  {finding.context[:80]}")
                lines.append(f"  Action:   {finding.suggestion}")

        if warning:
            lines.append("\n" + "-" * 80)
            lines.append(f"WARNINGS ({len(warning)}) - Should review")
            lines.append("-" * 80)
            for finding in warning:
                lines.append(f"\n{finding.file}:{finding.line}")
                lines.append(f"  Category: {finding.category}")
                lines.append(f"  Context:  {finding.context[:80]}")
                lines.append(f"  Action:   {finding.suggestion}")

        if info:
            lines.append("\n" + "-" * 80)
            lines.append(f"INFO ({len(info)}) - Review if time permits")
            lines.append("-" * 80)
            for finding in info:
                lines.append(f"\n{finding.file}:{finding.line}")
                lines.append(f"  Category: {finding.category}")
                lines.append(f"  Context:  {finding.context[:60]}")

        lines.append("\n" + "=" * 80)
        if critical:
            lines.append("RESULT: CRITICAL issues found - action required")
        elif warning:
            lines.append("RESULT: Warnings found - review recommended")
        else:
            lines.append("RESULT: Clean (no critical or warning issues)")
        lines.append("=" * 80)

        return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Audit agathos codebase for stale argus references')
    parser.add_argument('--strict', action='store_true', help='Exit with error if issues found')
    parser.add_argument('--json', action='store_true', help='Output JSON format')
    parser.add_argument('--root', type=Path, default=None, help='Root directory to scan')
    args = parser.parse_args()

    # Determine root directory
    if args.root:
        root = args.root
    else:
        # Default to agathos/ directory relative to script
        script_dir = Path(__file__).parent
        root = script_dir

    if not root.exists():
        print(f"Error: Directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    auditor = ArgusAuditor(root)
    auditor.audit()

    output_format = 'json' if args.json else 'text'
    print(auditor.report(output_format))

    # Exit code
    if args.strict:
        critical = [f for f in auditor.findings if f.severity == 'critical']
        warning = [f for f in auditor.findings if f.severity == 'warning']
        if critical or warning:
            sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
