#!/usr/bin/env python3
"""
Security scanner for detecting hardcoded credentials in code.

Scans a directory for common credential patterns and reports findings.
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Common secret patterns (regex)
SECRET_PATTERNS = {
    'aws_access_key': r'(?i)AKIA[0-9A-Z]{16}',
    'aws_secret_key': r'(?i)(?:aws_secret|aws.{0,20}secret).{0,20}[=:]\s*["\']?([0-9a-zA-Z/+=]{40})["\']?',
    'supabase_url': r'https://[a-z]{20}\.supabase\.co',
    'supabase_anon_key': r'eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*',
    'stripe_key': r'(?:sk|pk)_(live|test)_[0-9a-zA-Z]{24,}',
    'cloudflare_api_token': r'(?i)(?:cloudflare|cf).{0,20}(?:token|key).{0,20}[=:]\s*["\']?([a-zA-Z0-9_-]{40,})["\']?',
    'turnstile_key': r'0x[0-9A-F]{22}',
    'generic_api_key': r'(?i)(?:api[_-]?key|apikey).{0,20}[=:]\s*["\']?([0-9a-zA-Z_\-]{20,})["\']?',
    'r2_account_id': r'[0-9a-f]{32}(?=\.r2\.cloudflarestorage\.com)',
    'jwt_token': r'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
    'private_key': r'-----BEGIN (?:RSA|EC|OPENSSH|DSA) PRIVATE KEY-----',
    'oauth_secret': r'(?i)(?:client_secret|oauth).{0,20}[=:]\s*["\']?([0-9a-zA-Z_\-]{20,})["\']?',
}

# File extensions to scan
SCANNABLE_EXTENSIONS = {
    '.ts', '.tsx', '.js', '.jsx', '.py', '.md', '.json', '.yaml', '.yml',
    '.env', '.env.example', '.env.local', '.env.production', '.env.development',
    '.sh', '.bash', '.zsh', '.sql', '.go', '.java', '.rb', '.php', '.cs'
}

# Directories to skip
SKIP_DIRS = {
    'node_modules', '.git', '.venv', 'venv', '__pycache__', 'dist', 'build',
    '.next', '.nuxt', 'vendor', 'target', 'bin', 'obj', '.terraform'
}

class SecretFinding:
    """Represents a detected secret."""
    def __init__(self, file_path: str, line_num: int, pattern_name: str,
                 matched_text: str, line_content: str):
        self.file_path = file_path
        self.line_num = line_num
        self.pattern_name = pattern_name
        self.matched_text = matched_text
        self.line_content = line_content.strip()

    def to_dict(self) -> Dict:
        return {
            'file': self.file_path,
            'line': self.line_num,
            'type': self.pattern_name,
            'match': self.matched_text[:50] + '...' if len(self.matched_text) > 50 else self.matched_text,
            'context': self.line_content[:100] + '...' if len(self.line_content) > 100 else self.line_content
        }

def scan_file(file_path: Path, base_dir: Path) -> List[SecretFinding]:
    """Scan a single file for secrets."""
    findings = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                for pattern_name, pattern in SECRET_PATTERNS.items():
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        # Skip common false positives
                        if should_skip_match(line, match.group()):
                            continue

                        findings.append(SecretFinding(
                            file_path=str(file_path.relative_to(base_dir)),
                            line_num=line_num,
                            pattern_name=pattern_name,
                            matched_text=match.group(),
                            line_content=line
                        ))
    except Exception as e:
        print(f"Warning: Could not scan {file_path}: {e}", file=sys.stderr)

    return findings

def should_skip_match(line: str, match: str) -> bool:
    """Check if a match should be skipped (likely false positive)."""
    # Skip example/placeholder values
    placeholders = [
        'your-', 'example', 'placeholder', 'xxx', 'yyy', 'zzz',
        'test-', 'demo-', 'sample-', '<YOUR_', '${', 'TODO'
    ]

    line_lower = line.lower()
    match_lower = match.lower()

    for placeholder in placeholders:
        if placeholder in match_lower or placeholder in line_lower:
            return True

    # Skip if in a comment
    if re.search(r'^\s*(?://|#|/\*|\*)', line):
        return True

    return False

def scan_directory(directory: Path, exclude_patterns: List[str] = None) -> List[SecretFinding]:
    """Scan a directory recursively for secrets."""
    findings = []
    exclude_patterns = exclude_patterns or []

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        root_path = Path(root)

        # Skip if matches exclude pattern
        if any(re.search(pattern, str(root_path)) for pattern in exclude_patterns):
            continue

        for file in files:
            file_path = root_path / file

            # Only scan relevant file types
            if file_path.suffix not in SCANNABLE_EXTENSIONS:
                continue

            # Skip if matches exclude pattern
            if any(re.search(pattern, str(file_path)) for pattern in exclude_patterns):
                continue

            file_findings = scan_file(file_path, directory)
            findings.extend(file_findings)

    return findings

def print_report(findings: List[SecretFinding], directory: Path):
    """Print a human-readable report."""
    if not findings:
        print("✅ No secrets detected!")
        return

    print(f"\n⚠️  Found {len(findings)} potential secrets in {directory}:\n")

    # Group by file
    by_file = {}
    for finding in findings:
        if finding.file_path not in by_file:
            by_file[finding.file_path] = []
        by_file[finding.file_path].append(finding)

    # Print grouped
    for file_path in sorted(by_file.keys()):
        print(f"📄 {file_path}")
        for finding in by_file[file_path]:
            print(f"   Line {finding.line_num}: {finding.pattern_name}")
            print(f"      Match: {finding.matched_text[:80]}")
            print(f"      Context: {finding.line_content[:80]}")
        print()

def main():
    if len(sys.argv) < 2:
        print("Usage: scan_secrets.py <directory> [--json] [--exclude pattern1 pattern2 ...]")
        print("\nExamples:")
        print("  scan_secrets.py ./my-project")
        print("  scan_secrets.py ./my-project --json")
        print("  scan_secrets.py ./my-project --exclude '.*test.*' '.*example.*'")
        sys.exit(1)

    directory = Path(sys.argv[1]).resolve()

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Parse arguments
    json_output = '--json' in sys.argv
    exclude_patterns = []

    if '--exclude' in sys.argv:
        exclude_idx = sys.argv.index('--exclude')
        exclude_patterns = [arg for arg in sys.argv[exclude_idx + 1:] if not arg.startswith('--')]

    # Scan
    findings = scan_directory(directory, exclude_patterns)

    # Output
    if json_output:
        print(json.dumps([f.to_dict() for f in findings], indent=2))
    else:
        print_report(findings, directory)

    # Exit code
    sys.exit(1 if findings else 0)

if __name__ == '__main__':
    main()
