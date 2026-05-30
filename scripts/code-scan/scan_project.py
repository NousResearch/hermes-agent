#!/usr/bin/env python3
"""Deterministic project scanner for the UA Flywheel code-scan module.

Walks a project tree, applies .hermesignore rules, classifies each file
by language and category, counts lines, and outputs a structured JSON report.

Usage:
    python scripts/code-scan/scan_project.py <target_dir> [--output file.json] [--verbose]

Exit codes: 0 = success, nonzero = error.
"""
import argparse
import fnmatch
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

# Ensure scripts/code-scan is on sys.path for sibling imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from language_registry import get_language, get_category, detect_frameworks

# Directories always excluded regardless of .hermesignore
ALWAYS_EXCLUDE: Set[str] = {'.git', '.svn', '.hg'}


def _load_hermesignore(project_root: str) -> List[str]:
    """Load .hermesignore patterns from project root, or return defaults."""
    ignore_path = os.path.join(project_root, '.hermesignore')
    patterns: List[str] = []

    # Default patterns that are always applied
    defaults = [
        '.git/', 'node_modules/', '__pycache__/',
        '.venv/', 'venv/', 'env/',
        'dist/', 'build/', 'out/',
        '*.pyc', '*.pyo', '*.egg-info/',
        '.pytest_cache/', '.mypy_cache/', '.ruff_cache/',
        '.vscode/', '.idea/', '*.swp', '*.swo',
        '.DS_Store', 'Thumbs.db',
        '.env', '.env.local',
        'vendor/',
    ]

    if os.path.isfile(ignore_path):
        try:
            with open(ignore_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and blank lines
                    if not line or line.startswith('#'):
                        continue
                    patterns.append(line)
            # Always include defaults even with a custom .hermesignore
            patterns.extend(defaults)
        except OSError as e:
            print(f"Warning: could not read .hermesignore: {e}", file=sys.stderr)
            patterns = defaults
    else:
        patterns = defaults

    return patterns


def _is_ignored(rel_path: str, patterns: List[str]) -> bool:
    """Check if a relative path matches any ignore pattern."""
    # Normalize path separators
    rel_path = rel_path.replace('\\', '/')
    for pattern in patterns:
        dir_pattern = pattern.rstrip('/')
        if pattern.endswith('/'):
            # Directory pattern: check if path starts with the directory
            if rel_path.startswith(pattern):
                return True
            # Also try fnmatch on the directory portion
            parts = rel_path.split('/')
            for i, part in enumerate(parts[:-1]):
                if fnmatch.fnmatch(part + '/', pattern):
                    return True
            # Check first component
            first_part = parts[0] + '/'
            if fnmatch.fnmatch(first_part, pattern):
                return True
        else:
            # File/glob pattern
            basename = os.path.basename(rel_path)
            if fnmatch.fnmatch(basename, pattern):
                return True
            if fnmatch.fnmatch(rel_path, pattern):
                return True
    return False


def _is_hardcoded_dir_excluded(dirname: str) -> bool:
    """Check if a directory is always excluded by hard-coded rules."""
    return dirname in ALWAYS_EXCLUDE


def _count_lines(filepath: str) -> int:
    """Count physical lines in a file. Returns 0 on read error."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return sum(1 for _ in f)
    except (OSError, IOError):
        return 0


def _walk_project(project_root: str, patterns: List[str]) -> List[dict]:
    """Walk the project tree and collect file records.

    Returns a list of dicts with keys:
        path, relative_path, language, category, lines, size_bytes
    """
    files = []
    root = os.path.realpath(project_root)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place
        dirnames[:] = [
            d for d in dirnames
            if not _is_hardcoded_dir_excluded(d)
        ]

        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root)

            if _is_ignored(rel_path, patterns):
                continue

            try:
                size_bytes = os.path.getsize(abs_path)
            except OSError:
                size_bytes = 0

            language = get_language(filename)
            category = get_category(rel_path)
            lines = _count_lines(abs_path)

            files.append({
                'path': abs_path,
                'relative_path': rel_path,
                'language': language,
                'category': category,
                'lines': lines,
                'size_bytes': size_bytes,
            })

    return files


def _build_summary(files: List[dict], frameworks: List[str],
                   project_root: str) -> dict:
    """Build the final JSON summary dict from scanned files."""
    # Count languages
    languages: dict = {}
    for f in files:
        lang = f['language']
        languages[lang] = languages.get(lang, 0) + 1

    # Count categories
    categories: dict = {}
    for f in files:
        cat = f['category']
        categories[cat] = categories.get(cat, 0) + 1

    total_lines = sum(f['lines'] for f in files)

    warnings: List[str] = []
    if not files:
        warnings.append("No files found in project root")

    return {
        'project_root': project_root,
        'scanned_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'total_files': len(files),
        'total_lines': total_lines,
        'languages': languages,
        'categories': categories,
        'frameworks': frameworks,
        'files': files,
        'warnings': warnings,
    }


def main() -> int:
    """CLI entry point. Returns exit code."""
    parser = argparse.ArgumentParser(
        description='Scan a project directory and emit structured JSON.'
    )
    parser.add_argument(
        'target_dir',
        help='Path to the project directory to scan',
    )
    parser.add_argument(
        '--output',
        help='Path to write JSON output to (default: print to stdout)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress information to stderr',
    )
    args = parser.parse_args()

    target_dir = os.path.realpath(args.target_dir)

    # Validate target is a directory
    if not os.path.isdir(target_dir):
        print(f"Error: '{args.target_dir}' is not a valid directory",
              file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Scanning: {target_dir}", file=sys.stderr)

    # Load ignore patterns
    patterns = _load_hermesignore(target_dir)
    if args.verbose:
        print(f"Loaded {len(patterns)} ignore patterns", file=sys.stderr)

    # Walk the project
    files = _walk_project(target_dir, patterns)
    if args.verbose:
        print(f"Found {len(files)} files", file=sys.stderr)

    # Detect frameworks
    frameworks = detect_frameworks(target_dir)
    if args.verbose:
        print(f"Detected frameworks: {frameworks or 'none'}", file=sys.stderr)

    # Build summary
    summary = _build_summary(files, frameworks, target_dir)

    # Output
    json_str = json.dumps(summary, indent=2)

    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json_str)
                f.write('\n')
            if args.verbose:
                print(f"Written to {args.output}", file=sys.stderr)
        except OSError as e:
            print(f"Error: could not write output file: {e}", file=sys.stderr)
            return 1
    else:
        print(json_str)

    return 0


if __name__ == '__main__':
    sys.exit(main())
