#!/usr/bin/env python3
"""Deterministic path standards scanner for BMad skills.

Validates all .md and .json files against BMad path conventions:
1. {project-root} for any project-scope path (not just _bmad)
2. Bare _bmad references must have {project-root} prefix
3. Config variables used directly — no double-prefix with {project-root}
4. ./ only for same-folder references — never ./subdir/ cross-directory
5. No ../ parent directory references
6. No absolute paths
7. Frontmatter allows only name and description
8. No .md files at skill root except SKILL.md
"""

# /// script
# requires-python = ">=3.9"
# ///

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


# Patterns to detect
# Double-prefix: {project-root}/{config-variable} — config vars already contain project-root
DOUBLE_PREFIX_RE = re.compile(r'\{project-root\}/\{[^}]+\}')
# Bare _bmad without {project-root} prefix — match _bmad at word boundary
# but not when preceded by {project-root}/
BARE_BMAD_RE = re.compile(r'(?<!\{project-root\}/)_bmad[/\s]')
# Absolute paths
ABSOLUTE_PATH_RE = re.compile(r'(?:^|[\s"`\'(])(/(?:Users|home|opt|var|tmp|etc|usr)/\S+)', re.MULTILINE)
HOME_PATH_RE = re.compile(r'(?:^|[\s"`\'(])(~/\S+)', re.MULTILINE)
# Parent directory reference (still invalid)
RELATIVE_DOT_RE = re.compile(r'(?:^|[\s"`\'(])(\.\./\S+)', re.MULTILINE)
# Cross-directory ./ — ./subdir/ is wrong because ./ means same folder only
CROSS_DIR_DOT_SLASH_RE = re.compile(r'(?:^|[\s"`\'(])\./(?:references|scripts|assets)/\S+', re.MULTILINE)

# Fenced code block detection (to skip examples showing wrong patterns)
FENCE_RE = re.compile(r'^```', re.MULTILINE)

# Valid frontmatter keys
VALID_FRONTMATTER_KEYS = {'name', 'description'}


def is_in_fenced_block(content: str, pos: int) -> bool:
    """Check if a position is inside a fenced code block."""
    fences = [m.start() for m in FENCE_RE.finditer(content[:pos])]
    # Odd number of fences before pos means we're inside a block
    return len(fences) % 2 == 1


def get_line_number(content: str, pos: int) -> int:
    """Get 1-based line number for a position in content."""
    return content[:pos].count('\n') + 1


def check_frontmatter(content: str, filepath: Path) -> list[dict]:
    """Validate SKILL.md frontmatter contains only allowed keys."""
    findings = []
    if filepath.name != 'SKILL.md':
        return findings

    if not content.startswith('---'):
        findings.append({
            'file': filepath.name,
            'line': 1,
            'severity': 'critical',
            'category': 'frontmatter',
            'title': 'SKILL.md missing frontmatter block',
            'detail': 'SKILL.md must start with --- frontmatter containing name and description',
            'action': 'Add frontmatter with name and description fields',
        })
        return findings

    # Find closing ---
    end = content.find('\n---', 3)
    if end == -1:
        findings.append({
            'file': filepath.name,
            'line': 1,
            'severity': 'critical',
            'category': 'frontmatter',
            'title': 'SKILL.md frontmatter block not closed',
            'detail': 'Missing closing --- for frontmatter',
            'action': 'Add closing --- after frontmatter fields',
        })
        return findings

    frontmatter = content[4:end]
    for i, line in enumerate(frontmatter.split('\n'), start=2):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            key = line.split(':', 1)[0].strip()
            if key not in VALID_FRONTMATTER_KEYS:
                findings.append({
                    'file': filepath.name,
                    'line': i,
                    'severity': 'high',
                    'category': 'frontmatter',
                    'title': f'Invalid frontmatter key: {key}',
                    'detail': f'Only {", ".join(sorted(VALID_FRONTMATTER_KEYS))} are allowed in frontmatter',
                    'action': f'Remove {key} from frontmatter — use as content field in SKILL.md body instead',
                })

    return findings


def check_root_md_files(skill_path: Path) -> list[dict]:
    """Check that no .md files exist at skill root except SKILL.md."""
    findings = []
    for md_file in skill_path.glob('*.md'):
        if md_file.name != 'SKILL.md':
            findings.append({
                'file': md_file.name,
                'line': 0,
                'severity': 'high',
                'category': 'structure',
                'title': f'Prompt file at skill root: {md_file.name}',
                'detail': 'All progressive disclosure content must be in ./references/ — only SKILL.md belongs at root',
                'action': f'Move {md_file.name} to references/{md_file.name}',
            })
    return findings


def scan_file(filepath: Path, skip_fenced: bool = True) -> list[dict]:
    """Scan a single file for path standard violations."""
    findings = []
    content = filepath.read_text(encoding='utf-8')
    rel_path = filepath.name

    checks = [
        (DOUBLE_PREFIX_RE, 'double-prefix', 'critical',
         'Double-prefix: {project-root}/{variable} — config variables already contain {project-root} at runtime'),
        (ABSOLUTE_PATH_RE, 'absolute-path', 'high',
         'Absolute path found — not portable across machines'),
        (HOME_PATH_RE, 'absolute-path', 'high',
         'Home directory path (~/) found — environment-specific'),
        (RELATIVE_DOT_RE, 'relative-prefix', 'high',
         'Parent directory reference (../) found — fragile, breaks with reorganization'),
        (CROSS_DIR_DOT_SLASH_RE, 'cross-dir-dot-slash', 'high',
         'Cross-directory ./ reference — ./ means same folder only; use bare skill-root relative path (e.g., references/foo.md not ./references/foo.md)'),
    ]

    for pattern, category, severity, message in checks:
        for match in pattern.finditer(content):
            pos = match.start()
            if skip_fenced and is_in_fenced_block(content, pos):
                continue
            line_num = get_line_number(content, pos)
            line_content = content.split('\n')[line_num - 1].strip()
            findings.append({
                'file': rel_path,
                'line': line_num,
                'severity': severity,
                'category': category,
                'title': message,
                'detail': line_content[:120],
                'action': '',
            })

    # Bare _bmad check — more nuanced, need to avoid false positives
    # inside {project-root}/_bmad which is correct
    for match in BARE_BMAD_RE.finditer(content):
        pos = match.start()
        if skip_fenced and is_in_fenced_block(content, pos):
            continue
        start = max(0, pos - 30)
        before = content[start:pos]
        if '{project-root}/' in before:
            continue
        line_num = get_line_number(content, pos)
        line_content = content.split('\n')[line_num - 1].strip()
        findings.append({
            'file': rel_path,
            'line': line_num,
            'severity': 'high',
            'category': 'bare-bmad',
            'title': 'Bare _bmad reference without {project-root} prefix',
            'detail': line_content[:120],
            'action': '',
        })

    return findings


def scan_skill(skill_path: Path, skip_fenced: bool = True) -> dict:
    """Scan all .md and .json files in a skill directory."""
    all_findings = []

    # Check for .md files at root that aren't SKILL.md
    all_findings.extend(check_root_md_files(skill_path))

    # Check SKILL.md frontmatter
    skill_md = skill_path / 'SKILL.md'
    if skill_md.exists():
        content = skill_md.read_text(encoding='utf-8')
        all_findings.extend(check_frontmatter(content, skill_md))

    # Find all .md and .json files
    md_files = sorted(list(skill_path.rglob('*.md')) + list(skill_path.rglob('*.json')))
    if not md_files:
        print(f"Warning: No .md or .json files found in {skill_path}", file=sys.stderr)

    files_scanned = []
    for md_file in md_files:
        rel = md_file.relative_to(skill_path)
        files_scanned.append(str(rel))
        file_findings = scan_file(md_file, skip_fenced)
        for f in file_findings:
            f['file'] = str(rel)
        all_findings.extend(file_findings)

    # Build summary
    by_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    by_category = {
        'double_prefix': 0,
        'bare_bmad': 0,
        'absolute_path': 0,
        'relative_prefix': 0,
        'cross_dir_dot_slash': 0,
        'frontmatter': 0,
        'structure': 0,
    }

    for f in all_findings:
        sev = f['severity']
        if sev in by_severity:
            by_severity[sev] += 1
        cat = f['category'].replace('-', '_')
        if cat in by_category:
            by_category[cat] += 1

    return {
        'scanner': 'path-standards',
        'script': 'scan-path-standards.py',
        'version': '3.0.0',
        'skill_path': str(skill_path),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'files_scanned': files_scanned,
        'status': 'pass' if not all_findings else 'fail',
        'findings': all_findings,
        'assessments': {},
        'summary': {
            'total_findings': len(all_findings),
            'by_severity': by_severity,
            'by_category': by_category,
            'assessment': 'Path standards scan complete',
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Scan BMad skill for path standard violations',
    )
    parser.add_argument(
        'skill_path',
        type=Path,
        help='Path to the skill directory to scan',
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Write JSON output to file instead of stdout',
    )
    parser.add_argument(
        '--include-fenced',
        action='store_true',
        help='Also check inside fenced code blocks (by default they are skipped)',
    )
    args = parser.parse_args()

    if not args.skill_path.is_dir():
        print(f"Error: {args.skill_path} is not a directory", file=sys.stderr)
        return 2

    result = scan_skill(args.skill_path, skip_fenced=not args.include_fenced)
    output = json.dumps(result, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
