#!/usr/bin/env python3
"""Deterministic pre-pass for workflow integrity scanner.

Extracts structural metadata from a BMad skill that the LLM scanner
can use instead of reading all files itself. Covers:
- Frontmatter parsing and validation
- Section inventory (H2/H3 headers)
- Template artifact detection
- Stage file cross-referencing
- Stage numbering validation
- Config header detection in prompts
- Language/directness pattern grep
- On Exit / Exiting section detection (invalid)
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


# Template artifacts that should NOT appear in finalized skills
TEMPLATE_ARTIFACTS = [
    r'\{if-complex-workflow\}', r'\{/if-complex-workflow\}',
    r'\{if-simple-workflow\}', r'\{/if-simple-workflow\}',
    r'\{if-simple-utility\}', r'\{/if-simple-utility\}',
    r'\{if-module\}', r'\{/if-module\}',
    r'\{if-headless\}', r'\{/if-headless\}',
    r'\{displayName\}', r'\{skillName\}',
]
# Runtime variables that ARE expected (not artifacts)
RUNTIME_VARS = {
    '{user_name}', '{communication_language}', '{document_output_language}',
    '{project-root}', '{output_folder}', '{planning_artifacts}',
}

# Directness anti-patterns
DIRECTNESS_PATTERNS = [
    (r'\byou should\b', 'Suggestive "you should" — use direct imperative'),
    (r'\bplease\b(?! note)', 'Polite "please" — use direct imperative'),
    (r'\bhandle appropriately\b', 'Ambiguous "handle appropriately" — specify how'),
    (r'\bwhen ready\b', 'Vague "when ready" — specify testable condition'),
]

# Invalid sections
INVALID_SECTIONS = [
    (r'^##\s+On\s+Exit\b', 'On Exit section found — no exit hooks exist in the system, this will never run'),
    (r'^##\s+Exiting\b', 'Exiting section found — no exit hooks exist in the system, this will never run'),
]


def parse_frontmatter(content: str) -> tuple[dict | None, list[dict]]:
    """Parse YAML frontmatter and validate."""
    findings = []
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not fm_match:
        findings.append({
            'file': 'SKILL.md', 'line': 1,
            'severity': 'critical', 'category': 'frontmatter',
            'issue': 'No YAML frontmatter found',
        })
        return None, findings

    try:
        # Frontmatter is YAML-like key: value pairs — parse manually
        fm = {}
        for line in fm_match.group(1).strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                key, _, value = line.partition(':')
                fm[key.strip()] = value.strip().strip('"').strip("'")
    except Exception as e:
        findings.append({
            'file': 'SKILL.md', 'line': 1,
            'severity': 'critical', 'category': 'frontmatter',
            'issue': f'Invalid frontmatter: {e}',
        })
        return None, findings

    if not isinstance(fm, dict):
        findings.append({
            'file': 'SKILL.md', 'line': 1,
            'severity': 'critical', 'category': 'frontmatter',
            'issue': 'Frontmatter is not a YAML mapping',
        })
        return None, findings

    # name check
    name = fm.get('name')
    if not name:
        findings.append({
            'file': 'SKILL.md', 'line': 1,
            'severity': 'critical', 'category': 'frontmatter',
            'issue': 'Missing "name" field in frontmatter',
        })
    elif not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', name):
        findings.append({
            'file': 'SKILL.md', 'line': 1,
            'severity': 'high', 'category': 'frontmatter',
            'issue': f'Name "{name}" is not kebab-case',
        })
    # bmad- prefix check removed — bmad- is reserved for official BMad creations only

    # description check
    desc = fm.get('description')
    if not desc:
        findings.append({
            'file': 'SKILL.md', 'line': 1,
            'severity': 'high', 'category': 'frontmatter',
            'issue': 'Missing "description" field in frontmatter',
        })
    elif 'Use when' not in desc and 'use when' not in desc:
        findings.append({
            'file': 'SKILL.md', 'line': 1,
            'severity': 'medium', 'category': 'frontmatter',
            'issue': 'Description missing "Use when..." trigger phrase',
        })

    # Extra fields check
    allowed = {'name', 'description', 'menu-code'}
    extra = set(fm.keys()) - allowed
    if extra:
        findings.append({
            'file': 'SKILL.md', 'line': 1,
            'severity': 'low', 'category': 'frontmatter',
            'issue': f'Extra frontmatter fields: {", ".join(sorted(extra))}',
        })

    return fm, findings


def extract_sections(content: str) -> list[dict]:
    """Extract all H2 headers with line numbers."""
    sections = []
    for i, line in enumerate(content.split('\n'), 1):
        m = re.match(r'^(#{2,3})\s+(.+)$', line)
        if m:
            sections.append({
                'level': len(m.group(1)),
                'title': m.group(2).strip(),
                'line': i,
            })
    return sections


def check_required_sections(sections: list[dict]) -> list[dict]:
    """Check for required and invalid sections."""
    findings = []
    h2_titles = [s['title'] for s in sections if s['level'] == 2]

    if 'Overview' not in h2_titles:
        findings.append({
            'file': 'SKILL.md', 'line': 1,
            'severity': 'high', 'category': 'sections',
            'issue': 'Missing ## Overview section',
        })

    if 'On Activation' not in h2_titles:
        findings.append({
            'file': 'SKILL.md', 'line': 1,
            'severity': 'high', 'category': 'sections',
            'issue': 'Missing ## On Activation section',
        })

    # Invalid sections
    for s in sections:
        if s['level'] == 2:
            for pattern, message in INVALID_SECTIONS:
                if re.match(pattern, f"## {s['title']}"):
                    findings.append({
                        'file': 'SKILL.md', 'line': s['line'],
                        'severity': 'high', 'category': 'invalid-section',
                        'issue': message,
                    })

    return findings


def find_template_artifacts(filepath: Path, rel_path: str) -> list[dict]:
    """Scan for orphaned template substitution artifacts."""
    findings = []
    content = filepath.read_text(encoding='utf-8')

    for pattern in TEMPLATE_ARTIFACTS:
        for m in re.finditer(pattern, content):
            matched = m.group()
            if matched in RUNTIME_VARS:
                continue
            line_num = content[:m.start()].count('\n') + 1
            findings.append({
                'file': rel_path, 'line': line_num,
                'severity': 'high', 'category': 'artifacts',
                'issue': f'Orphaned template artifact: {matched}',
                'fix': 'Resolve or remove this template conditional/placeholder',
            })

    return findings


def cross_reference_stages(skill_path: Path, skill_content: str) -> tuple[dict, list[dict]]:
    """Cross-reference stage files between SKILL.md and numbered prompt files at skill root."""
    findings = []

    # Get actual numbered prompt files at skill root (exclude SKILL.md)
    actual_files = set()
    for f in skill_path.iterdir():
        if f.is_file() and f.suffix == '.md' and f.name != 'SKILL.md' and re.match(r'^\d+-', f.name):
            actual_files.add(f.name)

    # Find stage references in SKILL.md — look for both old prompts/ style and new root style
    referenced = set()
    # Match `prompts/XX-name.md` (legacy) or bare `XX-name.md` references
    ref_pattern = re.compile(r'(?:prompts/)?(\d+-[^\s)`]+\.md)')
    for m in ref_pattern.finditer(skill_content):
        referenced.add(m.group(1))

    # Missing files (referenced but don't exist)
    missing = referenced - actual_files
    for f in sorted(missing):
        findings.append({
            'file': 'SKILL.md', 'line': 0,
            'severity': 'critical', 'category': 'missing-stage',
            'issue': f'Referenced stage file does not exist: {f}',
        })

    # Orphaned files (exist but not referenced)
    orphaned = actual_files - referenced
    for f in sorted(orphaned):
        findings.append({
            'file': f, 'line': 0,
            'severity': 'medium', 'category': 'naming',
            'issue': f'Stage file exists but not referenced in SKILL.md: {f}',
        })

    # Stage numbering check
    numbered = []
    for f in sorted(actual_files):
        m = re.match(r'^(\d+)-(.+)\.md$', f)
        if m:
            numbered.append((int(m.group(1)), f))

    if numbered:
        numbered.sort()
        nums = [n[0] for n in numbered]
        expected = list(range(nums[0], nums[0] + len(nums)))
        if nums != expected:
            gaps = set(expected) - set(nums)
            if gaps:
                findings.append({
                    'file': skill_path.name, 'line': 0,
                    'severity': 'medium', 'category': 'naming',
                    'issue': f'Stage numbering has gaps: missing {sorted(gaps)}',
                })

    stage_summary = {
        'total_stages': len(actual_files),
        'referenced': sorted(referenced),
        'actual': sorted(actual_files),
        'missing_stages': sorted(missing),
        'orphaned_stages': sorted(orphaned),
    }

    return stage_summary, findings


def check_prompt_basics(skill_path: Path) -> tuple[list[dict], list[dict]]:
    """Check each prompt file for config header and progression conditions."""
    findings = []
    prompt_details = []

    # Look for numbered prompt files at skill root
    prompt_files = sorted(
        f for f in skill_path.iterdir()
        if f.is_file() and f.suffix == '.md' and f.name != 'SKILL.md' and re.match(r'^\d+-', f.name)
    )
    if not prompt_files:
        return prompt_details, findings

    for f in prompt_files:
        content = f.read_text(encoding='utf-8')
        rel_path = f.name
        detail = {'file': f.name, 'has_config_header': False, 'has_progression': False}

        # Config header check
        if '{communication_language}' in content or '{document_output_language}' in content:
            detail['has_config_header'] = True
        else:
            findings.append({
                'file': rel_path, 'line': 1,
                'severity': 'medium', 'category': 'config-header',
                'issue': 'No config header with language variables found',
            })

        # Progression condition check (look for progression-related keywords near end)
        lower = content.lower()
        prog_keywords = ['progress', 'advance', 'move to', 'next stage', 'when complete',
                         'proceed to', 'transition', 'completion criteria']
        if any(kw in lower for kw in prog_keywords):
            detail['has_progression'] = True
        else:
            findings.append({
                'file': rel_path, 'line': len(content.split('\n')),
                'severity': 'high', 'category': 'progression',
                'issue': 'No progression condition keywords found',
            })

        # Directness checks
        for pattern, message in DIRECTNESS_PATTERNS:
            for m in re.finditer(pattern, content, re.IGNORECASE):
                line_num = content[:m.start()].count('\n') + 1
                findings.append({
                    'file': rel_path, 'line': line_num,
                    'severity': 'low', 'category': 'language',
                    'issue': message,
                })

        # Template artifacts
        findings.extend(find_template_artifacts(f, rel_path))

        prompt_details.append(detail)

    return prompt_details, findings


def detect_workflow_type(skill_content: str, has_prompts: bool) -> str:
    """Detect workflow type from SKILL.md content."""
    has_stage_refs = bool(re.search(r'(?:prompts/)?\d+-\S+\.md', skill_content))
    has_routing = bool(re.search(r'(?i)(rout|stage|branch|path)', skill_content))

    if has_stage_refs or (has_prompts and has_routing):
        return 'complex'
    elif re.search(r'(?m)^\d+\.\s', skill_content):
        return 'simple-workflow'
    else:
        return 'simple-utility'


def scan_workflow_integrity(skill_path: Path) -> dict:
    """Run all deterministic workflow integrity checks."""
    all_findings = []

    # Read SKILL.md
    skill_md = skill_path / 'SKILL.md'
    if not skill_md.exists():
        return {
            'scanner': 'workflow-integrity-prepass',
            'script': 'prepass-workflow-integrity.py',
            'version': '1.0.0',
            'skill_path': str(skill_path),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'fail',
            'issues': [{'file': 'SKILL.md', 'line': 1, 'severity': 'critical',
                         'category': 'missing-file', 'issue': 'SKILL.md does not exist'}],
            'summary': {'total_issues': 1, 'by_severity': {'critical': 1, 'high': 0, 'medium': 0, 'low': 0}},
        }

    skill_content = skill_md.read_text(encoding='utf-8')

    # Frontmatter
    frontmatter, fm_findings = parse_frontmatter(skill_content)
    all_findings.extend(fm_findings)

    # Sections
    sections = extract_sections(skill_content)
    section_findings = check_required_sections(sections)
    all_findings.extend(section_findings)

    # Template artifacts in SKILL.md
    all_findings.extend(find_template_artifacts(skill_md, 'SKILL.md'))

    # Directness checks in SKILL.md
    for pattern, message in DIRECTNESS_PATTERNS:
        for m in re.finditer(pattern, skill_content, re.IGNORECASE):
            line_num = skill_content[:m.start()].count('\n') + 1
            all_findings.append({
                'file': 'SKILL.md', 'line': line_num,
                'severity': 'low', 'category': 'language',
                'issue': message,
            })

    # Workflow type
    has_prompts = any(
        f.is_file() and f.suffix == '.md' and f.name != 'SKILL.md' and re.match(r'^\d+-', f.name)
        for f in skill_path.iterdir()
    )
    workflow_type = detect_workflow_type(skill_content, has_prompts)

    # Stage cross-reference
    stage_summary, stage_findings = cross_reference_stages(skill_path, skill_content)
    all_findings.extend(stage_findings)

    # Prompt basics
    prompt_details, prompt_findings = check_prompt_basics(skill_path)
    all_findings.extend(prompt_findings)

    # Build severity summary
    by_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    for f in all_findings:
        sev = f['severity']
        if sev in by_severity:
            by_severity[sev] += 1

    status = 'pass'
    if by_severity['critical'] > 0:
        status = 'fail'
    elif by_severity['high'] > 0:
        status = 'warning'

    return {
        'scanner': 'workflow-integrity-prepass',
        'script': 'prepass-workflow-integrity.py',
        'version': '1.0.0',
        'skill_path': str(skill_path),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'status': status,
        'metadata': {
            'frontmatter': frontmatter,
            'sections': sections,
            'workflow_type': workflow_type,
        },
        'stage_summary': stage_summary,
        'prompt_details': prompt_details,
        'issues': all_findings,
        'summary': {
            'total_issues': len(all_findings),
            'by_severity': by_severity,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Deterministic pre-pass for workflow integrity scanning',
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
    args = parser.parse_args()

    if not args.skill_path.is_dir():
        print(f"Error: {args.skill_path} is not a directory", file=sys.stderr)
        return 2

    result = scan_workflow_integrity(args.skill_path)
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
