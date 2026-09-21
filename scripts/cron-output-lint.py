#!/usr/bin/env python3
"""Lint active Discord cron prompts/scripts for output-format regressions.

Default mode checks current prompts + scripts only. Use --latest to include the latest
historical cron output, which may still fail until a cron naturally reruns.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

JOBS = Path('/home/kensei/.hermes/cron/jobs.json')
SCRIPTS = Path('/home/kensei/.hermes/scripts')
OUTPUT = Path('/home/kensei/.hermes/cron/output')

BAD_PATTERNS = {
    'telegram_html_tags': re.compile(r'<(?:b|code|blockquote)(?:\s|>|/)', re.I),
    'raw_html_visible': re.compile(r'<!doctype html|<html[\s>]|<style[\s>]', re.I),
    'process_narration_literal': re.compile(r"Now I'll|I will write|I'll write", re.I),
    'memory_leak_literal': re.compile(r'<memory-context>|Mnemosyne Context', re.I),
}

# Visible-output-only check. This is intentionally NOT applied to prompts/scripts,
# because prompts need to mention examples and HTML attachment internals.
BAD_VISIBLE_DATE = re.compile(
    r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+'
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    r'|\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{1,2}\s+'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    r'|\d{4}-\d{2}-\d{2}'
)

FIRST_LINE_TS = re.compile(r'^.+ · \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}$')

# Evidence Gate (see cron-output-contract skill). Strong causal/state assertions in
# DELIVERED output must be backed by a nearby evidence citation. Applied to visible
# output only — never to prompts/scripts, which legitimately quote these as examples.
STRONG_CLAIM = re.compile(
    r'\bnot registered\b'
    r'|\bno scheduled jobs\b'
    r'|\b(?:does|did)\s*n[o\']?t\s+exist\b|\bdoesn\'?t exist\b'
    r'|\bnon-existent\b|\bnonexistent\b'
    r'|\b(?:tests?|ci|lint|build)\s+(?:are |is )?failing\b'
    r'|\bis down\b|\bservice down\b'
    r'|\bprocess not found\b|\bnot running\b'
    r'|\bzero\s+\w+\s+today\b'
    r'|\bpoints? to (?:a |the )?(?:wrong|non-existent|nonexistent)\b',
    re.I,
)
# An evidence citation is: the explicit Evidence: marker, an inline-code span (a command),
# an exit/conclusion token, a path with a verb, or the "cause unconfirmed" escape hatch.
EVIDENCE_TOKEN = re.compile(
    r'Evidence:'
    r'|`[^`]+`'
    r'|\bexit code\b|\bexit=\b|\bconclusion\b|\breturned\b|->'
    r'|\bcause unconfirmed\b|\bUNCONFIRMED\b'
    r'|\bhermes cron list\b|\bgh run\b|\bsystemctl\b|\bls -',
    re.I,
)


def find_unbacked_claims(text: str) -> bool:
    """True if any strong causal claim lacks an evidence citation within +/-2 lines."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not STRONG_CLAIM.search(line):
            continue
        window = '\n'.join(lines[max(0, i - 2):i + 3])
        if not EVIDENCE_TOKEN.search(window):
            return True
    return False


def scan_text(text: str) -> list[str]:
    return [name for name, rx in BAD_PATTERNS.items() if rx.search(text or '')]


# Context-aware scanning. The format patterns describe what may appear in the
# DELIVERED Discord message. Two legitimate cases must not be flagged:
#   - a script that builds an HTML *file* (the message itself uses a MEDIA: tag)
#   - a prompt that quotes a forbidden pattern as a prohibition/example
PRINT_LINE = re.compile(r'\b(?:print|sys\.stdout\.write)\s*\(')
NEGATION_CTX = re.compile(
    r"\b(?:no|not|never|avoid|without|don'?t|do not|e\.g\.|example|instead of|rather than)\b",
    re.I,
)


def scan_script(text: str) -> list[str]:
    """Only stdout-reachable lines can become the delivered message. HTML written
    to a file (rows.append("<html>...") -> write_text) is not delivered, so it is
    excluded; literal HTML inside a print() still flags."""
    printed = '\n'.join(l for l in (text or '').splitlines() if PRINT_LINE.search(l))
    return scan_text(printed)


def scan_prompt(text: str) -> list[str]:
    """Prompts legitimately quote forbidden patterns as examples/prohibitions
    ("No Telegram HTML tags (<b>...)"). Flag a pattern only when the match is NOT
    in a negation/example context or inside a fenced code block."""
    # Strip fenced code blocks (```...```) before scanning — HTML templates
    # inside code blocks are file-generation instructions, not Discord-visible.
    import re as _re
    stripped = _re.sub(r'```[a-z]*\n.*?```', '', text or '', flags=_re.DOTALL)
    hits: list[str] = []
    for name, rx in BAD_PATTERNS.items():
        for mobj in rx.finditer(stripped):
            ctx = stripped[max(0, mobj.start() - 50):mobj.start()]
            if not NEGATION_CTX.search(ctx):
                hits.append(name)
                break
    return hits


def latest_response(job_id: str) -> str:
    out_dir = OUTPUT / job_id
    if not out_dir.exists():
        return ''
    files = sorted(out_dir.glob('*.md'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return ''
    text = files[0].read_text(errors='ignore')
    if '## Response' in text:
        return text.split('## Response', 1)[1].strip()
    if '\n---\n\n' in text:
        return text.split('\n---\n\n', 1)[1].strip()
    # Failed run logs include prompt summaries/errors. Don't treat them as delivered output.
    if '(FAILED)' in text[:200] or '**Status:** script failed' in text:
        return ''
    return text.strip()


def scan_visible_output(text: str) -> list[str]:
    if not text:
        return []
    if text.strip() == '[SILENT]':
        return []
    hits = scan_text(text)
    if BAD_VISIBLE_DATE.search(text):
        hits.append('non_uk_visible_date')
    if find_unbacked_claims(text):
        hits.append('unbacked_causal_claim')
    first = next((line.strip() for line in text.splitlines() if line.strip()), '')
    if first and not first.startswith('#') and not FIRST_LINE_TS.match(first):
        # Allow local-only / failure metadata logs to pass; this linter targets delivered summaries.
        hits.append('first_line_missing_ddmmyy_hhmmss')
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--latest', action='store_true', help='also scan latest historical output files')
    args = parser.parse_args()

    data = json.loads(JOBS.read_text(encoding='utf-8'))
    rows = []
    for job in data.get('jobs', []):
        if not job.get('enabled'):
            continue
        deliver = job.get('deliver') or ''
        if not (deliver.startswith('discord') or deliver in {'origin', 'all'}):
            continue

        script_hits: list[str] = []
        script = job.get('script')
        if script:
            script_path = SCRIPTS / script
            if script_path.exists():
                script_hits = scan_script(script_path.read_text(errors='ignore'))

        prompt_hits = scan_prompt(job.get('prompt') or '')
        latest_hits = scan_visible_output(latest_response(job.get('id', ''))) if args.latest else []
        if prompt_hits or script_hits or latest_hits:
            rows.append({
                'job': job.get('name'),
                'id': job.get('id'),
                'mode': 'no_agent' if job.get('no_agent') else 'llm',
                'prompt': prompt_hits,
                'script': script_hits,
                'latest_output': latest_hits,
            })

    print(json.dumps({'issues': rows, 'count': len(rows), 'latest_checked': args.latest}, indent=2))
    return 1 if rows else 0


if __name__ == '__main__':
    raise SystemExit(main())
