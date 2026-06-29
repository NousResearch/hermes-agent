#!/usr/bin/env python3
"""Deterministic groundedness pre-scan for wiki pages.

Performs BM25-like keyword overlap between a wiki page's claims and its
source document(s). Returns candidate claims that have zero or weak source
correlation — these are escalated to LLM for final adjudication.

The script does NOT make quality decisions. It identifies potential issues
for a human or LLM to review.

Usage:
  python3 verify_claims.py wiki/path/to/page.md          # Single page
  python3 verify_claims.py --recent 7 wiki/               # Pages modified in last 7 days
  python3 verify_claims.py --json wiki/path/to/page.md    # JSON output

Output: JSON array of {claim, source_match_score, suggestion}
  score=0.0: no keyword overlap with source → highly suspicious
  score<0.3: weak overlap → candidate for review
  score>=0.3: reasonable overlap → likely grounded
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path


def extract_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from wiki page."""
    if not text.startswith('---'):
        return {}, text
    parts = text.split('---', 2)
    if len(parts) < 3:
        return {}, text
    try:
        import yaml
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        fm = {}
    return fm, parts[2]


def extract_claims(text: str) -> list[str]:
    """Extract claim-worthy sentences from wiki page body.

    Claims are sentences that are not headings, not wikilinks-only,
    not quality flags, and longer than 30 characters.
    """
    # Remove code blocks, YAML blocks, markdown formatting
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'^#.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[\[.*?\]\]', '', text)  # wikilinks
    text = re.sub(r'> \^Quality flag:.*$', '', text, flags=re.MULTILINE)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    claims = []
    for s in sentences:
        s = s.strip()
        # Filter: must be substantive (not a heading fragment, not a list bullet alone)
        if len(s) > 30 and not s.startswith('- [') and not s.startswith('|'):
            claims.append(s)
    return claims


def tokenize(text: str) -> set[str]:
    """Simple keyword tokenization — lowercase, >3 chars, remove stopwords."""
    stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they',
                 'their', 'will', 'would', 'could', 'should', 'about', 'also',
                 'which', 'when', 'where', 'what', 'them', 'then', 'than'}
    words = re.findall(r'[a-zA-Z]{4,}', text.lower())
    return {w for w in words if w not in stopwords}


def keyword_overlap(claim: str, source_text: str) -> float:
    """Compute keyword overlap ratio between claim and source.

    Returns 0.0 to 1.0. Higher = more likely grounded.
    """
    claim_tokens = tokenize(claim)
    if not claim_tokens:
        return 0.0
    source_tokens = tokenize(source_text)
    if not source_tokens:
        return 0.0
    overlap = claim_tokens & source_tokens
    return len(overlap) / len(claim_tokens)


def find_source_files(wiki_page_path: str, fm: dict) -> list[str]:
    """Find source files referenced by a wiki page."""
    sources = fm.get('sources', [])
    if not sources:
        return []
    wiki_dir = Path(wiki_page_path).parent.parent  # entities/ or concepts/ → wiki root
    source_paths = []
    for s in sources:
        sp = Path(wiki_dir) / s
        if sp.exists():
            source_paths.append(str(sp))
    return source_paths


def verify_page(wiki_page_path: str) -> list[dict]:
    """Verify a single wiki page against its sources."""
    path = Path(wiki_page_path)
    if not path.exists():
        return [{'claim': f'FILE NOT FOUND: {wiki_page_path}',
                 'score': -1.0,
                 'suggestion': 'Verify the file path and name.'}]

    text = path.read_text(encoding='utf-8')
    fm, body = extract_frontmatter(text)
    source_files = find_source_files(wiki_page_path, fm)

    if not source_files:
        return [{'claim': 'No source files found in frontmatter sources field',
                 'score': -1.0,
                 'suggestion': 'Add source references to the page frontmatter.'}]

    # Concatenate all source texts
    source_text = ''
    for sf in source_files:
        try:
            source_text += Path(sf).read_text(encoding='utf-8') + '\n'
        except Exception:
            continue

    if not source_text.strip():
        return []

    claims = extract_claims(body)
    candidates = []
    for claim in claims:
        score = keyword_overlap(claim, source_text)
        if score < 0.3:
            suggestion = (
                'No significant keyword overlap with source. '
                f'Verify this claim is supported by the referenced source(s).'
                if score < 0.1 else
                'Weak keyword overlap with source. Consider adding more specific citations.'
            )
            candidates.append({
                'claim': claim[:200],
                'score': round(score, 2),
                'suggestion': suggestion
            })
    return candidates


def find_recent_pages(wiki_dir: str, days: int = 7) -> list[str]:
    """Find wiki pages modified in the last N days."""
    cutoff = time.time() - days * 86400
    pages = []
    for subdir in ['entities', 'concepts', 'comparisons', 'queries']:
        sp = Path(wiki_dir) / subdir
        if not sp.exists():
            continue
        for f in sp.rglob('*.md'):
            if f.stat().st_mtime > cutoff:
                pages.append(str(f))
    return pages


def main():
    parser = argparse.ArgumentParser(
        description='Deterministic groundedness pre-scan for wiki pages'
    )
    parser.add_argument('target', nargs='?', help='Wiki page path or wiki directory')
    parser.add_argument('--recent', type=int, metavar='DAYS',
                       help='Scan pages modified in the last N days')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')
    args = parser.parse_args()

    if not args.target:
        parser.print_help()
        sys.exit(1)

    if args.recent:
        pages = find_recent_pages(args.target, args.recent)
        if not pages:
            print(json.dumps({'message': f'No pages modified in the last {args.recent} days'}))
            sys.exit(0)
    else:
        pages = [args.target]

    all_candidates = {}
    for page in pages:
        candidates = verify_page(page)
        if candidates:
            all_candidates[page] = candidates

    if args.json:
        print(json.dumps(all_candidates, indent=2, ensure_ascii=False))
    else:
        for page, candidates in all_candidates.items():
            print(f'\n=== {page} ===')
            for i, c in enumerate(candidates, 1):
                print(f'  [{i}] score={c["score"]:.2f} | {c["claim"][:100]}')
                print(f'      → {c["suggestion"]}')
        if not all_candidates:
            print('No candidate issues found.')
        print(f'\nScanned {len(pages)} page(s). {len(all_candidates)} page(s) flagged.')


if __name__ == '__main__':
    main()
