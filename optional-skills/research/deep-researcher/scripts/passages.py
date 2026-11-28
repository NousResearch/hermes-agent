#!/usr/bin/env python3
"""Select bounded literal-match passages from native cached text without fetching URLs."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from pathlib import Path
import re


def select(text: str, terms: list[str], max_chars: int = 4000, context: int = 200, max_passages: int = 6) -> dict:
    if max_chars <= 0 or context < 0 or max_passages <= 0:
        raise ValueError('max_chars/max_passages must be positive and context nonnegative')
    if not terms or any(not isinstance(term, str) or not term.strip() for term in terms):
        raise ValueError('Supply at least one nonempty search term')
    if '[... middle omitted — see footer ...]' in text and '[TRUNCATED]' in text:
        raise ValueError('This is a truncated native display. Use the Full text saved to path from its footer, not a cut copy, for passage selection.')
    # Regex offsets refer to the original Unicode text, unlike indexes into casefolded text.
    pattern = re.compile('|'.join(re.escape(term) for term in sorted(set(terms), key=len, reverse=True)), re.IGNORECASE)
    starts = [0, *[match.end() for match in re.finditer('\n', text)]]
    passages = []
    remaining = max_chars
    previous_end = 0
    matches = 0
    omitted = False
    for match in pattern.finditer(text):
        matches += 1
        if passages and match.end() <= previous_end:
            continue
        if len(passages) >= max_passages or remaining < match.end() - match.start():
            omitted = True
            continue
        room = remaining - (match.end() - match.start())
        before = min(context, room // 2)
        after = min(context, room - before)
        lo = max(0, match.start() - before)
        hi = min(len(text), match.end() + after)
        passages.append({'text': text[lo:hi], 'start': lo, 'end': hi,
                         'line_start': bisect_right(starts, lo),
                         'line_end': bisect_right(starts, max(lo, hi - 1)),
                         'clipped_before': lo > 0, 'clipped_after': hi < len(text)})
        remaining -= hi - lo
        previous_end = hi
    stored_truncated = '[... stored copy truncated at ' in text
    return {'status': 'matched' if matches else 'no-match', 'matches': matches,
            'input_truncated': stored_truncated,
            'omitted_matches': omitted, 'passages': passages,
            'note': ('Native stored copy is capped; omitted text remains unreviewed. ' if stored_truncated else '') +
                    'No match means absent from this extracted text, not proof that the original page lacks it.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--term', action='append', required=True)
    parser.add_argument('--max-chars', type=int, default=4000)
    parser.add_argument('--context', type=int, default=200)
    parser.add_argument('--max-passages', type=int, default=6)
    args = parser.parse_args()
    try:
        with args.cache.open(encoding='utf-8', newline='') as stream:
            text = stream.read()
        result = select(text, args.term, args.max_chars, args.context, args.max_passages)
        result['cache'] = str(args.cache)
        print(json.dumps(result, ensure_ascii=False))
    except (ValueError, OSError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
