"""Review signals are bounded heuristics, not named-entity recognition or truth tests."""
from __future__ import annotations

import calendar
import hashlib
import re

SIGNALS = {'finality': re.compile(r'\b(final|finalised|finalized|definitive)\b', re.IGNORECASE),
           'independence': re.compile(r'\b(independen\w*|corroborat\w*)\b', re.IGNORECASE)}
ENTITY = re.compile(r'\b(?:[A-Z][a-z]+(?:/[A-Za-z]+)+|[A-Z][A-Z0-9]{1,}|[A-Z][a-z]{2,}(?:[ -]+[A-Z][a-z]{2,})*)\b')
ORDINARY = {name.casefold() for name in (*calendar.month_name, *calendar.month_abbr)} | {
    'german', 'french', 'european', 'british', 'italian', 'spanish', 'danish', 'dutch',
    'polish', 'austrian', 'swedish', 'norwegian', 'finnish', 'swiss', 'portuguese'}


def normalized(text: str) -> str:
    return ' '.join(text.casefold().split())


def ordinary(text: str) -> bool:
    return normalized(text) in ORDINARY or bool(re.fullmatch(r'(?:H[12]|Q[1-4]|FY[0-9]{2,4})', text))



def label_spans(line: str) -> list[tuple[int, int]]:
    spans = []
    if line.lstrip().startswith('>'):
        spans.extend(match.span() for match in re.finditer(r'\b(?:Mode|Sources|Scope|As-of)(?:\*\*)?\s*:', line))
    spans.extend(match.span() for match in re.finditer(
        r'\bConfidence(?:\*\*)?\s*:\s*(?:\*\*)?(?:High|Medium|Low)(?:[–-](?:High|Medium|Low))?', line))
    label = re.match(r'^\s*(?:[0-9]+\.\s*|[-*]\s*)?\*\*([^*\n]+)\*\*', line)
    if label:
        words = label.group(1).strip(': .').split()
        ordinary_labels = {'Price', 'Policy', 'Affordability', 'Installer', 'Demand', 'Confidence'}
        if words and words[0] in ordinary_labels and not any(word[:1].isupper() for word in words[1:]):
            spans.append(label.span())
    return spans


def known_name(text: str, known: str) -> bool:
    names = text.split('/')
    return all(re.search(r'(?<!\w)' + re.escape(normalized(name)) + r'(?!\w)', known) for name in names)

def assertions(report: str, data: dict) -> list[dict]:
    known = normalized(' '.join(str(source.get(key, '')) for source in data['sources']
                               for key in ('title', 'publisher', 'author')) + ' ' +
                       ' '.join(e['excerpt'] for source in data['sources'] for e in source.get('evidence', [])))
    found = []
    in_fence = False
    for number, line in enumerate(report.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith('```'):
            in_fence = not in_fence
        if stripped.lower() in {'## references', '## citation links', '## bibliography'}:
            break
        if in_fence or stripped.startswith('#') or stripped.startswith('*Prepared with the Hermes Deep Researcher workflow.'):
            continue
        for kind, pattern in SIGNALS.items():
            matches = list(pattern.finditer(line))
            if matches:
                found.append(_assertion(kind, matches[0].group(), number, line))
        labels = label_spans(line)
        for match in ENTITY.finditer(line):
            if any(start <= match.start() and match.end() <= end for start, end in labels):
                continue
            text = re.sub(r'^(?:The|This|These|Those|Our|Their|Its) ', '', match.group())
            # Sentence-initial capitalization is not a named-entity assertion.
            initial = not line[:match.start()].strip(' -*|') or line[:match.start()].rstrip().endswith(('.', '!', '?'))
            if known_name(text, known) or ordinary(text) or (initial and len(text.split()) == 1 and not text.isupper() and '/' not in text):
                continue
            found.append(_assertion('entity', text, number, line))
    return list({(item['kind'], item['line'], normalized(item['text'])): item for item in found}.values())



def _assertion(kind: str, text: str, line: int, context: str) -> dict:
    key = hashlib.sha256(f'{kind}\0{line}\0{text}'.encode()).hexdigest()[:16]
    return {'id': key, 'kind': kind, 'text': text, 'line': line, 'context': context,
            'status': 'pending', 'evidence_ids': [], 'note': '', 'qualification': ''}

