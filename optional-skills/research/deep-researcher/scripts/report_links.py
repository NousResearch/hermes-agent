"""Extract report URL destinations without cutting balanced Markdown parentheses."""
from __future__ import annotations

import re

MARKDOWN_START = re.compile(r'\]\(\s*(<?https?://)')
BARE_URL = re.compile(r'https?://[^\s<>]+')


def _destination(text: str, start: int) -> str | None:
    if text[start] == '<':
        end = text.find('>', start + 1)
        return text[start + 1:end] if end >= 0 else None
    result = []
    depth = 0
    escaped = False
    for character in text[start:]:
        if escaped:
            result.append(character)
            escaped = False
            continue
        if character == '\\':
            escaped = True
        elif character == '(':
            depth += 1
            result.append(character)
        elif character == ')':
            if depth == 0:
                return ''.join(result)
            depth -= 1
            result.append(character)
        elif character.isspace() and depth == 0:
            return ''.join(result)
        else:
            result.append(character)
    return None


def first_url(text: str) -> str | None:
    for match in MARKDOWN_START.finditer(text):
        url = _destination(text, match.start(1))
        if url:
            return url
    match = BARE_URL.search(text)
    if not match:
        return None
    url = match.group().rstrip('.,;')
    while url.endswith(')') and url.count(')') > url.count('('):
        url = url[:-1]
    return url
