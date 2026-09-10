"""Markdown newline normalization for Telegram rich-message payloads."""

import re


# Rich-message regions whose internal newlines must stay bare (Telegram renders them natively):
# fenced code blocks OR GFM pipe-table blocks (header row, delimiter row, data rows).
_RICH_PROTECTED_REGION_RE = re.compile(
    r'(?:```[^\n]*\n[\s\S]*?```)'                       # fenced code block
    r'|(?:^[^\n]*\|[^\n]*\n'                            # table header row (has a pipe)
    r'[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*'  # delimiter
    r'(?:\n[^\n]*\|[^\n]*)*)',                          # data rows (newline-led, trailing \n left for prose)
    re.MULTILINE)


def _rich_normalize_linebreaks(text: str) -> str:
    """Convert lone ``\\n`` (a Markdown soft break) to hard breaks for sendRichMessage; ``\\n\\n``,
    fenced code and pipe tables are left untouched."""
    if not text or '\n' not in text:
        return text
    out: list[str] = []
    pos = 0
    for m in _RICH_PROTECTED_REGION_RE.finditer(text):
        out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', text[pos:m.start()]))
        out.append(m.group(0))  # protected region kept verbatim
        pos = m.end()
    out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', text[pos:]))
    return ''.join(out)
