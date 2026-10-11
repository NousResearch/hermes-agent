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


# A block-math span ($$...$$ with the delimiters each alone on their own line)
# also renders natively in the rich path, so a hard break injected between the
# rows of a multi-line aligned/cases/matrix environment is invalid LaTeX. This
# is matched only *within prose* — i.e. outside the fenced-code/table regions
# above — so a stray or inline ``$$`` can never pair across a protected region
# (which would otherwise let a ``$$`` inside a code block close a span and let
# the rest of the block receive hard breaks). The line-anchored delimiters also
# keep inline/currency ``$$`` out: only canonical display-math blocks match.
_RICH_BLOCK_MATH_RE = re.compile(
    r'^[ \t]*\$\$[ \t]*\n[\s\S]*?\n[ \t]*\$\$[ \t]*$',
    re.MULTILINE,
)


def _rich_hard_break_prose(prose: str) -> str:
    """Inject Markdown hard breaks into a prose run, leaving block-math spans
    ($$...$$ on their own lines) bare. Called only for prose between the
    fenced-code/table protected regions, so block math is recognized outside
    those regions and can never span into one."""
    out: list[str] = []
    pos = 0
    for m in _RICH_BLOCK_MATH_RE.finditer(prose):
        out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', prose[pos:m.start()]))
        out.append(m.group(0))  # block math kept verbatim
        pos = m.end()
    out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', prose[pos:]))
    return ''.join(out)


def _rich_normalize_linebreaks(text: str) -> str:
    """Convert lone ``\\n`` (a Markdown soft break) to hard breaks for sendRichMessage; ``\\n\\n``,
    fenced code, pipe tables and canonical display math are left untouched."""
    if not text or '\n' not in text:
        return text
    out: list[str] = []
    pos = 0
    for m in _RICH_PROTECTED_REGION_RE.finditer(text):
        out.append(_rich_hard_break_prose(text[pos:m.start()]))
        out.append(m.group(0))  # protected region kept verbatim
        pos = m.end()
    out.append(_rich_hard_break_prose(text[pos:]))
    return ''.join(out)
