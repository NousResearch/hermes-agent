"""Markdown spans the reply media extractors share (split from :mod:`gateway.platforms.base`):
code blocks and inline code, which hold quoted text and never a delivery directive, and the image
tags ``extract_images`` turns into attachments (the streamed display hides the same tags)."""
import re

MD_IMAGE_TAG_PATTERN = r'!\[([^\]]*)\]\((https?://[^\s\)]+)\)'
HTML_IMAGE_TAG_PATTERN = r'<img\s+src=["\']?(https?://[^\s"\'<>]+)["\']?\s*/?>\s*(?:</img>)?'
_IMAGE_URL_MARKERS = ('.png', '.jpg', '.jpeg', '.gif', '.webp', 'fal.media', 'fal-cdn', 'replicate.delivery')

# Code blocks as CommonMark reads them. A run of 3+ backticks opening a line (any indent, so fences
# nested in list items count; no backtick in the info string) closes only at a line holding nothing
# but a run at least as long: a ``` mid-line, or a shorter run inside a longer fence, is quoted text
# and cannot end the block to expose what follows. Unclosed, the block runs to the end of the text,
# as it renders, so a fence still streaming is already code. A run opening mid-line is a code span
# that closes at the next run of exactly its length.
_FENCED_CODE_RE = re.compile(
    r'^[ \t]*(`{3,})[^`\n]*\n.*?(?:^[ \t]*\1`*[ \t]*\r?$|\Z)'
    r'|(?<!`)(`{3,})(?!`)[^\n]*\n.*?(?<!`)\2(?!`)',
    re.DOTALL | re.MULTILINE)
_INLINE_CODE_RE = re.compile(r'`[^`\n]+`')


def _code_spans(content: str) -> list:
    """(start, end) spans of fenced code blocks and inline code in ``content``."""
    return [m.span() for rx in (_FENCED_CODE_RE, _INLINE_CODE_RE) for m in rx.finditer(content)]


def _blank_spans(text: str, spans: list) -> str:
    """Replace every non-newline char inside ``spans`` with a space (offsets preserved)."""
    chars = list(text)
    for start, end in spans:
        chars[start:end] = [c if c == '\n' else ' ' for c in chars[start:end]]
    return ''.join(chars)


def _delete_spans(text: str, spans: list) -> str:
    """Delete ``spans`` from ``text``, merging overlapping/nested ones first so multi-pattern
    matches over the same tag never double-delete adjacent text."""
    if not spans:
        return text
    merged: list = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    chars = list(text)
    for start, end in reversed(merged):
        del chars[start:end]
    return "".join(chars)


def mask_code_spans(content: str) -> str:
    """``content`` with fenced and inline code blanked, offsets preserved."""
    return _blank_spans(content, _code_spans(content))


def deliverable_image_tags(content: str) -> list[tuple[int, int, str, str]]:
    """``(start, end, url, alt)`` of the image tags ``extract_images`` delivers: markdown tags
    whose URL looks like an image, then ``<img src>`` tags. A tag in fenced or inline code is
    quoted text (markdown renders code literally), never an attachment."""
    masked = mask_code_spans(content)
    tags = [(m.start(), m.end(), m.group(2), content[m.start(1):m.end(1)])
            for m in re.finditer(MD_IMAGE_TAG_PATTERN, masked)
            if any(marker in m.group(2).lower() for marker in _IMAGE_URL_MARKERS)]
    tags.extend((m.start(), m.end(), m.group(1), "") for m in re.finditer(HTML_IMAGE_TAG_PATTERN, masked))
    return tags
