import re


_URL_RE = re.compile(r"https?://\S+")
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_PATH_RE = re.compile(r"(?:~?/|\.?\./|/)[^\s`]+")
_BULLET_RE = re.compile(r"^\s*[-*]\s+", re.M)
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+", re.M)
_FILLER_PATTERNS = [
    (re.compile(r"\bplease\b", re.I), ""),
    (re.compile(r"\bcarefully\b", re.I), ""),
    (re.compile(r"\bsimply\b", re.I), ""),
    (re.compile(r"\bbasically\b", re.I), ""),
    (re.compile(r"\bjust\b", re.I), ""),
    (re.compile(r"\bhelpfully\b", re.I), ""),
    (re.compile(r"\bin order to\b", re.I), "to"),
    (re.compile(r"\bdo not forget to\b", re.I), "remember to"),
    (re.compile(r"\bmake sure to\b", re.I), "ensure"),
    (re.compile(r"\bit is important to note that\b", re.I), "note:"),
    (re.compile(r"\bthe following\b", re.I), ""),
    (re.compile(r"\bshould be followed\b", re.I), "apply"),
]


def _protect(text: str):
    protected = []

    def repl(match):
        protected.append(match.group(0))
        return f"__CTXPROT_{len(protected)-1}__"

    for pattern in (_URL_RE, _INLINE_CODE_RE, _PATH_RE):
        text = pattern.sub(repl, text)
    return text, protected


def _restore(text: str, protected):
    for i, value in enumerate(protected):
        text = text.replace(f"__CTXPROT_{i}__", value)
    return text


def compact_context_prose(text: str) -> str:
    if not text or len(text) < 80:
        return text
    if _BULLET_RE.search(text) or _NUMBERED_RE.search(text):
        return text
    if text.lstrip().startswith("## "):
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            head, body = parts
            compact_body = compact_context_prose(body)
            return head + "\n\n" + compact_body
        return text

    working, protected = _protect(text)
    original = working

    for pattern, replacement in _FILLER_PATTERNS:
        working = pattern.sub(replacement, working)

    working = re.sub(r"[ \t]{2,}", " ", working)
    working = re.sub(r" ?\n ?", "\n", working)
    working = re.sub(r"\n{3,}", "\n\n", working)
    working = re.sub(r"\s+([,.;:])", r"\1", working)
    working = re.sub(r"\(\s+", "(", working)
    working = re.sub(r"\s+\)", ")", working)
    working = re.sub(r"\bNote:\s*note:\b", "note:", working, flags=re.I)
    working = re.sub(r"\s{2,}", " ", working).strip()

    if not working or len(working) >= len(original):
        return text

    restored = _restore(working, protected)
    if len(restored) >= len(text):
        return text
    return restored
