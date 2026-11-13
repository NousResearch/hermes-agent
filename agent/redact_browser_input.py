"""Mask the text a ``browser_exec`` snippet types into a page, for display only.

A password from a vault or the user lands in the snippet as a plain string literal
(``fill_input('#pass', 'hunter2')``) that matches no secret-shape pattern, so the
progress line, verbose args and callbacks would show it in chat (#72298). Typed
values are masked whatever they look like; selectors, URLs and the rest of the code
stay readable. The tool itself always receives the untouched code.
"""
import re

_TYPED_VALUE_PREFIXES = (
    # fill_input(selector, "value") / fill_input(selector, text="value")
    re.compile(r"\bfill_input\s*\(\s*[^,()\n]+?,\s*(?:text\s*=\s*)?"),
    # el.value = "value" (JS), the generic way a snippet sets an input
    re.compile(r"\.value\s*=\s*"),
    # CDP Input.insertText {"text": "value"} / text="value"
    re.compile(r"""(?:\\?["']text\\?["']\s*:\s*|\btext\s*=\s*)"""),
)
_QUOTES = ("'''", '"""', "\\'", '\\"', "\\`", "'", '"', "`")


def _literal_end(code: str, start: int) -> tuple[str, int] | None:
    """(quote, end) of the string literal at ``start``. Backslash escapes never end it; one that never
    closes runs to the line end (an unparsed value fails closed). None when no literal starts here."""
    quote = next((q for q in _QUOTES if code.startswith(q, start)), None)
    if quote is None:
        return None
    i, multiline = start + len(quote), len(quote) == 3
    while i < len(code):
        if code.startswith(quote, i):
            return quote, i + len(quote)
        if code[i] == "\n" and not multiline:
            break
        i += 2 if code[i] == "\\" else 1
    return "", i


def _mask_after(pattern: "re.Pattern[str]", code: str) -> str:
    out, pos = [], 0
    for m in pattern.finditer(code):
        lit = None if m.start() < pos else _literal_end(code, m.end())
        if lit is None:
            continue
        quote, end = lit
        out.append(f"{code[pos:m.end()]}{quote}***{quote}")
        pos = end
    return "".join(out) + code[pos:]


def mask_typed_browser_text(code: str) -> str:
    """``code`` with every value it types into a page replaced by ``***``."""
    for pattern in _TYPED_VALUE_PREFIXES:
        if pattern is _TYPED_VALUE_PREFIXES[2] and "insertText" not in code:
            continue
        code = _mask_after(pattern, code)
    return code
