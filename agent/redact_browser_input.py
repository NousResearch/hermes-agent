"""Mask the text a ``browser_exec`` snippet types into a page, for display only.

A password from a vault or the user lands in the snippet as a plain string literal
(``fill_input('#pass', 'hunter2')``) that matches no secret-shape pattern, so the
progress line, verbose args and callbacks would show it in chat (#72298). Typed
values are masked whatever they look like; selectors, URLs and the rest of the code
stay readable. The tool itself always receives the untouched code.
"""
import re

# One string literal on one line; the quote may be backslash-escaped when JS sits inside a Python string.
_LIT = r"""(?P<q>\\?["'`])(?:(?!(?P=q))[^\n])*(?P=q)"""
_TYPED_VALUE_RES = (
    # fill_input(selector, "value") / fill_input(selector, text="value")
    re.compile(rf"(\bfill_input\s*\(\s*[^,()\n]+?,\s*(?:text\s*=\s*)?){_LIT}"),
    # el.value = "value" (JS), the generic way a snippet sets an input
    re.compile(rf"(\.value\s*=\s*){_LIT}"),
    # CDP Input.insertText {"text": "value"} / text="value"
    re.compile(rf"""((?:\\?["']text\\?["']\s*:\s*|\btext\s*=\s*)){_LIT}"""),
)


def _mask_literal(m: "re.Match[str]") -> str:
    q = m.group("q")
    return f"{m.group(1)}{q}***{q}"


def mask_typed_browser_text(code: str) -> str:
    """``code`` with every value it types into a page replaced by ``***``."""
    for pattern in _TYPED_VALUE_RES:
        if pattern is _TYPED_VALUE_RES[2] and "insertText" not in code:
            continue
        code = pattern.sub(_mask_literal, code)
    return code
