"""Mask the text a ``browser_exec`` snippet types into a page, for display only.

A password from a vault or the user lands in the snippet as a plain string literal
(``fill_input('#pass', 'hunter2')``) that matches no secret-shape pattern, so the
progress line, verbose args and callbacks would show it in chat (#72298). Typed
values are masked whatever they look like; selectors, URLs and the rest of the code
stay readable. A typed value passed as a name (``pw = 'x'; fill_input(sel, pw)``) has
the literal assigned to that name masked instead. Values typed one key at a time
(``Input.dispatchKeyEvent`` in a loop) are not recognised. The tool itself always
receives the untouched code.
"""
import re

_FILL_INPUT = re.compile(r"\bfill_input\s*\(\s*")
_FILL_INPUT_BARE_SELECTOR = re.compile(r"[^,()\n]+?(?=,)")
_FILL_INPUT_VALUE = re.compile(r"\s*,\s*(?:text\s*=\s*)?")
_TYPE_TEXT = re.compile(r"\btype_text\s*\(\s*(?:text\s*=\s*)?")
# el.value = "value" (JS), the generic way a snippet sets an input; never a == / === comparison
_VALUE_ASSIGN = re.compile(r"\.value\s*=(?!=)\s*")
# CDP Input.insertText {"text": "value"} / text="value", read inside that call only
_INSERT_TEXT = re.compile(r"insertText")
_TEXT_KEY = re.compile(r"""(?:\\?["']text\\?["']\s*:\s*|\btext\s*=\s*)""")
_NAME_ARG = re.compile(r"[A-Za-z_]\w*(?=\s*[,)])")
# a pre-built dict handed to insertText: **params, {**params} or a positional params
_DICT_ARG = re.compile(r"(?:\*\*\s*|,\s*)([A-Za-z_]\w*)(?=\s*[,)}])")
_STR_PREFIX = re.compile(r"[fFrRbBuU]{1,2}(?=[\"'`\\])")
_QUOTES = ("'''", '"""', "\\'", '\\"', "\\`", "'", '"', "`")


def _literal_end(code: str, start: int) -> tuple[str, str, int] | None:
    """(prefix, quote, end) of the string literal at ``start``; ``prefix`` is an ``f``/``r``/``b``
    marker. Backslash escapes never end it; one that never closes runs to the line end (an unparsed
    value fails closed). None when no literal starts here."""
    pm = _STR_PREFIX.match(code, start)
    prefix = pm.group(0) if pm else ""
    qstart = start + len(prefix)
    quote = next((q for q in _QUOTES if code.startswith(q, qstart)), None)
    if quote is None:
        return None
    i, multiline = qstart + len(quote), len(quote) == 3
    while i < len(code):
        if code.startswith(quote, i):
            return prefix, quote, i + len(quote)
        if code[i] == "\n" and not multiline:
            break
        i += 2 if code[i] == "\\" else 1
    return prefix, "", i


def _fill_input_values(code: str):
    for m in _FILL_INPUT.finditer(code):
        sel = _literal_end(code, m.end())
        if sel is not None:
            sel_end = sel[2]
        else:
            bare = _FILL_INPUT_BARE_SELECTOR.match(code, m.end())
            if bare is None:
                continue
            sel_end = bare.end()
        sep = _FILL_INPUT_VALUE.match(code, sel_end)
        if sep is not None:
            yield sep.end()


def _call_end(code: str, i: int, pair: str = "()") -> int:
    """Index of the bracket closing the ``pair`` (call or dict) that ``i`` sits inside, skipping
    string literals."""
    depth, step = 1, {pair[0]: 1, pair[1]: -1}
    while i < len(code):
        lit = _literal_end(code, i) if code[i] in "'\"`\\" else None
        if lit is not None:
            i = max(lit[2], i + 1)
            continue
        depth += step.get(code[i], 0)
        if depth == 0:
            return i
        i += 1
    return i


def _insert_text_values(code: str):
    for m in _INSERT_TEXT.finditer(code):
        start = m.end()
        while start < len(code) and code[start] in "'\"`\\":
            start += 1  # the quote closing the 'Input.insertText' method literal
        window = code[:_call_end(code, start)]
        yield from (k.end() for k in _TEXT_KEY.finditer(window, start))
        args = _DICT_ARG.finditer(code, start, len(window) + 1)  # + the closing ")"
        for name in {a.group(1) for a in args}:
            yield from _dict_text_values(code, name)


def _dict_text_values(code: str, name: str):
    """Value starts of the ``text`` keys in each dict literal assigned to ``name``."""
    assign = re.compile(rf"(?<![\w.]){re.escape(name)}\s*=(?!=)\s*\{{")
    for m in assign.finditer(code):
        window = code[:_call_end(code, m.end(), "{}")]
        yield from (k.end() for k in _TEXT_KEY.finditer(window, m.end()))


def _typed_value_starts(code: str) -> list[int]:
    starts = list(_fill_input_values(code))
    starts += [m.end() for pat in (_TYPE_TEXT, _VALUE_ASSIGN) for m in pat.finditer(code)]
    starts += _insert_text_values(code)
    names = {n.group(0) for s in starts if (n := _NAME_ARG.match(code, s))}
    for name in names:
        assign = re.compile(rf"(?<![\w.]){re.escape(name)}\s*=(?!=)\s*")
        starts += [m.end() for m in assign.finditer(code)]
    return sorted(set(starts))


def mask_typed_browser_text(code: str) -> str:
    """``code`` with every value it types into a page replaced by ``***``."""
    out, pos = [], 0
    for start in _typed_value_starts(code):
        lit = None if start < pos else _literal_end(code, start)
        if lit is None:
            continue
        prefix, quote, end = lit
        out.append(f"{code[pos:start]}{prefix}{quote}***{quote}")
        pos = end
    return "".join(out) + code[pos:]
