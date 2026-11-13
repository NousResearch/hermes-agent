"""Mask the text a ``browser_exec`` snippet types into a page, for display only.

A password from a vault or the user lands in the snippet as a plain string literal
(``fill_input('#pass', 'hunter2')``) that matches no secret-shape pattern, so the
progress line, verbose args and callbacks would show it in chat (#72298). Typed
values are masked whatever they look like; selectors, URLs and the rest of the code
stay readable. The snippet is parsed as Python, so the typed argument is found by
keyword or position whatever its spelling (``('pw')``, ``'' 'pw'``, ``text=..., selector=...``,
f-strings, ``'a' + pw``, ``*args``/``**{'text': ...}`` splats); a name it uses has the literals
bound to that name masked too, whether assigned (``pw = 'x'``, ``sel, pw = '#p', 'x'``), looped
over (``for pw in ['x']``) or passed to a function the snippet defines (``def go(pw): ...``;
``go('x')``). Code that does not parse masks every literal inside each typing call (fail closed),
and a snippet the walk cannot finish has every string literal masked. Not recognised: values
typed one key at a time (``Input.dispatchKeyEvent`` in a loop), and values reached through
attributes, subscripts, imports or functions the snippet does not define. The tool itself always receives the untouched code.
"""
import ast
import re

# helper -> position of its typed value when passed positionally
_TYPING_CALLS = {"fill_input": 1, "type_text": 0}
_TYPING_CALL_RE = re.compile(r"\b(?:fill_input|type_text)\s*\(|insertText")
_MASK = "'***'"
# el.value = "value" (JS), the generic way a snippet sets an input; never a == / === comparison
_VALUE_ASSIGN = re.compile(r"\.value\s*=(?!=)\s*")
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


def _literal_mask(code: str, start: int) -> tuple[int, int, str] | None:
    lit = _literal_end(code, start)
    if lit is None:
        return None
    prefix, quote, end = lit
    return start, end, f"{prefix}{quote}***{quote}"


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


# A source line with its terminator: Python's tokenizer ends a line at \r\n, \r or \n alone.
_SOURCE_LINE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+")


class _Spans:
    """Char offsets of AST nodes in ``code`` (AST columns count UTF-8 bytes), with the line breaks
    the parser counts, so a bare ``\r`` cannot shift a span onto another line's text."""

    def __init__(self, code: str):
        self.lines = _SOURCE_LINE.findall(code) + [""]
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def _offset(self, lineno: int, col: int) -> int:
        return self.starts[lineno - 1] + len(self.lines[lineno - 1].encode()[:col].decode(errors="ignore"))

    def of(self, node: ast.AST) -> tuple[int, int]:
        return (self._offset(node.lineno, node.col_offset),
                self._offset(node.end_lineno, node.end_col_offset))


def _bind(out: dict, target: ast.AST, value: ast.AST) -> None:
    """Bind every name in *target* to *value*; a tuple target takes the whole value (fail closed)."""
    for node in ast.walk(target):
        if isinstance(node, ast.Name):
            out.setdefault(node.id, []).append(value)


def _call_bindings(tree: ast.AST, out: dict) -> None:
    """A parameter of a function defined in the snippet is bound to every argument passed to it."""
    params = {node.name: node.args for node in ast.walk(tree)
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in params):
            continue
        args = params[node.func.id]
        names = [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]
        names += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
        for value in [*node.args, *(kw.value for kw in node.keywords)]:
            for name in names:
                out.setdefault(name, []).append(value)


def _name_bindings(tree: ast.AST) -> dict[str, list[ast.AST]]:
    """Every value bound to each plain name anywhere in the snippet: assignments (tuple targets
    included), loop and comprehension variables, ``with ... as``, and parameters of functions the
    snippet defines and calls."""
    out: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _bind(out, target, node.value)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) and node.value is not None:
            _bind(out, node.target, node.value)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            _bind(out, node.target, node.iter)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            _bind(out, node.optional_vars, node.context_expr)
    _call_bindings(tree, out)
    return out


def _is_insert_text(node: ast.Call) -> bool:
    return any(isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.endswith("insertText")
               for a in [*node.args, *(kw.value for kw in node.keywords)])


class _TypedValues(ast.NodeVisitor):
    """The source spans of every string a parsed snippet types into a page."""

    def __init__(self, tree: ast.AST, code: str):
        self.spans, self.bindings = _Spans(code), _name_bindings(tree)
        self.out: list[tuple[int, int, str]] = []
        self._followed: set[str] = set()
        self._dict_followed: set[str] = set()  # params = {**params} rebinds a name to itself

    def _strings(self, expr: ast.AST) -> None:
        """Mask each string constant in *expr* (an f-string whole); a name has its bound values masked."""
        for node in ast.walk(expr):
            if isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                self.out.append((*self.spans.of(node), _MASK))
            elif isinstance(node, ast.Name) and node.id not in self._followed:
                self._followed.add(node.id)
                for bound in self.bindings.get(node.id, ()):
                    self._strings(bound)

    def _dict_text(self, expr: ast.AST) -> None:
        """Mask the ``text`` entry of a params dict given literally, by name or spread."""
        if isinstance(expr, ast.Dict):
            for key, value in zip(expr.keys, expr.values):
                if key is None:
                    self._dict_text(value)
                elif isinstance(key, ast.Constant) and key.value == "text":
                    self._strings(value)
        elif isinstance(expr, ast.Name) and expr.id not in self._dict_followed:
            self._dict_followed.add(expr.id)
            for bound in self.bindings.get(expr.id, ()):
                self._dict_text(bound)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name in _TYPING_CALLS:
            pos = _TYPING_CALLS[name]
            for i, arg in enumerate(node.args):
                if i == pos or isinstance(arg, ast.Starred):  # a splat may carry the value
                    self._strings(arg)
            for kw in node.keywords:
                if kw.arg == "text":
                    self._strings(kw.value)
                elif kw.arg is None:
                    self._dict_text(kw.value)
        elif _is_insert_text(node):
            for kw in node.keywords:
                (self._strings if kw.arg == "text" else self._dict_text)(kw.value)
            for arg in node.args:
                self._dict_text(arg)
        self.generic_visit(node)


def _literal_regions(code: str, start: int, end: int) -> list[tuple[int, int, str]]:
    """A mask for every string literal in code[start:end], found by scanning the raw text."""
    out, i = [], start
    while i < end:
        lit = _literal_mask(code, i) if code[i] in "'\"`\\" else None
        if lit is None:
            i += 1
            continue
        out.append(lit)
        i = max(lit[1], i + 1)
    return out


def _unparsed_masks(code: str) -> list[tuple[int, int, str]]:
    """Every literal inside each typing call region of a snippet that does not parse."""
    return [lit for m in _TYPING_CALL_RE.finditer(code)
            for lit in _literal_regions(code, m.end(), _call_end(code, m.end()))]


def _typed_masks(code: str) -> list[tuple[int, int, str]]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        masks = _unparsed_masks(code)
    else:
        visitor = _TypedValues(tree, code)
        visitor.visit(tree)
        masks = visitor.out
    return masks + [lit for m in _VALUE_ASSIGN.finditer(code) if (lit := _literal_mask(code, m.end()))]


def mask_typed_browser_text(code: str) -> str:
    """``code`` with every value it types into a page replaced by ``***``. A snippet the walk
    cannot finish has every string literal masked by a raw-text scan that shares nothing with the
    walk, so display masking never raises before the browser call dispatches."""
    try:
        masks = _typed_masks(code)
    except Exception:  # health: allow BLE001 -- display masking must never block the call; mask everything
        masks = _literal_regions(code, 0, len(code))
    out, pos = [], 0
    for start, end, replacement in sorted(masks, key=lambda s: (s[0], -s[1])):
        if start < pos:  # inside a span already masked
            continue
        out.append(code[pos:start] + replacement)
        pos = end
    return "".join(out) + code[pos:]
