"""A small evaluator for the GitHub Actions expression subset our workflows gate on.

Enough to replay ``detect`` -> ``ci.yaml`` job ``if:``/``with:`` -> a called
workflow's job ``if:`` and its ``${{ }}``-rendered step text, so a test can feed
the real classifier's output through the real workflow files instead of
asserting booleans in isolation. Anything outside the subset raises
:class:`Unsupported`: a new construct must be modelled here, never ignored.

Semantics follow the Actions docs: ``&&``/``||`` return an operand, ``!``
negates truthiness, ``==``/``!=`` compare strings case-insensitively and coerce
mismatched types to numbers (so ``'true' == true`` is false), and a missing
property is ``null``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

_TOKEN = re.compile(r"""\s*(?:
    (?P<str>'(?:[^']|'')*')
  | (?P<num>-?\d+(?:\.\d+)?)
  | (?P<op>==|!=|&&|\|\||<=|>=|[!()<>,])
  | (?P<path>[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_*][A-Za-z0-9_-]*)*)
)""", re.VERBOSE)
_TEMPLATE = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)


class Unsupported(ValueError):
    """An expression construct this evaluator does not model."""


def _tokens(text: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    text = text.strip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match or match.end() == pos:
            raise Unsupported(f"cannot tokenize {text[pos:]!r} in {text!r}")
        kind = match.lastgroup or ""
        out.append((kind, match.group(kind)))
        pos = match.end()
    return out


def truthy(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0 and not math.isnan(value)
    if isinstance(value, str):
        return value != ""
    return True


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip()) if value.strip() else 0.0
        except ValueError:
            return math.nan
    return math.nan


def _equal(a: Any, b: Any) -> bool:
    if type(a) is type(b) or (a is None and b is None):
        if isinstance(a, str) and isinstance(b, str):
            return a.lower() == b.lower()
        return a == b
    return _number(a) == _number(b)


def to_string(value: Any) -> str:
    """How ``${{ }}`` renders a value into text (``true``/``false``, ``''`` for null)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


class _Parser:
    def __init__(self, text: str, context: Mapping[str, Any]):
        self.text, self.toks, self.i, self.ctx = text, _tokens(text), 0, context

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, value: str | None = None) -> tuple[str, str]:
        tok = self.peek()
        if tok is None or (value is not None and tok[1] != value):
            raise Unsupported(f"expected {value or 'a token'} in {self.text!r}")
        self.i += 1
        return tok

    def parse(self) -> Any:
        value = self.or_()
        tok = self.peek()
        if tok is not None:
            raise Unsupported(f"trailing {tok[1]!r} in {self.text!r}")
        return value

    def or_(self) -> Any:
        value = self.and_()
        while self.peek() == ("op", "||"):
            self.take()
            right = self.and_()
            value = value if truthy(value) else right
        return value

    def and_(self) -> Any:
        value = self.cmp()
        while self.peek() == ("op", "&&"):
            self.take()
            right = self.cmp()
            value = right if truthy(value) else value
        return value

    def cmp(self) -> Any:
        value = self.unary()
        while self.peek() in (("op", "=="), ("op", "!=")):
            op = self.take()[1]
            right = self.unary()
            value = _equal(value, right) if op == "==" else not _equal(value, right)
        tok = self.peek()
        if tok is not None and tok[1] in ("<", ">", "<=", ">="):
            raise Unsupported(f"ordering comparison in {self.text!r}")
        return value

    def unary(self) -> Any:
        if self.peek() == ("op", "!"):
            self.take()
            return not truthy(self.unary())
        return self.atom()

    def atom(self) -> Any:
        kind, text = self.take()
        if kind == "op" and text == "(":
            value = self.or_()
            self.take(")")
            return value
        if kind == "str":
            return text[1:-1].replace("''", "'")
        if kind == "num":
            return float(text)
        if kind != "path":
            raise Unsupported(f"unexpected {text!r} in {self.text!r}")
        if text in ("true", "false"):
            return text == "true"
        if text == "null":
            return None
        if self.peek() == ("op", "("):
            return self.call(text)
        return self.lookup(text)

    def call(self, name: str) -> Any:
        self.take("(")
        args = []
        while self.peek() != ("op", ")"):
            args.append(self.or_())
            if self.peek() == ("op", ","):
                self.take()
        self.take(")")
        if name in ("always", "success", "failure", "cancelled"):
            fn = self.ctx.get("__status__", {}).get(name)
            if fn is None:
                raise Unsupported(f"{name}() outside a modelled job status in {self.text!r}")
            return fn
        if name == "toJSON":
            return json.dumps(args[0])
        if name == "fromJSON":
            return json.loads(args[0])
        if name == "contains":
            hay, needle = args
            if isinstance(hay, list):
                return any(_equal(h, needle) for h in hay)
            return to_string(needle).lower() in to_string(hay).lower()
        raise Unsupported(f"function {name}() in {self.text!r}")

    def lookup(self, path: str) -> Any:
        node: Any = self.ctx
        for part in path.split("."):
            if part == "*":
                raise Unsupported(f"object filter in {self.text!r}")
            node = node.get(part) if isinstance(node, Mapping) else None
        return node


def evaluate(expr: str, context: Mapping[str, Any]) -> Any:
    """Value of a bare expression or a single ``${{ expr }}``."""
    text = expr.strip()
    whole = _TEMPLATE.fullmatch(text)
    return _Parser(whole.group(1) if whole else text, context).parse()


def condition(expr: Any, context: Mapping[str, Any]) -> bool:
    """A job/step ``if:`` (absent means ``success()``, which the caller models)."""
    if expr is None:
        return True
    if isinstance(expr, bool):
        return expr
    return truthy(evaluate(str(expr), context))


def render(text: Any, context: Mapping[str, Any]) -> Any:
    """``with:``/``env:``/``run:`` value: a lone ``${{ x }}`` keeps its type, embedded ones stringify."""
    if not isinstance(text, str):
        return text
    whole = _TEMPLATE.fullmatch(text.strip())
    if whole:
        return evaluate(whole.group(1), context)
    return _TEMPLATE.sub(lambda m: to_string(evaluate(m.group(1), context)), text)
