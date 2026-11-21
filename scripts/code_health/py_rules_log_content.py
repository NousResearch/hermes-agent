"""HX017: an INFO+ log call in gateway/adapter code that prints user content or identities.

Heuristic and name-based (registered in ``py_rules.CHECKERS``). A ``logger.info/warning/error/
critical/exception`` (or ``log``/``logging``) call is flagged when one of its arguments, or a value
interpolated into an f-string / ``%`` message, IS a value named like a message body or a person:
``text``, ``body``, ``content``, ``subject``, ``caption``, any ``*_text`` local, ``user_name``,
``display_name``, ``sender_name``, an e-mail address or phone number. The value may be sliced, ``or``-ed
with a default, wrapped in ``str``/``repr``/``json.dumps`` or passed through a string method
(``.strip()``, ``.replace(...)``): those still print it. ``len(text)``, ``bool(...)``, a hash or a
redaction helper call is not flagged, and neither is an HTTP payload read off a response or error
object (``resp.text``, ``exc.body``) or named for one (``error_text``, ``stderr_text``), which is
the remote side's answer, not a user's message.

Limits: chat/room/user ids (and the ambiguous ``sender``, an id on Matrix and a dict on WeCom) are
not matched: they are everywhere, and an operator needs a user id to edit an allowlist. A body a body that reaches the call under any other name (``msg``, ``payload``,
``response``) or through a generator is missed. A value named like content that holds none (a
config string, an API error body) takes ``# health: allow HX017 -- <why>``.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

from scripts.code_health.py_ast import Ctx, _dotted

_LEVELS = {"info", "warning", "warn", "error", "critical", "exception"}
_CONTENT = re.compile(
    r"(?:text|body|content|subject|caption|(?!(?:error|err|stderr|stdout|body|status|resp|response)_)\w+_text"
    r"|transcript|user_?name|user_display|display_name|sender_name|nickname|first_name|last_name|full_name"
    r"|sender_addr|from_addr|to_addr|email|email_address|sender_email|from_number|to_number"
    r"|phone|phone_number)",
    re.IGNORECASE,
)
# Bases whose ``.text`` / ``.body`` / ``.content`` is a transport payload, not a user's message.
_TRANSPORT = re.compile(r"(?:resp|response|res|r|http_response|raw_response|exc|err|error|e|ex)",
                        re.IGNORECASE)
_PASSTHROUGH_CALLS = {"str", "repr", "dumps", "json.dumps", "format"}
_STRING_METHODS = {"strip", "lstrip", "rstrip", "lower", "upper", "casefold", "title", "replace",
                   "encode", "decode", "expandtabs"}


def _named_content(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return bool(_CONTENT.fullmatch(node.id))
    if isinstance(node, ast.Attribute) and _CONTENT.fullmatch(node.attr):
        base = node.value
        return not (isinstance(base, ast.Name) and _TRANSPORT.fullmatch(base.id))
    return False


def _prints_content(node: ast.AST | None) -> bool:
    """True when evaluating ``node`` yields (a piece of) a content-named value as text."""
    if node is None:
        return False
    if _named_content(node):
        return True
    if isinstance(node, ast.Subscript):
        return _prints_content(node.value)
    if isinstance(node, ast.BoolOp):
        return any(map(_prints_content, node.values))
    if isinstance(node, ast.IfExp):
        return _prints_content(node.body) or _prints_content(node.orelse)
    if isinstance(node, (ast.FormattedValue, ast.Starred)):
        return _prints_content(node.value)
    if isinstance(node, ast.JoinedStr):
        return any(_prints_content(v) for v in node.values if isinstance(v, ast.FormattedValue))
    if isinstance(node, ast.BinOp):  # "..." % (a, b) and "a" + text
        right = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
        return _prints_content(node.left) or any(map(_prints_content, right))
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _STRING_METHODS:
            return _prints_content(func.value)
        if _dotted(func) in _PASSTHROUGH_CALLS:
            return bool(node.args) and _prints_content(node.args[0])
        if _dotted(func) == "getattr" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            return _named_content(ast.Attribute(node.args[0], str(node.args[1].value), ast.Load()))
    return False


def _is_log_call(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LEVELS and node.args):
        return False
    return "log" in _dotted(node.func.value).lower()


def user_content_in_log(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_log_call(node) and any(map(_prints_content, node.args)):
            yield node.lineno
