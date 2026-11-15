"""HX014: a def/lambda deciding from an exception parameter's status code or text (``py_rules.CHECKERS``)."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

from scripts.code_health.py_ast import _FUNCS, Ctx, _call_name
from scripts.code_health.py_rules_sleep import _own

_EXC_NAME = re.compile(r"(?:exc|err|error|e|ex|exception|\w+_(?:err|error|exc))$")
_EXC_TYPE = re.compile(r"(?!.*ClassifiedError)(?:(?:typing\.)?Optional\[)?[\w.]*(?:Exception|Error)\]?(?: \| None)?")
_PROBES = {"startswith", "endswith", "search", "match", "fullmatch", "_contains_any"}


def _exc_params(args: ast.arguments) -> str:
    return "|".join(p.arg for p in (*args.posonlyargs, *args.args, *args.kwonlyargs)
                    if (_EXC_TYPE.fullmatch(ast.unparse(p.annotation).strip("'\"")) if p.annotation
                        else _EXC_NAME.match(p.arg)))


def private_error_predicate(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for func in ast.walk(tree):
        if not isinstance(func, (*_FUNCS, ast.Lambda)) or not (exc := _exc_params(func.args)):
            continue
        own = rf"(?:{exc})(?:\.response)?|getattr\((?:{exc}), 'response'[^()]*\)"
        status = re.compile(rf"(?:{own})\.status_code|getattr\((?:{own}), 'status_code'[^()]*\)")
        nodes, texts = list(_own(func)), rf"(?:str|repr)\((?:{exc})\)"
        for _ in range(4):  # text locals, chained: msg = str(exc); low = msg.lower()
            fold = re.compile(rf"(?:{texts})(?:\.(?:lower|upper|casefold|strip)\(\))*")
            texts = "|".join({texts, *(t.id for n in nodes if isinstance(n, (ast.Assign, ast.NamedExpr))
                                       for t in getattr(n, "targets", [getattr(n, "target", None)])
                                       if isinstance(t, ast.Name) and fold.fullmatch(ast.unparse(n.value)))})
        is_text = lambda n: n is not None and bool(fold.fullmatch(ast.unparse(n)))
        for n in nodes:
            if (isinstance(n, (ast.Attribute, ast.Call)) and status.fullmatch(ast.unparse(n))
                    or isinstance(n, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) and is_text(r)
                                                          for op, r in zip(n.ops, n.comparators))
                    or isinstance(n, ast.Call) and _call_name(n).rsplit(".", 1)[-1] in _PROBES
                    and any(map(is_text, (getattr(n.func, "value", None), *n.args)))):
                yield n.lineno
