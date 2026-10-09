"""The scripted-prelude type alias must stay importable on every supported Python.

``agent/turn_scripted_prelude.py`` declares the alias

    Prelude = Generator[tuple[str, str, dict], Optional[str], None]

at module level, so it is evaluated at import time - ``from __future__ import
annotations`` does not defer it. ``typing.Generator`` only defaults its third
argument (``ReturnType``) on Python 3.13+, so the two-argument spelling raises

    TypeError: Too few arguments for typing.Generator; actual 2, expected 3

on 3.11 and 3.12, which ``pyproject.toml`` still declares supported
(``requires-python = ">=3.11,<3.15"``). The whole gateway then fails to import,
not just this module.

This reads the alias out of the source instead of importing it: importing the
module drags in the whole agent import chain, and a plain ``import`` would only
catch the regression on the interpreter running the suite, while CI runs one
version. Checking the parsed alias fails on every interpreter the moment the
third argument is dropped again - which is exactly what a linter autofix did
(UP043 assumes a 3.13+ target).
"""

import ast
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "agent" / "turn_scripted_prelude.py"


def _prelude_annotation() -> ast.expr:
    """Return the value assigned to the module-level ``Prelude`` alias."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "Prelude":
                return node.value

    raise AssertionError("agent/turn_scripted_prelude.py no longer defines a module-level Prelude alias")


def test_prelude_alias_keeps_its_third_type_argument() -> None:
    """Generator[...] must carry all three arguments, not the 3.13+ defaulted form."""
    annotation = _prelude_annotation()

    assert isinstance(annotation, ast.Subscript), "Prelude must stay a typing.Generator[...] alias"
    args = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]

    assert len(args) == 3, (
        "Prelude must be Generator[YieldType, SendType, ReturnType]; a two-argument "
        "Generator[...] is only valid on Python 3.13+ and makes the gateway "
        "unimportable on 3.11/3.12 (see issue #135587)"
    )
