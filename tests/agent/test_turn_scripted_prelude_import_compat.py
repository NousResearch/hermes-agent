"""Regression test for #135587: agent/turn_scripted_prelude.py imports on every
Python the project supports (requires-python >= 3.11).

``Prelude = Generator[...]`` is a module-level alias, so ``from __future__ import
annotations`` does not defer its evaluation: typing computes the subscript at
import time. Before Python 3.13, ``typing.Generator`` has no defaulted return
type and requires all three arguments; the two-argument form raised ``TypeError:
Too few arguments for typing.Generator; actual 2, expected 3`` at import time,
taking down every importer of the module — ``tui_gateway.server``,
``gateway.run``, ``agent.conversation_loop`` — on the 3.11/3.12 installs
``requires-python`` still promises to support.

CI runs on 3.14, where the two-argument form succeeds, so a plain import cannot
catch a regression of this shape. The check below pins the contract itself:
module-level aliases of strict-arity typing generics (``Generator`` — 3 args,
``AsyncGenerator`` — 2, ``Coroutine`` — 3) must always pass the full argument
count, on every Python. It fails on the unfixed file and passes once the
explicit ``None`` is present, regardless of interpreter version.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import AsyncGenerator, Generator, Optional

import agent.turn_scripted_prelude

REPO_ROOT = Path(__file__).resolve().parents[2]
# Subpackages whose module-level aliases must stay 3.11-importable.
_PACKAGES = ("agent", "gateway", "tui_gateway", "hermes_cli", "tools", "cron", "acp_adapter", "providers")
# typing generics with a fixed argument count before 3.13 (no defaulted
# trailing parameter there). Everything not listed accepts its short form on
# every supported version.
_STRICT_ARITY = {"Generator": 3, "AsyncGenerator": 2, "Coroutine": 3}


def _module_level_alias_subscripts(tree: ast.Module):
    """``Name = Generic[...]`` plain assignments — evaluated at import time."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Subscript):
                yield target.id, node.value


def _arity_offenders() -> list[str]:
    offenders = []
    for package in _PACKAGES:
        for path in sorted((REPO_ROOT / package).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, OSError):
                continue
            for _alias, subscript in _module_level_alias_subscripts(tree):
                base = ast.unparse(subscript.value).rsplit(".", 1)[-1]
                expected = _STRICT_ARITY.get(base)
                if expected is None:
                    continue
                args = subscript.slice
                count = len(args.elts) if isinstance(args, ast.Tuple) else 1
                if count != expected:
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}: {base} needs {expected} arguments "
                        f"on Python < 3.13, found {count}: {ast.unparse(subscript)}"
                    )
    return offenders


def test_module_level_aliases_keep_full_arity_on_all_supported_pythons():
    """A short ``Generator[...]`` alias breaks every 3.11/3.12 import of the module."""
    assert not _arity_offenders()


def test_prelude_alias_carries_explicit_none_return():
    """The alias must equal the 3-argument form 3.11 evaluates."""
    expected = Generator[tuple[str, str, dict], Optional[str], None]
    assert agent.turn_scripted_prelude.Prelude == expected


def test_module_imports():
    """The import path that regressed: the module-level alias must not raise."""
    assert agent.turn_scripted_prelude.Prelude is not None
