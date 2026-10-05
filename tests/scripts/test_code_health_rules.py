"""Rule and structure precision of the code-health ratchet (scripts/code_health): each case is a
reviewer's repro paired with the control that must keep its verdict."""

from __future__ import annotations

import ast

import pytest

from scripts.code_health.config import RULES_BY_ID
from scripts.code_health.py_rules import CHECKERS, Ctx, canonical_tree
from scripts.code_health.py_structure import body_hash, nesting_depth
from tests.scripts.test_code_health import _commit, _repo, _verdict


def _hits(rule: str, src: str) -> list[int]:
    """Lines ``rule`` reports for ``src``, through the same canonical tree the measurer uses."""
    return sorted(set(CHECKERS[rule](canonical_tree(ast.parse(src)), Ctx())))


def _judge(tmp_path, capsys, base_files: dict[str, str], head_files: dict[str, str | None]):
    repo, base = _repo(tmp_path)
    if base_files:
        base = _commit(repo, base_files)
    return _verdict(repo, base, head_files, capsys)


# --- F12: an import alias is shadowed only in the scope that rebinds it ---

_ALIAS = "from subprocess import run as execute\n\n\n{}\n\n\ndef launch(cmd):\n{}    return execute(cmd)\n"


@pytest.mark.parametrize("elsewhere, local, flagged", [
    ("def identity(execute):\n    return execute", "", True),  # unrelated parameter
    ("def other():\n    execute = 42\n    return execute", "", True),  # unrelated local
    ("def other():\n    def execute():\n        return 1\n    return execute", "", True),  # nested def
    ("def identity(x):\n    return x", "", True),
    # genuine shadowing where the call is made still stops resolution
    ("def identity(x):\n    return x", "    execute = print\n", False),
    ("execute = print", "", False),  # the module itself rebinds it: ambiguous
])
def test_import_alias_resolves_per_scope(elsewhere, local, flagged):
    src = _ALIAS.format(elsewhere, local)
    assert bool(_hits("HX006", src)) is flagged, src


def test_import_alias_parameter_of_consuming_function_shadows():
    src = "from subprocess import run as execute\n\n\ndef launch(cmd, execute):\n    return execute(cmd)\n"
    assert _hits("HX006", src) == []


def test_import_alias_with_unrelated_parameter_blocks_through_the_verdict(tmp_path, capsys):
    head = {"pkg/c.py": _ALIAS.format("def identity(execute):\n    return execute", "")}
    code, out = _judge(tmp_path, capsys, {}, head)
    assert code == 1 and "HX006" in out, out
