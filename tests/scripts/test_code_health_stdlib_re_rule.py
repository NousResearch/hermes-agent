"""HX018: a new stdlib ``re`` import in product code blocks (existing ones are grandfathered until
their file moves to ``hermes_re``); HX019: ``hermes_re`` may not be imported by a file that keeps
stdlib ``re`` because google-re2 can be missing where it runs."""

from __future__ import annotations

import ast

import pytest

from scripts.code_health.config import RULES_BY_ID, rule_applies
from scripts.code_health.py_rules import CHECKERS, Ctx, canonical_tree
from scripts.code_health.stdlib_re_exempt import RE_EXEMPT, RE_EXEMPT_FILES
from tests.scripts.test_code_health import _commit, _repo, _verdict


def _hits(rule: str, src: str) -> list[int]:
    return sorted(set(CHECKERS[rule](canonical_tree(ast.parse(src)), Ctx())))


@pytest.mark.parametrize("src, flagged", [
    ("import re\n", True),
    ("import re as _re\n", True),
    ("from re import compile, IGNORECASE\n", True),
    ("import re._parser\n", True),
    ("import os, re\n", True),
    ("def f(text):\n    import re\n    return re.sub('a', 'b', text)\n", True),
    ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import re\n", True),
    ("import sre_parse\n", True),
    ("from sre_constants import MAXREPEAT\n", True),
    ("import sre_compile as compiler\n", True),
    # the dynamic spellings with a literal name, through any alias of import_module
    ("__import__('re')\n", True),
    ("import importlib\nimportlib.import_module('re')\n", True),
    ("from importlib import import_module as load\nload('sre_parse')\n", True),
    ("import importlib\nimportlib.import_module(name='re')\n", True),
    ("__import__(name='re')\n", True),
    # the facade, and names that only look like the module
    ("import hermes_re as re\nre.compile('a')\n", False),
    ("import regex\n", False),
    ("import retry\n", False),
    ("from . import re\n", False),          # a sibling module of the package
    ("from .re import helper\n", False),
    ("PROGRAM = 'import re'\n", False),     # program text for another interpreter
    ("import importlib\ndef f(name):\n    return importlib.import_module(name)\n", False),
])
def test_stdlib_re_import_shapes(src, flagged):
    assert bool(_hits("HX018", src)) is flagged, src


@pytest.mark.parametrize("src, flagged", [
    ("import hermes_re as re\n", True),
    ("import hermes_re\n", True),
    ("from hermes_re import compile\n", True),
    ("import hermes_re._engine\n", True),
    ("def f():\n    import hermes_re as re\n    return re\n", True),
    ("import importlib\nimportlib.import_module('hermes_re')\n", True),
    ("import importlib\nimportlib.import_module(name='hermes_re')\n", True),
    ("import re\n", False),
    ("import hermes_requests\n", False),
    ("from . import hermes_re\n", False),
])
def test_facade_import_shapes(src, flagged):
    assert bool(_hits("HX019", src)) is flagged, src


@pytest.mark.parametrize("path, applies", [
    ("agent/x.py", True), ("gateway/platforms/x.py", True), ("plugins/platforms/slack/adapter.py", True),
    ("evals/run.py", True), ("cli.py", True),
    ("tests/agent/test_x.py", False), ("plugins/x/tests/test_y.py", False), ("agent/conftest.py", False),
    ("scripts/check_x.py", False), ("skills/x/scripts/helper.py", False), ("pm/cli.py", False),
    ("hermes_re/_errors.py", False),  # the facade derives its error class from the stdlib one
])
def test_stdlib_re_rule_scope(path, applies):
    assert rule_applies(RULES_BY_ID["HX018"], path) is applies


def test_one_list_decides_both_rules():
    assert RE_EXEMPT_FILES, "the exemption list names no single file"
    for entry in RE_EXEMPT:
        path = entry.replace("*", "sub/module.py")
        assert not rule_applies(RULES_BY_ID["HX018"], path), entry
    for path in RE_EXEMPT_FILES:
        assert rule_applies(RULES_BY_ID["HX019"], path), path
    # pm/ runs before dependencies exist; other exempt directories hold scripts for a developer's
    # environment or another interpreter, where importing the facade is their own business.
    assert rule_applies(RULES_BY_ID["HX019"], "pm/environment.py")
    assert not rule_applies(RULES_BY_ID["HX019"], "scripts/check_re2_patterns.py")
    assert not rule_applies(RULES_BY_ID["HX019"], "agent/x.py")


_OLD = "import re\n\n\ndef old(text):\n    return re.sub('a', 'b', text)\n"
_FACADE = _OLD.replace("import re\n", "import hermes_re as re\n")


@pytest.mark.parametrize("head, blocks", [
    ({"agent/b.py": _OLD}, True),                                             # a new file
    ({"agent/a.py": _OLD + "\n\ndef new(text):\n    import re as _re\n    return _re.escape(text)\n"}, True),
    ({"agent/a.py": _OLD + "\n\ndef unrelated():\n    return 1\n"}, False),   # the old import untouched
    ({"agent/a.py": _FACADE}, False),                                         # the migration itself
    ({"agent/b.py": _FACADE}, False),
    ({"tests/agent/test_b.py": _OLD, "scripts/b.py": _OLD, "pm/b.py": _OLD}, False),
    ({RE_EXEMPT_FILES[0]: _OLD}, False),
])
def test_stdlib_re_ratchet(tmp_path, capsys, head, blocks):
    repo, _ = _repo(tmp_path)
    base = _commit(repo, {"agent/a.py": _OLD})
    code, out = _verdict(repo, base, head, capsys)
    assert (code != 0) is blocks, out
    assert ("HX018" in out and "hermes_re" in out) is blocks, out


def test_waiver_keeps_a_stdlib_pattern_a_third_party_api_requires(tmp_path, capsys):
    repo, _ = _repo(tmp_path)
    base = _commit(repo, {"agent/a.py": _OLD})
    waived = _FACADE.replace(
        "import hermes_re as re\n",
        "import hermes_re as re\nimport re as _stdlib_re  # health: allow HX018 -- slack_bolt requires stdlib re.Pattern\n")
    code, out = _verdict(repo, base, {"agent/a.py": waived}, capsys)
    assert code == 0 and "slack_bolt requires stdlib re.Pattern" in out, out


@pytest.mark.parametrize("path, blocks", [
    (RE_EXEMPT_FILES[0], True), ("pm/b.py", True), ("scripts/b.py", False), ("agent/b.py", False),
])
def test_facade_is_refused_where_the_engine_may_be_absent(tmp_path, capsys, path, blocks):
    repo, base = _repo(tmp_path)
    code, out = _verdict(repo, base, {path: _FACADE}, capsys)
    assert (code != 0) is blocks, out
    assert ("HX019" in out) is blocks, out
