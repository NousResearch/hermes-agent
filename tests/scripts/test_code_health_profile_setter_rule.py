"""HX014: a new direct call to a legacy profile setter blocks; existing calls are grandfathered and
migrate to ``agent.runtime_scope.bind_profile_scope`` when their function is next edited."""

from __future__ import annotations

import ast

import pytest

from scripts.code_health.config import RULES_BY_ID, rule_applies
from scripts.code_health.py_rules import CHECKERS, Ctx, canonical_tree
from tests.scripts.test_code_health import _commit, _repo, _verdict


def _hits(src: str) -> list[int]:
    return sorted(set(CHECKERS["HX014"](canonical_tree(ast.parse(src)), Ctx())))


@pytest.mark.parametrize("src, flagged", [
    ("from hermes_constants import set_hermes_home_override\nset_hermes_home_override('/h')\n", True),
    ("from hermes_constants import set_hermes_home_override as s\ns('/h')\n", True),
    ("import hermes_constants\nhermes_constants.set_hermes_home_override('/h')\n", True),
    ("import agent.secret_scope as ss\nss.set_secret_scope({})\n", True),
    ("def f():\n    from tools.terminal_scope import install_profile_terminal_scope\n"
     "    install_profile_terminal_scope('/h')\n", True),
    ("from tools.terminal_scope import set_terminal_scope\nset_terminal_scope({})\n", True),
    # the replacement, and unrelated names that merely share a spelling
    ("from agent.runtime_scope import bind_profile_scope\nwith bind_profile_scope('/h'):\n    pass\n", False),
    ("def f(set_hermes_home_override):\n    set_hermes_home_override('/h')\n", False),
    ("from hermes_constants import reset_hermes_home_override\nreset_hermes_home_override(t)\n", False),
    # relative imports inside the package: the AST cannot know the package, so a whole-component tail
    ("from .secret_scope import set_secret_scope\nset_secret_scope({})\n", True),
    ("from . import terminal_scope\nterminal_scope.install_profile_terminal_scope('/h')\n", True),
    ("from .. import hermes_constants\nhermes_constants.set_hermes_home_override('/h')\n", True),
    ("from .other_secret_scope import set_secret_scope\nset_secret_scope({})\n", False),
    ("from . import set_secret_scope\nset_secret_scope({})\n", False),  # names no module
    # a local object that merely shares the module's name is not the module
    ("hermes_constants = Fake()\nhermes_constants.set_hermes_home_override('/h')\n", False),
    ("def f(hermes_constants):\n    hermes_constants.set_hermes_home_override('/h')\n", False),
])
def test_detects_direct_setter_calls(src, flagged):
    assert bool(_hits(src)) is flagged


@pytest.mark.parametrize("path, applies", [
    ("gateway/run.py", True),
    ("tui_gateway/model_switch.py", True),
    ("agent/runtime_scope.py", False),  # the one module that turns a scope into profile state
    ("hermes_constants.py", False),     # the setters' own definitions
    ("agent/secret_scope.py", False),
    ("tools/terminal_scope.py", False),
])
def test_scope(path, applies):
    assert rule_applies(RULES_BY_ID["HX014"], path) is applies


_OLD = ("from hermes_constants import set_hermes_home_override\n\n\n"
        "def old(home):\n    return set_hermes_home_override(home)\n")
_NEW_CALL = "\n\ndef new(home):\n    return set_hermes_home_override(home)\n"


@pytest.mark.parametrize("head, blocks", [
    ({"gateway/a.py": _OLD + _NEW_CALL}, True),                     # a new call in an old file
    ({"gateway/b.py": _OLD}, True),                                 # a new file
    ({"gateway/a.py": _OLD + "\n\ndef unrelated():\n    return 1\n"}, False),  # old call untouched
    ({"gateway/a.py": "def old(home):\n    return home\n"}, False),  # migrating one away
])
def test_ratchet(tmp_path, capsys, head, blocks):
    repo, _ = _repo(tmp_path)
    base = _commit(repo, {"gateway/a.py": _OLD})
    code, out = _verdict(repo, base, head, capsys)
    assert (code != 0) is blocks, out
    if blocks:
        assert "HX014" in out
