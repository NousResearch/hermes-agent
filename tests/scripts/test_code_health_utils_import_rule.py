"""HX018: a ``utils/`` module may not reach first-party product code, by import or by
``sys.modules``; first-party names come from the judged tree's tracked files."""

from __future__ import annotations

import ast
import re

import pytest

from scripts.code_health.py_rules import CHECKERS, Ctx, canonical_tree
from tests.scripts.test_code_health import _repo, _verdict

_UTILS_X = """\
import json
import agent.redact
from hermes_constants import get_hermes_home
from utils import sib
from . import sib as sib2
import sys
import yaml


def f():
    from hermes_cli.config import load_config
    return sys.modules.get("gateway.run")


def g():
    # health: allow HX018 -- the waiver form keeps a real exception possible
    from hermes_cli import config
"""


def test_utils_product_reach_blocks_and_honours_waiver(tmp_path, capsys):
    repo, base = _repo(tmp_path)
    tree: dict[str, str | None] = {
        "agent/__init__.py": "", "agent/redact.py": "", "gateway/__init__.py": "",
        "hermes_cli/__init__.py": "", "hermes_constants.py": "",
        "utils/__init__.py": "", "utils/sib.py": "", "utils/x.py": _UTILS_X,
        "pkg/b.py": "import agent\n",  # outside utils: not this rule's business
    }
    code, out = _verdict(repo, base, tree, capsys)
    flagged = {(m[1], int(m[2])) for m in re.finditer(r"^(\S+):(\d+)  HX018", out, re.MULTILINE)}
    assert code == 1
    assert flagged == {("utils/x.py", 2), ("utils/x.py", 11), ("utils/x.py", 12)}
    assert "utils/x.py:17  HX018" in out.split("Allowed by comment")[1]


@pytest.mark.parametrize("src, flagged", [
    ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from agent import x\n", True),
    ("import importlib\nimportlib.import_module('agent.x')\n", True),
    ("m = __import__('gateway')\n", True),
    ("import sys as s\nm = s.modules['agent']\n", True),
    ("import sys\nok = 'agent' in sys.modules\n", True),
    ("import importlib\nimportlib.import_module('hermes_platform')\n", False),
    ("import sys\nm = sys.modules.get('json')\n", False),
    ("import importlib\ndef f(name):\n    return importlib.import_module(name)\n", False),
])
def test_utils_product_reach_shapes(src, flagged):
    ctx = Ctx(first_party=frozenset({"agent", "gateway", "hermes_platform", "utils"}))
    assert bool(list(CHECKERS["HX018"](canonical_tree(ast.parse(src)), ctx))) is flagged, src
