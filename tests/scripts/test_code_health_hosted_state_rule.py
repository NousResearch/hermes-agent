"""HX015: process-wide mutable state is refused in every module the hosted entry point imports.

A planted module-level dict reachable from the hosted roots blocks the ratchet; the same dict in
a module the roots never import does not; state that existed before is baselined; the two
declarations (``ScopedCache``, ``tenant_neutral`` with a reason) pass.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess

import pytest

from scripts.code_health import config
from scripts.code_health.hosted_graph import reachable
from scripts.code_health.py_rules import CHECKERS, Ctx, canonical_tree
from tests.scripts.test_code_health import REPO, _commit, _repo, _verdict


def _hits(src: str) -> list[int]:
    return sorted(set(CHECKERS["HX015"](canonical_tree(ast.parse(src)), Ctx())))


@pytest.mark.parametrize("src, flagged", [
    ("_CACHE = {}\n", True),
    ("_SEEN: set = set()\n", True),
    ("import collections\n_BY = collections.defaultdict(list)\n", True),
    ("_ITEMS = []\n", True),
    ("_TABLE = {'a': 1}\ndef put(k):\n    _TABLE[k] = 2\n", True),
    ("_NAMES = ['a']\ndef add(n):\n    _NAMES.append(n)\n", True),
    ("_MEMO = {'x': 1}\ndef reset():\n    global _MEMO\n    _MEMO = {}\n", True),
    ("try:\n    _X = {}\nexcept ImportError:\n    pass\n", True),
    ("import functools\n@functools.lru_cache(maxsize=8)\ndef f(x):\n    return x\n", True),
    ("from functools import cache\n@cache\ndef f(x):\n    return x\n", True),
    ("import functools\nclass C:\n    @functools.lru_cache\n    def m(self):\n        return 1\n", True),
    ("class C:\n    _live = {}\n    def add(self, k):\n        self._live[k] = 1\n", True),
    ("class C:\n    _live = []\n    @classmethod\n    def add(cls, k):\n        cls._live.append(k)\n", True),
    ("class C:\n    current = None\n    @classmethod\n    def set(cls, v):\n        cls.current = v\n", True),
    # constants, per-instance state, and the two declarations
    ("_TABLE = {'a': 1, 'b': 2}\n", False),
    ("_NAMES = ('a', 'b')\n", False),
    ("__all__ = []\n", False),
    ("class C:\n    _live = {}\n    def __init__(self):\n        self._live = {}\n    def add(self, k):\n"
     "        self._live[k] = 1\n", False),
    ("def f():\n    seen = {}\n    seen[1] = 2\n", False),
    ("from hermes_scope_cache import ScopedCache\n_CACHE = ScopedCache('c')\n", False),
    ("from hermes_scope_cache import tenant_neutral\n"
     "_CATALOG = tenant_neutral({}, reason='model catalog, same for every tenant')\n", False),
    ("import functools\nfrom hermes_scope_cache import tenant_neutral\n"
     "@tenant_neutral(reason='pure parse of a constant')\n@functools.cache\ndef f(x):\n    return x\n", False),
    # a declaration needs a literal, non-empty reason
    ("from hermes_scope_cache import tenant_neutral\n_C = tenant_neutral({}, reason='')\n", True),
    ("from hermes_scope_cache import tenant_neutral\n_C = tenant_neutral({}, reason=WHY)\n", True),
])
def test_detects_process_wide_state(src, flagged):
    assert bool(_hits(src)) is flagged


def _graph_repo(tmp_path, files):
    root = tmp_path / "g"
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=60)
    return root


def test_graph_follows_import_time_edges_only(tmp_path):
    root = _graph_repo(tmp_path, {
        "entry.py": "import pkg.sub\nfrom pkg import leaf\nfrom typing import TYPE_CHECKING\n"
                    "if TYPE_CHECKING:\n    import typed_only\ntry:\n    import guarded\nexcept ImportError:\n"
                    "    pass\ndef later():\n    import lazy_only\n",
        "pkg/__init__.py": "", "pkg/sub.py": "from . import rel\n", "pkg/rel.py": "",
        "pkg/leaf.py": "", "typed_only.py": "", "guarded.py": "", "lazy_only.py": "",
        "island.py": "",
    })
    got = reachable(root, None, ("entry",))
    assert got == {"entry.py", "pkg/__init__.py", "pkg/sub.py", "pkg/rel.py", "pkg/leaf.py", "guarded.py"}


def test_every_hosted_root_names_a_module():
    """A renamed root would silently shrink the judged graph."""
    missing = [r for r in config.HOSTED_ENTRY if importlib.util.find_spec(r) is None]
    assert not missing
    assert {f"{r.replace('.', '/')}.py" for r in ("hermes_scope", "tools.registry")} <= reachable(REPO, None)


_ENTRY = "import pkg.state\n"
_OLD_STATE = "_OLD = {}\n"


@pytest.fixture
def hosted_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOSTED_ENTRY", ("pkg.entry",))
    repo, _ = _repo(tmp_path)
    base = _commit(repo, {"pkg/__init__.py": "", "pkg/entry.py": _ENTRY, "pkg/state.py": _OLD_STATE,
                          "pkg/island.py": _OLD_STATE})
    return repo, base


@pytest.mark.parametrize("head, blocks", [
    ({"pkg/state.py": _OLD_STATE + "_PLANTED = {}\n"}, True),       # reachable: blocks
    ({"pkg/island.py": _OLD_STATE + "_PLANTED = {}\n"}, False),     # nothing hosted imports it
    ({"pkg/state.py": _OLD_STATE + "\n\ndef f():\n    return 1\n"}, False),  # old state baselined
    # a new import pulls an old module onto the hosted path while editing it: its debt is kept
    ({"pkg/entry.py": _ENTRY + "import pkg.island\n",
      "pkg/island.py": _OLD_STATE + "\n\ndef f():\n    return 1\n"}, False),
    ({"pkg/state.py": _OLD_STATE + "from hermes_scope_cache import ScopedCache\n"
                                   "_NEW = ScopedCache('new')\n"}, False),
])
def test_ratchet(hosted_repo, capsys, head, blocks):
    repo, base = hosted_repo
    code, out = _verdict(repo, base, head, capsys)
    assert (code != 0) is blocks, out
    if blocks:
        assert "HX015" in out
