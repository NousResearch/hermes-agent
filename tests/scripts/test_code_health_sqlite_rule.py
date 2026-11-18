"""HX015: a raw ``sqlite3.connect`` on ``state.db`` is invisible to ``has_live_connection()``;
it warns (heuristic, not yet replayed), existing hits are grandfathered, and a waiver silences it."""

from __future__ import annotations

import ast

import pytest

from scripts.code_health.config import RULES_BY_ID, rule_applies
from scripts.code_health.py_rules import CHECKERS, Ctx, canonical_tree
from tests.scripts.test_code_health import _commit, _repo, _verdict


def _hits(src: str) -> list[int]:
    return sorted(set(CHECKERS["HX015"](canonical_tree(ast.parse(src)), Ctx())))


@pytest.mark.parametrize("src, flagged", [
    # red: the target names the session store
    ("import sqlite3\nsqlite3.connect('state.db')\n", True),
    ("import sqlite3\nsqlite3.connect(str(home / 'state.db'), timeout=1)\n", True),
    ("import sqlite3 as s\ndef f(state_db_path):\n    return s.connect(state_db_path)\n", True),
    ("from sqlite3 import connect\ndef f(self):\n    connect(self._state_db)\n", True),
    ("import sqlite3\ndef f(p):\n    return sqlite3.connect(read_only_db_uri(p), uri=True)\n", True),
    ("import sqlite3\nsqlite3.connect(database=f'file:{home}/state.db?mode=ro', uri=True)\n", True),
    ("import sqlite3, contextlib\nwith contextlib.closing(sqlite3.connect('state.db')) as c:\n"
     "    pass\n", True),
    # red: a local bound from a state.db expression, directly, transitively or from an outer scope
    ("import sqlite3\ndef f(home):\n    db_path = home / 'state.db'\n    sqlite3.connect(str(db_path))\n",
     True),
    ("import sqlite3\ndef f(home):\n    a = home / 'state.db'\n    b = a.resolve()\n"
     "    sqlite3.connect(b.as_uri() + '?mode=ro', uri=True)\n", True),
    ("import sqlite3\ndb = HOME / 'state.db'\ndef f():\n    sqlite3.connect(db)\n", True),
    # a class body is its own scope: its binding taints its methods, not the module
    ("import sqlite3\nclass C:\n    db = HOME / 'state.db'\n    def f(self):\n"
     "        sqlite3.connect(db)\n", True),
    ("import sqlite3\nclass C:\n    db = HOME / 'state.db'\ndef g(db):\n    sqlite3.connect(db)\n",
     False),
    # green: other databases, :memory:, and the tracked helper's plumbing
    ("import sqlite3\nsqlite3.connect(':memory:')\n", False),
    ("import sqlite3\nsqlite3.connect(str(home / 'kanban.db'))\n", False),
    ("import sqlite3\ndef f(path):\n    return sqlite3.connect(str(path))\n", False),
    ("import sqlite3\nconnect_tracked(state_db, connect_fn=sqlite3.connect)\n", False),
    ("import sqlite3\ndef f():\n    return _connect_tracked_db(state_db_path, timeout=1)\n", False),
    ("def connect(x):\n    return x\nconnect('state.db')\n", False),  # a local helper, not the API
    # a taint does not leak into a sibling scope
    ("import sqlite3\ndef f(h):\n    p = h / 'state.db'\ndef g(p):\n    sqlite3.connect(p)\n", False),
])
def test_untracked_state_connect_shapes(src, flagged):
    assert bool(_hits(src)) is flagged, src


@pytest.mark.parametrize("path, applies", [
    ("hermes_cli/backup.py", True), ("gateway/x.py", True),
    ("tests/test_x.py", False), ("evals/run.py", False),
])
def test_scope_skips_tests_and_evals(path, applies):
    assert rule_applies(RULES_BY_ID["HX015"], path) is applies


def test_rule_warns_until_replayed():
    assert RULES_BY_ID["HX015"].blocking is False


_OLD = "import sqlite3\n\n\ndef old(home):\n    return sqlite3.connect(str(home / 'state.db'))\n"
_NEW = "import sqlite3\n\n\ndef new(home):\n    return sqlite3.connect(str(home / 'state.db'))\n"


def _judge(tmp_path, capsys, head):
    repo, _ = _repo(tmp_path)
    base = _commit(repo, {"pkg/s.py": _OLD})
    return _verdict(repo, base, head, capsys)


def test_new_hit_warns_without_blocking(tmp_path, capsys):
    code, out = _judge(tmp_path, capsys, {"pkg/t.py": _NEW})
    assert code == 0, out
    assert "HX015" in out and "connect_tracked" in out and "1 advisory" in out, out


def test_existing_hit_is_grandfathered(tmp_path, capsys):
    code, out = _judge(tmp_path, capsys, {"pkg/s.py": _OLD + "\n\ndef more():\n    return 1\n"})
    assert code == 0 and "HX015" not in out, out


def test_waiver_silences(tmp_path, capsys):
    waived = _NEW.replace("'state.db'))", "'state.db'))  # health: allow HX015 -- offline snapshot")
    code, out = _judge(tmp_path, capsys, {"pkg/t.py": waived})
    assert code == 0 and "0 advisory" in out and "offline snapshot" in out, out
