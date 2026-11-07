"""The SQL mapping checker on how calls reach the functions they run and on declared inherited labels: each rule
passes on a matching pair of trees and the label it guards stays sharp."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parents[3] / "persistence" / "sqlite" / "_sqlmap" / "check.py"
_spec = importlib.util.spec_from_file_location("sqlmap_check", CHECK)
check = importlib.util.module_from_spec(_spec)
sys.modules["sqlmap_check"] = check
_spec.loader.exec_module(check)


def write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def run(legacy: Path, branch: Path, manifest: dict) -> "check.Report":
    write(branch, {"persistence/sqlite/_sqlmap/x.json": json.dumps(manifest)})
    return check.Checker(legacy, branch, check.load_manifests(branch / check.MANIFEST_DIR)).run()


LEAF = '''
    def leaf(self):
        self._conn.execute("SELECT 1 FROM items")
'''
MIDDLE = '''
    def c3(self):
        self._conn.execute("SELECT 3 FROM items")
        self.c2()
'''
CHAIN = '''
    def c1(self):
        self.leaf()

    def c2(self):
        self.c1()

    def c4(self):
        self.c3()

    def c5(self):
        self.c4()

    def c6(self):
        self.c5()

    def c7(self):
        self._execute_write(lambda conn: self.c6())
'''


def test_an_inherited_label_depends_on_the_caller_chain_not_on_which_statement_asks_first(tmp_path):
    for order, body in (("leaf_first", LEAF + MIDDLE), ("middle_first", MIDDLE + LEAF)):
        legacy = write(tmp_path / order, {"hermes_state_x.py": "\nclass Store:" + body + CHAIN})
        tree = check.Tree(legacy, ["hermes_state_x.py"], dict(check.DEFAULT_WRAPPERS))
        (leaf,) = tree.sites[("hermes_state_x.py", "Store.leaf")]
        (middle,) = tree.sites[("hermes_state_x.py", "Store.c3")]
        assert (order, leaf.view()["transaction"], middle.view()["transaction"]) == (order, "none", "write")


LEGACY_FLAG = '''
class Store:
    def read_flag(self, key):
        return self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()

    def sweep(self):
        def _do(conn):
            self.read_flag("swept")
            conn.execute("DELETE FROM meta WHERE key = 'swept'")
        self._execute_write(_do, patience_s=self._patience("sweep"))
'''
BACKEND_FLAG = '''
class Repo:
    def read_flag(self, key):
        return self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
'''
FLAG_UNIT = {"backend": "persistence/sqlite/x.py::Repo.read_flag", "legacy": "hermes_state_x.py::Store.read_flag"}


def test_a_declared_inherited_label_compares_with_or_without_the_receiver(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_FLAG})
    tree = check.Tree(legacy, ["hermes_state_x.py"], dict(check.DEFAULT_WRAPPERS))
    assert tree.inherited(FLAG_UNIT["legacy"]) == "write(patience_s=_patience('sweep'))"
    for n, label in enumerate(("write(patience_s=self._patience('sweep'))", "write(patience_s=_patience('sweep'))")):
        branch = write(tmp_path / f"branch{n}", {"persistence/sqlite/x.py": BACKEND_FLAG})
        report = run(legacy, branch, {"family": "x", "units": [{**FLAG_UNIT, "inherits": label}]})
        assert (label, report.failures) == (label, [])
    found = run(legacy, branch, {"family": "x", "units": [{**FLAG_UNIT, "inherits": "write(patience_s=self.patience)"}]})
    assert any("but the legacy function inherits" in f for f in found.failures)


LEGACY_META = '''
class Store:
    def get_meta(self, key):
        return self._conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()

    def touch(self, key):
        def body(conn):
            conn.execute("UPDATE state_meta SET value = value WHERE key = ?", (key,))
        return self._execute_write(body)
'''
BACKEND_META = '''
from .meta import get_meta


class Repo:
    def get_meta(self, key):
        return self._conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()

    def touch(self, key):
        def body(conn):
            conn.execute("UPDATE state_meta SET value = value WHERE key = ?", (key,))
            return LOOKUP
        return self._execute_write(body)
'''
META_MANIFEST = {
    "family": "x",
    "units": [{"backend": "persistence/sqlite/x.py::Repo.get_meta", "legacy": "hermes_state_x.py::Store.get_meta"},
              {"backend": "persistence/sqlite/x.py::Repo.touch", "legacy": "hermes_state_x.py::Store.touch"}],
}


def test_a_bare_name_call_is_not_a_call_of_a_method(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_META})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": BACKEND_META.replace("LOOKUP", "get_meta(conn, key)")})
    assert run(legacy, branch, META_MANIFEST).failures == []
    planted = write(tmp_path / "planted", {"persistence/sqlite/x.py": BACKEND_META.replace("LOOKUP", "self.get_meta(key)")})
    assert any("Repo.get_meta transaction" in f for f in run(legacy, planted, META_MANIFEST).failures)


BACKEND_SEEN = '''
from . import seen


class Repo:
    def lookup(self, sid):
        return self._conn.execute("UPDATE sessions SET seen = 1 WHERE id = ?", (sid,))

    def mark(self, sid):
        return self._execute_write(lambda conn: self.lookup(sid))

    def peek(self, sid):
        with self._read_ctx() as conn:
            return CALL
'''
BACKEND_SEEN_MODULE = '''
def lookup(conn, sid):
    return conn.execute("SELECT seen FROM sessions WHERE id = ?", (sid,)).fetchone()
'''


@pytest.mark.parametrize("call, labels", [
    ("seen.lookup(conn, sid)", ("write", "read")),
    ("self.lookup(sid)", ("mixed(read | write)", "none")),
], ids=["module", "method"])
def test_a_call_on_an_imported_module_runs_its_function_not_a_method_of_the_same_name(tmp_path, call, labels):
    rels = ["persistence/sqlite/x.py", "persistence/sqlite/seen.py"]
    root = write(tmp_path, dict(zip(rels, (BACKEND_SEEN.replace("CALL", call), BACKEND_SEEN_MODULE))))
    tree = check.Tree(root, rels, dict(check.DEFAULT_WRAPPERS))
    ((method,), (function,)) = (tree.sites[key] for key in (("persistence/sqlite/x.py", "Repo.lookup"),
                                                           ("persistence/sqlite/seen.py", "lookup")))
    assert (method.view()["transaction"], function.view()["transaction"]) == labels


LEGACY_CLOSE = '''
class Store:
    def close(self):
        self._conn.execute("PRAGMA optimize")

    def prune(self, owner):
        def body(conn):
            rows = conn.execute("SELECT id FROM items WHERE owner = ?", (owner,))
            return rows.fetchall()
        return self._execute_write(body)
'''
BACKEND_CLOSE = '''
class Repo:
    def close(self):
        self._conn.execute("PRAGMA optimize")

    def prune(self, owner):
        def body(conn):
            rows = conn.execute("SELECT id FROM items WHERE owner = ?", (owner,))
            try:
                return rows.fetchall()
            finally:
                CLOSE
        return self._execute_write(body)
'''
CLOSE_MANIFEST = {
    "family": "x",
    "units": [{"backend": "persistence/sqlite/x.py::Repo.close", "legacy": "hermes_state_x.py::Store.close"},
              {"backend": "persistence/sqlite/x.py::Repo.prune", "legacy": "hermes_state_x.py::Store.prune"}],
}


def test_a_method_of_a_cursor_local_is_not_a_call_of_the_tree(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_CLOSE})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": BACKEND_CLOSE.replace("CLOSE", "rows.close()")})
    assert run(legacy, branch, CLOSE_MANIFEST).failures == []
    planted = write(tmp_path / "planted", {"persistence/sqlite/x.py": BACKEND_CLOSE.replace("CLOSE", "self.close()")})
    assert any("Repo.close transaction" in f for f in run(legacy, planted, CLOSE_MANIFEST).failures)


LEGACY_STORE = '''
class SessionDB:
    def _read_one(self, sql, params=()):
        return self._conn.execute(sql, params).fetchone()

    def get_title(self, sid):
        return self._read_one("SELECT title FROM sessions WHERE id = ?", (sid,))


class Snapshot:
    def _read_one(self, sql, params=()):
        return self.conn.execute(sql, params).fetchone()


def source_of(db, sid):
    return db._read_one("SELECT source FROM sessions WHERE id = ?", (sid,))
'''
BACKEND_STORE = '''
class StoreHolder:
    def _read_one(self, sql, params=()):
        return self._conn.execute(sql, params).fetchone()


class _View:
    def _read_one(self, sql, params=()):
        return self.conn.execute(sql, params).fetchone()


def get_title(holder, sid):
    return holder._read_one("SELECT title FROM sessions WHERE id = ?", (sid,))


def source_of(holder, sid):
    return SOURCE
'''
STORE_MANIFEST = {
    "family": "x",
    "units": [{"backend": "persistence/sqlite/x.py::StoreHolder._read_one", "legacy": "hermes_state_x.py::SessionDB._read_one"},
              {"backend": "persistence/sqlite/x.py::_View._read_one", "legacy": "hermes_state_x.py::Snapshot._read_one"},
              {"backend": "persistence/sqlite/x.py::get_title", "legacy": "hermes_state_x.py::SessionDB.get_title"},
              {"backend": "persistence/sqlite/x.py::source_of", "legacy": "hermes_state_x.py::source_of"}],
}
SOURCE = 'holder._read_one("SELECT source FROM sessions WHERE id = ?", (sid,))'


def test_a_call_on_an_untyped_receiver_reaches_the_store_class_not_a_second_class_with_its_primitives(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_STORE})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": BACKEND_STORE.replace("SOURCE", SOURCE)})
    assert run(legacy, branch, STORE_MANIFEST).failures == []
    for root, rel, func in ((legacy, "hermes_state_x.py", "source_of"), (branch, "persistence/sqlite/x.py", "get_title")):
        tree = check.Tree(root, [rel], dict(check.DEFAULT_WRAPPERS))
        assert (func, len(tree.sites[(rel, func)])) == (func, 1)
    planted = write(tmp_path / "planted", {"persistence/sqlite/x.py": BACKEND_STORE.replace("SOURCE", SOURCE + " or " + SOURCE)})
    assert any("x.py::source_of count: 2 backend statements, 1 legacy" in f for f in run(legacy, planted, STORE_MANIFEST).failures)


LEGACY_DB = '''
def _fence(conn, key):
    conn.execute("SELECT value FROM meta WHERE key = ?", (key,))


def stamp(db, key):
    def write(conn):
        _fence(conn, key)
    return db._execute_write(write, patience_s=db._PATIENCE_S)
'''
BACKEND_DB = '''
def _fence(conn, key):
    conn.execute("SELECT value FROM meta WHERE key = ?", (key,))


def stamp(holder, key):
    def write(conn):
        _fence(conn, key)
    return holder._execute_write(write, patience_s=PATIENCE)
'''
DB_UNIT = {"backend": "persistence/sqlite/x.py::_fence", "legacy": "hermes_state_x.py::_fence"}


def test_a_legacy_module_functions_db_receiver_reads_like_the_backend_holder(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_DB})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": BACKEND_DB.replace("PATIENCE", "holder._PATIENCE_S")})
    assert run(legacy, branch, {"family": "x", "units": [DB_UNIT]}).failures == []
    planted = write(tmp_path / "planted", {"persistence/sqlite/x.py": BACKEND_DB.replace("PATIENCE", "holder._OTHER_S")})
    (failure,) = run(legacy, planted, {"family": "x", "units": [DB_UNIT]}).failures
    assert "x.py::_fence transaction:" in failure and "_OTHER_S" in failure


def test_a_units_names_rename_the_label_its_statements_inherit(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_DB})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": BACKEND_DB.replace("PATIENCE", "holder._WAIT_S")})
    renamed = {**DB_UNIT, "names": {"db._PATIENCE_S": "_WAIT_S"}}
    assert run(legacy, branch, {"family": "x", "units": [renamed]}).failures == []
    (failure,) = run(legacy, branch, {"family": "x", "units": [DB_UNIT]}).failures
    assert "x.py::_fence transaction:" in failure and "_WAIT_S" in failure
    elsewhere = {**DB_UNIT, "names": {"db._PATIENCE_S": "_OTHER_S"}}
    assert any("x.py::_fence transaction:" in f for f in run(legacy, branch, {"family": "x", "units": [elsewhere]}).failures)


LEGACY_COLUMNS = '''
class SessionDB:
    _COLUMNS = "id, title"


class Snapshot:
    _BODY_COLUMNS = "id, body"


def titles(db, sid):
    return db._conn.execute(f"SELECT {db._COLUMNS} FROM sessions WHERE id = ?", (sid,))


def bodies(db, sid):
    return db._conn.execute(f"SELECT {db._BODY_COLUMNS} FROM messages WHERE id = ?", (sid,))
'''
BACKEND_COLUMNS = '''
_COLUMNS = "id, title"


def titles(holder, sid):
    return holder._conn.execute(f"SELECT {_COLUMNS} FROM sessions WHERE id = ?", (sid,))
'''
TITLES_UNIT = {"backend": "persistence/sqlite/x.py::titles", "legacy": "hermes_state_x.py::titles"}


def test_a_constant_read_on_an_untyped_parameter_resolves_in_the_store_class(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_COLUMNS})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": BACKEND_COLUMNS})
    assert run(legacy, branch, {"family": "x", "units": [TITLES_UNIT]}).failures == []
    planted = write(tmp_path / "planted", {"persistence/sqlite/x.py": BACKEND_COLUMNS.replace('"id, title"', '"id, name"')})
    (failure,) = run(legacy, planted, {"family": "x", "units": [TITLES_UNIT]}).failures
    assert "x.py::titles text:" in failure


def test_a_constant_only_another_class_defines_stays_unresolved(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_COLUMNS})
    tree = check.Tree(legacy, ["hermes_state_x.py"], dict(check.DEFAULT_WRAPPERS))
    ((titles,), (bodies,)) = (tree.sites[("hermes_state_x.py", f)] for f in ("titles", "bodies"))
    assert (titles.view()["text"], bodies.view()["text"]) == (
        "SELECT id, title FROM sessions WHERE id = ?", "SELECT {_BODY_COLUMNS} FROM messages WHERE id = ?")


LEGACY_COLS_BUILDER = '''
def cols(n):
    return "id, name"


def items(db, owner):
    sql = "SELECT " + cols(len(owner)) + " FROM items WHERE owner = ?"
    return db._conn.execute(sql, (owner,))
'''
BACKEND_ITEMS = '''
from .shapes import cols


def items(holder, owner):
    sql = "SELECT " + cols(len(owner)) + " FROM items WHERE owner = ?"
    return holder._conn.execute(sql, (owner,))
'''
BACKEND_SHAPES = '''
def cols(n):
    return "id, name"
'''
BACKEND_OTHER = '''
def cols(row):
    return list(row)
'''
COLS_MANIFEST = {
    "family": "x",
    "units": [{"backend": "persistence/sqlite/x.py::items", "legacy": "hermes_state_x.py::items"}],
    "builders": [{"backend": "persistence/sqlite/shapes.py::cols", "legacy": "hermes_state_x.py::cols"}],
}


def cols_trees(tmp_path: Path, name: str, items: str = BACKEND_ITEMS, shapes: str = BACKEND_SHAPES):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_COLS_BUILDER})
    branch = write(tmp_path / name, {"persistence/sqlite/x.py": items, "persistence/sqlite/shapes.py": shapes,
                                     "persistence/sqlite/other.py": BACKEND_OTHER})
    return legacy, branch


def test_a_bare_name_call_reaches_the_function_its_import_binds_not_every_function_of_that_name(tmp_path):
    legacy, branch = cols_trees(tmp_path, "branch")
    report = run(legacy, branch, COLS_MANIFEST)
    assert (report.failures, report.counts["backend_builders"]) == ([], 1)


def test_a_changed_builder_reached_through_an_import_still_fails(tmp_path):
    legacy, branch = cols_trees(tmp_path, "planted", shapes=BACKEND_SHAPES.replace("id, name", "id, title"))
    (failure,) = run(legacy, branch, COLS_MANIFEST).failures
    assert "builder persistence/sqlite/shapes.py::cols builder:" in failure


def test_a_bare_name_bound_to_a_parameter_still_reaches_every_function_of_that_name(tmp_path):
    items = BACKEND_ITEMS.replace("def items(holder, owner):", "def items(holder, owner, cols=cols):")
    legacy, branch = cols_trees(tmp_path, "parameter", items=items)
    failures = run(legacy, branch, COLS_MANIFEST).failures
    assert "backend builder persistence/sqlite/other.py::cols has no manifest entry" in failures


LEGACY_FENCE = '''
def _fence(conn, key):
    conn.execute("SELECT value FROM meta WHERE key = ?", (key,))


def check_key(db, key):
    return db._read_retrying_ioerr(lambda conn: _fence(conn, key))
'''
LEGACY_ROOM_FENCE = '''
def _room_fence(conn, room):
    conn.execute("SELECT owner FROM rooms WHERE id = ?", (room,))


def claim(db, room):
    return db._execute_write(lambda conn: _room_fence(conn, room))
'''
BACKEND_FENCE = '''
def _fence(conn, key):
    conn.execute("SELECT value FROM meta WHERE key = ?", (key,))


def check_key(holder, key):
    return holder._read_retrying_ioerr(lambda conn: _fence(conn, key))
'''
BACKEND_ROOM_FENCE = '''
def _fence(conn, room):
    conn.execute("SELECT owner FROM rooms WHERE id = ?", (room,))


def claim(holder, room):
    return holder._execute_write(lambda conn: _fence(conn, room))
'''
FENCE_UNIT = {"backend": "persistence/sqlite/x.py::_fence", "legacy": "hermes_state_x.py::_fence"}
ROOM_FENCE_UNIT = {"backend": "persistence/sqlite/rooms.py::_fence", "legacy": "hermes_state_rooms.py::_room_fence"}


def fence_trees(tmp_path: Path, name: str, rooms: str) -> tuple[Path, Path]:
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_FENCE, "hermes_state_rooms.py": LEGACY_ROOM_FENCE})
    return legacy, write(tmp_path / name, {"persistence/sqlite/x.py": BACKEND_FENCE, "persistence/sqlite/rooms.py": rooms})


def test_a_bare_name_call_runs_only_the_function_its_scopes_bind_for_inherited_transactions(tmp_path):
    legacy, branch = fence_trees(tmp_path, "branch", BACKEND_ROOM_FENCE)
    assert run(legacy, branch, {"family": "x", "units": [FENCE_UNIT, ROOM_FENCE_UNIT]}).failures == []


@pytest.mark.parametrize("rooms", [
    "from .x import _fence\n" + BACKEND_ROOM_FENCE.split("\n\n\n")[1],
    BACKEND_ROOM_FENCE.replace("def claim(holder, room):", "def claim(holder, room, _fence=_fence):"),
], ids=["imported", "parameter"])
def test_a_bare_name_bound_to_the_function_or_to_a_parameter_still_hands_on_its_transaction(tmp_path, rooms):
    legacy, branch = fence_trees(tmp_path, "planted", rooms)
    units = [FENCE_UNIT] if rooms.startswith("from") else [FENCE_UNIT, ROOM_FENCE_UNIT]
    failures = run(legacy, branch, {"family": "x", "units": units}).failures
    assert any("x.py::_fence transaction:" in f and "write" in f for f in failures)


def test_a_builder_entry_whose_backend_function_feeds_no_sql_text_fails(tmp_path):
    legacy, branch = cols_trees(tmp_path, "branch")
    assert run(legacy, branch, COLS_MANIFEST).failures == []
    built = '"SELECT " + cols(len(owner)) + " FROM items WHERE owner = ?"'
    plain = '"SELECT id, name FROM items WHERE owner = ?"'
    _, stale = cols_trees(tmp_path, "stale", items=BACKEND_ITEMS.replace(built, plain))
    write(legacy, {"hermes_state_x.py": LEGACY_COLS_BUILDER.replace(built, plain)})
    (failure,) = run(legacy, stale, COLS_MANIFEST).failures
    assert failure == "x.json: builder persistence/sqlite/shapes.py::cols: stale entry, persistence/sqlite/shapes.py::cols feeds no SQL text"


LEGACY_VIEW_COLS = '''
class SessionDB:
    def _cols(self):
        return "id, name"

    def items(self, owner):
        return self._conn.execute("SELECT " + self._cols() + " FROM items WHERE owner = ?", (owner,))


class Snapshot:
    def _cols(self):
        return ", ".join(self.names)
'''
BACKEND_VIEW_COLS = LEGACY_VIEW_COLS.replace("SessionDB", "StoreHolder").replace("Snapshot", "_View")
VIEW_MANIFEST = {
    "family": "x",
    "legacy_modules": ["hermes_state_x.py"],
    "units": [{"backend": "persistence/sqlite/x.py::StoreHolder.items", "legacy": "hermes_state_x.py::SessionDB.items"}],
    "builders": [{"backend": "persistence/sqlite/x.py::StoreHolder._cols", "legacy": "hermes_state_x.py::SessionDB._cols"}],
}


def test_a_method_call_in_sql_text_reaches_only_the_builder_of_its_class_group(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_VIEW_COLS})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": BACKEND_VIEW_COLS})
    report = run(legacy, branch, VIEW_MANIFEST)
    assert (report.failures, report.counts["backend_builders"]) == ([], 1)


def test_a_changed_builder_of_the_class_group_still_fails(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_VIEW_COLS})
    branch = write(tmp_path / "planted", {"persistence/sqlite/x.py": BACKEND_VIEW_COLS.replace('"id, name"', '"id, title"')})
    (failure,) = run(legacy, branch, VIEW_MANIFEST).failures
    assert "builder persistence/sqlite/x.py::StoreHolder._cols builder:" in failure
