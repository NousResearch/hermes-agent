"""The SQL mapping checker on code whose legacy surroundings have not moved yet: each rule passes on a
matching pair of trees and still fails on a planted change."""

import importlib.util
import json
import sys
from pathlib import Path

CHECK = Path(__file__).resolve().parents[3] / "persistence" / "sqlite" / "_sqlmap" / "check.py"
_spec = importlib.util.spec_from_file_location("sqlmap_check", CHECK)
check = importlib.util.module_from_spec(_spec)
sys.modules["sqlmap_check"] = check
_spec.loader.exec_module(check)

READ_ONE = '''
    def _read_one(self, sql, params=()):
        with self._read_ctx() as conn:
            return conn.execute(sql, params).fetchone()
'''
LEGACY_COUNT = '''
class CountMixin:
    def count(self, owner):
        return self._read_one("SELECT COUNT(*) FROM items WHERE owner = ?", (owner,))


class Store(CountMixin):''' + READ_ONE + '''

class Adapter:
    def _read_one(self, sql, params=()):
        return self.conn.execute(sql, params).fetchone()
'''
BACKEND_COUNT = '''
class Repo:''' + READ_ONE + '''
    def count(self, owner):
        return self._read_one("SELECT COUNT(*) FROM items WHERE owner = ?", (owner,))
'''
COUNT_MANIFEST = {
    "family": "x",
    "legacy_modules": ["hermes_state_x.py::Store._read_one", "hermes_state_x.py::CountMixin.count"],
    "units": [{"backend": "persistence/sqlite/x.py::Repo._read_one", "legacy": "hermes_state_x.py::Store._read_one"},
              {"backend": "persistence/sqlite/x.py::Repo.count", "legacy": "hermes_state_x.py::CountMixin.count"}],
}


def write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def run(legacy: Path, branch: Path, manifest: dict) -> "check.Report":
    write(branch, {"persistence/sqlite/_sqlmap/x.json": json.dumps(manifest)})
    return check.Checker(legacy, branch, check.load_manifests(branch / check.MANIFEST_DIR)).run()


def test_a_method_call_reaches_only_the_helper_of_its_class_group(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_COUNT})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": BACKEND_COUNT})
    assert run(legacy, branch, COUNT_MANIFEST).failures == []
    tree = check.Tree(legacy, ["hermes_state_x.py"], dict(check.DEFAULT_WRAPPERS))
    (site,) = tree.sites[("hermes_state_x.py", "CountMixin.count")]
    assert site.view()["transaction"] == "read"


def test_a_changed_statement_behind_a_helper_still_fails(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_COUNT})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": BACKEND_COUNT.replace("owner = ?", "owner_id = ?")})
    assert any("Repo.count text:" in f and "owner_id" in f for f in run(legacy, branch, COUNT_MANIFEST).failures)


KEYS = 'CRON_PRINCIPAL = "cron"\n'
LEGACY_CRON = '''
from hermes_state_x_keys import CRON_PRINCIPAL

CRON_SQL = "DELETE FROM sessions WHERE principal = '" + CRON_PRINCIPAL + "'"


class Store:
    def drop_cron(self):
        self._execute_write(lambda conn: conn.execute(CRON_SQL))
'''
CRON_MANIFEST = {
    "family": "x",
    "units": [{"backend": "persistence/sqlite/x.py::Repo.drop_cron", "legacy": "hermes_state_x.py::Store.drop_cron"}],
}


def cron_trees(tmp_path: Path, branch_keys: str = KEYS) -> tuple[Path, Path]:
    legacy = write(tmp_path / "m1", {"hermes_state_x_keys.py": KEYS, "hermes_state_x.py": LEGACY_CRON})
    branch = write(tmp_path / "branch", {
        "hermes_state_x_keys.py": branch_keys,
        "persistence/sqlite/_ddl.py": LEGACY_CRON.split("\n\n\nclass")[0] + "\n",
        "persistence/sqlite/x.py": '''
from ._ddl import CRON_SQL


class Repo:
    def drop_cron(self):
        self._execute_write(lambda conn: conn.execute(CRON_SQL))
''',
    })
    return legacy, branch


def test_a_constant_imported_from_outside_the_backend_resolves(tmp_path):
    legacy, branch = cron_trees(tmp_path)
    assert run(legacy, branch, CRON_MANIFEST).failures == []
    tree = check.Tree(branch, ["persistence/sqlite/_ddl.py", "persistence/sqlite/x.py"], dict(check.DEFAULT_WRAPPERS))
    (site,) = tree.sites[("persistence/sqlite/x.py", "Repo.drop_cron")]
    assert site.view()["text"] == "DELETE FROM sessions WHERE principal = 'cron'"
    assert "hermes_state_x_keys.py" not in tree.modules


def test_a_changed_constant_outside_the_backend_still_fails(tmp_path):
    legacy, branch = cron_trees(tmp_path, branch_keys='CRON_PRINCIPAL = "crontab"\n')
    assert any("Repo.drop_cron text:" in f and "crontab" in f for f in run(legacy, branch, CRON_MANIFEST).failures)


LEGACY_FLAG = '''
class Store:
    def read_flag(self, key):
        return self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()

    def cleanup(self):
        self._execute_write(lambda conn: conn.execute("DELETE FROM meta"))
'''
BACKEND_FLAG = '''
_seen = {}


class Repo:
    def read_flag(self, key):
        return self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()

    def get(self, scope, key):
        return self.read_flag(key)

    def cleanup(self):
        def _do(conn):
            _seen.get(conn)
            conn.execute("DELETE FROM meta")
        self._execute_write(_do)
'''
FLAG_MANIFEST = {
    "family": "x",
    "units": [{"backend": "persistence/sqlite/x.py::Repo.read_flag", "legacy": "hermes_state_x.py::Store.read_flag"},
              {"backend": "persistence/sqlite/x.py::Repo.cleanup", "legacy": "hermes_state_x.py::Store.cleanup"}],
}


def test_a_call_whose_arguments_do_not_fit_passes_on_no_transaction(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_FLAG})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": BACKEND_FLAG})
    assert run(legacy, branch, FLAG_MANIFEST).failures == []


def test_a_real_call_inside_a_transaction_still_passes_it_on(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_FLAG})
    branch = write(tmp_path / "branch", {
        "persistence/sqlite/x.py": BACKEND_FLAG.replace("_seen.get(conn)", 'self.get(None, "k")')})
    assert any("Repo.read_flag transaction:" in f for f in run(legacy, branch, FLAG_MANIFEST).failures)


READ_FLAG = '''
    def read_flag(self, key):
        return self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
'''
SWEEP = '''
    def sweep(self):
        def _do(conn):
            self.read_flag("swept")
            conn.execute("DELETE FROM meta WHERE key = 'swept'")
        self._execute_write(_do)
'''
FLAG_UNIT = {"backend": "persistence/sqlite/x.py::Repo.read_flag", "legacy": "hermes_state_x.py::Store.read_flag",
             "inherits": "write"}
SWEEP_UNIT = {"backend": "persistence/sqlite/x.py::Repo.sweep", "legacy": "hermes_state_x.py::Store.sweep"}


def sweep_trees(tmp_path: Path, backend: str) -> tuple[Path, Path]:
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": "\nclass Store:" + READ_FLAG + SWEEP})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": "\nclass Repo:" + backend})
    return legacy, branch


def test_a_declared_transaction_stands_in_for_callers_not_moved(tmp_path):
    legacy, branch = sweep_trees(tmp_path, READ_FLAG)
    plain = run(legacy, branch, {"family": "x", "units": [{k: v for k, v in FLAG_UNIT.items() if k != "inherits"}]})
    assert any("Repo.read_flag transaction:" in f for f in plain.failures)
    report = run(legacy, branch, {"family": "x", "units": [FLAG_UNIT]})
    assert report.failures == []
    assert any("Repo.read_flag: write from 1 legacy callers, first hermes_state_x.py::Store.sweep" in line
               for line in report.inherited)


def test_a_declaration_other_than_the_legacy_transaction_fails(tmp_path):
    legacy, branch = sweep_trees(tmp_path, READ_FLAG)
    found = run(legacy, branch, {"family": "x", "units": [{**FLAG_UNIT, "inherits": "read"}]}).failures
    assert any("declares inherited 'read', but the legacy function inherits 'write'" in f for f in found)


def test_a_declaration_fails_once_every_legacy_caller_has_moved(tmp_path):
    legacy, branch = sweep_trees(tmp_path, READ_FLAG + SWEEP)
    found = run(legacy, branch, {"family": "x", "units": [FLAG_UNIT, SWEEP_UNIT]}).failures
    assert any("Repo.read_flag transaction: declares inherited 'write', but every legacy caller has moved" in f
               for f in found)


def test_a_declaration_does_not_hide_a_backend_caller_in_another_transaction(tmp_path):
    reader = '''
    def peek(self):
        with self._read_ctx():
            return self.read_flag("swept")
'''
    legacy, branch = sweep_trees(tmp_path, READ_FLAG + reader)
    found = run(legacy, branch, {"family": "x", "units": [FLAG_UNIT]}).failures
    assert any("Repo.read_flag transaction:" in f and "mixed(read | write)" in f for f in found)


def test_a_declaration_the_backend_callers_give_already_fails(tmp_path):
    audit = SWEEP.replace("def sweep", "def audit").replace("'swept'", "'audited'").replace('"swept"', '"audited"')
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": "\nclass Store:" + READ_FLAG + SWEEP + audit})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": "\nclass Repo:" + READ_FLAG + SWEEP})
    found = run(legacy, branch, {"family": "x", "units": [FLAG_UNIT, SWEEP_UNIT]}).failures
    assert any("Repo.read_flag transaction: declares inherited 'write', but its backend callers and the other "
               "declarations give it already" in f for f in found)


CONTAINER_LEGACY = '''
class Store:
    def get_meta(self, key):
        return self._conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()

    def touch(self, key):
        def body(conn):
            conn.execute("UPDATE state_meta SET value = value WHERE key = ?", (key,))
        return self._execute_write(body)
'''
CONTAINER_BACKEND = '''
_open: dict = {}


class Holder:
    def get_meta(self, key):
        return self._conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()

    def touch(self, key):
        def body(conn):
            conn.execute("UPDATE state_meta SET value = value WHERE key = ?", (key,))
            return LOOKUP
        return self._execute_write(body)


class Repo:
    def get(self, key):
        return self.holder.get_meta(key)
'''
CONTAINER_MANIFEST = {
    "family": "x",
    "units": [{"backend": "persistence/sqlite/x.py::Holder.get_meta", "legacy": "hermes_state_x.py::Store.get_meta"},
              {"backend": "persistence/sqlite/x.py::Holder.touch", "legacy": "hermes_state_x.py::Store.touch"}],
}


def test_a_method_of_a_module_level_container_is_not_a_call_of_the_tree(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": CONTAINER_LEGACY})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": CONTAINER_BACKEND.replace("LOOKUP", "_open.get(key)")})
    assert run(legacy, branch, CONTAINER_MANIFEST).failures == []
    planted = write(tmp_path / "planted", {"persistence/sqlite/x.py": CONTAINER_BACKEND.replace("LOOKUP", "self.repo.get(key)")})
    assert any("Holder.get_meta transaction" in f for f in run(legacy, planted, CONTAINER_MANIFEST).failures)


LEGACY_WHERE = '''
def _where(db, owner):
    db._conn.execute("UPDATE items SET seen = 1 WHERE owner = ?", (owner,))
    return "owner = ?"


def count_items(db, owner):
    return db._conn.execute("SELECT COUNT(*) FROM items WHERE " + _where(db, owner), (owner,))


def items(db, owner):
    return db._conn.execute("SELECT id FROM items WHERE owner = ?", (owner,))
'''
BACKEND_ITEMS = '''
def items(holder, owner):
    return holder._conn.execute("SELECT id FROM items WHERE owner = ?", (owner,))
'''
STAYS = "the count stays in the legacy module"


def test_a_not_moved_builder_leaves_the_statements_of_its_function_to_entries_of_their_own(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_state_x.py": LEGACY_WHERE})
    branch = write(tmp_path / "branch", {"persistence/sqlite/x.py": BACKEND_ITEMS})
    manifest = {
        "family": "x",
        "legacy_modules": ["hermes_state_x.py"],
        "units": [{"backend": "persistence/sqlite/x.py::items", "legacy": "hermes_state_x.py::items"}],
        "not_moved": [{"legacy": "hermes_state_x.py::count_items", "reason": STAYS},
                      {"legacy": "hermes_state_x.py::_where", "reason": STAYS},
                      {"legacy": "hermes_state_x.py::_where#0", "reason": STAYS}],
    }
    assert run(legacy, branch, manifest).failures == []
    manifest["not_moved"].pop()
    (failure,) = run(legacy, branch, manifest).failures
    assert failure.startswith("legacy statement hermes_state_x.py::_where#0 (line 3) is unmapped")


SUPPORT = '''
def open_conn(conn, foreign_keys):
    if foreign_keys:
        conn.execute("PRAGMA foreign_keys=ON")
'''
SUPPORT_UNIT = {"backend": "sqlite_support.py::open_conn", "legacy": "hermes_cli/sqlite_util.py::open_conn"}


def test_sqlite_support_is_a_backend_source_mapped_to_the_helper_it_copies(tmp_path):
    legacy = write(tmp_path / "m1", {"hermes_cli/sqlite_util.py": SUPPORT})
    branch = write(tmp_path / "branch", {"sqlite_support.py": SUPPORT, "persistence/sqlite/__init__.py": ""})
    report = run(legacy, branch, {"family": "x", "units": [SUPPORT_UNIT]})
    assert (report.failures, report.counts["backend_statements"]) == ([], 1)
    (failure,) = run(legacy, branch, {"family": "x"}).failures
    assert failure.startswith("backend statement sqlite_support.py::open_conn#0 (line 4) has no manifest entry")
    write(branch, {"sqlite_support.py": SUPPORT.replace("foreign_keys=ON", "foreign_keys=OFF")})
    (failure,) = run(legacy, branch, {"family": "x", "units": [SUPPORT_UNIT]}).failures
    assert "sqlite_support.py::open_conn text:" in failure
