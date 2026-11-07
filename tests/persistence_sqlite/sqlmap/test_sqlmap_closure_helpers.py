"""The SQL mapping checker on helpers that run the SQL they are handed inside a closure: the statement is
checked at each call of the helper, and a planted change there fails."""

import importlib.util
import json
import sys
from pathlib import Path

CHECK = Path(__file__).resolve().parents[3] / "persistence" / "sqlite" / "_sqlmap" / "check.py"
_spec = importlib.util.spec_from_file_location("sqlmap_check", CHECK)
check = importlib.util.module_from_spec(_spec)
sys.modules["sqlmap_check"] = check
_spec.loader.exec_module(check)

STORE = '''
class {name}:
    def _write_sql(self, sql, params=(), *, many=False, patience_s=None):
        def _do(conn):
            (conn.executemany if many else conn.execute)(sql, params)
        self._execute_write(_do, patience_s=patience_s)

    def _write_rowcount(self, sql, params=(), *, patience_s=None):
        def _do(conn):
            rowcount = conn.execute(sql, params).rowcount
            if rowcount is None or rowcount < 0:
                rowcount = conn.execute("SELECT changes()").fetchone()[0]
            return rowcount
        return self._execute_write(_do, patience_s=patience_s)

    def _write_patch(self, item_id, sql="UPDATE items SET body = ? WHERE id = ?", params=None):
        def _do(conn):
            body = conn.execute("SELECT body FROM items WHERE id = ?", (item_id,)).fetchone()[0]
            conn.execute(sql, params(body) if params else (body, item_id))
        self._execute_write(_do)

    def rename(self, item_id, name):
        return self._write_rowcount("UPDATE items SET name = ? WHERE id = ?", (name, item_id))

    def drop(self, owner, keys):
        self._write_sql("DELETE FROM items WHERE owner = ? AND key = ?", [(owner, k) for k in keys], many=True)

    def retitle(self, item_id, title):
        self._write_patch(item_id, "UPDATE items SET title = ?, body = ? WHERE id = ?", lambda body: (title, body, item_id))

    def touch(self, item_id):
        self._write_patch(item_id)
'''
FUNCS = ("_write_sql", "_write_rowcount", "_write_patch", "rename", "drop", "retitle", "touch")
MANIFEST = {
    "family": "x",
    "legacy_modules": ["hermes_state_x.py"],
    "units": [{"backend": f"persistence/sqlite/x.py::Repo.{f}", "legacy": f"hermes_state_x.py::Store.{f}"} for f in FUNCS],
}


def run(tmp_path: Path, backend: str) -> "check.Report":
    files = {"m1/hermes_state_x.py": STORE.format(name="Store"), "branch/persistence/sqlite/x.py": backend,
             "branch/persistence/sqlite/_sqlmap/x.json": json.dumps(MANIFEST)}
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    branch = tmp_path / "branch"
    return check.Checker(tmp_path / "m1", branch, check.load_manifests(branch / check.MANIFEST_DIR)).run()


def planted(tmp_path: Path, old: str, new: str) -> list[str]:
    backend = STORE.format(name="Repo")
    assert backend.count(old) == 1
    return run(tmp_path, backend.replace(old, new)).failures


def test_sql_run_inside_a_helper_closure_is_checked_at_each_call(tmp_path):
    assert run(tmp_path, STORE.format(name="Repo")).failures == []
    tree = check.Tree(tmp_path / "m1", ["hermes_state_x.py"], dict(check.DEFAULT_WRAPPERS))
    views = {f: [s.view() for s in tree.sites[("hermes_state_x.py", f"Store.{f}")]] for f in FUNCS}
    (rename,) = views["rename"]
    assert (rename["text"], rename["binding"], rename["transaction"]) == (
        "UPDATE items SET name = ? WHERE id = ?", "(name, item_id)", "write(patience_s=None)")
    (drop,) = views["drop"]
    assert (drop["method"], drop["text"]) == ("executemany", "DELETE FROM items WHERE owner = ? AND key = ?")
    assert views["_write_sql"][0]["method"] == "executemany if many else execute"
    (retitle,) = views["retitle"]
    assert retitle["text"] == "UPDATE items SET title = ?, body = ? WHERE id = ?"
    assert retitle["binding"].startswith("(lambda body: (title, body, item_id))(body)")
    (touch,) = views["touch"]
    assert (touch["text"], touch["transaction"]) == ("UPDATE items SET body = ? WHERE id = ?", "write")


def test_a_changed_statement_passed_to_a_closure_helper_fails_as_text(tmp_path):
    failures = planted(tmp_path, "SET name = ?", "SET label = ?")
    assert len(failures) == 1 and "Repo.rename text:" in failures[0]


def test_a_swapped_parameter_passed_to_a_closure_helper_fails_as_binding(tmp_path):
    failures = planted(tmp_path, "(name, item_id))", "(item_id, name))")
    assert len(failures) == 1 and "Repo.rename binding:" in failures[0]


def test_a_swapped_parameter_built_by_a_passed_callable_fails_as_binding(tmp_path):
    failures = planted(tmp_path, "lambda body: (title, body, item_id)", "lambda body: (body, title, item_id)")
    assert len(failures) == 1 and "Repo.retitle binding:" in failures[0]


def test_a_dropped_many_flag_fails_as_method(tmp_path):
    failures = planted(tmp_path, ", many=True)", ")")
    assert len(failures) == 1 and "Repo.drop method:" in failures[0]
