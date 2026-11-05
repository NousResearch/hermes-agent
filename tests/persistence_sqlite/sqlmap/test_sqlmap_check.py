"""The SQL mapping checker: a matching fixture passes, and every planted change fails it."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CHECK = ROOT / "persistence" / "sqlite" / "_sqlmap" / "check.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
LEGACY = FIXTURES / "m1"
REPO = "persistence/sqlite/demo/repo.py"
MANIFEST = "persistence/sqlite/_sqlmap/demo.json"

_spec = importlib.util.spec_from_file_location("sqlmap_check", CHECK)
check = importlib.util.module_from_spec(_spec)
sys.modules["sqlmap_check"] = check
_spec.loader.exec_module(check)


@pytest.fixture
def branch(tmp_path):
    target = tmp_path / "branch"
    shutil.copytree(FIXTURES / "branch", target)
    return target


def plant(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new), encoding="utf-8")


def edit_manifest(branch: Path, change) -> None:
    path = branch / MANIFEST
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def run(branch: Path) -> "check.Report":
    manifests = check.load_manifests(branch / check.MANIFEST_DIR)
    return check.Checker(LEGACY, branch, manifests).run()


def failures(branch: Path) -> list[str]:
    report = run(branch)
    assert report.failures, "the planted change went unnoticed"
    return report.failures


def test_matching_fixture_passes_and_lists_ruled_exceptions(branch):
    report = run(branch)
    assert report.failures == []
    assert report.counts["backend_statements"] == 9
    assert report.counts["legacy_statements"] == 12
    assert len(report.exceptions) == 1 and "[stated difference] transaction" in report.exceptions[0]
    assert len(report.not_moved) == 2


def test_reordered_statement_fails_as_order(branch):
    plant(branch / REPO, """            conn.execute(_TOUCH_SQL, (now, cur.lastrowid))
            conn.execute("UPDATE owners SET item_count = item_count + 1 WHERE id = ?", (owner,))""",
          """            conn.execute("UPDATE owners SET item_count = item_count + 1 WHERE id = ?", (owner,))
            conn.execute(_TOUCH_SQL, (now, cur.lastrowid))""")
    assert any("add_item order:" in f for f in failures(branch))


def test_changed_column_fails_as_text(branch):
    plant(branch / REPO, "SET item_count = item_count + 1", "SET items_count = items_count + 1")
    assert any("add_item text:" in f and "items_count" in f for f in failures(branch))


def test_changed_shared_column_list_fails_every_statement_using_it(branch):
    plant(branch / "persistence/sqlite/demo/_sql.py", '"id, name, size"', '"id, name, bytes"')
    found = failures(branch)
    assert any("add_item text:" in f for f in found)
    assert any("find_items text:" in f for f in found)


def test_swapped_parameter_fails_as_binding(branch):
    plant(branch / REPO, "(None, name, size, owner)", "(None, size, name, owner)")
    assert any("add_item binding:" in f for f in failures(branch))


def test_reordered_parameter_building_fails_as_builder(branch):
    plant(branch / REPO, """            clauses.append(f"name IN ({_placeholders(len(names))})")
            params.extend(names)""", """            params.extend(names)
            clauses.append(f"name IN ({_placeholders(len(names))})")""")
    assert any("find_items builder:" in f for f in failures(branch))


def test_changed_condition_fails_as_builder(branch):
    plant(branch / REPO, 'clauses.append("size >= ?")', 'clauses.append("size > ?")')
    assert any("find_items builder:" in f for f in failures(branch))


def test_changed_sql_builder_function_fails(branch):
    plant(branch / "persistence/sqlite/demo/_sql.py", 'return ", ".join', 'return ",".join')
    assert any("builder persistence/sqlite/demo/_sql.py::_placeholders builder:" in f for f in failures(branch))


def test_changed_transaction_kind_fails(branch):
    plant(branch / REPO, """        self._holder.execute_write(lambda conn: conn.executemany("UPDATE items SET name = ? WHERE id = ?", pairs))""",
          """        with self._holder.read_ctx() as conn:
            conn.executemany("UPDATE items SET name = ? WHERE id = ?", pairs)""")
    assert any("rename_many transaction:" in f and "read" in f for f in failures(branch))


def test_split_transaction_fails(branch):
    plant(branch / REPO, """            conn.execute("UPDATE owners SET item_count = item_count + 1 WHERE id = ?", (owner,))
            return cur.lastrowid

        return self._holder.execute_write(_do)""", """            return cur.lastrowid

        def _bump(conn):
            conn.execute("UPDATE owners SET item_count = item_count + 1 WHERE id = ?", (owner,))

        rowid = self._holder.execute_write(_do)
        self._holder.execute_write(_bump)
        return rowid""")
    assert any("is split over backend transactions" in f for f in failures(branch))


def test_changed_execute_method_fails(branch):
    plant(branch / REPO, 'conn.executescript("DELETE FROM items; DELETE FROM owners;")',
          'conn.execute("DELETE FROM items; DELETE FROM owners;")')
    assert any("reset method:" in f for f in failures(branch))


def test_backend_statement_without_entry_fails(branch):
    plant(branch / REPO, "    def drain_ledger(self):", """    def purge(self):
        self._holder.execute_write(lambda conn: conn.execute("DELETE FROM items WHERE size = 0"))

    def drain_ledger(self):""")
    assert any("DemoRepository.purge#0" in f and "has no manifest entry" in f for f in failures(branch))


def test_backend_builder_without_entry_fails(branch):
    edit_manifest(branch, lambda data: data.update(builders=[]))
    found = failures(branch)
    assert any(f.startswith("backend builder persistence/sqlite/demo/_sql.py::_placeholders") for f in found)
    assert any(f.startswith("legacy builder hermes_state_demo.py::_placeholders") for f in found)


def test_unmapped_legacy_statement_of_a_moved_family_fails(branch):
    edit_manifest(branch, lambda data: data.update(not_moved=data["not_moved"][1:]))
    found = failures(branch)
    assert any("legacy statement hermes_state_demo.py::DemoMixin.drain_ledger#0" in f for f in found)
    assert any("legacy statement hermes_state_demo.py::DemoMixin.drain_ledger#2" in f for f in found)


def test_unruled_difference_fails_and_unused_exception_fails(branch):
    def move_exception(data):
        unit = data["units"][6]
        data["units"][5]["exception"] = unit.pop("exception")
    edit_manifest(branch, move_exception)
    found = failures(branch)
    assert any("drain_ledger transaction:" in f for f in found)
    assert any("DemoRepository.reset: exception stated difference excuses no transaction difference" in f for f in found)


def test_statement_with_no_legacy_source_is_a_ruled_exception(branch):
    plant(branch / REPO, "    def drain_ledger(self):", """    def purge(self):
        self._holder.execute_write(lambda conn: conn.execute("DELETE FROM items WHERE size = 0"))

    def drain_ledger(self):""")
    unit = {"backend": f"{REPO}::DemoRepository.purge", "legacy": None,
            "exception": {"ruling": "boundary translation", "aspects": ["source"], "reason": "a new verb"}}
    edit_manifest(branch, lambda data: data["units"].append(unit))
    report = run(branch)
    assert report.failures == []
    assert any("[boundary translation] source:" in e and "DELETE FROM items WHERE size = 0" in e for e in report.exceptions)


def test_exception_ruling_outside_the_categories_is_refused(branch):
    edit_manifest(branch, lambda data: data["units"][6]["exception"].update(ruling="Q25"))
    with pytest.raises(check.ManifestError, match=r"units\[6\]: ruling 'Q25' is not one of"):
        check.load_manifests(branch / check.MANIFEST_DIR)


def test_not_moved_ruling_outside_the_categories_is_refused(branch):
    edit_manifest(branch, lambda data: data["not_moved"][0].update(ruling="pending"))
    with pytest.raises(check.ManifestError, match=r"not_moved\[0\]: ruling 'pending' is not one of"):
        check.load_manifests(branch / check.MANIFEST_DIR)


def test_renamed_name_needs_a_names_entry(branch):
    plant(branch / REPO, """    def rename_many(self, pairs):
        self._holder.execute_write(lambda conn: conn.executemany("UPDATE items SET name = ? WHERE id = ?", pairs))""",
          """    def rename_many(self, renames):
        self._holder.execute_write(lambda conn: conn.executemany("UPDATE items SET name = ? WHERE id = ?", renames))""")
    assert any("rename_many binding:" in f for f in failures(branch))
    edit_manifest(branch, lambda data: data["units"][4].update(names={"pairs": "renames"}))
    assert run(branch).failures == []


def test_reference_to_a_missing_function_fails(branch):
    edit_manifest(branch, lambda data: data["units"][1].update(legacy="hermes_state_demo.py::DemoMixin.add"))
    assert any("DemoMixin.add: no SQL statements there" in f for f in failures(branch))


def test_whitespace_and_comments_outside_literals_do_not_matter():
    assert check.normalize_sql("SELECT  a ,b\n FROM t -- note\n WHERE x = 'a  b' /* c */") == \
        "SELECT a, b FROM t WHERE x = 'a  b'"
    assert check.normalize_sql("SELECT 'a  b'") != check.normalize_sql("SELECT 'a b'")


def test_sql_passed_to_a_helper_is_checked_at_the_call():
    tree = check.Tree(LEGACY, ["hermes_state_demo.py", "hermes_state_core.py"], dict(check.DEFAULT_WRAPPERS))
    (site,) = tree.sites[("hermes_state_demo.py", "DemoMixin.count_items")]
    view = site.view()
    assert view["text"] == "SELECT COUNT (*) FROM items WHERE owner = ?"
    assert view["binding"] == "(owner,)"
    assert view["transaction"] == "read"


@pytest.mark.parametrize("content", ["{not json", json.dumps({"family": "demo", "unit": []})])
def test_unusable_manifest_exits_2(branch, content):
    (branch / MANIFEST).write_text(content, encoding="utf-8")
    result = subprocess.run([sys.executable, str(CHECK), "check", "--legacy", str(LEGACY), "--branch", str(branch)],
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 2, result.stdout + result.stderr


def test_cli_exit_codes_and_json(branch):
    command = [sys.executable, str(CHECK), "check", "--legacy", str(LEGACY), "--branch", str(branch), "--json"]
    passed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert json.loads(passed.stdout)["ok"] is True
    plant(branch / REPO, "ORDER BY id", "ORDER BY name")
    failed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    assert failed.returncode == 1
    assert any("find_items text:" in f for f in json.loads(failed.stdout)["failures"])


def test_scan_prints_every_statement_reference(branch):
    result = subprocess.run([sys.executable, str(CHECK), "scan", "--tree", str(branch), "--wrapper", "execute_write=write",
                             "persistence/sqlite/demo"], capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    assert f"{REPO}::DemoRepository.add_item#2" in result.stdout
    assert "builder persistence/sqlite/demo/_sql.py::_placeholders" in result.stdout


def test_draft_proposes_the_fixture_units(branch):
    result = subprocess.run([sys.executable, str(CHECK), "draft", "--legacy", str(LEGACY), "--branch", str(branch),
                             "--family", "demo", "--legacy-module", "hermes_state_demo.py", f"{REPO}"],
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    units = {u["backend"]: u["legacy"] for u in json.loads(result.stdout)["units"]}
    assert units[f"{REPO}::DemoRepository.add_item"] == "hermes_state_demo.py::DemoMixin.add_item#0,1,2"
    assert units[f"{REPO}::DemoRepository.drain_ledger"] == "hermes_state_demo.py::DemoMixin.drain_ledger#1"
