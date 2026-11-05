"""The SQL mapping checker on begin-mode wrapper entries: a transaction label keeps only the kind and the begin
mode, a file-scoped entry wins over the bare name, and a ``with`` block that begins ``BEGIN IMMEDIATE`` itself
counts as immediate; a planted begin mode fails.
"""

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


LEGACY_ROOMS = '''
def create(db_path, name):
    with _transaction(db_path, immediate=True) as conn:
        conn.execute("INSERT INTO rooms (name) VALUES (?)", (name,))


def rename(db_path, name):
    with _transaction(db_path) as conn:
        conn.execute("UPDATE rooms SET name = ?", (name,))


def sync(db_path, name):
    with _transaction(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM rooms WHERE name = ?", (name,))
'''
LEGACY_DRIVER = '''
def lease(db_path, room):
    with _transaction(db_path) as conn:
        conn.execute("UPDATE leases SET room = ?", (room,))
'''
BACKEND = '''
class Rooms:
    def create(self, placement, name):
        def body(conn):
            conn.execute("INSERT INTO rooms (name) VALUES (?)", (name,))
        return self._files._hosted_write(placement, body, immediate=CREATE)

    def rename(self, placement, name):
        def body(conn):
            conn.execute("UPDATE rooms SET name = ?", (name,))
        return self._files._hosted_write(placement, body, immediate=RENAME)

    def sync(self, placement, name):
        def body(conn):
            conn.execute("DELETE FROM rooms WHERE name = ?", (name,))
        return self._files._hosted_write(placement, body, immediate=SYNC)

    def lease(self, placement, room):
        def body(conn):
            conn.execute("UPDATE leases SET room = ?", (room,))
        return self._files._hosted_write(placement, body, immediate=LEASE)
'''
WRAPPERS = {
    "_hosted_write": {"kind": "write", "begin": "immediate"},
    "gateway/rooms.py::_transaction": {"kind": "write", "begin": "immediate", "default": False},
    "gateway/driver.py::_transaction": {"kind": "write", "begin": "immediate", "default": True},
}
UNITS = [
    {"backend": "persistence/sqlite/rooms.py::Rooms.create", "legacy": "gateway/rooms.py::create"},
    {"backend": "persistence/sqlite/rooms.py::Rooms.rename", "legacy": "gateway/rooms.py::rename"},
    {"backend": "persistence/sqlite/rooms.py::Rooms.sync", "legacy": "gateway/rooms.py::sync#1"},
    {"backend": "persistence/sqlite/rooms.py::Rooms.lease", "legacy": "gateway/driver.py::lease"},
]
MATCHING = {"CREATE": "True", "RENAME": "False", "SYNC": "True", "LEASE": "True"}


def backend(root: Path, **modes: str) -> Path:
    text = BACKEND
    for name, value in {**MATCHING, **modes}.items():
        text = text.replace(name, value)
    return write(root, {"persistence/sqlite/rooms.py": text})


@pytest.fixture
def legacy(tmp_path):
    return write(tmp_path / "m1", {"gateway/rooms.py": LEGACY_ROOMS, "gateway/driver.py": LEGACY_DRIVER})


def test_labels_keep_only_the_begin_mode(tmp_path, legacy):
    report = run(legacy, backend(tmp_path / "branch"), {"family": "x", "wrappers": WRAPPERS, "units": UNITS})
    assert report.failures == []


@pytest.mark.parametrize("planted, verb", [
    ({"CREATE": "False"}, "Rooms.create"),
    ({"RENAME": "True"}, "Rooms.rename"),
    ({"SYNC": "False"}, "Rooms.sync"),
    ({"LEASE": "False"}, "Rooms.lease"),
])
def test_a_planted_begin_mode_fails(tmp_path, legacy, planted, verb):
    report = run(legacy, backend(tmp_path / "planted", **planted), {"family": "x", "wrappers": WRAPPERS, "units": UNITS})
    assert [f for f in report.failures if "transaction" in f] and all(verb in f for f in report.failures)


def test_the_file_scoped_entry_wins_over_the_bare_name(tmp_path, legacy):
    wrappers = {**WRAPPERS, "_transaction": {"kind": "write", "begin": "immediate", "default": False}}
    del wrappers["gateway/driver.py::_transaction"]
    report = run(legacy, backend(tmp_path / "bare"), {"family": "x", "wrappers": wrappers, "units": UNITS})
    assert report.failures and all("Rooms.lease" in f for f in report.failures)


def test_plain_kind_entries_still_compare_every_argument(tmp_path, legacy):
    wrappers = {"_hosted_write": "write", "_transaction": "write"}
    report = run(legacy, backend(tmp_path / "plain"), {"family": "x", "wrappers": wrappers, "units": UNITS})
    assert any("Rooms.create transaction" in f for f in report.failures)


@pytest.mark.parametrize("entry", [{"kind": "write", "begin": 1}, {"kind": "write", "default": "no"}, {"begin": "immediate"},
                                   {"kind": "write", "mode": "immediate"}])
def test_a_malformed_begin_mode_entry_is_unusable_input(entry):
    with pytest.raises(check.ManifestError):
        check.validate("x.json", {"family": "x", "wrappers": {"_hosted_write": entry}})
