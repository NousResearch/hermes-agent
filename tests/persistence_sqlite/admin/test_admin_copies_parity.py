"""The backend's private copies of application helpers against their originals: the admin's page-level salvage lane
and its column history, the profile label and gateway lock record the reconcile messages name, the hooks profile
repair calls, and the zeroed-file probe the state file helpers run; and each waived line of the admin's copies
against main's function, through the table in ``core/_moved.py``."""

from __future__ import annotations

import ast
import inspect
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from gateway import runtime_ownership
from gateway import session as gateway_session
from gateway import status as gateway_status
from hermes_cli import backup
from persistence.sqlite._schema_ddl import SCHEMA_SQL
from persistence.sqlite._state import dbfile
from persistence.sqlite.admin import _lost_and_found, _reconcile, _schema_history, profile_repair
from tests.persistence_sqlite._legacy_store import oracle_tree
from tests.persistence_sqlite.authority.test_coordination_obligations import main_ledger
from tests.persistence_sqlite.core._moved import WAIVER_IDS, WAIVERS, assert_copy

_ADMIN_WAIVERS = [(case, case_id) for case, case_id in zip(WAIVERS, WAIVER_IDS)
                  if case[0].backend.startswith("persistence/sqlite/admin/")]

# Where each copy imports its dependencies from differs on purpose.
_IMPORTS = (ast.Import, ast.ImportFrom)


def _definitions(module) -> dict[str, str]:
    """Every top-level statement of ``module`` (a module, or the path of main's file) but its imports and docstring,
    keyed by the name it binds."""
    tree = ast.parse(module.read_text(encoding="utf-8") if isinstance(module, Path) else inspect.getsource(module))
    body = tree.body[1:] if ast.get_docstring(tree) else tree.body
    out = {}
    for node in body:
        if isinstance(node, _IMPORTS):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign):
            name = ",".join(ast.unparse(target) for target in node.targets)
        elif isinstance(node, ast.AnnAssign):
            name = ast.unparse(node.target)
        else:
            name = ast.unparse(node)
        out[name] = ast.dump(node)
    return out


def test_the_salvage_lane_copy_defines_what_the_original_defines():
    copy, original = _definitions(_lost_and_found), _definitions(oracle_tree() / "hermes_cli" / "session_lost_and_found.py")
    assert copy.pop("logger") != original.pop("logger")
    assert copy == original


def test_the_column_history_copy_defines_what_the_original_defines():
    assert _definitions(_schema_history) == _definitions(oracle_tree() / "hermes_cli" / "session_schema_history.py")


def test_the_zeroed_file_probe_copy_is_the_original_text():
    assert inspect.getsource(dbfile.is_zeroed_sqlite_file) == inspect.getsource(backup.is_zeroed_sqlite_file)


def _declared_now(table: str) -> tuple[str, ...]:
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript(SCHEMA_SQL)
        return tuple(str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")'))


@pytest.mark.parametrize("table", sorted(_schema_history.SCHEMA_HISTORY))
def test_the_replayed_column_history_ends_at_the_backend_schema(table):
    assert _schema_history.current_declared_columns(table) == _declared_now(table)


def test_the_salvage_lane_logs_under_the_original_logger():
    assert _lost_and_found.logger.name == "hermes_cli.session_lost_and_found"


def test_the_auxiliary_table_initializer_creates_the_gateway_ledger_table(tmp_path):
    shapes = []
    for name, init in (("legacy.db", main_ledger()._initialize_schema), ("admin.db", _lost_and_found._AUXILIARY_TABLE_SCHEMAS["delivery_obligations"])):
        with closing(sqlite3.connect(tmp_path / name)) as conn:
            conn.execute("CREATE TABLE delivery_obligations (obligation_id TEXT PRIMARY KEY, session_key TEXT NOT NULL, "
                         "platform TEXT NOT NULL, chat_id TEXT NOT NULL, thread_id TEXT, content TEXT NOT NULL, "
                         "state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, "
                         "updated_at REAL NOT NULL, owner_pid INTEGER, owner_started_at INTEGER, last_error TEXT)")
            init(conn)
            init(conn)
            shapes.append(conn.execute("PRAGMA table_info(delivery_obligations)").fetchall())
        with closing(sqlite3.connect(tmp_path / f"fresh-{name}")) as conn:
            init(conn)
            shapes.append(conn.execute("SELECT sql FROM sqlite_master").fetchall())
    assert shapes[0] == shapes[2] and shapes[1] == shapes[3]
    assert "adapter_profile" in {row[1] for row in shapes[0]}


def _homes(root: Path) -> list[Path]:
    return [root, root / "profiles" / "work", root / "profiles" / "Bad Name", root / "profiles" / "main",
            root / "elsewhere", root / ".." / root.name / "profiles" / "work", Path("~/relative")]


def test_the_profile_label_copy_names_every_home_as_the_original(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    for home in _homes(root):
        assert _reconcile._profile_label_for_home(home) == gateway_status._profile_label_for_home(home), home
    assert _reconcile._profile_label_for_home(root / "profiles" / "work") == "work"


def test_the_canonical_home_copy_matches_the_original(tmp_path):
    for home in _homes(tmp_path):
        assert _reconcile.canonical_home(home) == runtime_ownership.canonical_home(home)


@pytest.mark.parametrize("content", [None, "", "  \n", '{"pid": 12, "hermes_home": "/x"}', "4321", "[1, 2]",
                                     "not json", b"\xff\xfe\x00garbage", '\ufeff{"pid": 7}'])
def test_the_lock_record_reader_copy_reads_every_file_as_the_original(tmp_path, content):
    path = tmp_path / "gateway.lock"
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif content is not None:
        path.write_text(content, encoding="utf-8")
    assert _reconcile._read_json_file(path, bare_pid_ok=True) == gateway_status._read_gateway_lock_record(path)


@pytest.mark.parametrize("profile", [None, "", "default", "main", "work", "main~"])
def test_profile_repair_writes_the_gateway_key_namespace(profile):
    assert profile_repair._session_key_namespace(profile) == gateway_session._session_key_namespace(profile)


@pytest.mark.parametrize(("copy", "index"), [case for case, _ in _ADMIN_WAIVERS], ids=[i for _, i in _ADMIN_WAIVERS])
def test_each_waived_line_is_a_line_of_mains_function(copy, index):
    assert_copy(copy, index, oracle_tree())
