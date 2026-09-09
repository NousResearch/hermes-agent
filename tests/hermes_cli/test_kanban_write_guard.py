"""#69283: kanban write guard prevents tests from writing to real ~/.hermes."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_cli import kanban_db
from hermes_cli import kanban_db_connect as kbc


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("relative_db", "pinned_board"),
    [
        (Path("kanban.db"), "default"),
        (
            Path("kanban") / "boards" / "private-operations" / "kanban.db",
            "private-operations",
        ),
    ],
    ids=["custom-default-board", "custom-named-board"],
)
def test_external_pytest_probe_cannot_write_pinned_custom_operational_db(
    tmp_path, relative_db, pinned_board
):
    """An out-of-tree pytest process must be stopped at ``connect()``.

    The subprocess deliberately runs a test file outside ``tests/``, so the
    repository conftest cannot remove its inherited board pins or patch
    ``kanban_db.connect``. Pre-existing paired ``HERMES_HOME`` and
    ``HERMES_KANBAN_HOME`` values mark the synthetic custom root as operational;
    no live board is opened.
    """
    operational_root = tmp_path / "synthetic-operational-hermes"
    operational_db = operational_root / relative_db
    with kanban_db.connect_closing(operational_db) as conn:
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0

    external_test = tmp_path / "external-review" / "test_review_probe.py"
    external_test.parent.mkdir()
    external_test.write_text(
        textwrap.dedent(
            """
            import os
            import sys

            from hermes_cli import kanban_db

            assert "tests.conftest" not in sys.modules
            assert os.environ["HERMES_HOME"] == os.environ["HERMES_KANBAN_HOME"]
            assert os.environ.get("HERMES_TEST_ISOLATION") != os.environ["HERMES_HOME"]
            assert os.environ["HERMES_KANBAN_DB"] == os.environ["SYNTHETIC_OPERATIONAL_DB"]

            def test_two_review_probe_creates_are_refused():
                errors = []
                for _ in range(2):
                    try:
                        with kanban_db.connect(board="misleading-board") as conn:
                            kanban_db.create_task(
                                conn,
                                title="probe",
                                assignee="reviewer",
                                tenant="sensitive-tenant",
                            )
                    except RuntimeError as exc:
                        errors.append(str(exc))

                assert len(errors) == 2
                for message in errors:
                    assert os.environ["SYNTHETIC_OPERATIONAL_ROOT"] not in message
                    assert os.environ["SYNTHETIC_OPERATIONAL_DB"] not in message
                    assert os.environ["PINNED_BOARD"] not in message
                    assert "misleading-board" not in message
                    assert "sensitive-tenant" not in message
                    assert "temporary HERMES_HOME" in message
                    assert "explicit temporary db_path" in message
            """
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.update(
        {
            "HERMES_HOME": str(operational_root),
            "HERMES_KANBAN_HOME": str(operational_root),
            "HERMES_KANBAN_DB": str(operational_db),
            "HERMES_KANBAN_BOARD": pinned_board,
            "PYTHONPATH": str(REPO_ROOT),
            "PINNED_BOARD": pinned_board,
            "SYNTHETIC_OPERATIONAL_DB": str(operational_db),
            "SYNTHETIC_OPERATIONAL_ROOT": str(operational_root),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(external_test), "-q"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    with sqlite3.connect(operational_db) as conn:
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


@pytest.mark.windows_only
def test_external_pytest_scrubbed_child_keeps_guard_via_ancestry(tmp_path):
    operational_root = tmp_path / "synthetic-operational-hermes"
    operational_db = operational_root / "kanban.db"
    with kanban_db.connect_closing(operational_db) as conn:
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0

    external_test = tmp_path / "external-review" / "test_scrubbed_child.py"
    external_test.parent.mkdir()
    external_test.write_text(
        textwrap.dedent(
            """
            import os
            import subprocess
            import sys

            CHILD = (
                "import os;"
                "from pathlib import Path;"
                "from hermes_cli import kanban_db;"
                "kanban_db._KANBAN_TEST_ISOLATION_EXTRA_DENY_ROOTS="
                "(Path(os.environ['SYNTHETIC_OPERATIONAL_ROOT']),);"
                "conn=kanban_db.connect(board='misdirection');"
                "kanban_db.create_task(conn,title='probe',assignee='reviewer');"
                "conn.close()"
            )

            def test_scrubbed_child_is_refused():
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith("PYTEST_")
                    and key != "HERMES_TEST_ISOLATION"
                }
                result = subprocess.run(
                    [sys.executable, "-c", CHILD],
                    cwd=os.environ["REPO_ROOT"],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                assert result.returncode != 0, result.stdout + result.stderr
                assert "temporary HERMES_HOME" in result.stderr
            """
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.update(
        {
            "HERMES_KANBAN_DB": str(operational_db),
            "HERMES_KANBAN_BOARD": "synthetic-board",
            "PYTHONPATH": str(REPO_ROOT),
            "REPO_ROOT": str(REPO_ROOT),
            "SYNTHETIC_OPERATIONAL_ROOT": str(operational_root),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(external_test), "-q"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    with sqlite3.connect(operational_db) as conn:
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


@pytest.mark.parametrize(
    "relative_db",
    [
        Path("kanban.db"),
        Path("kanban") / "boards" / "project-one" / "kanban.db",
    ],
    ids=["default-board", "named-board"],
)
def test_test_context_refuses_canonical_operational_paths_before_mkdir(
    tmp_path, monkeypatch, relative_db
):
    operational_root = tmp_path / "synthetic-operational-hermes"
    target = operational_root / relative_db
    monkeypatch.setattr(
        kanban_db,
        "_KANBAN_TEST_ISOLATION_EXTRA_DENY_ROOTS",
        (operational_root,),
    )

    with pytest.raises(RuntimeError) as exc_info:
        kanban_db.connect(target)

    message = str(exc_info.value)
    assert str(target) not in message
    assert str(operational_root) not in message
    assert "project-one" not in message
    assert "temporary HERMES_HOME" in message
    assert "explicit temporary db_path" in message
    assert not operational_root.exists()


@pytest.mark.parametrize(
    "relative_db",
    [
        Path("kanban.db"),
        Path("kanban") / "boards" / "project-one" / "kanban.db",
    ],
    ids=["default-board", "named-board"],
)
def test_init_db_refuses_canonical_operational_paths_before_mkdir(
    tmp_path, monkeypatch, relative_db
):
    operational_root = tmp_path / "synthetic-operational-hermes"
    target = operational_root / relative_db
    monkeypatch.setattr(
        kanban_db,
        "_KANBAN_TEST_ISOLATION_EXTRA_DENY_ROOTS",
        (operational_root,),
    )

    with pytest.raises(RuntimeError, match="temporary HERMES_HOME"):
        kanban_db.init_db(target)

    assert not operational_root.exists()


def _assert_filesystem_guard_error_is_neutral(
    exc_info: pytest.ExceptionInfo[RuntimeError],
    operational_root: Path,
    *sensitive_values: str,
) -> None:
    message = str(exc_info.value)
    assert str(operational_root) not in message
    for value in sensitive_values:
        assert value not in message
    assert "temporary HERMES_HOME" in message
    assert "explicit temporary db_path" in message


@pytest.mark.parametrize(
    ("board", "operation"),
    [
        ("default", "write"),
        ("private-operations", "write"),
        ("private-operations", "create"),
    ],
    ids=["default-metadata", "named-metadata", "create-board"],
)
def test_board_metadata_mutators_refuse_operational_root_before_write(
    tmp_path, monkeypatch, board, operation
):
    operational_root = tmp_path / "synthetic-operational-hermes"
    metadata_path = (
        operational_root / "kanban" / "boards" / board / "board.json"
    )
    before = None
    if operation == "write":
        metadata_path.parent.mkdir(parents=True)
        metadata_path.write_text('{"name": "original"}\n', encoding="utf-8")
        before = metadata_path.read_bytes()

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(operational_root))
    monkeypatch.setattr(
        kanban_db,
        "_KANBAN_TEST_ISOLATION_EXTRA_DENY_ROOTS",
        (operational_root,),
    )

    with pytest.raises(RuntimeError) as exc_info:
        if operation == "create":
            kanban_db.create_board(board, name="sensitive-display-name")
        else:
            kanban_db.write_board_metadata(
                board,
                name="sensitive-display-name",
                description="sensitive-tenant",
            )

    _assert_filesystem_guard_error_is_neutral(
        exc_info,
        operational_root,
        board,
        "sensitive-display-name",
        "sensitive-tenant",
    )
    if before is None:
        assert not operational_root.exists()
    else:
        assert metadata_path.read_bytes() == before


def test_set_current_board_refuses_operational_root_before_mkdir(
    tmp_path, monkeypatch
):
    operational_root = tmp_path / "synthetic-operational-hermes"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(operational_root))
    monkeypatch.setattr(
        kanban_db,
        "_KANBAN_TEST_ISOLATION_EXTRA_DENY_ROOTS",
        (operational_root,),
    )

    with pytest.raises(RuntimeError) as exc_info:
        kanban_db.set_current_board("sensitive-board")

    _assert_filesystem_guard_error_is_neutral(
        exc_info, operational_root, "sensitive-board", "sensitive-tenant"
    )
    assert not operational_root.exists()


def test_clear_current_board_refuses_operational_root_before_unlink(
    tmp_path, monkeypatch
):
    operational_root = tmp_path / "synthetic-operational-hermes"
    selector = operational_root / "kanban" / "current"
    selector.parent.mkdir(parents=True)
    selector.write_text("sensitive-board\n", encoding="utf-8")
    before = selector.read_bytes()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(operational_root))
    monkeypatch.setattr(
        kanban_db,
        "_KANBAN_TEST_ISOLATION_EXTRA_DENY_ROOTS",
        (operational_root,),
    )

    with pytest.raises(RuntimeError) as exc_info:
        kanban_db.clear_current_board()

    _assert_filesystem_guard_error_is_neutral(
        exc_info, operational_root, "sensitive-board", "sensitive-tenant"
    )
    assert selector.read_bytes() == before


@pytest.mark.parametrize("archive", [True, False], ids=["archive", "delete"])
def test_remove_board_refuses_operational_root_before_any_mutation(
    tmp_path, monkeypatch, archive
):
    operational_root = tmp_path / "synthetic-operational-hermes"
    board = "sensitive-board"
    board_path = operational_root / "kanban" / "boards" / board
    metadata_path = board_path / "board.json"
    payload_path = board_path / "payload.txt"
    selector = operational_root / "kanban" / "current"
    board_path.mkdir(parents=True)
    metadata_path.write_text('{"name": "Sensitive"}\n', encoding="utf-8")
    payload_path.write_text("preserve me\n", encoding="utf-8")
    selector.parent.mkdir(parents=True, exist_ok=True)
    selector.write_text(board + "\n", encoding="utf-8")
    before = {
        "metadata": metadata_path.read_bytes(),
        "payload": payload_path.read_bytes(),
        "selector": selector.read_bytes(),
    }
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(operational_root))
    monkeypatch.setattr(
        kanban_db,
        "_KANBAN_TEST_ISOLATION_EXTRA_DENY_ROOTS",
        (operational_root,),
    )

    with pytest.raises(RuntimeError) as exc_info:
        kanban_db.remove_board(board, archive=archive)

    _assert_filesystem_guard_error_is_neutral(
        exc_info, operational_root, board, "sensitive-tenant"
    )
    assert metadata_path.read_bytes() == before["metadata"]
    assert payload_path.read_bytes() == before["payload"]
    assert selector.read_bytes() == before["selector"]
    assert not (operational_root / "kanban" / "boards" / "_archived").exists()


@pytest.mark.require_symlinks
@pytest.mark.parametrize("archive", [True, False], ids=["archive", "delete"])
def test_remove_board_refuses_outbound_operational_symlink_before_any_mutation(
    tmp_path, monkeypatch, archive
):
    operational_root = tmp_path / "synthetic-operational-hermes"
    board = "sensitive-board"
    board_link = operational_root / "kanban" / "boards" / board
    external_board = tmp_path / "external-board"
    external_board.mkdir()
    metadata = external_board / "board.json"
    payload = external_board / "payload.txt"
    metadata.write_text('{"name": "Sensitive"}\n', encoding="utf-8")
    payload.write_text("preserve me\n", encoding="utf-8")
    board_link.parent.mkdir(parents=True)
    board_link.symlink_to(external_board, target_is_directory=True)
    selector = operational_root / "kanban" / "current"
    selector.write_text("other-board\n", encoding="utf-8")
    before = {
        "metadata": metadata.read_bytes(),
        "payload": payload.read_bytes(),
        "selector": selector.read_bytes(),
    }
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(operational_root))
    monkeypatch.setattr(
        kanban_db,
        "_KANBAN_TEST_ISOLATION_EXTRA_DENY_ROOTS",
        (operational_root,),
    )

    with pytest.raises(RuntimeError) as exc_info:
        kanban_db.remove_board(board, archive=archive)

    _assert_filesystem_guard_error_is_neutral(exc_info, operational_root, board)
    assert board_link.is_symlink()
    assert metadata.read_bytes() == before["metadata"]
    assert payload.read_bytes() == before["payload"]
    assert selector.read_bytes() == before["selector"]
    assert not (operational_root / "kanban" / "boards" / "_archived").exists()


@pytest.mark.require_symlinks
@pytest.mark.parametrize(
    "operation",
    ["connect", "init", "metadata", "selector"],
)
def test_operational_outbound_symlink_sibling_mutators_fail_before_write(
    tmp_path, monkeypatch, operation
):
    operational_root = tmp_path / "synthetic-operational-hermes"
    external_root = tmp_path / "external-target"
    external_root.mkdir()
    kanban_link = operational_root / "kanban"
    operational_root.mkdir()
    kanban_link.symlink_to(external_root, target_is_directory=True)
    before = tuple(external_root.iterdir())
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(operational_root))
    monkeypatch.setattr(
        kanban_db,
        "_KANBAN_TEST_ISOLATION_EXTRA_DENY_ROOTS",
        (operational_root,),
    )

    with pytest.raises(RuntimeError) as exc_info:
        if operation == "connect":
            kbc.connect(board="sensitive-board")
        elif operation == "init":
            kbc.init_db(board="sensitive-board")
        elif operation == "metadata":
            kanban_db.write_board_metadata("sensitive-board", name="Sensitive")
        else:
            kanban_db.set_current_board("sensitive-board")

    _assert_filesystem_guard_error_is_neutral(
        exc_info, operational_root, "sensitive-board"
    )
    assert kanban_link.is_symlink()
    assert tuple(external_root.iterdir()) == before


@pytest.mark.parametrize("path_form", ["relative", "symlink"])
def test_filesystem_mutation_guard_resolves_relative_and_symlink_roots(
    tmp_path, monkeypatch, path_form
):
    operational_root = tmp_path / "synthetic-operational-hermes"
    if path_form == "relative":
        monkeypatch.chdir(tmp_path)
        configured_root = Path("synthetic-operational-hermes")
    else:
        operational_root.mkdir()
        configured_root = tmp_path / "operational-alias"
        configured_root.symlink_to(operational_root, target_is_directory=True)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(configured_root))
    monkeypatch.setattr(
        kanban_db,
        "_KANBAN_TEST_ISOLATION_EXTRA_DENY_ROOTS",
        (operational_root,),
    )

    with pytest.raises(RuntimeError, match="temporary HERMES_HOME"):
        kanban_db.write_board_metadata("sensitive-board", name="sensitive-name")

    assert not (
        operational_root
        / "kanban"
        / "boards"
        / "sensitive-board"
        / "board.json"
    ).exists()


@pytest.mark.require_symlinks
def test_test_context_resolves_symlink_before_operational_path_check(
    tmp_path, monkeypatch
):
    operational_root = tmp_path / "synthetic-operational-hermes"
    operational_root.mkdir()
    alias = tmp_path / "operational-alias"
    alias.symlink_to(operational_root, target_is_directory=True)
    monkeypatch.setattr(
        kanban_db,
        "_KANBAN_TEST_ISOLATION_EXTRA_DENY_ROOTS",
        (operational_root,),
    )

    with pytest.raises(RuntimeError, match="temporary HERMES_HOME"):
        kanban_db.connect(alias / "kanban.db")

    assert not (operational_root / "kanban.db").exists()


def test_explicit_temporary_db_can_create_task(tmp_path):
    temporary_db = tmp_path / "explicit" / "kanban.db"

    with kanban_db.connect_closing(temporary_db) as conn:
        task_id = kanban_db.create_task(conn, title="safe probe", assignee="reviewer")
        assert kanban_db.get_task(conn, task_id) is not None


def test_isolated_temporary_hermes_home_can_create_named_board_task(
    tmp_path, monkeypatch
):
    home = tmp_path / "isolated-hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)

    kanban_db.create_board("safe-board", name="Safe Board")
    kanban_db.set_current_board("safe-board")
    with kanban_db.connect_closing(board="safe-board") as conn:
        task_id = kanban_db.create_task(conn, title="safe", assignee="reviewer")
        assert kanban_db.get_task(conn, task_id) is not None

    kanban_db.write_board_metadata("safe-board", description="still isolated")
    kanban_db.clear_current_board()
    assert kanban_db.kanban_db_path(board="safe-board").is_file()
    assert kanban_db.read_board_metadata("safe-board")["description"] == "still isolated"


def test_non_test_runtime_keeps_canonical_board_behavior(tmp_path, monkeypatch):
    import hermes_state

    operational_root = tmp_path / "synthetic-operational-hermes"
    target = operational_root / "kanban.db"
    monkeypatch.setattr(hermes_state, "_in_test_context", lambda: False)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(operational_root))
    monkeypatch.setattr(
        kanban_db,
        "_KANBAN_TEST_ISOLATION_EXTRA_DENY_ROOTS",
        (operational_root,),
    )

    with kanban_db.connect_closing(target) as conn:
        task_id = kanban_db.create_task(conn, title="runtime", assignee="worker")
        assert kanban_db.get_task(conn, task_id) is not None

    kanban_db.create_board("runtime-board", name="Runtime Board")
    kanban_db.set_current_board("runtime-board")
    kanban_db.write_board_metadata("runtime-board", description="allowed")
    kanban_db.clear_current_board()
    result = kanban_db.remove_board("runtime-board", archive=False)

    assert target.is_file()
    assert result["action"] == "deleted"
    assert not kanban_db.board_dir("runtime-board").exists()


def test_connect_succeeds_under_test_home(tmp_path, monkeypatch):
    """When HERMES_HOME is a temp dir, kanban connect succeeds normally."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = kbc.connect()
    try:
        assert str(kanban_db.kanban_db_path()).startswith(str(home))
    finally:
        conn.close()


def test_connect_raises_when_kanban_home_is_real_root(monkeypatch):
    """When kanban paths resolve to the REAL root, connect raises RuntimeError."""
    import tests.conftest as _conftest

    monkeypatch.setattr(
        kanban_db, "kanban_home", lambda: _conftest._REAL_KANBAN_ROOT
    )
    monkeypatch.setattr(
        kanban_db,
        "kanban_db_path",
        lambda board=None: _conftest._REAL_KANBAN_ROOT / "kanban.db",
    )
    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect()


def test_connect_raises_for_explicit_db_path_under_real_root():
    """Explicit db_path pointing under the real root is also refused."""
    import tests.conftest as _conftest

    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect(_conftest._REAL_KANBAN_ROOT / "kanban.db")
