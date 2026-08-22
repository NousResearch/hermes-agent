import json
import hashlib
import os
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.verification_evidence import (
    _ensure_schema,
    classify_verification_command,
    mark_workspace_edited,
    record_terminal_result,
    verification_status,
    verification_status_readonly,
)


def _node_project(root: Path) -> None:
    (root / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest", "lint": "eslint .", "dev": "vite"}})
    )
    (root / "pnpm-lock.yaml").write_text("")
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / "run_tests.sh").write_text("#!/bin/sh\n")


def _python_project(root: Path) -> None:
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")

def _create_verification_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        _ensure_schema(conn)


def _sidecar_paths(db_path: Path) -> list[Path]:
    return [
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
        db_path.with_name(f"{db_path.name}-journal"),
    ]


def _readonly_schema_snapshot(db_path: Path) -> dict[str, object]:
    with sqlite3.connect(db_path) as conn:
        sqlite_master = conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        table_info = {
            table_name: conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            for table_name in ("verification_state", "verification_events")
        }
    return {
        "sha256": hashlib.sha256(db_path.read_bytes()).hexdigest(),
        "sqlite_master": sqlite_master,
        "table_info": table_info,
        "sidecars": {str(path): path.exists() for path in _sidecar_paths(db_path)},
    }


def _assert_readonly_snapshot_unchanged(
    db_path: Path,
    before: dict[str, object],
) -> None:
    after = _readonly_schema_snapshot(db_path)
    assert after["sha256"] == before["sha256"]
    assert after["sqlite_master"] == before["sqlite_master"]
    assert after["table_info"] == before["table_info"]
    assert after["sidecars"] == before["sidecars"]
    assert not any(path.exists() for path in _sidecar_paths(db_path))


def _insert_verification_event(
    db_path: Path,
    *,
    session_id: str = "s1",
    root: Path,
    status: str = "passed",
    created_at: str = "2026-01-01T00:00:00+00:00",
    last_edit_at: str | None = None,
    changed_paths_json: str = "[]",
    state_event_id: int | None = None,
) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO verification_events(
                created_at, session_id, cwd, root, command, canonical_command,
                kind, scope, status, exit_code, output_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                session_id,
                str(root),
                str(root),
                "python -m pytest",
                "pytest",
                "test",
                "full",
                status,
                0 if status == "passed" else 1,
                "summary",
            ),
        )
        if cur.lastrowid is None:
            raise RuntimeError("verification event insert did not return an id")
        event_id = int(cur.lastrowid)
        conn.execute(
            """
            INSERT INTO verification_state(
                session_id, root, last_event_id, last_edit_at, changed_paths_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, str(root), state_event_id if state_event_id is not None else event_id, last_edit_at, changed_paths_json),
        )
        conn.commit()
        return event_id


def test_classifies_targeted_project_verify_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    evidence = classify_verification_command(
        "scripts/run_tests.sh tests/test_widget.py -q",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="1 passed",
    )

    assert evidence is not None
    assert evidence.canonical_command == "scripts/run_tests.sh"
    assert evidence.kind == "test"
    assert evidence.scope == "targeted"
    assert evidence.status == "passed"






def test_lint_and_typecheck_are_not_reported_as_full_tests(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    lint = classify_verification_command(
        "pnpm run lint",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )
    test = classify_verification_command(
        "pnpm run test -- tests/button.test.tsx",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )

    assert lint is not None
    assert lint.kind == "lint"
    assert lint.scope == "full"
    assert test is not None
    assert test.kind == "test"
    assert test.scope == "targeted"




def test_shell_wrappers_match_but_echo_does_not(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    wrapped = classify_verification_command(
        "env CI=1 bash scripts/run_tests.sh tests/test_widget.py",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )
    echoed = classify_verification_command(
        "echo scripts/run_tests.sh tests/test_widget.py",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )

    assert wrapped is not None
    assert wrapped.canonical_command == "scripts/run_tests.sh"
    assert wrapped.scope == "targeted"
    assert echoed is None


@pytest.mark.parametrize(
    "command",
    [
        "pytest || true",
        "pytest ; true",
        "pytest | tee test.log",
        "pytest &",
    ],
)
def test_masking_shell_control_is_not_verification_evidence(
    tmp_path, monkeypatch, command
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        command,
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )

    assert evidence is None


@pytest.mark.parametrize("command", ["prepare && pytest", "pytest && report"])
def test_successful_and_chain_preserves_passing_evidence(
    tmp_path, monkeypatch, command
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        command,
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )

    assert evidence is not None
    assert evidence.status == "passed"


@pytest.mark.parametrize("exit_code, expected", [(0, "passed"), (1, "failed")])
def test_final_verifier_after_sequence_owns_shell_exit_status(
    tmp_path, monkeypatch, exit_code, expected
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        "prepare; pytest",
        cwd=tmp_path,
        session_id="s1",
        exit_code=exit_code,
    )

    assert evidence is not None
    assert evidence.status == expected


def test_quoted_shell_operator_remains_a_verifier_argument(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        "pytest -k 'passes || fails'",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )

    assert evidence is not None
    assert evidence.status == "passed"


@pytest.mark.parametrize("redirect", ["2>&1", "&> test.log"])
def test_shell_redirection_does_not_hide_simple_verifier(tmp_path, monkeypatch, redirect):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        f"pytest {redirect}",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )

    assert evidence is not None
    assert evidence.status == "passed"


def test_masked_ad_hoc_script_is_not_verification_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    script = Path(tempfile.gettempdir()) / f"hermes-ad-hoc-{tmp_path.name}.py"
    script.write_text("raise SystemExit(1)\n", encoding="utf-8")
    try:
        evidence = classify_verification_command(
            f"python {script} || true",
            cwd=tmp_path,
            session_id="s1",
            exit_code=0,
        )
    finally:
        script.unlink(missing_ok=True)

    assert evidence is None


def test_masked_verifier_does_not_clear_edited_ledger_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)
    record_terminal_result(
        command="pytest",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="passed",
    )
    mark_workspace_edited(
        session_id="s1",
        cwd=tmp_path,
        paths=[str(tmp_path / "changed.py")],
    )

    result = record_terminal_result(
        command="pytest || true",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="1 failed",
    )

    assert result is None
    assert verification_status(session_id="s1", cwd=tmp_path)["status"] == "stale"




def test_temp_script_records_ad_hoc_evidence_without_canonical_suite(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    script = Path(tempfile.gettempdir()) / f"hermes-ad-hoc-{tmp_path.name}.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    try:
        evidence = classify_verification_command(
            f"python {script}",
            cwd=tmp_path,
            session_id="s1",
            exit_code=0,
            output="ok",
        )
    finally:
        script.unlink(missing_ok=True)

    assert evidence is not None
    assert evidence.canonical_command == "ad-hoc verification script"
    assert evidence.kind == "ad_hoc"
    assert evidence.scope == "targeted"
    assert evidence.status == "passed"












def test_file_tool_stales_evidence_by_session_id_for_absolute_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    target = tmp_path / "src" / "app.ts"
    target.parent.mkdir()

    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="conversation",
        exit_code=0,
        output="green",
    )

    from tools.file_tools import write_file_tool

    result = json.loads(
        write_file_tool(
            str(target),
            "export const ok = true\n",
            task_id="turn",
            session_id="conversation",
        )
    )

    assert result["files_modified"] == [str(target.resolve())]
    assert verification_status(session_id="conversation", cwd=tmp_path)["status"] == "stale"
    assert verification_status(session_id="turn", cwd=tmp_path)["status"] == "unverified"






def test_recording_expires_old_edit_only_state(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _node_project(tmp_path)

    mark_workspace_edited(
        session_id="old-session",
        cwd=tmp_path,
        paths=[str(tmp_path / "src" / "app.ts")],
    )
    cutoff = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    with sqlite3.connect(home / "verification_evidence.db") as conn:
        conn.execute("UPDATE verification_state SET last_edit_at = ?", (cutoff,))
        conn.commit()

    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="new-session",
        exit_code=0,
        output="new green",
    )

    status = verification_status(session_id="old-session", cwd=tmp_path)
    assert status["status"] == "unverified"
    assert status["changed_paths"] == []


def test_windows_backslash_ad_hoc_script_path_is_matched(tmp_path, monkeypatch):
    """Ad-hoc verification scripts with Windows backslash paths must be
    matched by ``_find_ad_hoc_match`` trying ``posix=False`` in addition to
    the default ``posix=True``. (#53553 / #65919)

    On Linux, ``Path`` doesn't parse Windows backslash paths, so we mock
    ``_is_temp_script_path`` to simulate the Windows environment where the
    path resolves correctly. The test verifies the posix=False splitting
    fallback — the actual fix from #53553.
    """
    from agent.verification_evidence import _find_ad_hoc_match

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")

    # On Windows, shlex.split(posix=True) eats backslashes as escape chars;
    # posix=False preserves them. Mock _is_temp_script_path so the test
    # focuses on the splitting fallback without needing a real Windows FS.
    def mock_is_temp_script(token, root):
        return "hermes-ad-hoc" in token and ".py" in token

    monkeypatch.setattr(
        "agent.verification_evidence._is_temp_script_path",
        mock_is_temp_script,
    )

    win_script = r"C:\Users\test\AppData\Local\Temp\hermes-ad-hoc-check.py"
    result = _find_ad_hoc_match(f"python {win_script}", tmp_path)
    assert result is not None, (
        "Windows backslash path should be matched via posix=False fallback"
    )


def test_readonly_status_returns_unverified_for_missing_db_without_sidecars(tmp_path):
    db_path = tmp_path / "missing" / "verification_evidence.db"

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status == {
        "status": "unverified",
        "evidence": None,
        "root": str(tmp_path),
        "session_id": "s1",
        "changed_paths": [],
    }
    assert not db_path.exists()
    assert not db_path.parent.exists()
    assert not (tmp_path / "missing" / "verification_evidence.db-wal").exists()
    assert not (tmp_path / "missing" / "verification_evidence.db-shm").exists()
    assert not (tmp_path / "missing" / "verification_evidence.db-journal").exists()


def test_readonly_status_returns_unverified_for_missing_schema_without_creating_tables(tmp_path):
    db_path = tmp_path / "verification_evidence.db"
    with sqlite3.connect(db_path):
        pass

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status["status"] == "unverified"
    with sqlite3.connect(db_path) as conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    assert tables == []


def test_readonly_status_returns_unverified_for_partial_state_schema_without_repair(tmp_path):
    db_path = tmp_path / "verification_evidence.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE verification_events (
                id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL,
                session_id TEXT NOT NULL,
                cwd TEXT NOT NULL,
                root TEXT NOT NULL,
                command TEXT NOT NULL,
                canonical_command TEXT NOT NULL,
                kind TEXT NOT NULL,
                scope TEXT NOT NULL,
                status TEXT NOT NULL,
                exit_code INTEGER NOT NULL,
                output_summary TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE verification_state (
                session_id TEXT NOT NULL,
                root TEXT NOT NULL,
                last_event_id INTEGER,
                last_edit_at TEXT,
                PRIMARY KEY (session_id, root)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO verification_events(
                id, created_at, session_id, cwd, root, command, canonical_command,
                kind, scope, status, exit_code, output_summary
            ) VALUES (1, '2026-01-01T00:00:00+00:00', 's1', ?, ?,
                      'python -m pytest', 'pytest', 'test', 'full', 'passed', 0, 'summary')
            """,
            (str(tmp_path), str(tmp_path)),
        )
        conn.execute(
            """
            INSERT INTO verification_state(
                session_id, root, last_event_id, last_edit_at
            ) VALUES ('s1', ?, 1, NULL)
            """,
            (str(tmp_path),),
        )
        conn.commit()
    before = _readonly_schema_snapshot(db_path)

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status == {
        "status": "unverified",
        "evidence": None,
        "root": str(tmp_path),
        "session_id": "s1",
        "changed_paths": [],
    }
    _assert_readonly_snapshot_unchanged(db_path, before)


def test_readonly_status_returns_unverified_for_partial_events_schema_without_repair(tmp_path):
    db_path = tmp_path / "verification_evidence.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE verification_events (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                cwd TEXT NOT NULL,
                root TEXT NOT NULL,
                command TEXT NOT NULL,
                canonical_command TEXT NOT NULL,
                kind TEXT NOT NULL,
                scope TEXT NOT NULL,
                status TEXT NOT NULL,
                exit_code INTEGER NOT NULL,
                output_summary TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE verification_state (
                session_id TEXT NOT NULL,
                root TEXT NOT NULL,
                last_event_id INTEGER,
                last_edit_at TEXT,
                changed_paths_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (session_id, root)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO verification_events(
                id, session_id, cwd, root, command, canonical_command,
                kind, scope, status, exit_code, output_summary
            ) VALUES (1, 's1', ?, ?, 'python -m pytest', 'pytest', 'test',
                      'full', 'passed', 0, 'summary')
            """,
            (str(tmp_path), str(tmp_path)),
        )
        conn.execute(
            """
            INSERT INTO verification_state(
                session_id, root, last_event_id, last_edit_at, changed_paths_json
            ) VALUES ('s1', ?, 1, NULL, '[]')
            """,
            (str(tmp_path),),
        )
        conn.commit()
    before = _readonly_schema_snapshot(db_path)

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status == {
        "status": "unverified",
        "evidence": None,
        "root": str(tmp_path),
        "session_id": "s1",
        "changed_paths": [],
    }
    _assert_readonly_snapshot_unchanged(db_path, before)


def test_readonly_status_returns_unverified_when_state_is_missing(tmp_path):
    db_path = tmp_path / "verification_evidence.db"
    _create_verification_db(db_path)

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status["status"] == "unverified"
    assert status["evidence"] is None
    assert status["changed_paths"] == []


def test_readonly_status_returns_unverified_when_referenced_event_is_missing(tmp_path):
    db_path = tmp_path / "verification_evidence.db"
    _create_verification_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO verification_state(
                session_id, root, last_event_id, last_edit_at, changed_paths_json
            ) VALUES (?, ?, 999, NULL, '[]')
            """,
            ("s1", str(tmp_path)),
        )
        conn.commit()

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status["status"] == "unverified"
    assert status["evidence"] is None


def test_readonly_status_returns_passed_evidence_and_changed_paths(tmp_path):
    db_path = tmp_path / "verification_evidence.db"
    _create_verification_db(db_path)
    event_id = _insert_verification_event(
        db_path,
        root=tmp_path,
        status="passed",
        changed_paths_json=json.dumps(["b.py", "a.py"]),
    )

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status["status"] == "passed"
    assert status["changed_paths"] == ["b.py", "a.py"]
    assert status["evidence"]["id"] == event_id
    assert status["evidence"]["status"] == "passed"
    assert status["evidence"]["canonical_command"] == "pytest"


def test_readonly_status_returns_failed_evidence(tmp_path):
    db_path = tmp_path / "verification_evidence.db"
    _create_verification_db(db_path)
    _insert_verification_event(db_path, root=tmp_path, status="failed")

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status["status"] == "failed"
    assert status["evidence"]["status"] == "failed"


def test_readonly_status_returns_stale_when_edit_is_after_evidence(tmp_path):
    db_path = tmp_path / "verification_evidence.db"
    _create_verification_db(db_path)
    _insert_verification_event(
        db_path,
        root=tmp_path,
        status="passed",
        created_at="2026-01-01T00:00:00+00:00",
        last_edit_at="2026-01-01T00:00:01+00:00",
    )

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status["status"] == "stale"


def test_readonly_status_uses_empty_changed_paths_for_corrupt_json_without_repair(tmp_path):
    db_path = tmp_path / "verification_evidence.db"
    _create_verification_db(db_path)
    _insert_verification_event(db_path, root=tmp_path, changed_paths_json="not-json")

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status["changed_paths"] == []
    with sqlite3.connect(db_path) as conn:
        stored = conn.execute("SELECT changed_paths_json FROM verification_state").fetchone()[0]
    assert stored == "not-json"


def test_readonly_status_opens_sqlite_uri_readonly_and_executes_no_ddl_or_dml(tmp_path, monkeypatch):
    import agent.verification_evidence as verification_evidence

    db_path = tmp_path / "verification_evidence.db"
    _create_verification_db(db_path)
    _insert_verification_event(db_path, root=tmp_path)
    original_connect = sqlite3.connect
    calls = []
    statements = []

    class CursorProxy:
        def __init__(self, cursor):
            self._cursor = cursor

        def fetchone(self):
            return self._cursor.fetchone()

        def fetchall(self):
            return self._cursor.fetchall()

    class ConnectionProxy:
        def __init__(self, conn):
            self._conn = conn

        @property
        def row_factory(self):
            return self._conn.row_factory

        @row_factory.setter
        def row_factory(self, value):
            self._conn.row_factory = value

        def execute(self, sql, parameters=()):
            statements.append(sql.strip())
            first = sql.lstrip().split(None, 1)[0].upper()
            assert first not in {"CREATE", "INSERT", "UPDATE", "DELETE", "REPLACE"}
            return CursorProxy(self._conn.execute(sql, parameters))

        def close(self):
            self._conn.close()

    def connect_proxy(database, *args, **kwargs):
        calls.append((database, args, kwargs))
        return ConnectionProxy(original_connect(database, *args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", connect_proxy)
    monkeypatch.setattr(verification_evidence, "_connect", lambda: (_ for _ in ()).throw(AssertionError("_connect called")))
    monkeypatch.setattr(verification_evidence, "_ensure_schema", lambda conn: (_ for _ in ()).throw(AssertionError("_ensure_schema called")))

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status["status"] == "passed"
    assert len(calls) == 1
    database, _, kwargs = calls[0]
    assert kwargs["uri"] is True
    assert "mode=ro" in str(database)
    assert all(not sql.upper().startswith("PRAGMA JOURNAL_MODE") for sql in statements)


def test_readonly_status_does_not_call_project_discovery(tmp_path, monkeypatch):
    import agent.coding_context as coding_context

    db_path = tmp_path / "verification_evidence.db"
    _create_verification_db(db_path)
    _insert_verification_event(db_path, root=tmp_path)
    monkeypatch.setattr(coding_context, "project_facts_for", lambda cwd: (_ for _ in ()).throw(AssertionError("project discovery called")))

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status["status"] == "passed"


def test_readonly_status_does_not_use_cwd_home_or_hermes_home(tmp_path, monkeypatch):
    db_path = tmp_path / "verification_evidence.db"
    _create_verification_db(db_path)
    _insert_verification_event(db_path, root=tmp_path)
    monkeypatch.setattr(os, "getcwd", lambda: (_ for _ in ()).throw(AssertionError("cwd discovery called")))
    monkeypatch.setattr(Path, "cwd", lambda: (_ for _ in ()).throw(AssertionError("Path.cwd called")))
    monkeypatch.setattr(Path, "home", lambda: (_ for _ in ()).throw(AssertionError("Path.home called")))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "different-home"))

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status["status"] == "passed"


def test_readonly_status_does_not_use_subprocess(tmp_path, monkeypatch):
    db_path = tmp_path / "verification_evidence.db"
    _create_verification_db(db_path)
    _insert_verification_event(db_path, root=tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess.run called")))
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess.Popen called")))
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess.check_output called")))

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status["status"] == "passed"


def test_readonly_status_returns_unverified_for_corrupt_database_without_sidecars(tmp_path):
    db_path = tmp_path / "verification_evidence.db"
    db_path.write_text("not sqlite", encoding="utf-8")

    status = verification_status_readonly(session_id="s1", root=tmp_path, db_path=db_path)

    assert status["status"] == "unverified"
    assert status["evidence"] is None
    assert db_path.read_text(encoding="utf-8") == "not sqlite"
    assert not (tmp_path / "verification_evidence.db-wal").exists()
    assert not (tmp_path / "verification_evidence.db-shm").exists()
    assert not (tmp_path / "verification_evidence.db-journal").exists()
