import json
import hashlib
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.verification_evidence import (
    classify_verification_command,
    mark_workspace_edited,
    record_terminal_result,
    record_verify_run,
    verification_status,
    verification_status_readonly,
    verification_status_readonly_for_cwd,
)

@pytest.fixture(autouse=True)
def _ledger_on(monkeypatch):
    """The ledger is inert unless verify-on-stop is enabled; these tests exercise the ledger."""
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")



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


# ---------------------------------------------------------------------------
# Strict-A integration tests (PR74986 WAL3)
# ---------------------------------------------------------------------------


@pytest.fixture
def _hermes_home(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    import agent.verification_evidence as ve
    import agent.verification_lock as vl
    monkeypatch.setattr(ve, "get_hermes_home", lambda: home)
    monkeypatch.setattr(vl, "get_hermes_home", lambda: home)
    return home


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _list_family(home: Path) -> dict[str, Path]:
    base = home / "verification_evidence.db"
    paths = {"db": base}
    for sfx in ("-wal", "-shm", "-journal"):
        p = Path(str(base) + sfx)
        if p.exists():
            paths[sfx.lstrip("-")] = p
    return paths


def test_strict_writer_creates_lockfile(_hermes_home, monkeypatch):
    """A25/A26: a single writer creates the persistent lockfile with 1 byte."""
    # The writer path triggers a record_verify_run. We stub classify to
    # avoid the project-facts step which requires a real project layout.
    from agent import verification_evidence as ve

    monkeypatch.setattr(
        ve, "_project_facts",
        lambda cwd: {"root": "/r"} if cwd else None,
    )
    monkeypatch.setattr(ve, "_root_for", lambda facts, cwd: "/r")

    record_verify_run(root="/r", session_id="s1", ok=True)

    lf = _hermes_home / ".locks" / "verification_evidence.lock"
    assert lf.is_file()
    assert lf.stat().st_size == 1
    assert lf.read_bytes() == b"\x00"


def test_strict_reader_does_not_create_lockfile(_hermes_home, monkeypatch):
    """A20/A24: the reader MUST NOT create the lockfile or parent dir."""
    from agent import verification_evidence as ve

    monkeypatch.setattr(
        ve, "_project_facts",
        lambda cwd: {"root": "/r"} if cwd else None,
    )
    monkeypatch.setattr(ve, "_root_for", lambda facts, cwd: "/r")

    db_path = _hermes_home / "verification_evidence.db"
    result = verification_status_readonly_for_cwd(
        session_id="s1", cwd="/r"
    )

    assert not (_hermes_home / ".locks").exists()
    assert result["status"] == "unverified"


def test_strict_reader_leaves_source_bytes_unchanged(_hermes_home, monkeypatch):
    """A6: every primitive of the ledger family stays byte-stable."""
    from agent import verification_evidence as ve

    monkeypatch.setattr(
        ve, "_project_facts",
        lambda cwd: {"root": "/r"} if cwd else None,
    )
    monkeypatch.setattr(ve, "_root_for", lambda facts, cwd: "/r")

    record_verify_run(root="/r", session_id="s1", ok=True)
    db = _hermes_home / "verification_evidence.db"
    # Force-checkpoint to consolidate WAL.
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    before = {k: (_sha(p), p.stat().st_size if p.exists() else 0) for k, p in _list_family(_hermes_home).items()}

    result = verification_status_readonly_for_cwd(
        session_id="s1", cwd="/r"
    )
    after = {k: (_sha(p), p.stat().st_size if p.exists() else 0) for k, p in _list_family(_hermes_home).items()}

    assert result["status"] == "passed"
    assert before == after


def test_strict_reader_missing_db_returns_unverified(_hermes_home, monkeypatch):
    """A1/A24: missing DB → unverified, zero filesystem creation."""
    from agent import verification_evidence as ve

    monkeypatch.setattr(
        ve, "_project_facts",
        lambda cwd: {"root": "/r"} if cwd else None,
    )
    monkeypatch.setattr(ve, "_root_for", lambda facts, cwd: "/r")

    before_listing = sorted(p.name for p in _hermes_home.glob("*"))
    result = verification_status_readonly_for_cwd(
        session_id="s1", cwd="/r"
    )
    after_listing = sorted(p.name for p in _hermes_home.glob("*"))

    assert result["status"] == "unverified"
    assert before_listing == after_listing


def test_strict_reader_corrupt_db_returns_unverified(_hermes_home, monkeypatch):
    """A2: corrupt DB → unverified, bytes unchanged."""
    from agent import verification_evidence as ve

    monkeypatch.setattr(
        ve, "_project_facts",
        lambda cwd: {"root": "/r"} if cwd else None,
    )
    monkeypatch.setattr(ve, "_root_for", lambda facts, cwd: "/r")

    db = _hermes_home / "verification_evidence.db"
    db.write_bytes(b"NOT A SQLITE FILE")
    # Materialize lockfile via writer. The writer will run _transaction()
    # which calls _connect; with corrupt bytes the connect succeeds (sqlite
    # is permissive) and _ensure_schema silently no-ops because the tables
    # already exist as part of the corrupt content. To avoid relying on the
    # corrupt-content path of the writer, materialize the lockfile manually.
    lf = _hermes_home / ".locks" / "verification_evidence.lock"
    lf.parent.mkdir(parents=True, exist_ok=True)
    lf.write_bytes(b"\x00")

    before = _sha(db)
    result = verification_status_readonly_for_cwd(
        session_id="s1", cwd="/r"
    )
    after = _sha(db)
    assert result["status"] == "unverified"
    assert before == after


def test_strict_reader_incomplete_schema_returns_unverified(_hermes_home, monkeypatch):
    """A3: incomplete schema → unverified, no repair."""
    from agent import verification_evidence as ve

    monkeypatch.setattr(
        ve, "_project_facts",
        lambda cwd: {"root": "/r"} if cwd else None,
    )
    monkeypatch.setattr(ve, "_root_for", lambda facts, cwd: "/r")

    db = _hermes_home / "verification_evidence.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE foo (x INTEGER)")  # not the required tables
    conn.commit()
    conn.close()
    lf = _hermes_home / ".locks" / "verification_evidence.lock"
    lf.parent.mkdir(parents=True, exist_ok=True)
    lf.write_bytes(b"\x00")

    before = _sha(db)
    result = verification_status_readonly_for_cwd(
        session_id="s1", cwd="/r"
    )
    after = _sha(db)
    assert result["status"] == "unverified"
    assert before == after


def test_strict_reader_lockfile_byte_stable(_hermes_home, monkeypatch):
    """A27: reader leaves lockfile size and SHA256 unchanged."""
    from agent import verification_evidence as ve

    monkeypatch.setattr(
        ve, "_project_facts",
        lambda cwd: {"root": "/r"} if cwd else None,
    )
    monkeypatch.setattr(ve, "_root_for", lambda facts, cwd: "/r")

    record_verify_run(root="/r", session_id="s1", ok=True)
    lf = _hermes_home / ".locks" / "verification_evidence.lock"
    size_before = lf.stat().st_size
    sha_before = hashlib.sha256(lf.read_bytes()).hexdigest()

    for _ in range(10):
        result = verification_status_readonly_for_cwd(
            session_id="s1", cwd="/r"
        )
        assert result["status"] == "passed"

    assert lf.stat().st_size == size_before
    assert hashlib.sha256(lf.read_bytes()).hexdigest() == sha_before


def test_strict_writer_timeout_propagates(_hermes_home, monkeypatch):
    """A11: writer lock acquisition timeout raises LockTimeout; no bypass."""
    from agent import verification_evidence as ve
    from agent.verification_lock import LockTimeout, coordinated_lock

    monkeypatch.setattr(
        ve, "_project_facts",
        lambda cwd: {"root": "/r"} if cwd else None,
    )
    monkeypatch.setattr(ve, "_root_for", lambda facts, cwd: "/r")

    # First, materialize the lockfile via a writer.
    record_verify_run(root="/r", session_id="s1", ok=True)

    # Hold the lock in a separate coordination context to simulate an
    # in-flight writer. The next record_verify_run must raise.
    held = coordinated_lock("writer")
    held.__enter__()
    try:
        with pytest.raises(LockTimeout):
            record_verify_run(root="/r", session_id="s1", ok=True)
    finally:
        held.__exit__(None, None, None)


def test_strict_reader_does_not_open_source(_hermes_home, monkeypatch):
    """A21: SOURCE_SQLITE_CONNECT_COUNT=0 in the reader path.

    Instruments sqlite3.connect at the agent.verification_evidence module
    level and counts only paths that point at the source ledger.
    """
    from agent import verification_evidence as ve

    monkeypatch.setattr(
        ve, "_project_facts",
        lambda cwd: {"root": "/r"} if cwd else None,
    )
    monkeypatch.setattr(ve, "_root_for", lambda facts, cwd: "/r")

    record_verify_run(root="/r", session_id="s1", ok=True)
    db = _hermes_home / "verification_evidence.db"
    # Checkpoint to remove sidecars.
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    real_connect = sqlite3.connect
    source_open_count = {"n": 0}
    home_str = str(_hermes_home)

    def spy_connect(database, *args, **kwargs):
        path_str = str(database)
        # Source path: under hermes_home + verification_evidence.db,
        # NOT a tempfile-prefixed snapshot path.
        if (
            home_str in path_str
            and "verification_evidence.db" in path_str
            and "verification_strict_snapshot_" not in path_str
        ):
            source_open_count["n"] += 1
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(ve.sqlite3, "connect", spy_connect)

    verification_status_readonly_for_cwd(session_id="s1", cwd="/r")
    assert source_open_count["n"] == 0, (
        "Reader must never sqlite3.connect the source ledger"
    )
