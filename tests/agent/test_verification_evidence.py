import json
import os
import sqlite3
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.verification_evidence import (
    classify_verification_command,
    latest_verification_status,
    mark_workspace_edited,
    record_terminal_result,
    verification_status,
    workspace_artifact_fingerprint,
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


def _init_git_project(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Hermes Test"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "hermes@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)


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


def test_classifies_python_module_pytest_as_detected_pytest(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        "python -m pytest tests/test_calc.py::test_even -q",
        cwd=tmp_path,
        session_id="s1",
        exit_code=1,
        output="failed",
    )

    assert evidence is not None
    assert evidence.canonical_command == "pytest"
    assert evidence.kind == "test"
    assert evidence.scope == "targeted"
    assert evidence.status == "failed"


def test_records_passed_then_marks_stale_after_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    event = record_terminal_result(
        command="scripts/run_tests.sh",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="all green",
    )

    assert event is not None
    assert verification_status(session_id="s1", cwd=tmp_path)["status"] == "passed"

    mark_workspace_edited(
        session_id="s1",
        cwd=tmp_path,
        paths=[str(tmp_path / "src" / "app.ts")],
    )

    status = verification_status(session_id="s1", cwd=tmp_path)
    assert status["status"] == "stale"
    assert status["changed_paths"] == [str(tmp_path / "src" / "app.ts")]


def test_pass_is_bound_to_file_content_even_for_unrecorded_external_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    source = tmp_path / "src" / "app.ts"
    source.parent.mkdir()
    source.write_text("export const value = 1;\n", encoding="utf-8")
    _init_git_project(tmp_path)
    mark_workspace_edited(session_id="s-hash", cwd=tmp_path, paths=[str(source)])

    event = record_terminal_result(
        command="scripts/run_tests.sh",
        cwd=tmp_path,
        session_id="s-hash",
        exit_code=0,
        output="all green",
    )
    before = verification_status(session_id="s-hash", cwd=tmp_path)

    assert event is not None
    assert event["artifact_hash"].startswith("sha256:")
    assert before["status"] == "passed"
    assert before["artifact_hash"] == before["current_artifact_hash"]

    # Bypass mark_workspace_edited on purpose. Recomputing the fingerprint must
    # still invalidate the cached pass.
    source.write_text("export const value = 2;\n", encoding="utf-8")
    after = verification_status(session_id="s-hash", cwd=tmp_path)

    assert after["status"] == "stale"
    assert after["artifact_hash"] != after["current_artifact_hash"]


def test_pass_is_invalidated_by_unrecorded_gitignored_artifact_edit(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _node_project(project)
    (project / ".gitignore").write_text("ignored-artifact.bin\n", encoding="utf-8")
    artifact = project / "ignored-artifact.bin"
    artifact.write_bytes(b"version one")
    _init_git_project(project)

    event = record_terminal_result(
        command="scripts/run_tests.sh",
        cwd=project,
        session_id="s-ignored",
        exit_code=0,
        output="green",
    )
    assert event is not None
    assert "ignored-artifact.bin" in event["changed_paths"]
    assert verification_status(session_id="s-ignored", cwd=project)["status"] == "passed"

    artifact.write_bytes(b"version two")
    status = verification_status(session_id="s-ignored", cwd=project)

    assert status["status"] == "stale"
    assert status["artifact_hash"] != status["current_artifact_hash"]


def test_fingerprint_requires_git_head(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    artifact_hash, paths = workspace_artifact_fingerprint(tmp_path)

    assert artifact_hash is None
    assert paths == []


def test_fingerprint_fails_closed_when_git_inventory_fails(tmp_path, monkeypatch):
    from agent import verification_evidence as module

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    _init_git_project(tmp_path)
    real_capture = module._git_capture

    def _capture(root, *args):
        if args and args[0] == "ls-files":
            return None
        return real_capture(root, *args)

    monkeypatch.setattr(module, "_git_capture", _capture)

    artifact_hash, paths = workspace_artifact_fingerprint(tmp_path)

    assert artifact_hash is None
    assert paths == []


def test_tracked_index_flags_cannot_hide_content_changes(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    assumed = project / "assumed.txt"
    skipped = project / "skipped.txt"
    assumed.write_text("version one\n", encoding="utf-8")
    skipped.write_text("version one\n", encoding="utf-8")
    _init_git_project(project)
    subprocess.run(
        ["git", "update-index", "--assume-unchanged", "assumed.txt"],
        cwd=project,
        check=True,
    )
    subprocess.run(
        ["git", "update-index", "--skip-worktree", "skipped.txt"],
        cwd=project,
        check=True,
    )

    before, _ = workspace_artifact_fingerprint(project)
    assumed.write_text("version two\n", encoding="utf-8")
    skipped.write_text("version two\n", encoding="utf-8")
    after, paths = workspace_artifact_fingerprint(project)

    assert before and after and before != after
    assert "assumed.txt" in paths
    assert "skipped.txt" in paths


def test_fingerprint_bounds_fail_closed_without_partial_hash(tmp_path, monkeypatch):
    from agent import verification_evidence as module

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    _init_git_project(tmp_path)
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"123456")
    second.write_bytes(b"abcdef")

    monkeypatch.setattr(module, "_MAX_ARTIFACT_PATHS", 1)
    artifact_hash, paths = workspace_artifact_fingerprint(tmp_path)
    assert artifact_hash is None
    assert paths == []

    monkeypatch.setattr(module, "_MAX_ARTIFACT_PATHS", 10)
    monkeypatch.setattr(module, "_MAX_ARTIFACT_HASH_BYTES", 5)
    artifact_hash, paths = workspace_artifact_fingerprint(tmp_path)
    assert artifact_hash is None
    assert paths


def test_git_identity_query_output_is_bounded(tmp_path, monkeypatch):
    from agent import verification_evidence as module

    payload = tmp_path / "payload.txt"
    payload.write_text("more than eight bytes\n", encoding="utf-8")
    _init_git_project(tmp_path)
    monkeypatch.setattr(module, "_MAX_GIT_QUERY_BYTES", 8)

    assert module._git_capture(tmp_path, "show", "HEAD:payload.txt") is None


def test_inventory_change_during_hash_fails_closed(tmp_path, monkeypatch):
    from agent import verification_evidence as module

    tracked = tmp_path / "tracked.txt"
    tracked.write_text("stable\n", encoding="utf-8")
    _init_git_project(tmp_path)
    inventories = iter(({"tracked.txt"}, {"tracked.txt", "late.txt"}))
    monkeypatch.setattr(module, "_git_workspace_paths", lambda _root: next(inventories))

    artifact_hash, paths = module.workspace_artifact_fingerprint(tmp_path)

    assert artifact_hash is None
    assert "tracked.txt" in paths


def test_external_explicit_path_fails_closed(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    tracked = project / "tracked.txt"
    tracked.write_text("stable\n", encoding="utf-8")
    _init_git_project(project)
    outside = tmp_path / "outside.txt"
    outside.write_text("external\n", encoding="utf-8")

    artifact_hash, paths = workspace_artifact_fingerprint(
        project,
        changed_paths=[str(outside)],
    )

    assert artifact_hash is None
    assert paths == []


def test_same_size_mtime_restored_mutation_during_hash_fails_closed(
    tmp_path, monkeypatch
):
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("stable\n", encoding="utf-8")
    _init_git_project(tmp_path)
    initial_stat = tracked.stat()
    original_open = Path.open
    mutated = False

    class MutatingReader:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self._handle.close()

        def read(self, size=-1):
            nonlocal mutated
            data = self._handle.read(size)
            if not data and not mutated:
                mutated = True
                time.sleep(0.01)
                with original_open(tracked, "wb") as writer:
                    writer.write(b"mutate\n")
                os.utime(
                    tracked,
                    ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns),
                )
            return data

    def _open(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        handle = original_open(path, *args, **kwargs)
        if path == tracked and mode == "rb":
            return MutatingReader(handle)
        return handle

    monkeypatch.setattr(Path, "open", _open)

    artifact_hash, paths = workspace_artifact_fingerprint(tmp_path)

    assert mutated is True
    assert artifact_hash is None
    assert "tracked.txt" in paths


def test_fingerprint_tracks_dirty_submodule_contents_and_head(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    component = tmp_path / "component"
    project = tmp_path / "project"
    component.mkdir()
    project.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    tracked = component / "tracked.txt"
    tracked.write_text("version one\n", encoding="utf-8")
    _init_git_project(component)
    _node_project(project)
    _init_git_project(project)
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(component),
            "vendor/component",
        ],
        cwd=project,
        check=True,
    )
    subprocess.run(["git", "commit", "-qam", "add submodule"], cwd=project, check=True)
    nested = project / "vendor" / "component" / "tracked.txt"

    nested.write_text("version two\n", encoding="utf-8")
    first_hash, first_paths = workspace_artifact_fingerprint(project)
    nested.write_text("version three\n", encoding="utf-8")
    second_hash, second_paths = workspace_artifact_fingerprint(project)

    assert first_hash and second_hash and first_hash != second_hash
    assert "vendor/component/tracked.txt" in first_paths
    assert "vendor/component/tracked.txt" in second_paths

    subprocess.run(["git", "add", "tracked.txt"], cwd=nested.parent, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Hermes Test", "-c", "user.email=hermes@example.invalid",
         "commit", "-qm", "advance submodule"],
        cwd=nested.parent,
        check=True,
    )
    committed_hash, _ = workspace_artifact_fingerprint(project)
    assert committed_hash and committed_hash != second_hash


def test_project_under_hermes_home_is_still_fingerprinted(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    project = home / "projects" / "demo"
    project.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _node_project(project)
    source = project / "src" / "app.ts"
    source.parent.mkdir()
    source.write_text("one\n", encoding="utf-8")
    _init_git_project(project)
    mark_workspace_edited(session_id="s-home-project", cwd=project, paths=[str(source)])
    event = record_terminal_result(
        command="scripts/run_tests.sh",
        cwd=project,
        session_id="s-home-project",
        exit_code=0,
        output="green",
    )

    assert event is not None
    assert event["artifact_hash"]
    source.write_text("two\n", encoding="utf-8")
    assert verification_status(session_id="s-home-project", cwd=project)["status"] == "stale"


def test_clean_git_pass_is_invalidated_by_a_new_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    _init_git_project(tmp_path)

    event = record_terminal_result(
        command="scripts/run_tests.sh",
        cwd=tmp_path,
        session_id="s-commit",
        exit_code=0,
        output="all green",
    )
    assert event is not None
    assert verification_status(session_id="s-commit", cwd=tmp_path)["status"] == "passed"
    assert latest_verification_status(session_id="s-commit")["status"] == "passed"

    (tmp_path / "README.md").write_text("new commit\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "advance"], cwd=tmp_path, check=True)

    assert verification_status(session_id="s-commit", cwd=tmp_path)["status"] == "stale"


def test_schema_v1_is_migrated_in_place(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _node_project(tmp_path)
    db_path = home / "verification_evidence.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO meta(key, value) VALUES ('schema_version', '1');
            CREATE TABLE verification_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            );
            CREATE TABLE verification_state (
                session_id TEXT NOT NULL,
                root TEXT NOT NULL,
                last_event_id INTEGER,
                last_edit_at TEXT,
                changed_paths_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (session_id, root)
            );
            """
        )

    assert verification_status(session_id="migrate", cwd=tmp_path)["status"] == "unverified"
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(verification_events)")}
        version = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
    assert {"artifact_hash", "changed_paths_json"} <= columns
    assert version == "2"


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


def test_package_script_shorthand_matches_canonical_verify_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    evidence = classify_verification_command(
        "pnpm test -- tests/button.test.tsx",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )

    assert evidence is not None
    assert evidence.canonical_command == "pnpm run test"
    assert evidence.scope == "targeted"


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


def test_uv_run_pytest_matches_detected_pytest(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        "uv run pytest tests/test_calc.py",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )

    assert evidence is not None
    assert evidence.canonical_command == "pytest"
    assert evidence.scope == "targeted"


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


def test_unprefixed_temp_script_is_not_ad_hoc_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    script = Path(tempfile.gettempdir()) / f"random-check-{tmp_path.name}.py"
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

    assert evidence is None


def test_temp_script_does_not_replace_detected_suite(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
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

    assert evidence is None


def test_non_temp_script_is_not_ad_hoc_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    script = tmp_path / "scripts" / "repro.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="utf-8")

    evidence = classify_verification_command(
        f"python {script}",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="ok",
    )

    assert evidence is None


def test_status_is_unverified_without_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    assert verification_status(session_id="s1", cwd=tmp_path)["status"] == "unverified"


def test_edit_without_prior_evidence_stays_unverified(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    mark_workspace_edited(
        session_id="s1",
        cwd=tmp_path,
        paths=[str(tmp_path / "src" / "app.ts")],
    )

    status = verification_status(session_id="s1", cwd=tmp_path)
    assert status["status"] == "unverified"
    assert status["changed_paths"] == [str(tmp_path / "src" / "app.ts")]


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


def test_recording_prunes_old_events_but_keeps_latest_state(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _node_project(tmp_path)

    for index in range(120):
        record_terminal_result(
            command="pnpm test",
            cwd=tmp_path,
            session_id="s1",
            exit_code=0,
            output=f"green {index}",
        )

    with sqlite3.connect(home / "verification_evidence.db") as conn:
        event_count = conn.execute("SELECT COUNT(*) FROM verification_events").fetchone()[0]
        latest_summary = conn.execute(
            """
            SELECT output_summary
            FROM verification_events
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()[0]

    assert event_count == 100
    assert latest_summary == "green 119"
    assert verification_status(session_id="s1", cwd=tmp_path)["status"] == "passed"


def test_recording_expires_old_current_evidence(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _node_project(tmp_path)

    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="old-session",
        exit_code=0,
        output="old green",
    )
    cutoff = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    with sqlite3.connect(home / "verification_evidence.db") as conn:
        conn.execute("UPDATE verification_events SET created_at = ?", (cutoff,))
        conn.commit()

    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="new-session",
        exit_code=0,
        output="new green",
    )

    assert verification_status(session_id="old-session", cwd=tmp_path)["status"] == "unverified"
    assert verification_status(session_id="new-session", cwd=tmp_path)["status"] == "passed"
    with sqlite3.connect(home / "verification_evidence.db") as conn:
        old_rows = conn.execute(
            "SELECT COUNT(*) FROM verification_events WHERE session_id = 'old-session'"
        ).fetchone()[0]
    assert old_rows == 0


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
