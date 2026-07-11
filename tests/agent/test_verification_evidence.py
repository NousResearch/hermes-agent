import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.verification_evidence import (
    classify_verification_command,
    mark_workspace_edited,
    record_terminal_result,
    verification_status,
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


# --- Outcome/shape inference (issue #62728): a test run that the canonical
# --- verify-command list does not enumerate must still be recognized. ---


def _python_project_with_tests(tmp_path: Path) -> None:
    """A python project whose canonical command list is empty of pytest spellings."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_core.py").write_text("def test_ok():\n    assert True\n")


def test_inference_recognizes_venv_pytest_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project_with_tests(tmp_path)

    evidence = classify_verification_command(
        ".venv/bin/pytest tests/",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="1 passed",
    )
    assert evidence is not None
    assert evidence.kind == "test"
    assert evidence.status == "passed"
    assert evidence.canonical_command == "inferred verification run"


def test_inference_recognizes_cargo_go_mix_bun_test(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project_with_tests(tmp_path)

    for cmd in ("cargo test", "go test ./...", "mix test", "bun test"):
        evidence = classify_verification_command(
            cmd, cwd=tmp_path, session_id="s1", exit_code=0, output="ok"
        )
        assert evidence is not None, cmd
        assert evidence.kind == "test", cmd
        assert evidence.canonical_command == "inferred verification run", cmd


def test_inference_recognizes_make_verify_target(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project_with_tests(tmp_path)
    (tmp_path / "Makefile").write_text("verify:\n\techo ok\ncheck-all:\n\t@true\n")

    for cmd in ("make verify", "make check-all"):
        evidence = classify_verification_command(
            cmd, cwd=tmp_path, session_id="s1", exit_code=0, output="ok"
        )
        assert evidence is not None, cmd
        assert evidence.kind == "test", cmd
        assert evidence.canonical_command == "inferred verification run", cmd


def test_inference_recognizes_named_smoke_script_in_test_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project_with_tests(tmp_path)

    evidence = classify_verification_command(
        "python3 smoke_test.py",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="ok",
    )
    assert evidence is not None
    assert evidence.kind == "test"
    assert evidence.canonical_command == "inferred verification run"


def test_inference_requires_test_shape_not_just_keyword(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    # No test tree: a stray ``./check status`` must NOT be inferred.
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")

    evidence = classify_verification_command(
        "./check status",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="ok",
    )
    assert evidence is None


def test_inference_requires_zero_exit_code(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project_with_tests(tmp_path)

    evidence = classify_verification_command(
        ".venv/bin/pytest tests/",
        cwd=tmp_path,
        session_id="s1",
        exit_code=1,
        output="1 failed",
    )
    assert evidence is None


def test_inference_does_not_mistake_pm_install_for_test(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project_with_tests(tmp_path)

    evidence = classify_verification_command(
        "pip install pytest",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="ok",
    )
    assert evidence is None


def test_inference_does_not_mistake_git_commit_of_test_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project_with_tests(tmp_path)

    evidence = classify_verification_command(
        "git commit tests/test_core.py -m wip",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="ok",
    )
    assert evidence is None


def test_inferred_run_satisfies_freshness_check(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project_with_tests(tmp_path)

    event = record_terminal_result(
        command=".venv/bin/pytest tests/test_core.py",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="1 passed",
    )
    assert event is not None
    status = verification_status(session_id="s1", cwd=tmp_path)
    assert status["status"] == "passed"
    assert status["evidence"]["kind"] == "test"


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
