import json
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hermes_constants import get_scratch_dir

from agent.verification_evidence import (
    classify_verification_command,
    mark_workspace_edited,
    record_terminal_result,
    verification_status,
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


def _manifestless_unittest_project(root: Path) -> None:
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_widget.py").write_text(
        "import unittest\n\nclass WidgetTests(unittest.TestCase):\n    def test_widget(self):\n        pass\n"
    )


def _hermes_scratch(root: Path) -> Path:
    return get_scratch_dir(root / ".hermes", prune=False)








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
    script = _hermes_scratch(tmp_path) / f"hermes-ad-hoc-{tmp_path.name}.py"
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
    script = _hermes_scratch(tmp_path) / f"hermes-ad-hoc-{tmp_path.name}.py"
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


@pytest.mark.parametrize(
    ("exit_code", "output", "expected_status"),
    [
        (0, "Ran 1 test in 0.001s\n\nOK", "passed"),
        (1, "Ran 1 test in 0.001s\n\nFAILED (failures=1)", "failed"),
        (0, "Ran 0 tests in 0.000s\n\nOK", None),
    ],
)
def test_manifestless_unittest_requires_observed_nonempty_test_run(
    tmp_path, monkeypatch, exit_code, output, expected_status
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _manifestless_unittest_project(tmp_path)

    evidence = classify_verification_command(
        "python -m unittest",
        cwd=tmp_path,
        session_id="s1",
        exit_code=exit_code,
        output=output,
    )

    if expected_status is None:
        assert evidence is None
    else:
        assert evidence is not None
        assert evidence.canonical_command == "python -m unittest"
        assert evidence.status == expected_status


@pytest.mark.parametrize("command", [
    "python -c \"print('Ran 1 test')\" && python -m unittest",
    "python -m unittest && python -c \"print('Ran 1 test')\"",
])
def test_unittest_compound_output_is_not_attributed_to_test_run(tmp_path, command):
    _manifestless_unittest_project(tmp_path)
    assert classify_verification_command(
        command, cwd=tmp_path, exit_code=0,
        output="Ran 1 test\nRan 0 tests in 0.000s\nOK\n",
    ) is None


@pytest.mark.parametrize("output", [
    "Ran 1 test\nRan 0 tests in 0.000s\n\nOK\n",
    "Ran 1 test in 0.001s\nRan 0 tests in 0.000s\n\nOK\n",
    "Ran 0 tests in 0.000s\n\nOK\nRan 1 test in 0.001s\n",
])
def test_unittest_rejects_misleading_or_multiple_summaries(tmp_path, output):
    _manifestless_unittest_project(tmp_path)
    assert classify_verification_command(
        "python -m unittest", cwd=tmp_path, exit_code=0, output=output,
    ) is None


def test_unittest_filename_does_not_claim_canonical_suite(tmp_path):
    from agent.coding_context import project_facts_for

    _manifestless_unittest_project(tmp_path)
    (tmp_path / "testutils.py").write_text("HELPER = True\n")
    facts = project_facts_for(tmp_path)
    assert "python -m unittest" not in facts["verifyCommands"]
    script = tmp_path / "hermes-verify-fallback.py"
    script.write_text("assert 1 + 1 == 2\n")
    assert classify_verification_command(
        f'python "{script.as_posix()}"', cwd=tmp_path, exit_code=0, output=""
    ) is not None


@pytest.mark.parametrize("interpreter", ["node", "bash", "ruby", "$PY"])
def test_non_python_unittest_invocation_is_not_evidence(tmp_path, interpreter):
    _manifestless_unittest_project(tmp_path)
    assert classify_verification_command(
        f"{interpreter} -m unittest", cwd=tmp_path,
        exit_code=0, output="Ran 1 test in 0.001s\nOK"
    ) is None


def test_unittest_does_not_bypass_declared_pytest_suite(tmp_path):
    _manifestless_unittest_project(tmp_path)
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    assert classify_verification_command(
        "python -m unittest", cwd=tmp_path,
        exit_code=0, output="Ran 1 test in 0.001s\nOK"
    ) is None


def test_real_unittest_discovery_records_merged_stderr(tmp_path, monkeypatch):
    import sys

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _manifestless_unittest_project(tmp_path)
    args = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    run = subprocess.run(args, cwd=tmp_path, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, timeout=20)
    assert run.returncode == 0
    assert "Ran 1 test" in run.stdout
    event = record_terminal_result(
        command=f'"{Path(sys.executable).as_posix()}" -m unittest discover -s tests',
        cwd=tmp_path, session_id="actual-unittest", exit_code=run.returncode,
        output=run.stdout,
    )
    assert event is not None and event["status"] == "passed"
    assert verification_status(session_id="actual-unittest", cwd=tmp_path)["status"] == "passed"


def test_legacy_system_temp_script_remains_evidence(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.json").write_text("{}", encoding="utf-8")
    legacy_temp = tmp_path / "legacy-temp"
    legacy_temp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(legacy_temp))
    script = legacy_temp / "hermes-verify-legacy.py"
    script.write_text("assert True\n", encoding="utf-8")

    evidence = classify_verification_command(
        f'python "{script.as_posix()}"', cwd=project, exit_code=0
    )

    assert evidence is not None
    assert evidence.kind == "ad_hoc"


def test_hermes_scratch_ad_hoc_script_is_allowed(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    script = get_scratch_dir(home, prune=False) / "hermes-verify-scratch.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    evidence = classify_verification_command(
        f"python {script}", cwd=tmp_path, session_id="s1", exit_code=0, output="ok"
    )

    assert evidence is not None
    assert evidence.kind == "ad_hoc"


def test_workspace_prefixed_ad_hoc_script_is_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    script = tmp_path / "hermes-verify-workspace.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    evidence = classify_verification_command(
        f"python {script}", cwd=tmp_path, session_id="s1", exit_code=0, output="ok"
    )

    assert evidence is not None
    assert evidence.kind == "ad_hoc"


@pytest.mark.parametrize(
    "invocation",
    [
        "python3.12 {script}",
        "/usr/bin/python3 {script}",
        "/usr/bin/python3.12 {script}",
        "python3 -u {script}",
        "/usr/bin/env python3 {script}",
    ],
)
def test_versioned_or_absolute_interpreter_records_ad_hoc_evidence(tmp_path, monkeypatch, invocation):
    """A versioned, absolute or `env`-prefixed interpreter names the same interpreter as a bare
    `python3`. Matching only the bare token left the ad-hoc branch blind to the invocation shapes
    the verify-on-stop nudge itself hands the agent, so a passing run recorded no evidence and
    every later turn re-nudged the same workspace as unverified.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    script = _hermes_scratch(tmp_path) / f"hermes-verify-{tmp_path.name}.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    try:
        evidence = classify_verification_command(
            invocation.format(script=script),
            cwd=tmp_path,
            session_id="s1",
            exit_code=0,
            output="ok",
        )
    finally:
        script.unlink(missing_ok=True)

    assert evidence is not None
    assert evidence.kind == "ad_hoc"
    assert evidence.status == "passed"


def test_non_interpreter_command_touching_the_temp_script_is_not_evidence(tmp_path, monkeypatch):
    """The nudge also tells the agent to clean the temp script up. A command that merely names
    the script — `rm`, `chmod`, `cat` — runs no verification and must not satisfy the gate.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    script = _hermes_scratch(tmp_path) / f"hermes-verify-{tmp_path.name}.py"

    for command in (f"rm -f {script}", f"/usr/bin/chmod +x {script}", f"/usr/bin/cat {script}"):
        evidence = classify_verification_command(command, cwd=tmp_path, session_id="s1", exit_code=0)
        assert evidence is None, f"{command!r} must not be recorded as verification evidence"












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


@pytest.mark.parametrize(
    "interpreter",
    [r"C:\Users\me\venv\Scripts\python.exe", "python.EXE", "py -3"],
)
def test_windows_exe_interpreter_records_ad_hoc_evidence(tmp_path, monkeypatch, interpreter):
    """A venv's absolute ``Scripts\\python.exe`` (or the ``py`` launcher) is the interpreter shape
    Windows hands the agent; ``.exe`` must not hide it from the ad-hoc branch (review follow-up)."""
    from agent.verification_evidence import _find_ad_hoc_match

    monkeypatch.setattr(
        "agent.verification_evidence._is_temp_script_path",
        lambda token, root: "hermes-verify-" in token and token.endswith(".py"),
    )
    win_script = r"C:\Users\me\AppData\Local\Temp\hermes-verify-x.py"
    assert _find_ad_hoc_match(f"{interpreter} {win_script}", tmp_path) == []


def test_quoted_windows_interpreter_and_script_are_matched(tmp_path, monkeypatch):
    from agent.verification_evidence import _find_ad_hoc_match

    monkeypatch.setattr(
        "agent.verification_evidence._is_temp_script_path",
        lambda token, root: "hermes-verify-" in token and token.replace('"', "").endswith(".py"),
    )
    interpreter = r'"C:\Program Files\Python\python.exe"'
    script = r'"C:\Users\me\AppData\Local\hermes\cache\scratch\hermes-verify-x.py"'

    assert _find_ad_hoc_match(f"{interpreter} {script}", tmp_path) == []


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
