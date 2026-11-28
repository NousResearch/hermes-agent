import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path

import pytest

from agent.verification_evidence import (
    classify_verification_command,
    mark_workspace_edited,
    record_terminal_result,
    record_verify_run,
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


@pytest.mark.parametrize("pipeline_status, expected", [([0, 0], "passed"), ([1, 0], "failed")])
def test_piped_verifier_uses_its_own_pipeline_status(tmp_path, monkeypatch, pipeline_status, expected):
    """``pytest | tail`` exits with tail's status; PIPESTATUS carries pytest's own."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        "pytest -q 2>&1 | tail -20", cwd=tmp_path, session_id="s1",
        exit_code=0, output="12 passed", pipeline_status=pipeline_status,
    )

    assert evidence is not None
    assert evidence.canonical_command == "pytest"
    assert evidence.status == expected


def test_piped_verifier_without_pipeline_status_is_not_evidence(tmp_path, monkeypatch):
    """Without PIPESTATUS the shell's code belongs to the filter, so nothing is proven."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    assert classify_verification_command(
        "pytest -q 2>&1 | tail -20", cwd=tmp_path, session_id="s1", exit_code=0,
    ) is None


@pytest.mark.parametrize("command, exit_code, pipeline_status, expected", [
    ("true | false && pytest -q | tail -5", 1, [0, 1], None),
    ("cd . && pytest -q | tail -5", 0, [1, 0], "failed"),
])
def test_piped_verifier_behind_and_counts_only_when_the_chain_ran(
    tmp_path, monkeypatch, command, exit_code, pipeline_status, expected,
):
    """A short-circuited ``&&`` skips the verifier and leaves an earlier pipeline's PIPESTATUS."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        command, cwd=tmp_path, session_id="s1", exit_code=exit_code, pipeline_status=pipeline_status,
    )

    assert (evidence and evidence.status) == expected


@pytest.mark.parametrize("pipeline_status", [[0], [1]])
def test_piped_verifier_with_single_pipeline_status_is_not_evidence(
    tmp_path, monkeypatch, pipeline_status,
):
    """A one-element PIPESTATUS belongs to a compound command (``!``, ``if``), not a pipe."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    assert classify_verification_command(
        "pytest -q 2>&1 | tail -20", cwd=tmp_path, session_id="s1",
        exit_code=0, pipeline_status=pipeline_status,
    ) is None


@pytest.mark.parametrize(
    "command",
    [
        "./venv/bin/python -m pytest tests/test_widget.py -q",
        "venv/bin/python3 -m pytest tests/test_widget.py -q",
        "/opt/venv/bin/python3.12 -m pytest tests/test_widget.py -q",
        ".venv/bin/python -I -B -m pytest -o pythonpath=.",
        "/usr/bin/python3.12 -u -m pytest tests/test_widget.py -q",
        "python -W ignore -X dev -m pytest tests/test_widget.py -q",
        "py -3.11 -m pytest tests/test_widget.py -q",
    ],
)
def test_venv_interpreter_path_matches_detected_pytest(tmp_path, monkeypatch, command):
    """A venv interpreter path is the same ``pytest`` run: the agent's usual spelling."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        command, cwd=tmp_path, session_id="s1", exit_code=0, output="12 passed",
    )

    assert evidence is not None
    assert evidence.canonical_command == "pytest"
    assert evidence.status == "passed"


@pytest.mark.parametrize(
    "command",
    [
        "python --version -m pytest",
        "python -c pass -m pytest",
        "python -h -m pytest",
    ],
)
def test_interpreter_flag_that_skips_the_module_is_not_evidence(tmp_path, monkeypatch, command):
    """A flag that exits before ``-m`` runs cannot produce passing evidence."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    assert classify_verification_command(
        command, cwd=tmp_path, session_id="s1", exit_code=0,
    ) is None


def test_dot_slash_script_invocation_matches_detected_script(tmp_path, monkeypatch):
    """``./scripts/run_tests.sh`` is the detected command run the obvious way."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    evidence = classify_verification_command(
        "./scripts/run_tests.sh", cwd=tmp_path, session_id="s1", exit_code=0,
    )

    assert evidence is not None
    assert evidence.canonical_command == "scripts/run_tests.sh"


@pytest.mark.parametrize("command", [
    "bash -s scripts/run_tests.sh", "sh -s scripts/run_tests.sh",
    "nice --help scripts/run_tests.sh", "timeout --help 300 scripts/run_tests.sh",
    "nohup --version scripts/run_tests.sh", "stdbuf --help scripts/run_tests.sh",
])
def test_flag_that_skips_the_script_is_not_evidence(tmp_path, monkeypatch, command):
    """``bash -s`` reads stdin and ``--help`` exits 0: the script named after it never runs."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    assert classify_verification_command(
        command, cwd=tmp_path, session_id="s1", exit_code=0,
    ) is None


@pytest.mark.parametrize(
    "command",
    ["timeout 300 pytest -q", "nice -n 5 pytest -q", "timeout 5m pytest -q", "timeout -s KILL 300 pytest -q",
     "timeout --signal=KILL -k 10 5m pytest -q", "stdbuf -oL nohup pytest -q"],
)
def test_status_preserving_wrapper_still_attributes_the_run(tmp_path, monkeypatch, command):
    """``timeout``/``nice`` return the child's status, so the run proves the verifier's."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        command, cwd=tmp_path, session_id="s1", exit_code=0,
    )

    assert evidence is not None
    assert evidence.canonical_command == "pytest"
    assert evidence.status == "passed"


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
    script = Path(tempfile.gettempdir()) / f"hermes-verify-{tmp_path.name}.py"
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
    script = Path(tempfile.gettempdir()) / f"hermes-verify-{tmp_path.name}.py"

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






def test_workspace_verify_clears_stale_on_other_session(tmp_path, monkeypatch):
    """A hermes verify pass is a property of the workspace, not the session
    that ran the CLI (#103271)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)
    changed = str(tmp_path / "changed.py")
    mark_workspace_edited(session_id="s-edit", cwd=tmp_path, paths=[changed])
    assert verification_status(session_id="s-edit", cwd=tmp_path)["status"] == "unverified"

    event = record_verify_run(root=tmp_path, session_id="default", ok=True, output="all green")
    assert event is not None
    status = verification_status(session_id="s-edit", cwd=tmp_path)
    assert status["status"] == "passed"
    assert status["evidence"]["canonical_command"] == "hermes verify"


def test_targeted_partial_verify_does_not_clear_other_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)
    record_verify_run(root=tmp_path, session_id="s-edit", ok=True)
    mark_workspace_edited(session_id="s-edit", cwd=tmp_path, paths=[str(tmp_path / "a.py")])
    assert verification_status(session_id="s-edit", cwd=tmp_path)["status"] == "stale"

    record_verify_run(root=tmp_path, session_id=None, ok=True, scope="targeted")

    status = verification_status(session_id="s-edit", cwd=tmp_path)
    assert status["status"] == "stale"


def test_failed_verify_does_not_erase_other_session_edit_record(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _python_project(tmp_path)
    mark_workspace_edited(session_id="s-edit", cwd=tmp_path, paths=[str(tmp_path / "a.py")])

    record_verify_run(root=tmp_path, session_id=None, ok=False)

    row = sqlite3.connect(home / "verification_evidence.db").execute(
        "SELECT last_edit_at, changed_paths_json FROM verification_state WHERE session_id='s-edit'"
    ).fetchone()
    assert row is not None
    assert row[0] is not None
    assert verification_status(session_id="s-edit", cwd=tmp_path)["status"] != "passed"


def test_older_workspace_verify_does_not_clear_later_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)
    record_verify_run(root=tmp_path, session_id="default", ok=True, output="old pass")
    mark_workspace_edited(
        session_id="s-edit",
        cwd=tmp_path,
        paths=[str(tmp_path / "changed.py")],
    )
    assert verification_status(session_id="s-edit", cwd=tmp_path)["status"] == "unverified"


def test_recording_expires_old_edit_only_state(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _node_project(tmp_path)

    mark_workspace_edited(
        session_id="old-session",
        cwd=tmp_path,
        paths=[str(tmp_path / "src" / "app.ts")],
    )
    cutoff = (datetime.now(UTC) - timedelta(days=31)).isoformat()
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
    assert _find_ad_hoc_match(f"{interpreter} {win_script}", tmp_path) == ([], 0)


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


# --- a status the command echoed is the honest one when the shell hid its own --------------


def test_echoed_status_after_a_sequence_attributes_a_failing_verifier(tmp_path, monkeypatch):
    """`V; echo "TESTS_RC=$?"` leaves the shell's code to the echo, so the verifier's real
    status only exists in the output. Reading it back is what makes the shape honest instead
    of silently passing a failing run.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        'pytest -q; echo "TESTS_RC=$?"',
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="1 failed, 5 passed\nTESTS_RC=1",
    )

    assert evidence is not None
    assert evidence.canonical_command == "pytest"
    assert evidence.kind == "test"
    assert evidence.status == "failed"


def test_echoed_pipeline_status_attributes_the_leading_verifier(tmp_path, monkeypatch):
    """`V | tail; echo "RC=${PIPESTATUS[0]}"` names the verifier's own slot, so the pipeline's
    status survives the filter even when the tool handed over no PIPESTATUS list.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        'pytest -q | tail -5; echo "RC=${PIPESTATUS[0]}"',
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="1 failed, 5 passed\nRC=1",
    )

    assert evidence is not None
    assert evidence.canonical_command == "pytest"
    assert evidence.status == "failed"


def test_echoed_status_also_attributes_an_ad_hoc_script(tmp_path, monkeypatch):
    """The nudge itself suggests running a temp script; wrapping it in `; echo "RC=$?"` must
    still record the run, otherwise the advice the agent is given yields nothing.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    script = Path(tempfile.gettempdir()) / f"hermes-verify-{tmp_path.name}.py"
    script.write_text("raise SystemExit(1)\n", encoding="utf-8")
    try:
        evidence = classify_verification_command(
            f'python {script}; echo "RC=$?"',
            cwd=tmp_path,
            session_id="s1",
            exit_code=0,
            output="boom\nRC=1",
        )
    finally:
        script.unlink(missing_ok=True)

    assert evidence is not None
    assert evidence.kind == "ad_hoc"
    assert evidence.status == "failed"


def test_pipefail_makes_the_shell_code_describe_the_verifier(tmp_path, monkeypatch):
    """Under `set -o pipefail` the pipeline's code is the rightmost non-zero member, so it
    describes the verifier and both directions become readable.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    failing = classify_verification_command(
        "set -o pipefail; pytest -q | tail -5",
        cwd=tmp_path,
        session_id="s1",
        exit_code=1,
        output="1 failed, 5 passed",
    )
    passing = classify_verification_command(
        "set -o pipefail; pytest -q | tail -5",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="5 passed",
    )

    assert failing is not None and failing.status == "failed"
    assert passing is not None and passing.status == "passed"


@pytest.mark.parametrize(
    "command, exit_code",
    [
        ("pytest -q | tail -5", 1),
        ("false && pytest -q | tail -5", 1),
    ],
)
def test_without_pipefail_or_a_ran_chain_the_pipeline_code_is_not_evidence(
    tmp_path, monkeypatch, command, exit_code
):
    """The mirror of the fix: a plain pipeline's code belongs to the last member, and a chain
    that never reached the verifier must not claim its run.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    assert (
        classify_verification_command(
            command, cwd=tmp_path, session_id="s1", exit_code=exit_code, output="1 failed"
        )
        is None
    )


@pytest.mark.parametrize(
    "command, output",
    [
        ('pytest -q | tail -5; echo "RC=${PIPESTATUS[1]}"', "RC=1"),
        ('pytest -q | tail -5; echo "RC=$?"', "RC=1"),
        ('pytest -q; echo "done"', "done"),
        ('pytest -q; echo "TESTS_RC=$?"', "5 passed"),
        ('pytest -q; ls; echo "RC=$?"', "RC=1"),
    ],
)
def test_echoed_status_is_ignored_when_it_cannot_belong_to_the_verifier(tmp_path, monkeypatch, command, output):
    """Everything the echo can legitimately mean something else: a later member's PIPESTATUS
    slot, `$?` after the filter, an echo that carries no status, a label that never printed,
    and a command that ran between the verifier and the echo.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    assert (
        classify_verification_command(
            command, cwd=tmp_path, session_id="s1", exit_code=0, output=output
        )
        is None
    )


# --- the end directory may carry the evidence when the start directory cannot ---------------


def test_alt_cwd_recovers_evidence_when_the_start_directory_names_no_workspace(tmp_path, monkeypatch):
    """`cd project && pytest` starts outside the project, so the start directory classifies
    nothing while the shell's code describes a run inside it (#151339).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    project = tmp_path / "project"
    project.mkdir()
    _python_project(project)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    assert classify_verification_command("pytest -q", cwd=elsewhere, session_id="s1", exit_code=0) is None

    evidence = classify_verification_command(
        "pytest -q", cwd=elsewhere, session_id="s1", exit_code=0, alt_cwd=project
    )

    assert evidence is not None
    assert evidence.canonical_command == "pytest"
    assert evidence.root == str(project.resolve())


def test_alt_cwd_does_not_replace_a_start_directory_that_already_answers(tmp_path, monkeypatch):
    """The start directory stays first: a command that classifies there is not re-attributed
    to wherever the shell happened to end up.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    started = tmp_path / "started"
    started.mkdir()
    _python_project(started)
    ended = tmp_path / "ended"
    ended.mkdir()
    _python_project(ended)

    evidence = classify_verification_command(
        "pytest -q", cwd=started, session_id="s1", exit_code=0, alt_cwd=ended
    )

    assert evidence is not None
    assert evidence.root == str(started.resolve())


def test_alt_cwd_records_a_terminal_result_from_the_end_of_the_command(tmp_path, monkeypatch):
    """`terminal_tool_result` hands over the cwd the command ended in; without it the run is
    dropped and every later turn re-nudges the same workspace as unverified.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    project = tmp_path / "project"
    project.mkdir()
    _python_project(project)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    recorded = record_terminal_result(
        command="pytest -q",
        cwd=elsewhere,
        session_id="s1",
        exit_code=0,
        output="5 passed",
        alt_cwd=project,
    )

    assert recorded is not None
    assert recorded["status"] == "passed"
    assert recorded["root"] == str(project.resolve())


def test_a_cd_into_another_workspace_outranks_the_starting_one(tmp_path, monkeypatch):
    """`cd ended && pytest` run from a different workspace: the exit status describes the
    run inside `ended`, so that workspace claims the evidence and the starting directory
    stays unverified (#151861 review).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    started = tmp_path / "started"
    started.mkdir()
    _python_project(started)
    ended = tmp_path / "ended"
    ended.mkdir()
    _python_project(ended)

    evidence = classify_verification_command(
        f"cd {ended.as_posix()} && pytest -q",
        cwd=started, session_id="s1", exit_code=0, alt_cwd=ended,
    )

    assert evidence is not None
    assert evidence.canonical_command == "pytest"
    assert evidence.root == str(ended.resolve())


def test_a_trailing_cd_leaves_the_starting_workspace_in_charge(tmp_path, monkeypatch):
    """`pytest && cd ended`: the tests ran before the shell moved, so the starting directory
    still describes that run and keeps the evidence.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    started = tmp_path / "started"
    started.mkdir()
    _python_project(started)
    ended = tmp_path / "ended"
    ended.mkdir()
    _python_project(ended)

    evidence = classify_verification_command(
        f"pytest -q && cd {ended.as_posix()}",
        cwd=started, session_id="s1", exit_code=0, alt_cwd=ended,
    )

    assert evidence is not None
    assert evidence.root == str(started.resolve())


def test_a_cd_into_another_workspace_records_the_terminal_result_there(tmp_path, monkeypatch):
    """The ledger follows the same precedence: a run that `cd`ed elsewhere is recorded for
    the workspace it ran in, so the start directory keeps its verify nudge.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    started = tmp_path / "started"
    started.mkdir()
    _python_project(started)
    ended = tmp_path / "ended"
    ended.mkdir()
    _python_project(ended)

    recorded = record_terminal_result(
        command=f"cd {ended.as_posix()} && pytest -q",
        cwd=started,
        session_id="s1",
        exit_code=0,
        output="5 passed",
        alt_cwd=ended,
    )

    assert recorded is not None
    assert recorded["status"] == "passed"
    assert recorded["root"] == str(ended.resolve())
