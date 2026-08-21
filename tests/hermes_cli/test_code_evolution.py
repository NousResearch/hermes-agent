"""Behavior tests for bounded Hermes code-evolution campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import code_evolution as ce
from hermes_cli import code_evolution_process_guard as cepg
from hermes_cli import code_evolution_verifier as cev
from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Hermes Tests")
    (repo / "src").mkdir()
    (repo / "src" / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_agent.py").write_text(
        "def test_value():\n    assert True\n", encoding="utf-8"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


@pytest.fixture
def profile_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "profiles" / "reviewer").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _project_for_repo(repo: Path) -> str:
    with pdb.connect_closing() as conn:
        return pdb.create_project(
            conn,
            name=f"Code Evolution {repo.parent.name}",
            folders=[str(repo)],
            primary_path=str(repo),
        )


def _prepare_contract(repo: Path) -> ce.PreparedContract:
    return ce.prepare_contract(
        repository=repo,
        project=_project_for_repo(repo),
        objective="Make retries deterministic",
        evidence="tests/test_agent.py reproduces nondeterministic retry ordering",
        success_metric="The frozen regression gate exits zero on the final candidate",
        allowed_paths=("src", "tests/test_agent.py"),
        quality_gates=((f'"{sys.executable}" -c "import sys; sys.exit(0)"', 30),),
        assignee="default",
        reviewer="reviewer",
        goal_max_turns=7,
        max_runtime_seconds=600,
    )


def _write_frozen_files(
    tmp_path: Path, prepared: ce.PreparedContract
) -> tuple[Path, Path]:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    contract_path = frozen / ce.CONTRACT_FILENAME
    verifier_path = frozen / ce.VERIFIER_FILENAME
    contract_path.write_bytes(prepared.contract_bytes)
    verifier_path.write_bytes(prepared.verifier_bytes)
    return contract_path, verifier_path


def _run_frozen_verifier(
    repository: Path,
    contract_path: Path,
    verifier_path: Path,
    contract_sha256: str,
    *,
    run_gates: bool,
    expected_workspace: Path | None = None,
    expected_branch: str | None = None,
) -> dict:
    mode = "--run-gates" if run_gates else "--preflight"
    argv = [
        sys.executable,
        str(verifier_path),
        "--contract",
        str(contract_path),
        "--expected-contract-sha256",
        contract_sha256,
        "--repo",
        str(repository),
    ]
    if expected_workspace is not None:
        argv.extend(["--expected-workspace", str(expected_workspace)])
    if expected_branch is not None:
        argv.extend(["--expected-branch", expected_branch])
    argv.append(mode)
    result = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def test_prepare_contract_freezes_repo_identity_and_policy(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)

    prepared = ce.prepare_contract(
        repository=repo,
        project=_project_for_repo(repo),
        objective="Make retries deterministic",
        evidence="tests/test_agent.py reproduces nondeterministic retry ordering",
        success_metric="The frozen regression gate exits zero on the final candidate",
        allowed_paths=("src/", "tests/test_agent.py"),
        quality_gates=((f'"{sys.executable}" -c "import sys; sys.exit(0)"', 30),),
        assignee="default",
        reviewer="Reviewer",
        goal_max_turns=7,
        max_runtime_seconds=600,
    )

    payload = prepared.payload
    assert payload["kind"] == "hermes-code-evolution"
    assert payload["schema_version"] == 1
    assert payload["repository"] == str(repo.resolve())
    assert payload["git_common_dir"] == _git(
        repo, "rev-parse", "--path-format=absolute", "--git-common-dir"
    )
    assert payload["base_commit"] == _git(repo, "rev-parse", "HEAD")
    assert payload["base_tree"] == _git(repo, "rev-parse", "HEAD^{tree}")
    assert payload["success_metric"] == (
        "The frozen regression gate exits zero on the final candidate"
    )
    assert payload["allowed_paths"] == ["src", "tests/test_agent.py"]
    assert payload["allowed_path_rules"] == [
        {"kind": "tree", "path": "src"},
        {"kind": "file", "path": "tests/test_agent.py"},
    ]
    assert payload["assignee"] == "default"
    assert payload["reviewer"] == "reviewer"
    assert payload["budgets"] == {
        "goal_max_turns": 7,
        "max_runtime_seconds": 600,
        "max_retries": 1,
    }
    assert payload["verifier"]["filename"] == ce.VERIFIER_FILENAME
    assert (
        payload["verifier"]["sha256"]
        == hashlib.sha256(prepared.verifier_bytes).hexdigest()
    )

    envelope = json.loads(prepared.contract_bytes)
    canonical_payload = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert envelope == {"contract": payload, "sha256": prepared.sha256}
    assert prepared.sha256 == hashlib.sha256(canonical_payload).hexdigest()
    assert prepared.contract_id == f"ce_{prepared.sha256[:16]}"


def test_prepare_contract_rejects_empty_success_metric(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)

    with pytest.raises(ce.CodeEvolutionError, match="success metric is required"):
        ce.prepare_contract(
            repository=repo,
            project=_project_for_repo(repo),
            objective="Make retries deterministic",
            evidence="the failure reproduces",
            success_metric=" ",
            allowed_paths=("src",),
            quality_gates=((sys.executable, 30),),
            assignee="default",
            reviewer="reviewer",
        )


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            r"C:\Python\python.exe -m pytest",
            [r"C:\Python\python.exe", "-m", "pytest"],
        ),
        (
            r'"C:\Program Files\Python\python.exe" -m pytest',
            [r"C:\Program Files\Python\python.exe", "-m", "pytest"],
        ),
        (
            r'tool --output="C:\Program Files\artifact"',
            ["tool", r"--output=C:\Program Files\artifact"],
        ),
    ],
)
def test_quality_gate_parser_preserves_windows_executable_paths(
    command: str, expected: list[str]
) -> None:
    assert ce._split_quality_gate_command(command, windows=True) == expected


def _listener_gate_parent_code(
    ready_path: Path,
    *,
    wait_for_timeout: bool,
    child_pid_path: Path | None = None,
    child_start_new_session: bool = False,
) -> str:
    child_code = (
        "import os, socket, time; from pathlib import Path; "
        "listener=socket.socket(); listener.bind(('127.0.0.1', 0)); "
        "listener.listen(); "
        f"Path({str(ready_path)!r}).write_text(str(listener.getsockname()[1]), "
        "encoding='utf-8'); "
    )
    if child_pid_path is not None:
        child_code += (
            f"Path({str(child_pid_path)!r}).write_text(str(os.getpid()), "
            "encoding='utf-8'); "
        )
    child_code += "time.sleep(30)"
    parent_code = (
        "import subprocess, sys, time; "
        f"ready={str(ready_path)!r}; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, "
        f"start_new_session={child_start_new_session!r}); "
        "deadline=time.monotonic()+5; "
        "exec(\"while not __import__('os').path.exists(ready):\\n"
        '    assert time.monotonic() < deadline\\n    time.sleep(0.01)"); '
    )
    if wait_for_timeout:
        parent_code += "time.sleep(30)"
    return parent_code


def _assert_listener_closed(ready_path: Path) -> None:
    port = int(ready_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 3
    while True:
        with socket.socket() as probe:
            probe.settimeout(0.1)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return
        if time.monotonic() >= deadline:
            pytest.fail("quality-gate descendant retained its listening socket")
        time.sleep(0.05)


def test_posix_process_guard_refuses_unsupported_platform_before_spawn(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "must-not-run"

    returncode = cepg.run_guarded(
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        platform="darwin",
        cleanup_timeout=0.1,
    )

    assert returncode == cepg._CLEANUP_FAILURE_EXIT
    assert not marker.exists()


@pytest.mark.parametrize(
    ("os_name", "platform", "expected"),
    [
        ("nt", "win32", True),
        ("posix", "linux", True),
        ("posix", "darwin", False),
        ("posix", "freebsd14", False),
    ],
)
def test_strict_verifier_containment_support_is_explicit(
    os_name: str,
    platform: str,
    expected: bool,
) -> None:
    assert (
        ce._strict_verifier_containment_available(
            os_name=os_name,
            platform=platform,
        )
        is expected
    )


def test_tracked_process_cleanup_reports_unverified_survivor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Survivor:
        pid = 424242

        def __init__(self) -> None:
            self.kill_calls = 0

        def is_running(self) -> bool:
            return True

        def create_time(self) -> float:
            return 1.0

        def status(self) -> str:
            return "running"

        def kill(self) -> None:
            self.kill_calls += 1

    survivor = Survivor()
    observed_timeouts: list[float] = []

    def retain_processes(processes, *, timeout):
        observed_timeouts.append(timeout)
        return [], list(processes)

    monkeypatch.setattr(cepg.psutil, "wait_procs", retain_processes)
    started = time.monotonic()

    error = cepg._terminate_tracked_processes(
        [survivor],
        deadline=started + 0.25,
        label="verifier descendant",
    )

    assert survivor.kill_calls == 1
    assert observed_timeouts and 0 <= observed_timeouts[0] <= 0.25
    assert "survived cleanup" in (error or "")
    assert time.monotonic() - started < 0.25


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_process_guard_cleans_verifier_after_inspection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "verifier.pid"

    def fail_after_verifier_starts(_parent, _tracked) -> None:
        deadline = time.monotonic() + 2
        while not pid_path.exists():
            assert time.monotonic() < deadline
            time.sleep(0.005)
        raise OSError("synthetic process-table failure")

    monkeypatch.setattr(cepg, "_capture_descendants", fail_after_verifier_starts)
    try:
        returncode = cepg.run_guarded(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import os, time; "
                f"Path({str(pid_path)!r}).write_text(str(os.getpid())); "
                "time.sleep(30)",
            ],
            cleanup_timeout=0.5,
        )

        assert returncode == cepg._CLEANUP_FAILURE_EXIT
        pid = int(pid_path.read_text(encoding="utf-8"))
        with pytest.raises(cepg.psutil.NoSuchProcess):
            cepg.psutil.Process(pid).status()
    finally:
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text(encoding="utf-8")), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_quality_gate_timeout_terminates_descendant_processes(tmp_path: Path) -> None:
    ready = tmp_path / "timeout-child-port.txt"

    result = cev._run_gate(
        tmp_path,
        {
            "argv": [
                sys.executable,
                "-c",
                _listener_gate_parent_code(ready, wait_for_timeout=True),
            ],
            "timeout_seconds": 3,
        },
    )

    assert result["timed_out"] is True
    _assert_listener_closed(ready)


def test_quality_gate_cleans_descendants_after_success(tmp_path: Path) -> None:
    ready = tmp_path / "success-child-port.txt"

    result = cev._run_gate(
        tmp_path,
        {
            "argv": [
                sys.executable,
                "-c",
                _listener_gate_parent_code(ready, wait_for_timeout=False),
            ],
            "timeout_seconds": 5,
        },
    )

    assert result["passed"] is True
    _assert_listener_closed(ready)


@pytest.mark.windows_only
def test_outer_verifier_job_cleans_descendants_after_success(tmp_path: Path) -> None:
    ready = tmp_path / "outer-verifier-child-port.txt"

    result = ce._run_verifier_process(
        [
            sys.executable,
            "-c",
            _listener_gate_parent_code(ready, wait_for_timeout=False),
        ],
        timeout=5,
    )

    assert result.returncode == 0
    _assert_listener_closed(ready)


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_outer_verifier_cleans_posix_descendants_after_success(tmp_path: Path) -> None:
    ready = tmp_path / "outer-verifier-posix-success-port.txt"
    child_pid = tmp_path / "outer-verifier-posix-success.pid"

    try:
        result = ce._run_verifier_process(
            [
                sys.executable,
                "-c",
                _listener_gate_parent_code(
                    ready,
                    wait_for_timeout=False,
                    child_pid_path=child_pid,
                    child_start_new_session=True,
                ),
            ],
            timeout=5,
        )

        assert result.returncode == 0
        _assert_listener_closed(ready)
    finally:
        if child_pid.exists():
            try:
                os.kill(int(child_pid.read_text(encoding="utf-8")), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_outer_verifier_timeout_cleans_posix_tree_within_deadline(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "outer-verifier-posix-timeout-port.txt"
    child_pid = tmp_path / "outer-verifier-posix-timeout.pid"

    started = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            ce._run_verifier_process(
                [
                    sys.executable,
                    "-c",
                    _listener_gate_parent_code(
                        ready,
                        wait_for_timeout=True,
                        child_pid_path=child_pid,
                        child_start_new_session=True,
                    ),
                ],
                timeout=2,
            )
        elapsed = time.monotonic() - started

        assert elapsed < 3
        _assert_listener_closed(ready)
    finally:
        if child_pid.exists():
            try:
                os.kill(int(child_pid.read_text(encoding="utf-8")), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_quality_gate_timeout_budget_includes_pipe_cleanup(tmp_path: Path) -> None:
    escaped_pid = tmp_path / "escaped-child.pid"
    child_code = (
        "import os, time; from pathlib import Path; "
        f"Path({str(escaped_pid)!r}).write_text(str(os.getpid()), encoding='utf-8'); "
        "time.sleep(8)"
    )
    parent_code = (
        "import os, subprocess, sys, time; "
        f"pid_path={str(escaped_pid)!r}; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "start_new_session=True); "
        "deadline=time.monotonic()+5; "
        'exec("while not os.path.exists(pid_path):\\n'
        '    assert time.monotonic() < deadline\\n    time.sleep(0.01)"); '
        "time.sleep(30)"
    )

    started = time.monotonic()
    result = cev._run_gate(
        tmp_path,
        {
            "argv": [sys.executable, "-c", parent_code],
            "timeout_seconds": 2,
        },
    )
    elapsed = time.monotonic() - started

    try:
        assert result["timed_out"] is True
        assert "retained output pipes" in result.get("cleanup_error", "")
        assert elapsed < 4
    finally:
        if escaped_pid.exists():
            try:
                os.kill(int(escaped_pid.read_text(encoding="utf-8")), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_launch_campaign_creates_ready_goal_task_with_frozen_evidence(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)

    result = ce.launch_campaign(
        prepared,
        board="default",
        priority=4,
        created_by="test-controller",
    )

    assert result.created is True
    assert result.task_id
    assert result.status == "ready"
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, result.task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "default"
        assert task.priority == 4
        assert task.project_id == prepared.payload["project_id"]
        assert task.workspace_kind == "worktree"
        assert task.workspace_path == str(
            (repo.resolve() / ".worktrees" / result.task_id).resolve(strict=False)
        )
        assert task.branch_name == (f"evolve/{result.task_id}-{prepared.sha256[:12]}")
        assert task.goal_mode is True
        assert task.goal_max_turns == 7
        assert task.max_runtime_seconds == 600
        assert task.max_retries == 1
        assert task.skills == [ce.SKILL_NAME]
        assert prepared.sha256 in (task.body or "")
        assert "kanban_request_review" in (task.body or "")
        assert "Do not commit, push, merge, deploy, or restart" in (task.body or "")
        attachments = kb.list_attachments(conn, result.task_id)
        assert [attachment.filename for attachment in attachments] == [
            ce.CONTRACT_FILENAME,
            ce.VERIFIER_FILENAME,
        ]
        assert Path(attachments[0].stored_path).read_bytes() == prepared.contract_bytes
        assert Path(attachments[1].stored_path).read_bytes() == prepared.verifier_bytes

        comments = kb.list_comments(conn, result.task_id)
        assert len(comments) == 1
        assert prepared.sha256 in comments[0].body
        assert "reviewer" in comments[0].body


@pytest.mark.macos_only
def test_launch_campaign_fails_before_dispatch_without_strict_process_containment(
    tmp_path: Path,
    profile_home: Path,
) -> None:
    prepared = _prepare_contract(_make_repo(tmp_path))

    with pytest.raises(
        ce.CodeEvolutionError,
        match="cannot guarantee verifier descendant cleanup",
    ):
        ce.launch_campaign(prepared, board="default")


def test_frozen_contract_rejects_task_that_lost_its_project_link(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None

    with kb.connect_closing(board="default") as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET project_id = NULL WHERE id = ?",
                (launched.task_id,),
            )
        task = kb.get_task(conn, launched.task_id)
        with pytest.raises(ce.CodeEvolutionError, match="project-linked worktree"):
            ce.load_frozen_task_contract(conn, task)


def test_frozen_contract_rejects_project_primary_path_drift(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    replacement = tmp_path / "replacement-project-root"
    replacement.mkdir()

    with pdb.connect_closing() as project_conn:
        project_conn.execute(
            "UPDATE projects SET primary_path = ? WHERE id = ?",
            (str(replacement), prepared.payload["project_id"]),
        )
        project_conn.commit()

    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        with pytest.raises(ce.CodeEvolutionError, match="anchored"):
            ce.load_frozen_task_contract(conn, task)


@pytest.mark.linux_only
def test_frozen_contract_rejects_symlink_substituted_worktree(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    launched = ce.launch_campaign(_prepare_contract(repo), board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        workspace = kb.resolve_workspace(task, board="default")

    _git(repo, "worktree", "remove", "--force", str(workspace))
    workspace.symlink_to(repo, target_is_directory=True)

    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        with pytest.raises(ce.CodeEvolutionError, match="project-linked worktree"):
            ce.load_frozen_task_contract(conn, task)


@pytest.mark.linux_only
def test_workspace_validator_rejects_symlinked_repository_ancestor(
    tmp_path: Path, profile_home: Path
) -> None:
    bound_parent = tmp_path / "bound-parent"
    bound_parent.mkdir()
    repo = _make_repo(bound_parent)
    launched = ce.launch_campaign(_prepare_contract(repo), board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        payload = ce.load_frozen_task_contract(conn, task)
        assert payload is not None

    moved_parent = tmp_path / "moved-parent"
    bound_parent.rename(moved_parent)
    bound_parent.symlink_to(moved_parent, target_is_directory=True)

    with pytest.raises(ce.CodeEvolutionError, match="project-linked worktree"):
        ce._validate_frozen_workspace_path(payload, task, must_exist=True)


@pytest.mark.linux_only
def test_frozen_verifier_rejects_symlinked_repository_ancestor(
    tmp_path: Path, profile_home: Path
) -> None:
    bound_parent = tmp_path / "verifier-bound-parent"
    bound_parent.mkdir()
    repo = _make_repo(bound_parent)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)

    moved_parent = tmp_path / "verifier-moved-parent"
    bound_parent.rename(moved_parent)
    bound_parent.symlink_to(moved_parent, target_is_directory=True)

    report = _run_frozen_verifier(
        repo,
        contract_path,
        verifier_path,
        prepared.sha256,
        run_gates=False,
    )

    assert report["passed"] is False
    assert [issue["code"] for issue in report["issues"]] == ["unsafe_repository_path"]


def test_launch_campaign_is_idempotent_for_same_frozen_contract(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    kb.init_db()

    first = ce.launch_campaign(prepared, board="default")
    second = ce.launch_campaign(prepared, board="default")

    assert first.task_id == second.task_id
    assert first.created is True
    assert second.created is False
    with kb.connect_closing(board="default") as conn:
        assert len(kb.list_attachments(conn, first.task_id)) == 2
        assert len(kb.list_comments(conn, first.task_id)) == 1


def test_launch_campaign_remains_idempotent_after_verification_evidence(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    first = ce.launch_campaign(prepared, board="default")
    assert first.task_id is not None

    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, first.task_id)
        assert task is not None
        workspace = kb.resolve_workspace(task, board="default")
        (workspace / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")
        assert (
            ce.enforce_frozen_task_verifier(
                conn,
                task,
                phase="implementation",
                board="default",
            )
            is None
        )

    second = ce.launch_campaign(prepared, board="default")

    assert second.created is False
    assert second.task_id == first.task_id


def test_launched_campaign_materializes_clean_worktree_that_passes_preflight(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    kb.init_db()
    result = ce.launch_campaign(prepared, board="default")
    assert result.task_id is not None

    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, result.task_id)
        assert task is not None
        workspace = kb.resolve_workspace(task, board="default")
        attachments = kb.list_attachments(conn, result.task_id)

    assert workspace == repo / ".worktrees" / result.task_id
    assert _git(workspace, "branch", "--show-current") == task.branch_name
    assert _git(workspace, "rev-parse", "HEAD") == prepared.payload["base_commit"]

    verification = subprocess.run(
        [
            sys.executable,
            attachments[1].stored_path,
            "--contract",
            attachments[0].stored_path,
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(workspace),
            "--preflight",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    report = json.loads(verification.stdout)
    assert verification.returncode == 0, report
    assert report["passed"] is True


def test_frozen_verifier_accepts_allowed_diff_and_runs_exact_gates(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    (repo / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(repo),
            "--run-gates",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    report = json.loads(result.stdout)
    assert result.returncode == 0, report
    assert report["passed"] is True
    assert report["contract_sha256"] == prepared.sha256
    assert report["changed_paths"] == ["src/agent.py"]
    assert report["issues"] == []
    assert report["quality_gates"][0]["passed"] is True


def test_frozen_verifier_rejects_out_of_scope_changes(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    (repo / "README.md").write_text("outside scope\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(repo),
            "--run-gates",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["passed"] is False
    assert report["changed_paths"] == ["README.md"]
    assert [issue["code"] for issue in report["issues"]] == ["path_scope_violation"]
    assert report["quality_gates"] == []


@pytest.mark.linux_only
def test_frozen_verifier_preserves_posix_backslashes_in_changed_paths(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    escaped = repo / r"src\escape.py"
    escaped.write_text("outside lexical src tree\n", encoding="utf-8")

    report = _run_frozen_verifier(
        repo,
        contract_path,
        verifier_path,
        prepared.sha256,
        run_gates=True,
    )

    assert not report["passed"]
    assert r"src\escape.py" in report["changed_paths"]
    assert any(issue["code"] == "path_scope_violation" for issue in report["issues"])


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_frozen_verifier_rejects_index_flags_that_hide_tracked_changes(
    tmp_path: Path,
    profile_home: Path,
    index_flag: str,
) -> None:
    repo = _make_repo(tmp_path)
    hidden = repo / "outside.py"
    hidden.write_text("original\n", encoding="utf-8")
    _git(repo, "add", "outside.py")
    _git(repo, "commit", "-m", "add outside file")
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)

    _git(repo, "update-index", index_flag, "outside.py")
    hidden.write_text("hidden mutation\n", encoding="utf-8")
    (repo / "src" / "agent.py").write_text("allowed mutation\n", encoding="utf-8")

    report = _run_frozen_verifier(
        repo,
        contract_path,
        verifier_path,
        prepared.sha256,
        run_gates=True,
    )

    assert not report["passed"]
    assert any(issue["code"] == "unsafe_index_flag" for issue in report["issues"])


def test_frozen_verifier_rejects_ignored_out_of_scope_changes(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    (repo / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore generated files")
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    (repo / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "ignored").mkdir()
    (repo / "ignored" / "outside.txt").write_text("bypass\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(repo),
            "--run-gates",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert "ignored/outside.txt" in report["changed_paths"]
    assert report["issues"][0]["code"] == "path_scope_violation"
    assert report["quality_gates"] == []


@pytest.mark.linux_only
def test_frozen_verifier_rejects_changed_symlink_entries(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    outside = tmp_path / "outside.txt"
    outside.write_text("external\n", encoding="utf-8")
    (repo / "src" / "escape-link").symlink_to(outside)

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(repo),
            "--run-gates",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["passed"] is False
    assert report["issues"][0]["code"] == "unsafe_changed_entry"
    assert "src/escape-link" in report["issues"][0]["message"]


@pytest.mark.linux_only
def test_frozen_verifier_rejects_changed_hardlink_entries(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    outside = tmp_path / "outside.txt"
    outside.write_text("external\n", encoding="utf-8")
    os.link(outside, repo / "src" / "escape-hardlink")

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(repo),
            "--run-gates",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["issues"][0]["code"] == "unsafe_changed_entry"
    assert "src/escape-hardlink" in report["issues"][0]["message"]


@pytest.mark.macos_only
def test_frozen_verifier_rejects_changed_symlink_entries_on_macos(
    tmp_path: Path, profile_home: Path
) -> None:
    test_frozen_verifier_rejects_changed_symlink_entries(tmp_path, profile_home)


@pytest.mark.macos_only
def test_frozen_verifier_rejects_changed_hardlink_entries_on_macos(
    tmp_path: Path, profile_home: Path
) -> None:
    test_frozen_verifier_rejects_changed_hardlink_entries(tmp_path, profile_home)


def _assert_special_entry_rejected(
    repo: Path,
    contract_path: Path,
    verifier_path: Path,
    contract_sha256: str,
    relative_path: str,
) -> None:
    report = _run_frozen_verifier(
        repo,
        contract_path,
        verifier_path,
        contract_sha256,
        run_gates=True,
    )
    assert not report["passed"]
    issue = next(
        item for item in report["issues"] if item["code"] == "unsafe_changed_entry"
    )
    assert relative_path in issue["message"]


def _exercise_fifo_rejection(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    fifo_path = repo / "src" / "unsafe-fifo"
    os.mkfifo(fifo_path)
    _assert_special_entry_rejected(
        repo,
        contract_path,
        verifier_path,
        prepared.sha256,
        "src/unsafe-fifo",
    )


@pytest.mark.linux_only
def test_frozen_verifier_rejects_changed_fifo_entries_on_linux(
    tmp_path: Path, profile_home: Path
) -> None:
    _exercise_fifo_rejection(tmp_path)


@pytest.mark.macos_only
def test_frozen_verifier_rejects_changed_fifo_entries_on_macos(
    tmp_path: Path, profile_home: Path
) -> None:
    _exercise_fifo_rejection(tmp_path)


def _exercise_socket_rejection(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    socket_path = repo / "src" / "unsafe.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(socket_path))
        _assert_special_entry_rejected(
            repo,
            contract_path,
            verifier_path,
            prepared.sha256,
            "src/unsafe.sock",
        )


@pytest.mark.linux_only
def test_frozen_verifier_rejects_changed_socket_entries_on_linux(
    tmp_path: Path, profile_home: Path
) -> None:
    _exercise_socket_rejection(tmp_path)


@pytest.mark.macos_only
def test_frozen_verifier_rejects_changed_socket_entries_on_macos(
    tmp_path: Path, profile_home: Path
) -> None:
    _exercise_socket_rejection(tmp_path)


@pytest.mark.windows_only
def test_frozen_verifier_rejects_changed_junction_entries_on_windows(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    (outside / "external.txt").write_text("external\n", encoding="utf-8")
    junction = repo / "src" / "escape-junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert created.returncode == 0, created.stderr
    try:
        _assert_special_entry_rejected(
            repo,
            contract_path,
            verifier_path,
            prepared.sha256,
            "src/escape-junction",
        )
    finally:
        os.rmdir(junction)


def test_frozen_verifier_rejects_candidate_mutation_during_quality_gate(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = ce.prepare_contract(
        repository=repo,
        project=_project_for_repo(repo),
        objective="Make retries deterministic",
        evidence="tests/test_agent.py reproduces nondeterministic retry ordering",
        success_metric="The quality gate passes without mutating the candidate",
        allowed_paths=("src",),
        quality_gates=(
            (
                (
                    sys.executable,
                    "-c",
                    "from pathlib import Path; "
                    "Path('src/agent.py').write_text('VALUE = 3\\n')",
                ),
                30,
            ),
        ),
        assignee="default",
        reviewer="reviewer",
    )
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    (repo / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(repo),
            "--run-gates",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["passed"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "candidate_changed_during_gates"
    }


@pytest.mark.linux_only
def test_frozen_verifier_rechecks_unsafe_entries_after_quality_gates(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    outside = tmp_path / "outside-agent.py"
    outside.write_text("VALUE = 2\n", encoding="utf-8")
    prepared = ce.prepare_contract(
        repository=repo,
        project=_project_for_repo(repo),
        objective="Keep the evaluated candidate isolated",
        evidence="the tracked candidate begins as an ordinary file",
        success_metric="The gate cannot replace candidate bytes with a hardlink",
        allowed_paths=("src",),
        quality_gates=(
            (
                (
                    sys.executable,
                    "-c",
                    "import os; os.unlink('src/agent.py'); "
                    f"os.link({str(outside)!r}, 'src/agent.py')",
                ),
                30,
            ),
        ),
        assignee="default",
        reviewer="reviewer",
    )
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    (repo / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(repo),
            "--run-gates",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert "unsafe_changed_entry" in {issue["code"] for issue in report["issues"]}


def test_frozen_verifier_does_not_treat_an_allowed_file_as_a_directory(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = ce.prepare_contract(
        repository=repo,
        project=_project_for_repo(repo),
        objective="Change one implementation file",
        evidence="the frozen gate reproduces the bug",
        success_metric="The exact frozen file remains the only changed path",
        allowed_paths=("src/agent.py",),
        quality_gates=((f'"{sys.executable}" -c "raise SystemExit(0)"', 30),),
        assignee="default",
        reviewer="reviewer",
    )
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    (repo / "src" / "agent.py").unlink()
    (repo / "src" / "agent.py").mkdir()
    (repo / "src" / "agent.py" / "escape.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(repo),
            "--run-gates",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert "src/agent.py/escape.py" in report["changed_paths"]
    assert "path_scope_violation" in {issue["code"] for issue in report["issues"]}


def test_frozen_verifier_fails_closed_when_quality_gate_fails(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = ce.prepare_contract(
        repository=repo,
        project=_project_for_repo(repo),
        objective="Make retries deterministic",
        evidence="tests/test_agent.py reproduces the failure",
        success_metric="The frozen failure gate exits zero",
        allowed_paths=("src",),
        quality_gates=(([sys.executable, "-c", "raise SystemExit(3)"], 30),),
        assignee="default",
        reviewer="reviewer",
    )
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    (repo / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(repo),
            "--run-gates",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert [issue["code"] for issue in report["issues"]] == ["quality_gate_failed"]
    assert report["quality_gates"][0]["passed"] is False


def test_frozen_verifier_reports_git_diff_check_failure_before_gates(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    (repo / "src" / "agent.py").write_text("VALUE = 2  \n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(repo),
            "--run-gates",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert [issue["code"] for issue in report["issues"]] == ["git_diff_check_failed"]
    assert report["quality_gates"] == []


def test_frozen_verifier_rejects_tampered_verifier(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    verifier_path.write_bytes(prepared.verifier_bytes + b"\n# tampered\n")

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(repo),
            "--preflight",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["issues"][0]["code"] == "verification_error"
    assert "verifier SHA-256" in report["issues"][0]["message"]


def test_frozen_verifier_rejects_self_consistent_contract_rewrite(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    envelope = json.loads(contract_path.read_text(encoding="utf-8"))
    envelope["contract"]["allowed_paths"] = ["README.md"]
    canonical = json.dumps(
        envelope["contract"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    envelope["sha256"] = hashlib.sha256(canonical).hexdigest()
    contract_path.write_text(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(repo),
            "--preflight",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["issues"][0]["code"] == "verification_error"
    assert "expected contract SHA-256" in report["issues"][0]["message"]


def test_frozen_verifier_rejects_contract_without_success_metric(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    envelope = json.loads(contract_path.read_text(encoding="utf-8"))
    envelope["contract"].pop("success_metric")
    canonical = json.dumps(
        envelope["contract"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    rewritten_digest = hashlib.sha256(canonical).hexdigest()
    envelope["sha256"] = rewritten_digest
    contract_path.write_text(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            rewritten_digest,
            "--repo",
            str(repo),
            "--preflight",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["issues"][0]["code"] == "verification_error"
    assert "success metric" in report["issues"][0]["message"]


def test_frozen_verifier_rejects_base_commit_drift(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    (repo / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "src/agent.py")
    _git(repo, "commit", "-m", "drift")

    result = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            prepared.sha256,
            "--repo",
            str(repo),
            "--preflight",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert [issue["code"] for issue in report["issues"]] == ["base_identity_mismatch"]


@pytest.mark.linux_only
def test_frozen_verifier_rejects_primary_checkout_substituted_for_worktree(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    contract_path, verifier_path = _write_frozen_files(tmp_path, prepared)
    (repo / "src" / "agent.py").write_text("candidate\n", encoding="utf-8")
    substituted = repo / ".worktrees" / "fake-task"
    substituted.parent.mkdir()
    substituted.symlink_to(repo, target_is_directory=True)

    report = _run_frozen_verifier(
        substituted,
        contract_path,
        verifier_path,
        prepared.sha256,
        run_gates=True,
        expected_workspace=substituted,
        expected_branch=f"evolve/fake-task-{prepared.sha256[:12]}",
    )

    assert not report["passed"]
    assert [issue["code"] for issue in report["issues"]] == ["unsafe_repository_path"]


def test_prepare_contract_rejects_dirty_repository(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    (repo / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ce.CodeEvolutionError, match="must be clean"):
        _prepare_contract(repo)


def test_prepare_contract_rejects_linked_worktree_as_source_anchor(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "-b", "linked-source", str(linked))

    with pytest.raises(ce.CodeEvolutionError, match="primary Git checkout"):
        _prepare_contract(linked)


def test_prepare_contract_requires_independent_existing_reviewer(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    project = _project_for_repo(repo)

    with pytest.raises(ce.CodeEvolutionError, match="must be different"):
        ce.prepare_contract(
            repository=repo,
            project=project,
            objective="x",
            evidence="y",
            success_metric="z",
            allowed_paths=("src",),
            quality_gates=((sys.executable, 30),),
            assignee="default",
            reviewer="default",
        )

    with pytest.raises(ce.CodeEvolutionError, match="unknown reviewer profile"):
        ce.prepare_contract(
            repository=repo,
            project=project,
            objective="x",
            evidence="y",
            success_metric="z",
            allowed_paths=("src",),
            quality_gates=((sys.executable, 30),),
            assignee="default",
            reviewer="missing-profile",
        )


def test_prepare_contract_rejects_project_for_another_repository(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    other = tmp_path / "other-repository"
    other.mkdir()
    with pdb.connect_closing() as conn:
        wrong_project = pdb.create_project(
            conn,
            name="Wrong repository",
            folders=[str(other)],
            primary_path=str(other),
        )

    with pytest.raises(ce.CodeEvolutionError, match="is anchored to"):
        ce.prepare_contract(
            repository=repo,
            project=wrong_project,
            objective="x",
            evidence="y",
            success_metric="z",
            allowed_paths=("src",),
            quality_gates=((sys.executable, 30),),
            assignee="default",
            reviewer="reviewer",
        )


@pytest.mark.parametrize(
    "unsafe_path", [".", "../outside", ".git/config", "C:\\outside"]
)
def test_prepare_contract_rejects_unbounded_or_unsafe_paths(
    tmp_path: Path,
    profile_home: Path,
    unsafe_path: str,
) -> None:
    repo = _make_repo(tmp_path)

    with pytest.raises(ce.CodeEvolutionError):
        ce.prepare_contract(
            repository=repo,
            project=_project_for_repo(repo),
            objective="x",
            evidence="y",
            success_metric="z",
            allowed_paths=(unsafe_path,),
            quality_gates=((sys.executable, 30),),
            assignee="default",
            reviewer="reviewer",
        )


def test_launch_campaign_removes_partial_task_when_freezing_fails(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    kb.init_db()
    original_store = kb.store_attachment_bytes
    calls = 0

    def fail_second_attachment(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated attachment failure")
        return original_store(*args, **kwargs)

    monkeypatch.setattr(kb, "store_attachment_bytes", fail_second_attachment)

    with pytest.raises(OSError, match="simulated attachment failure"):
        ce.launch_campaign(prepared, board="default")

    with kb.connect_closing(board="default") as conn:
        assert kb.list_tasks(conn, limit=100) == []
    assert list(kb.attachments_root(board="default").glob("*")) == []


def test_launch_campaign_rejects_mutated_prepared_payload(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    prepared.payload["reviewer"] = "default"

    with pytest.raises(ce.CodeEvolutionError, match="does not match frozen bytes"):
        ce.launch_campaign(prepared, board="default")


def test_code_evolution_implementer_cannot_complete_before_review(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as kt

    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        claimed = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert claimed is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", launched.task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "default")

    response = json.loads(kt._handle_complete({"summary": "implemented and verified"}))

    assert "error" in response
    assert "must request review" in response["error"]
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        assert task.status == "running"


def test_code_evolution_database_rejects_implementation_completion(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        claimed = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert claimed is not None

        assert not kb.complete_task(
            conn,
            launched.task_id,
            summary="self-approved",
            expected_run_id=claimed.current_run_id,
        )
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        assert task.status == "running"


def test_code_evolution_cli_rejects_implementation_completion(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        claimed = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", launched.task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "default")

    rc = kc._cmd_complete(
        argparse.Namespace(
            task_ids=[launched.task_id],
            result=None,
            summary="self-approved",
            metadata=None,
        )
    )

    assert rc == 1
    assert "must request review" in capsys.readouterr().err
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        assert task.status == "running"


def test_code_evolution_cli_request_review_rejects_reviewer_substitution(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        implementation = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert implementation is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", launched.task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(implementation.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "default")

    rc = kc._cmd_request_review(
        argparse.Namespace(
            task_id=launched.task_id,
            summary="implementation verified",
            metadata=None,
            reviewer="default",
            force=False,
        )
    )

    assert rc == 1
    assert "frozen reviewer 'reviewer'" in capsys.readouterr().err
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        assert task.status == "running"


def test_code_evolution_cli_request_review_runs_frozen_verifier(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        implementation = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert implementation is not None
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        kb.resolve_workspace(task, board="default")
    monkeypatch.setenv("HERMES_KANBAN_TASK", launched.task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(implementation.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "default")

    rc = kc._cmd_request_review(
        argparse.Namespace(
            task_id=launched.task_id,
            summary="implementation verified",
            metadata=None,
            reviewer=None,
            force=False,
        )
    )

    assert rc == 1
    assert "no_candidate_changes" in capsys.readouterr().err
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        assert task.status == "running"
        reports = [
            item
            for item in kb.list_attachments(conn, launched.task_id)
            if item.filename.startswith("code-evolution-implementation-verification-")
        ]
        assert len(reports) == 1


def test_code_evolution_cli_frozen_reviewer_can_approve_verified_candidate(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        implementation = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert implementation is not None
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        workspace = kb.resolve_workspace(task, board="default")
    (workspace / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_TASK", launched.task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(implementation.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "default")

    assert (
        kc._cmd_request_review(
            argparse.Namespace(
                task_id=launched.task_id,
                summary="implementation verified",
                metadata=None,
                reviewer=None,
                force=False,
            )
        )
        == 0
    )

    with kb.connect_closing(board="default") as conn:
        review = kb.claim_review_task(conn, launched.task_id, claimer="reviewer:1")
        assert review is not None
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "reviewer")

    assert (
        kc._cmd_complete(
            argparse.Namespace(
                task_ids=[launched.task_id],
                result=None,
                summary="approved",
                metadata=None,
            )
        )
        == 0
    )

    output = capsys.readouterr()
    assert "Requested review" in output.out
    assert "Completed" in output.out
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        assert task.status == "done"
        reports = [
            item
            for item in kb.list_attachments(conn, launched.task_id)
            if item.filename.startswith("code-evolution-")
            and "-verification-run-" in item.filename
        ]
        assert len(reports) == 2


def test_code_evolution_cli_verification_report_redacts_gate_output(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo(tmp_path)
    prepared = ce.prepare_contract(
        repository=repo,
        project=_project_for_repo(repo),
        objective="Make retries deterministic",
        evidence="tests/test_agent.py reproduces nondeterministic retry ordering",
        success_metric="The frozen regression gate exits zero on the final candidate",
        allowed_paths=("src",),
        quality_gates=(
            (
                (
                    sys.executable,
                    "-c",
                    "import os; print(os.environ['CODE_EVOLUTION_TEST_SECRET'])",
                ),
                30,
            ),
        ),
        assignee="default",
        reviewer="reviewer",
    )
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        implementation = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert implementation is not None
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        workspace = kb.resolve_workspace(task, board="default")
    (workspace / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")
    secret = "sk-test-abcdefghijklmnopqrstuvwxyz123456"
    monkeypatch.setenv("CODE_EVOLUTION_TEST_SECRET", secret)
    monkeypatch.setenv("HERMES_KANBAN_TASK", launched.task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(implementation.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "default")

    assert (
        kc._cmd_request_review(
            argparse.Namespace(
                task_id=launched.task_id,
                summary="implementation verified",
                metadata=None,
                reviewer=None,
                force=False,
            )
        )
        == 0
    )

    with kb.connect_closing(board="default") as conn:
        reports = [
            item
            for item in kb.list_attachments(conn, launched.task_id)
            if item.filename.startswith("code-evolution-implementation-verification-")
        ]
        assert len(reports) == 1
        retained = Path(reports[0].stored_path).read_text(encoding="utf-8")
        assert secret not in retained
        retained_report = json.loads(retained)
        assert "..." in retained_report["quality_gates"][0]["stdout"]


def test_code_evolution_database_rejects_reviewer_substitution(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        claimed = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert claimed is not None

        outcome = kb.request_review(
            conn,
            launched.task_id,
            reviewer="default",
            expected_run_id=claimed.current_run_id,
            with_reason=True,
        )
        assert isinstance(outcome, tuple)
        ok, reason = outcome
        assert not ok
        assert reason is not None
        assert "frozen reviewer 'reviewer'" in reason
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        assert task.status == "running"


def test_code_evolution_database_request_review_runs_frozen_verifier(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None

    with kb.connect_closing(board="default") as conn:
        claimed = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert claimed is not None
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        kb.resolve_workspace(task, board="default")

        forced = kb.request_review(
            conn,
            launched.task_id,
            reviewer="reviewer",
            force=True,
            with_reason=True,
            board="default",
        )
        assert forced == (
            False,
            "code-evolution review handoff requires ownership of the active "
            "implementation run",
        )

        outcome = kb.request_review(
            conn,
            launched.task_id,
            reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
            with_reason=True,
            board="default",
        )

        assert isinstance(outcome, tuple)
        ok, reason = outcome
        assert not ok
        assert reason is not None
        assert "no_candidate_changes" in reason
        landed = kb.get_task(conn, launched.task_id)
        assert landed is not None
        assert landed.status == "running"
        reports = [
            item
            for item in kb.list_attachments(conn, launched.task_id)
            if item.filename.startswith("code-evolution-implementation-verification-")
        ]
        assert len(reports) == 1


def test_code_evolution_database_completion_reverifies_and_requires_run_owner(
    tmp_path: Path, profile_home: Path
) -> None:
    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None

    with kb.connect_closing(board="default") as conn:
        implementation = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert implementation is not None
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        workspace = kb.resolve_workspace(task, board="default")
        (workspace / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")
        assert kb.request_review(
            conn,
            launched.task_id,
            reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
            board="default",
        )
        review = kb.claim_review_task(conn, launched.task_id, claimer="reviewer:1")
        assert review is not None

        ownership_errors: list[str] = []
        assert not kb.complete_task(
            conn,
            launched.task_id,
            summary="approved without run ownership",
            board="default",
            failure_reasons=ownership_errors,
        )
        assert ownership_errors == [
            "code-evolution completion requires ownership of the active review run"
        ]

        (workspace / "README.md").write_text(
            "late out-of-scope change\n", encoding="utf-8"
        )
        verifier_errors: list[str] = []
        assert not kb.complete_task(
            conn,
            launched.task_id,
            summary="approved after unsafe mutation",
            expected_run_id=review.current_run_id,
            board="default",
            failure_reasons=verifier_errors,
        )
        assert verifier_errors and "path_scope_violation" in verifier_errors[0]
        landed = kb.get_task(conn, launched.task_id)
        assert landed is not None
        assert landed.status == "running"
        reports = [
            item
            for item in kb.list_attachments(conn, launched.task_id)
            if item.filename.startswith("code-evolution-review-verification-")
        ]
        assert len(reports) == 1


def test_code_evolution_handoff_rejects_reviewer_override(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as kt

    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        claimed = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert claimed is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", launched.task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "default")

    response = json.loads(
        kt._handle_request_review({
            "summary": "implementation verified",
            "reviewer": "default",
        })
    )

    assert "error" in response
    assert "frozen reviewer 'reviewer'" in response["error"]
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        assert task.status == "running"
        assert task.assignee == "default"


def test_code_evolution_handoff_requires_passing_frozen_verifier(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as kt

    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        claimed = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert claimed is not None
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        kb.resolve_workspace(task, board="default")

    monkeypatch.setenv("HERMES_KANBAN_TASK", launched.task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "default")

    response = json.loads(
        kt._handle_request_review({"summary": "implementation verified"})
    )

    assert "error" in response
    assert "frozen verifier did not pass" in response["error"]
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        assert task.status == "running"
        reports = [
            item
            for item in kb.list_attachments(conn, launched.task_id)
            if item.filename.startswith("code-evolution-implementation-verification-")
        ]
        assert len(reports) == 1
        report = json.loads(Path(reports[0].stored_path).read_text(encoding="utf-8"))
        assert report["passed"] is False
        assert report["issues"][0]["code"] == "no_candidate_changes"


def test_code_evolution_handoff_uses_frozen_reviewer_when_omitted(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as kt

    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        claimed = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert claimed is not None
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        workspace = kb.resolve_workspace(task, board="default")
    (workspace / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_KANBAN_TASK", launched.task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "default")

    response = json.loads(
        kt._handle_request_review({"summary": "implementation verified"})
    )

    assert response["ok"] is True
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        assert task.status == "review"
        assert task.assignee == "reviewer"
        reports = [
            item
            for item in kb.list_attachments(conn, launched.task_id)
            if item.filename.startswith("code-evolution-implementation-verification-")
        ]
        assert len(reports) == 1
        report = json.loads(Path(reports[0].stored_path).read_text(encoding="utf-8"))
        assert report["passed"] is True


def test_code_evolution_completion_rejects_nonfrozen_review_profile(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as kt

    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        implementation = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert implementation is not None
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        workspace = kb.resolve_workspace(task, board="default")
        (workspace / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")
        assert kb.request_review(
            conn,
            launched.task_id,
            summary="implementation verified",
            reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
        )
        assert kb.assign_task(conn, launched.task_id, "default")
        review = kb.claim_review_task(conn, launched.task_id, claimer="default:2")
        assert review is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", launched.task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "default")

    response = json.loads(kt._handle_complete({"summary": "approved"}))

    assert "error" in response
    assert "frozen reviewer 'reviewer'" in response["error"]
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        assert task.status == "running"


def test_code_evolution_frozen_reviewer_can_complete_review(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as kt

    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        implementation = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert implementation is not None
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        workspace = kb.resolve_workspace(task, board="default")
    (workspace / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_KANBAN_TASK", launched.task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(implementation.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    handoff = json.loads(
        kt._handle_request_review({"summary": "implementation verified"})
    )
    assert handoff["ok"] is True

    with kb.connect_closing(board="default") as conn:
        review = kb.claim_review_task(conn, launched.task_id, claimer="reviewer:1")
        assert review is not None
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "reviewer")

    response = json.loads(kt._handle_complete({"summary": "approved"}))

    assert response["ok"] is True
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        assert task.status == "done"
        reports = [
            item
            for item in kb.list_attachments(conn, launched.task_id)
            if item.filename.startswith("code-evolution-")
            and "-verification-run-" in item.filename
        ]
        assert len(reports) == 2
        phases = {
            json.loads(Path(item.stored_path).read_text(encoding="utf-8"))[
                "enforcement_phase"
            ]
            for item in reports
        }
        assert phases == {"implementation", "review"}


def test_code_evolution_review_rechecks_candidate_after_handoff(
    tmp_path: Path,
    profile_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as kt

    repo = _make_repo(tmp_path)
    prepared = _prepare_contract(repo)
    launched = ce.launch_campaign(prepared, board="default")
    assert launched.task_id is not None
    with kb.connect_closing(board="default") as conn:
        implementation = kb.claim_task(conn, launched.task_id, claimer="default:1")
        assert implementation is not None
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        workspace = kb.resolve_workspace(task, board="default")
    (workspace / "src" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_KANBAN_TASK", launched.task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(implementation.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    handoff = json.loads(
        kt._handle_request_review({"summary": "implementation verified"})
    )
    assert handoff["ok"] is True

    (workspace / "README.md").write_text("late out-of-scope change\n", encoding="utf-8")
    with kb.connect_closing(board="default") as conn:
        review = kb.claim_review_task(conn, launched.task_id, claimer="reviewer:1")
        assert review is not None
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "reviewer")

    response = json.loads(kt._handle_complete({"summary": "approved"}))

    assert "error" in response
    assert "path_scope_violation" in response["error"]
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, launched.task_id)
        assert task is not None
        assert task.status == "running"
        reports = [
            item
            for item in kb.list_attachments(conn, launched.task_id)
            if item.filename.startswith("code-evolution-review-verification-")
        ]
        assert len(reports) == 1
        report = json.loads(Path(reports[0].stored_path).read_text(encoding="utf-8"))
        assert report["passed"] is False
        assert report["issues"][0]["code"] == "path_scope_violation"


def test_kanban_improve_dry_run_prints_frozen_plan_without_creating_task(
    tmp_path: Path,
    profile_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _make_repo(tmp_path)
    project = _project_for_repo(repo)
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    subparsers = parser.add_subparsers(dest="command")
    kc.build_parser(subparsers)
    args = parser.parse_args([
        "kanban",
        "improve",
        "Make retries deterministic",
        "--evidence",
        "tests/test_agent.py reproduces the failure",
        "--success-metric",
        "The frozen regression gate exits zero",
        "--repo",
        str(repo),
        "--project",
        project,
        "--assignee",
        "default",
        "--reviewer",
        "reviewer",
        "--allow",
        "src",
        "--allow",
        "tests/test_agent.py",
        "--gate",
        f'"{sys.executable}" -c "import sys; sys.exit(0)"',
        "--gate-timeout",
        "30",
        "--goal-max-turns",
        "7",
        "--max-runtime",
        "10m",
        "--dry-run",
        "--json",
    ])

    assert kc.kanban_command(args) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert output["task_id"] is None
    assert output["contract_id"].startswith("ce_")
    assert output["repository"] == str(repo.resolve())
    assert output["assignee"] == "default"
    assert output["reviewer"] == "reviewer"
    assert output["success_metric"] == "The frozen regression gate exits zero"
    assert output["budgets"]["max_runtime_seconds"] == 600
    with kb.connect_closing(board="default") as conn:
        assert kb.list_tasks(conn, limit=100) == []
