from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from plugins.mobile_bug_agent.config import RepoConfig
from plugins.mobile_bug_agent.repo_manager import RepoManager, RepoManagerError


def test_repo_manager_clones_missing_repo_and_creates_named_worktree(tmp_path):
    commands: list[list[str]] = []

    def run(cmd: list[str], cwd: Path | None = None) -> str:
        commands.append(cmd if cwd is None else ["cwd=" + str(cwd), *cmd])
        if cmd[:2] == ["git", "clone"]:
            Path(cmd[-1]).mkdir(parents=True)
        return ""

    manager = RepoManager(
        config=RepoConfig(
            url="git@github.com:acme/mobile-app.git",
            local_name="mobile-app",
            default_branch="main",
            branch_prefix="monica",
        ),
        workspace_root=tmp_path,
        run_command=run,
    )

    worktree = manager.prepare_worktree(
        linear_identifier="MOB-123",
        summary="Android checkout crash after promo code",
    )

    assert worktree.branch_name == "monica/MOB-123-android-checkout-crash-after-promo-code"
    assert worktree.path == tmp_path / "worktrees" / "monica-MOB-123-android-checkout-crash-after-promo-code"
    assert commands == [
        [
            "git",
            "clone",
            "git@github.com:acme/mobile-app.git",
            str(tmp_path / "repos" / "mobile-app"),
        ],
        [
            "git",
            "-C",
            str(tmp_path / "repos" / "mobile-app"),
            "worktree",
            "add",
            "-B",
            "monica/MOB-123-android-checkout-crash-after-promo-code",
            str(tmp_path / "worktrees" / "monica-MOB-123-android-checkout-crash-after-promo-code"),
            "origin/main",
        ],
    ]


def test_repo_manager_fetches_existing_repo_before_worktree(tmp_path):
    commands: list[list[str]] = []
    (tmp_path / "repos" / "mobile-app").mkdir(parents=True)

    def run(cmd: list[str], cwd: Path | None = None) -> str:
        commands.append(cmd)
        return ""

    manager = RepoManager(
        config=RepoConfig(url="git@github.com:acme/mobile-app.git", local_name="mobile-app"),
        workspace_root=tmp_path,
        run_command=run,
    )

    manager.prepare_worktree(linear_identifier="MOB-123", summary="Checkout crash")

    assert commands[0] == [
        "git",
        "-C",
        str(tmp_path / "repos" / "mobile-app"),
        "fetch",
        "origin",
        "main",
    ]


def test_repo_manager_rejects_existing_repo_file(tmp_path):
    commands: list[list[str]] = []
    repo_path = tmp_path / "repos" / "mobile-app"
    repo_path.parent.mkdir(parents=True)
    repo_path.write_text("not a git repo")

    manager = RepoManager(
        config=RepoConfig(url="git@github.com:acme/mobile-app.git", local_name="mobile-app"),
        workspace_root=tmp_path,
        run_command=lambda cmd, cwd=None: commands.append(cmd) or "",
    )

    with pytest.raises(RepoManagerError, match="repo path exists but is not a directory"):
        manager.prepare_worktree(linear_identifier="MOB-123", summary="Checkout crash")

    assert commands == []


def test_repo_manager_rejects_unsafe_local_name_before_commands(tmp_path):
    for local_name in ("../outside-runtime", "/tmp/outside-runtime", "nested/mobile-app", ".", "..", ""):
        commands: list[list[str]] = []
        manager = RepoManager(
            config=RepoConfig(url="git@github.com:acme/mobile-app.git", local_name=local_name),
            workspace_root=tmp_path,
            run_command=lambda cmd, cwd=None: commands.append(cmd) or "",
        )

        with pytest.raises(RepoManagerError, match="repo.local_name must be a simple directory name"):
            manager.prepare_worktree(linear_identifier="MOB-123", summary="Checkout crash")

        assert commands == []


def test_repo_manager_rejects_chandler_local_name_before_commands(tmp_path):
    commands: list[list[str]] = []
    manager = RepoManager(
        config=RepoConfig(url="git@github.com:acme/mobile-app.git", local_name="chandler"),
        workspace_root=tmp_path,
        run_command=lambda cmd, cwd=None: commands.append(cmd) or "",
    )

    with pytest.raises(RepoManagerError, match="repo.local_name must not point at a Chandler directory"):
        manager.prepare_worktree(linear_identifier="MOB-123", summary="Checkout crash")

    assert commands == []


def test_repo_manager_rejects_unsafe_branch_prefix_before_commands(tmp_path):
    unsafe_prefixes = (
        "",
        "../monica",
        "/monica",
        "monica branch",
        "monica..bad",
        "monica.lock",
        "monica~bad",
        "monica:bad",
        "monica^bad",
        "monica?bad",
        "monica*bad",
        "monica[bad",
        "monica\\bad",
    )
    for branch_prefix in unsafe_prefixes:
        commands: list[list[str]] = []
        manager = RepoManager(
            config=RepoConfig(
                url="git@github.com:acme/mobile-app.git",
                local_name="mobile-app",
                branch_prefix=branch_prefix,
            ),
            workspace_root=tmp_path,
            run_command=lambda cmd, cwd=None: commands.append(cmd) or "",
        )

        with pytest.raises(RepoManagerError, match="repo.branch_prefix must be a safe git branch prefix"):
            manager.prepare_worktree(linear_identifier="MOB-123", summary="Checkout crash")

        assert commands == []


def test_repo_manager_rejects_chandler_branch_prefix_before_commands(tmp_path):
    commands: list[list[str]] = []
    manager = RepoManager(
        config=RepoConfig(
            url="git@github.com:acme/mobile-app.git",
            local_name="mobile-app",
            branch_prefix="chandler",
        ),
        workspace_root=tmp_path,
        run_command=lambda cmd, cwd=None: commands.append(cmd) or "",
    )

    with pytest.raises(RepoManagerError, match="repo.branch_prefix must not point at Chandler"):
        manager.prepare_worktree(linear_identifier="MOB-123", summary="Checkout crash")

    assert commands == []


def test_repo_manager_rejects_unsafe_default_branch_before_commands(tmp_path):
    unsafe_branches = (
        "",
        "../main",
        "/main",
        "main branch",
        "main..bad",
        "main.lock",
        "main~bad",
        "main:bad",
        "main^bad",
        "main?bad",
        "main*bad",
        "main[bad",
        "main\\bad",
    )
    for default_branch in unsafe_branches:
        commands: list[list[str]] = []
        manager = RepoManager(
            config=RepoConfig(
                url="git@github.com:acme/mobile-app.git",
                local_name="mobile-app",
                default_branch=default_branch,
            ),
            workspace_root=tmp_path,
            run_command=lambda cmd, cwd=None: commands.append(cmd) or "",
        )

        with pytest.raises(RepoManagerError, match="repo.default_branch must be a safe git branch name"):
            manager.prepare_worktree(linear_identifier="MOB-123", summary="Checkout crash")

        assert commands == []


def test_repo_manager_reuses_existing_worktree_for_retry(tmp_path):
    commands: list[list[str]] = []
    (tmp_path / "repos" / "mobile-app").mkdir(parents=True)
    existing = tmp_path / "worktrees" / "monica-MOB-123-checkout-crash"
    existing.mkdir(parents=True)
    (existing / ".git").write_text("gitdir: /tmp/fake-worktree-git-dir")

    def run(cmd: list[str], cwd: Path | None = None) -> str:
        commands.append(cmd)
        if cmd == ["git", "-C", str(existing), "branch", "--show-current"]:
            return "monica/MOB-123-checkout-crash\n"
        return ""

    manager = RepoManager(
        config=RepoConfig(url="git@github.com:acme/mobile-app.git", local_name="mobile-app"),
        workspace_root=tmp_path,
        run_command=run,
    )

    worktree = manager.prepare_worktree(linear_identifier="MOB-123", summary="Checkout crash")

    assert worktree.branch_name == "monica/MOB-123-checkout-crash"
    assert worktree.path == existing
    assert not any(cmd[:4] == ["git", "-C", str(tmp_path / "repos" / "mobile-app"), "worktree"] for cmd in commands)
    assert commands[-2:] == [
        ["git", "-C", str(existing), "branch", "--show-current"],
        ["git", "-C", str(existing), "status", "--porcelain"],
    ]


def test_repo_manager_rejects_dirty_existing_worktree_for_retry(tmp_path):
    commands: list[list[str]] = []
    (tmp_path / "repos" / "mobile-app").mkdir(parents=True)
    existing = tmp_path / "worktrees" / "monica-MOB-123-checkout-crash"
    existing.mkdir(parents=True)
    (existing / ".git").write_text("gitdir: /tmp/fake-worktree-git-dir")

    def run(cmd: list[str], cwd: Path | None = None) -> str:
        commands.append(cmd)
        if cmd == ["git", "-C", str(existing), "branch", "--show-current"]:
            return "monica/MOB-123-checkout-crash\n"
        if cmd == ["git", "-C", str(existing), "status", "--porcelain"]:
            return " M android/app/build.gradle\n"
        return ""

    manager = RepoManager(
        config=RepoConfig(url="git@github.com:acme/mobile-app.git", local_name="mobile-app"),
        workspace_root=tmp_path,
        run_command=run,
    )

    with pytest.raises(RepoManagerError, match="worktree has uncommitted changes"):
        manager.prepare_worktree(linear_identifier="MOB-123", summary="Checkout crash")

    assert commands[-2:] == [
        ["git", "-C", str(existing), "branch", "--show-current"],
        ["git", "-C", str(existing), "status", "--porcelain"],
    ]


def test_repo_manager_rejects_existing_worktree_on_unexpected_branch(tmp_path):
    commands: list[list[str]] = []
    (tmp_path / "repos" / "mobile-app").mkdir(parents=True)
    existing = tmp_path / "worktrees" / "monica-MOB-123-checkout-crash"
    existing.mkdir(parents=True)
    (existing / ".git").write_text("gitdir: /tmp/fake-worktree-git-dir")

    def run(cmd: list[str], cwd: Path | None = None) -> str:
        commands.append(cmd)
        if cmd == ["git", "-C", str(existing), "branch", "--show-current"]:
            return "main\n"
        return ""

    manager = RepoManager(
        config=RepoConfig(url="git@github.com:acme/mobile-app.git", local_name="mobile-app"),
        workspace_root=tmp_path,
        run_command=run,
    )

    with pytest.raises(RepoManagerError, match="worktree branch mismatch"):
        manager.prepare_worktree(linear_identifier="MOB-123", summary="Checkout crash")

    assert commands == [
        [
            "git",
            "-C",
            str(tmp_path / "repos" / "mobile-app"),
            "fetch",
            "origin",
            "main",
        ],
        ["git", "-C", str(existing), "branch", "--show-current"],
    ]


def test_repo_manager_rejects_existing_worktree_directory_that_is_not_git_worktree(tmp_path):
    commands: list[list[str]] = []
    repo_path = tmp_path / "repos" / "mobile-app"
    repo_path.mkdir(parents=True)
    stale_path = tmp_path / "worktrees" / "monica-MOB-123-checkout-crash"
    stale_path.mkdir(parents=True)

    manager = RepoManager(
        config=RepoConfig(url="git@github.com:acme/mobile-app.git", local_name="mobile-app"),
        workspace_root=tmp_path,
        run_command=lambda cmd, cwd=None: commands.append(cmd) or "",
    )

    with pytest.raises(RepoManagerError, match="worktree path exists but is not a git worktree"):
        manager.prepare_worktree(linear_identifier="MOB-123", summary="Checkout crash")

    assert commands == [
        [
            "git",
            "-C",
            str(repo_path),
            "fetch",
            "origin",
            "main",
        ]
    ]


def test_repo_manager_rejects_existing_worktree_file(tmp_path):
    (tmp_path / "repos" / "mobile-app").mkdir(parents=True)
    stale_path = tmp_path / "worktrees" / "monica-MOB-123-checkout-crash"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text("not a worktree")

    manager = RepoManager(
        config=RepoConfig(url="git@github.com:acme/mobile-app.git", local_name="mobile-app"),
        workspace_root=tmp_path,
        run_command=lambda cmd, cwd=None: "",
    )

    with pytest.raises(RepoManagerError, match="worktree path exists but is not a directory"):
        manager.prepare_worktree(linear_identifier="MOB-123", summary="Checkout crash")


def test_repo_manager_error_includes_command_and_stderr(tmp_path):
    manager = RepoManager(
        config=RepoConfig(url="git@github.com:acme/mobile-app.git", local_name="mobile-app"),
        workspace_root=tmp_path,
    )

    try:
        manager._default_run(["git", "definitely-not-a-command"], None)
    except RepoManagerError as exc:
        message = str(exc)
    else:  # pragma: no cover - git invariant
        raise AssertionError("expected RepoManagerError")

    assert "git definitely-not-a-command" in message
    assert "stderr:" in message or "stdout:" in message


def test_repo_manager_timeout_raises_readable_error(tmp_path, monkeypatch):
    manager = RepoManager(
        config=RepoConfig(url="git@github.com:acme/mobile-app.git", local_name="mobile-app"),
        workspace_root=tmp_path,
        timeout_seconds=1,
    )

    def timeout_run(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["git", "fetch"], timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout_run)

    with pytest.raises(RepoManagerError, match="command timed out") as exc_info:
        manager._default_run(["git", "fetch"], None)

    assert "git fetch" in str(exc_info.value)
    assert "1s" in str(exc_info.value)


def test_repo_manager_missing_executable_raises_readable_error(tmp_path, monkeypatch):
    manager = RepoManager(
        config=RepoConfig(url="git@github.com:acme/mobile-app.git", local_name="mobile-app"),
        workspace_root=tmp_path,
    )

    def missing_run(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing_run)

    with pytest.raises(RepoManagerError, match="executable not found: git"):
        manager._default_run(["git", "fetch"], None)
