"""Behavior contracts for guarding every boot/chat/sidebar/doctor git caller against macOS's CLT stub."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from hermes_cli import _subprocess_compat
from hermes_platform.resolver import git as git_resolver


def _make_exe(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _link_bash(bin_dir: Path) -> None:
    bash = shutil.which("bash")
    assert bash is not None, "bash is required to execute test trap scripts"
    target = bin_dir / "bash"
    if not target.exists():
        target.symlink_to(bash)


def _init_repo(repo: Path, branch: str = "feature/clt-guard") -> Path:
    real_git = shutil.which("git")
    assert real_git is not None, "git is required for git-guard contract tests"
    repo.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    subprocess.run([real_git, "init", "-q", "-b", branch, str(repo)], check=True, env=env)
    (repo / "hello.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run([real_git, "-C", str(repo), "add", "hello.txt"], check=True, env=env)
    subprocess.run(
        [real_git, "-C", str(repo), "-c", "user.email=test@example.com", "-c", "user.name=Test",
         "commit", "-q", "-m", "initial"],
        check=True, env=env,
    )
    return repo


def _make_safe_directory_configs(tmp_path: Path) -> tuple[Path, Path]:
    system_cfg = tmp_path / "system-gitconfig"
    system_cfg.write_text("[safe]\n\tdirectory = *\n", encoding="utf-8")
    global_cfg = tmp_path / "global-gitconfig"
    global_cfg.write_text("[safe]\n\tdirectory = \n\tdirectory = /trusted/only\n", encoding="utf-8")
    return system_cfg, global_cfg


@pytest.fixture
def clt_stub_env(tmp_path, monkeypatch):
    """Plant a trap ``git`` executable on PATH and mark it as macOS's uninstalled CLT stub."""
    spawn_log = tmp_path / "git_spawns.log"
    trap_bin = tmp_path / "apple_usr_bin"
    trap_git = _make_exe(
        trap_bin / "git",
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> {str(spawn_log)!r}\nexit 1\n",
    )
    _link_bash(trap_bin)
    trap_real = os.path.realpath(trap_git)
    real_check = git_resolver.is_usable_git_executable

    def _stub_check(executable, env=None, **kwargs):
        if executable and os.path.realpath(os.fspath(executable)) == trap_real:
            return False
        return real_check(executable, env, **kwargs)

    git_resolver.clear_git_usability_cache()
    _subprocess_compat._safe_directory_cache.clear()
    monkeypatch.setattr(git_resolver, "is_usable_git_executable", _stub_check)
    monkeypatch.setenv("PATH", str(trap_bin))

    from tui_gateway import git_probe, methods_complete_helpers

    git_probe.invalidate()
    methods_complete_helpers._fuzzy_cache.clear()
    return trap_bin, trap_git, spawn_log


# ── Case 1: Unusable /usr/bin/git Apple CLT stub -> zero git subprocess spawns ──


@pytest.mark.platforms("posix")
@pytest.mark.real_safe_directory
def test_clt_stub_never_spawns_bounded_git_probe_or_safe_directories(tmp_path, clt_stub_env):
    repo = tmp_path / "repo"
    repo.mkdir()
    trap_bin, _trap_git, spawn_log = clt_stub_env
    system_cfg, global_cfg = _make_safe_directory_configs(tmp_path)
    base_env = {
        "PATH": str(trap_bin),
        "GIT_CONFIG_SYSTEM": str(system_cfg),
        "GIT_CONFIG_GLOBAL": str(global_cfg),
    }

    assert _subprocess_compat._user_safe_directories(base_env) == []
    assert _subprocess_compat.bounded_git_probe(
        ["git", "-C", str(repo), "branch", "--show-current"], timeout=5, env=base_env
    ) == ""
    assert _subprocess_compat.noninteractive_repo_git_env(repo, base=base_env) is None
    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"


@pytest.mark.platforms("posix")
@pytest.mark.real_safe_directory
def test_clt_stub_never_spawns_git_probe_branch_and_repo_root(tmp_path, clt_stub_env):
    from tui_gateway import git_probe

    repo = tmp_path / "repo"
    repo.mkdir()
    _trap_bin, _trap_git, spawn_log = clt_stub_env

    assert git_probe.branch(str(repo)) == ""
    assert git_probe.repo_root(str(repo)) == ""
    assert git_probe.resolve(str(repo)) is None
    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"


@pytest.mark.platforms("posix")
@pytest.mark.real_safe_directory
def test_clt_stub_never_spawns_completion_helpers_and_falls_back_to_walk(tmp_path, clt_stub_env):
    from tui_gateway import server

    repo = tmp_path / "workspace"
    repo.mkdir()
    (repo / "sample.py").write_text("x = 1\n", encoding="utf-8")
    _trap_bin, _trap_git, spawn_log = clt_stub_env

    assert list(server._git_repo_files(str(repo))) == []
    assert "sample.py" in server._list_repo_files(str(repo))
    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"


@pytest.mark.platforms("posix")
@pytest.mark.real_safe_directory
def test_clt_stub_never_spawns_host_supervisor_build_sha(tmp_path, monkeypatch, clt_stub_env):
    from tui_gateway import host_supervisor

    repo = tmp_path / "workspace"
    repo.mkdir()
    _trap_bin, _trap_git, spawn_log = clt_stub_env
    monkeypatch.setattr(host_supervisor, "_repo_root", lambda: repo)

    assert host_supervisor._build_sha() == "unknown"
    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"


@pytest.mark.platforms("posix")
@pytest.mark.real_safe_directory
def test_clt_stub_never_spawns_context_references_or_coding_context(tmp_path, clt_stub_env):
    from agent import coding_context, context_references

    repo = tmp_path / "workspace"
    (repo / ".git").mkdir(parents=True)
    _trap_bin, _trap_git, spawn_log = clt_stub_env

    ref = context_references.ContextReference(kind="diff", target="", raw="@diff", start=0, end=5)
    err, block = context_references._expand_git_reference(ref, repo, ["diff"], "git diff")
    assert err is not None and "git" in err.lower() and block is None
    coding_context.build_coding_workspace_block(repo)
    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"


@pytest.mark.platforms("posix")
@pytest.mark.real_safe_directory
def test_clt_stub_never_spawns_checkpoint_manager_or_profile_rename(tmp_path, clt_stub_env):
    from agent.checkpoints import manager as checkpoint_manager
    from agent.checkpoints import manager_profile_rename

    repo = tmp_path / "workspace"
    repo.mkdir()
    _trap_bin, _trap_git, spawn_log = clt_stub_env

    ok, out, cp_err = checkpoint_manager._run_git(["status"], tmp_path / "cp_store", str(repo))
    assert (ok, out) == (False, "") and "git" in cp_err.lower()
    init_err = checkpoint_manager._init_store(tmp_path / "cp_base" / "store", str(repo))
    assert init_err is not None and "git" in init_err.lower()

    new_profile = tmp_path / "new_profile"
    rename_store = checkpoint_manager._store_path(new_profile / "checkpoints")
    rename_store.mkdir(parents=True)
    (rename_store / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    rename_res = manager_profile_rename.migrate_profile_checkpoint_projects(tmp_path / "old_profile", new_profile)
    assert rename_res["errors"] == 1
    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"


@pytest.mark.platforms("posix")
@pytest.mark.real_safe_directory
def test_clt_stub_never_spawns_web_boot_and_auxiliary_git_callers(tmp_path, monkeypatch, clt_stub_env):
    from agent import trace_upload
    from hermes_cli import (
        boot_bootstrap,
        doctor,
        doctor_state,
        dump,
        git_credentials,
        main as hermes_main,
        main_tui_launch,
        source_check,
        web_git,
    )
    from hermes_cli.web_routers import files as web_files
    from tools import working_diff

    repo = tmp_path / "workspace"
    (repo / ".git").mkdir(parents=True)
    _trap_bin, _trap_git, spawn_log = clt_stub_env

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "install-stamp.json").write_text(
        json.dumps({"commit": "a" * 40, "distribution": "desktop-app", "updateMechanism": "electron-updater"}),
        encoding="utf-8",
    )
    legacy_home = tmp_path / "legacy_home"
    (legacy_home / "hermes-agent" / ".git").mkdir(parents=True)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", bundle)
    monkeypatch.setattr(doctor, "HERMES_HOME", legacy_home)
    doctor_state.check_legacy_desktop_checkout()

    assert web_files._fs_git_branch(str(repo)) == ""
    code, w_out, _w_err = web_git._git(str(repo), ["status"])
    assert code != 0 and w_out == ""
    code_w, _wo, _we = web_git._git_write(str(repo), ["status"])
    assert code_w != 0

    monkeypatch.chdir(repo)
    assert hermes_main._resolve_workspace_key() == str(repo)
    assert source_check._git_run(["rev-parse", "HEAD"], cwd=repo) is None
    with pytest.raises(FileNotFoundError):
        source_check.remote_git_run(["git", "ls-remote", "origin"], cwd=repo, timeout=2)

    assert trace_upload._git_branch(str(repo)) == ""
    assert working_diff._run(["diff"], str(repo)) == (127, "")
    assert dump._git_output(repo, "rev-parse", "HEAD") == ""
    assert git_credentials._credential_fill("https://github.com/NousResearch/hermes-agent") is None
    assert main_tui_launch._restore_tui_workspace(repo / "ui-tui") is False
    assert boot_bootstrap.read_git_head(repo) is None

    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"


@pytest.mark.platforms("posix")
def test_clt_stub_doctor_reports_honest_warning_and_install_hint(tmp_path, monkeypatch, capsys, clt_stub_env):
    from hermes_cli import doctor_tools

    _trap_bin, _trap_git, spawn_log = clt_stub_env
    monkeypatch.setattr(doctor_tools, "_pm_tool_path", lambda _name: None)

    doctor_tools._check_git_and_rg(False)
    output = capsys.readouterr().out

    assert "✓ git" not in output
    assert "git not found" in output
    assert "xcode-select --install" in output
    assert "brew install git" in output
    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"


@pytest.mark.platforms("posix")
def test_clt_stub_never_spawns_version_info_probes(tmp_path, clt_stub_env):
    """The startup banner, dashboard and ``hermes dump`` read an unstamped checkout's version from git."""
    from hermes_cli import version_info

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    _trap_bin, _trap_git, spawn_log = clt_stub_env

    info = version_info._git_version_info(checkout)
    assert (info.commit, info.dirty) == (None, False)
    assert version_info._run_git(checkout, "rev-parse", "HEAD") is None
    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"


@pytest.mark.platforms("posix")
def test_clt_stub_never_spawns_startup_dropped_directory_prune(tmp_path, clt_stub_env):
    """The launch-time bytecode sweep lists the files an update dropped; with the stub it takes the
    "recorded commit is gone" answer (what the stub's failed launch gave) instead of launching it."""
    from hermes_cli import main_web_build

    _trap_bin, _trap_git, spawn_log = clt_stub_env

    assert main_web_build._prune_dropped_directories(tmp_path, "git:" + "a" * 40) == 0
    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"


def test_startup_dropped_directory_prune_without_git_still_retries(tmp_path, monkeypatch):
    """A git that cannot be launched at all keeps raising, so the old fingerprint stays and the
    next launch retries the prune."""
    from hermes_cli import main_web_build

    empty = tmp_path / "no-git"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(OSError):
        main_web_build._prune_dropped_directories(tmp_path, "git:" + "a" * 40)


@pytest.mark.platforms("posix")
def test_clt_stub_plugin_git_falls_back_to_no_git_not_to_the_stub(tmp_path, monkeypatch, clt_stub_env):
    """Plugin update checks (gateway housekeeping) and doctor's provenance row resolve git through
    plugins_cmd, whose install-dir fallback names /usr/bin/git: the stub on such a Mac."""
    from hermes_cli import plugins_cmd, plugins_provenance, plugins_updates

    _trap_bin, _trap_git, spawn_log = clt_stub_env
    # Every candidate reads as the stub, as /usr/bin/git does on a Mac without the tools.
    monkeypatch.setattr(git_resolver, "is_usable_git_executable", lambda _exe, _env=None, **_kw: False)
    launched: list = []
    monkeypatch.setattr(plugins_cmd, "_run_plugin_git", lambda *a, **kw: launched.append(a))
    # Record a git launch and never run it: on a Mac without the tools it would be the stub.
    monkeypatch.setattr(
        plugins_provenance.subprocess, "run",
        lambda cmd, *a, **kw: launched.append(cmd) or subprocess.CompletedProcess(cmd, 1, "", ""),
    )
    plugin = tmp_path / "plugin"
    (plugin / ".git").mkdir(parents=True)
    (plugin / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://example.invalid/p.git\n', encoding="utf-8",
    )

    assert plugins_cmd._resolve_git_executable() is None
    with pytest.raises(RuntimeError):
        plugins_updates.default_ls_remote("https://example.invalid/p.git")
    assert plugins_provenance._git_origin_url(plugin) == "https://example.invalid/p.git"
    assert not launched, f"plugin git launched: {launched}"
    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"


@pytest.mark.platforms("posix")
def test_clt_stub_skips_gh_calls_that_make_gh_run_git(tmp_path, clt_stub_env):
    """The review pane's PR refresh, its Create PR and the sidebar's PR badges ask gh about "this
    repo", which gh answers by running git. With the stub the pane reports gh as not ready, which
    disables Create PR, and a direct create is refused before gh runs."""
    from hermes_cli import web_git

    trap_bin, _trap_git, spawn_log = clt_stub_env
    gh_log = tmp_path / "gh.log"
    _make_exe(
        trap_bin / "gh",
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> {str(gh_log)!r}\n"
        "[ \"$1\" = auth ] && exit 0\ngit rev-parse --abbrev-ref HEAD >/dev/null 2>&1\nexit 1\n",
    )
    repo = tmp_path / "repo"
    repo.mkdir()

    assert web_git.review_ship_info(str(repo)) == {"ghReady": False, "pr": None}
    assert web_git.review_pr_list(str(repo), ["main"]) == {"ghReady": False, "prs": []}
    with pytest.raises(RuntimeError, match="git is not available"):
        web_git.review_create_pr(str(repo))
    assert not gh_log.exists(), f"gh was run: {gh_log.read_text()}"
    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"


@pytest.mark.platforms("posix")
def test_clt_stub_never_spawns_review_commit_identity_or_commit(tmp_path, clt_stub_env):
    """The review pane's commit reads the identity with ``git var`` before any ``_git_in`` call."""
    from hermes_cli import web_git

    repo = tmp_path / "repo"
    repo.mkdir()
    _trap_bin, _trap_git, spawn_log = clt_stub_env

    with pytest.raises(RuntimeError):
        web_git.review_commit(str(repo), "message", False)
    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"


@pytest.mark.platforms("posix")
def test_resume_workspace_key_keeps_users_git_dir_and_work_tree(tmp_path, monkeypatch):
    """``--resume latest`` groups by the repo root git reports for the user's own environment:
    a dotfiles-style ``GIT_DIR``/``GIT_WORK_TREE`` pair must still name the work tree."""
    from hermes_cli import main as hermes_main

    work_tree = _init_repo(tmp_path / "work-tree")
    git_dir = tmp_path / "separate.git"
    (work_tree / ".git").rename(git_dir)
    subdir = work_tree / "nested"
    subdir.mkdir()
    monkeypatch.setenv("GIT_DIR", str(git_dir))
    monkeypatch.setenv("GIT_WORK_TREE", str(work_tree))
    monkeypatch.chdir(subdir)

    key = hermes_main._resolve_workspace_key()
    assert key is not None and os.path.realpath(key) == os.path.realpath(work_tree)


# ── Case 2: Working Apple git (<dev_dir>/usr/bin/git) -> normal branch + ordered safe.directory ──


@pytest.mark.platforms("posix")
@pytest.mark.real_safe_directory
def test_working_apple_git_returns_branch_and_ordered_safe_directories(tmp_path, monkeypatch, capsys):
    from hermes_cli import doctor_tools
    from tui_gateway import git_probe

    real_git = shutil.which("git")
    assert real_git is not None
    repo = _init_repo(tmp_path / "repo", branch="feature/apple-clt-ok")
    system_cfg, global_cfg = _make_safe_directory_configs(tmp_path)

    apple_bin = tmp_path / "apple_usr_bin"
    apple_bin.mkdir()
    apple_git = apple_bin / "git"
    apple_git.symlink_to(real_git)
    dev_dir = tmp_path / "clt"
    _make_exe(dev_dir / "usr" / "bin" / "git", f"#!/usr/bin/env bash\nexec {real_git!r} \"$@\"\n")

    env = {
        "PATH": str(apple_bin),
        "DEVELOPER_DIR": str(dev_dir),
        "GIT_CONFIG_SYSTEM": str(system_cfg),
        "GIT_CONFIG_GLOBAL": str(global_cfg),
    }
    assert git_resolver.resolve_usable_git(
        env, platform="darwin", apple_stub_path=str(apple_git), cached=False
    ) == str(apple_git)

    git_resolver.clear_git_usability_cache()
    _subprocess_compat._safe_directory_cache.clear()
    git_probe.invalidate()
    monkeypatch.setenv("PATH", str(apple_bin))
    monkeypatch.setenv("DEVELOPER_DIR", str(dev_dir))
    monkeypatch.setattr(doctor_tools, "_pm_tool_path", lambda _name: None)

    assert git_probe.branch(str(repo)) == "feature/apple-clt-ok"
    assert _subprocess_compat._user_safe_directories(env) == ["*", "", "/trusted/only"]

    hardened = _subprocess_compat.noninteractive_git_env(env)
    replayed = [
        hardened[f"GIT_CONFIG_VALUE_{i}"]
        for i in range(int(hardened["GIT_CONFIG_COUNT"]))
        if hardened[f"GIT_CONFIG_KEY_{i}"] == "safe.directory"
    ]
    assert replayed == ["*", "", "/trusted/only"]

    doctor_tools._check_git_and_rg(False)
    assert "✓ git" in capsys.readouterr().out


# ── Case 3: Custom/Homebrew git on PATH with CLT absent -> not blocked ──


@pytest.mark.platforms("posix")
@pytest.mark.real_safe_directory
def test_custom_homebrew_git_with_clt_absent_works_normally(tmp_path, monkeypatch, capsys):
    from hermes_cli import doctor_tools
    from tui_gateway import git_probe

    real_git = shutil.which("git")
    assert real_git is not None
    repo = _init_repo(tmp_path / "repo", branch="feature/homebrew-git")
    system_cfg, global_cfg = _make_safe_directory_configs(tmp_path)

    spawn_log = tmp_path / "stub_spawns.log"
    stub_bin = tmp_path / "apple_usr_bin"
    stub_git = _make_exe(
        stub_bin / "git",
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> {str(spawn_log)!r}\nexit 1\n",
    )
    _link_bash(stub_bin)
    homebrew_bin = tmp_path / "homebrew" / "bin"
    homebrew_bin.mkdir(parents=True)
    homebrew_git = _make_exe(
        homebrew_bin / "git",
        f"#!/usr/bin/env bash\nexec {real_git!r} \"$@\"\n",
    )
    _link_bash(homebrew_bin)

    missing_dev = str(tmp_path / "nonexistent-clt")
    env = {
        "PATH": f"{homebrew_bin}{os.pathsep}{stub_bin}",
        "DEVELOPER_DIR": missing_dev,
        "GIT_CONFIG_SYSTEM": str(system_cfg),
        "GIT_CONFIG_GLOBAL": str(global_cfg),
    }

    assert git_resolver.resolve_usable_git(
        env, platform="darwin", apple_stub_path=str(stub_git), cached=False
    ) == str(homebrew_git)

    git_resolver.clear_git_usability_cache()
    _subprocess_compat._safe_directory_cache.clear()
    git_probe.invalidate()
    monkeypatch.setenv("PATH", env["PATH"])
    monkeypatch.setenv("DEVELOPER_DIR", missing_dev)
    monkeypatch.setattr(doctor_tools, "_pm_tool_path", lambda _name: None)

    assert git_probe.branch(str(repo)) == "feature/homebrew-git"
    assert _subprocess_compat._user_safe_directories(env) == ["*", "", "/trusted/only"]

    doctor_tools._check_git_and_rg(False)
    assert "✓ git" in capsys.readouterr().out
    assert not spawn_log.exists(), "Apple stub must never be invoked when Homebrew git is first on PATH"


# ── Native macOS end-to-end checks (no resolver mocking, real sys.platform == 'darwin') ──


@pytest.mark.platforms("macos")
@pytest.mark.real_safe_directory
def test_macos_native_absent_clt_prevents_all_git_spawns_and_warns_doctor(tmp_path, monkeypatch, capsys):
    from agent import context_references
    from agent.checkpoints import manager as checkpoint_manager
    from hermes_cli import doctor_tools
    from tui_gateway import git_probe, host_supervisor, server

    repo = tmp_path / "repo"
    repo.mkdir()
    git_resolver.clear_git_usability_cache()
    _subprocess_compat._safe_directory_cache.clear()
    git_probe.invalidate()

    # The resolver runs its own `xcode-select -p`; only its answer is replaced, with the empty
    # developer dir a macOS upgrade leaves behind when it drops the Command Line Tools.
    missing_dev = str(tmp_path / "nonexistent-clt")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("DEVELOPER_DIR", raising=False)
    monkeypatch.setattr(git_resolver, "_probe_xcode_select_developer_dir", lambda _env=None: missing_dev)
    monkeypatch.setattr(doctor_tools, "_pm_tool_path", lambda _name: None)
    monkeypatch.setattr(host_supervisor, "_repo_root", lambda: repo)

    spawned: list[list[str]] = []
    real_run = subprocess.run
    real_popen = subprocess.Popen

    # A git launch here is the stub: record it and never run it, so a regression fails the test
    # instead of opening Apple's install dialog on the machine running it.
    def spy_run(cmd, *args, **kwargs):
        if cmd and "git" in str(cmd[0]):
            spawned.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 1, "", "")
        return real_run(cmd, *args, **kwargs)

    def spy_popen(cmd, *args, **kwargs):
        if cmd and "git" in str(cmd[0]):
            spawned.append(list(cmd))
            raise FileNotFoundError(f"trapped git launch: {cmd[0]}")
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy_run)
    monkeypatch.setattr(subprocess, "Popen", spy_popen)

    assert _subprocess_compat.bounded_git_probe(["git", "-C", str(repo), "status"], timeout=2) == ""
    assert _subprocess_compat._user_safe_directories(os.environ) == []
    assert git_probe.branch(str(repo)) == ""
    assert list(server._git_repo_files(str(repo))) == []
    assert host_supervisor._build_sha() == "unknown"
    ref = context_references.ContextReference(kind="diff", target="", raw="@diff", start=0, end=5)
    err, block = context_references._expand_git_reference(ref, repo, ["diff"], "git diff")
    assert err is not None and block is None
    ok, _out, _cp_err = checkpoint_manager._run_git(["status"], tmp_path / "store", str(repo))
    assert ok is False

    doctor_tools._check_git_and_rg(False)
    out = capsys.readouterr().out
    assert "✓ git" not in out
    assert "xcode-select --install" in out
    assert not spawned, f"macOS /usr/bin/git stub was spawned: {spawned}"


@pytest.mark.platforms("macos")
@pytest.mark.real_safe_directory
def test_macos_native_working_apple_git_works_normally(tmp_path, monkeypatch, capsys):
    from hermes_cli import doctor_tools
    from tui_gateway import git_probe

    proc = subprocess.run(
        ["/usr/bin/xcode-select", "-p"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False, timeout=2,
    )
    dev_dir = proc.stdout.strip()
    if proc.returncode != 0 or not os.access(os.path.join(dev_dir, "usr", "bin", "git"), os.X_OK):
        pytest.skip("host has no active macOS Command Line Tools git")

    repo = _init_repo(tmp_path / "repo", branch="feature/mac-apple-git")
    system_cfg, global_cfg = _make_safe_directory_configs(tmp_path)

    git_resolver.clear_git_usability_cache()
    _subprocess_compat._safe_directory_cache.clear()
    git_probe.invalidate()

    # No DEVELOPER_DIR: the resolver must reach the host's CLT through its own `xcode-select -p`.
    env = {
        "PATH": "/usr/bin:/bin",
        "GIT_CONFIG_SYSTEM": str(system_cfg),
        "GIT_CONFIG_GLOBAL": str(global_cfg),
    }
    monkeypatch.setenv("PATH", env["PATH"])
    monkeypatch.delenv("DEVELOPER_DIR", raising=False)
    monkeypatch.setattr(doctor_tools, "_pm_tool_path", lambda _name: None)

    assert git_probe.branch(str(repo)) == "feature/mac-apple-git"
    assert _subprocess_compat._user_safe_directories(env) == ["*", "", "/trusted/only"]
    doctor_tools._check_git_and_rg(False)
    assert "✓ git" in capsys.readouterr().out


@pytest.mark.platforms("macos")
@pytest.mark.real_safe_directory
def test_macos_native_custom_git_works_with_absent_clt(tmp_path, monkeypatch, capsys):
    from hermes_cli import doctor_tools
    from tui_gateway import git_probe

    real_git = shutil.which("git")
    if real_git is None:
        pytest.skip("git not installed")
    if os.path.realpath(real_git) == "/usr/bin/git":
        proc = subprocess.run(
            ["/usr/bin/xcode-select", "-p"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False, timeout=2,
        )
        backing = Path(proc.stdout.strip()) / "usr" / "bin" / "git"
        if not backing.is_file():
            pytest.skip("no working git binary on host")
        real_git = str(backing)

    repo = _init_repo(tmp_path / "repo", branch="feature/mac-custom")
    system_cfg, global_cfg = _make_safe_directory_configs(tmp_path)
    custom_bin = tmp_path / "homebrew" / "bin"
    custom_bin.mkdir(parents=True)
    (custom_bin / "git").symlink_to(real_git)

    git_resolver.clear_git_usability_cache()
    _subprocess_compat._safe_directory_cache.clear()
    git_probe.invalidate()

    env = {
        "PATH": f"{custom_bin}:/usr/bin:/bin",
        "DEVELOPER_DIR": str(tmp_path / "nonexistent-clt"),
        "GIT_CONFIG_SYSTEM": str(system_cfg),
        "GIT_CONFIG_GLOBAL": str(global_cfg),
    }
    monkeypatch.setenv("PATH", env["PATH"])
    monkeypatch.setenv("DEVELOPER_DIR", env["DEVELOPER_DIR"])
    monkeypatch.setattr(doctor_tools, "_pm_tool_path", lambda _name: None)

    assert git_probe.branch(str(repo)) == "feature/mac-custom"
    assert _subprocess_compat._user_safe_directories(env) == ["*", "", "/trusted/only"]
    doctor_tools._check_git_and_rg(False)
    assert "✓ git" in capsys.readouterr().out


def test_subprocess_compat_imports_in_cold_bootstrap_and_git_guard_degrades_open(tmp_path):
    """Cold bootstrap (python -I -S with only hermes_cli, no hermes_platform) must import
    _subprocess_compat, and the git guard degrades open: without the resolver it refuses nothing,
    so source_check's direct git reads still run. (bounded_probe_run itself needs psutil, which
    -S leaves out, so a probe through it proves nothing here.)"""
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    cold_root = tmp_path / "cold"
    cold_root.mkdir()
    shutil.copytree(
        repo_root / "hermes_cli",
        cold_root / "hermes_cli",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    script = (
        f"import sys; sys.path.insert(0, {str(cold_root)!r}); "
        "from hermes_cli import _subprocess_compat as sc; "
        "sc.suppress_platform_ver_console(); "
        "assert sc.windows_hide_flags() >= 0; "
        "assert sc._git_resolver() is None, 'hermes_platform was importable'; "
        "assert sc._git_argv0_unusable(['git']) is False; "
        "assert sc._git_usable({'PATH': ''}) is True"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
