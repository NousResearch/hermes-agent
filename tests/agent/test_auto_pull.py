"""Gates for agent.auto_pull — fast-forward a clean default branch only."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agent.auto_pull import (
    maybe_auto_pull,
    reset_auto_pull_state_for_tests,
)


GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(not GIT, reason="git is required")


def _git_env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    })
    return env


def _run(cwd: Path, *args: str, env: dict[str, str]) -> None:
    subprocess.run([GIT, "-C", str(cwd), *args], check=True, env=env, capture_output=True)


def _write_commit(repo: Path, name: str, body: str, env: dict[str, str], message: str) -> None:
    (repo / name).write_text(body, encoding="utf-8")
    _run(repo, "add", name, env=env)
    _run(repo, "commit", "-q", "-m", message, env=env)


@pytest.fixture
def pair(tmp_path):
    """A bare origin and a clone on main, one commit ahead on origin."""
    reset_auto_pull_state_for_tests()
    env = _git_env(tmp_path)
    origin = tmp_path / "origin.git"
    origin.mkdir()
    subprocess.run(
        [GIT, "init", "-q", "--bare", "-b", "main", str(origin)],
        check=True, env=env, capture_output=True,
    )
    clone = tmp_path / "clone"
    subprocess.run(
        [GIT, "clone", "-q", str(origin), str(clone)],
        check=True, env=env, capture_output=True,
    )
    _write_commit(clone, "app.py", "print(1)\n", env, "init")
    _run(clone, "push", "-q", "-u", "origin", "main", env=env)

    # Advance origin via a second clone so the first clone is behind.
    other = tmp_path / "other"
    subprocess.run(
        [GIT, "clone", "-q", str(origin), str(other)],
        check=True, env=env, capture_output=True,
    )
    _write_commit(other, "app.py", "print(2)\n", env, "upstream")
    _run(other, "push", "-q", "origin", "main", env=env)

    return clone, env


def test_disabled_is_a_noop(pair):
    clone, _env = pair
    result = maybe_auto_pull(clone, enabled=False)
    assert result.action == "skipped"
    assert result.reason == "disabled"
    assert (clone / "app.py").read_text(encoding="utf-8") == "print(1)\n"


def test_pulls_clean_default_branch_that_is_behind(pair):
    clone, _env = pair
    result = maybe_auto_pull(clone, enabled=True, running_root=clone.parent)
    assert result.action == "pulled"
    assert result.commits >= 1
    assert (clone / "app.py").read_text(encoding="utf-8") == "print(2)\n"
    assert "origin/main" in result.upstream
    assert "Auto-pulled" in result.snapshot_line()


def test_skips_dirty_tree(pair):
    clone, _env = pair
    (clone / "app.py").write_text("print('dirty')\n", encoding="utf-8")
    result = maybe_auto_pull(clone, enabled=True, running_root=clone.parent)
    assert result.action == "skipped"
    assert result.reason == "dirty"
    # Local edit is untouched, and we did not fast-forward over it.
    assert (clone / "app.py").read_text(encoding="utf-8") == "print('dirty')\n"


def test_skips_untracked_file(pair):
    clone, _env = pair
    (clone / "notes.txt").write_text("wip\n", encoding="utf-8")
    result = maybe_auto_pull(clone, enabled=True, running_root=clone.parent)
    assert result.action == "skipped"
    assert result.reason == "dirty"


def test_skips_local_commits(pair):
    clone, env = pair
    _write_commit(clone, "local.py", "x = 1\n", env, "local")
    result = maybe_auto_pull(clone, enabled=True, running_root=clone.parent)
    assert result.action == "skipped"
    assert result.reason == "local commits"
    assert (clone / "local.py").is_file()


def test_skips_feature_branch(pair):
    clone, env = pair
    _run(clone, "checkout", "-q", "-b", "feature", env=env)
    _run(clone, "branch", "-q", "-u", "origin/main", env=env)
    result = maybe_auto_pull(clone, enabled=True, running_root=clone.parent)
    assert result.action == "skipped"
    assert result.reason == "not default branch"


def test_skips_running_hermes_checkout(pair):
    clone, _env = pair
    result = maybe_auto_pull(clone, enabled=True, running_root=clone)
    assert result.action == "skipped"
    assert result.reason == "running hermes checkout"
    assert (clone / "app.py").read_text(encoding="utf-8") == "print(1)\n"


def test_second_call_debounces_without_another_fetch(pair):
    clone, _env = pair
    first = maybe_auto_pull(clone, enabled=True, running_root=clone.parent)
    assert first.action == "pulled"
    second = maybe_auto_pull(clone, enabled=True, running_root=clone.parent)
    assert second.action == "skipped"
    assert second.reason == "recently fetched"


def test_up_to_date_after_reset_debounce(pair):
    clone, _env = pair
    maybe_auto_pull(clone, enabled=True, running_root=clone.parent)
    reset_auto_pull_state_for_tests()
    again = maybe_auto_pull(clone, enabled=True, running_root=clone.parent)
    assert again.action == "skipped"
    assert again.reason == "up to date"


def test_not_a_repo(tmp_path):
    reset_auto_pull_state_for_tests()
    (tmp_path / "app.py").write_text("print(1)\n", encoding="utf-8")
    result = maybe_auto_pull(tmp_path, enabled=True)
    assert result.action == "skipped"
    assert result.reason == "not a git repo"


def test_global_config_enables_without_a_project(pair):
    clone, _env = pair
    result = maybe_auto_pull(
        clone,
        config={"agent": {"auto_pull": True}},
        running_root=clone.parent,
    )
    assert result.action == "pulled"


def test_project_flag_enables_when_global_is_off(pair, tmp_path):
    from hermes_cli import projects_db as pdb

    clone, _env = pair
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="App", folders=[str(clone)])
        pdb.update_project(conn, pid, auto_pull=True)

    result = maybe_auto_pull(
        clone,
        config={"agent": {"auto_pull": False}},
        running_root=clone.parent,
    )
    assert result.action == "pulled"


def test_coding_snapshot_records_the_pull(pair):
    from agent.coding_context import resolve_runtime_mode

    clone, _env = pair
    mode = resolve_runtime_mode(
        platform="cli",
        cwd=clone,
        config={"agent": {"coding_context": "auto", "auto_pull": True}},
    )
    _prefix, workspace, _trailing = mode.system_prompt_parts()
    text = "\n".join(workspace)
    assert "Auto-pulled" in text
    assert (clone / "app.py").read_text(encoding="utf-8") == "print(2)\n"
