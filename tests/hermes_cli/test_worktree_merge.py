"""``hermes worktree merge``: a branch lands only when every check passes, else nothing changes.

Each test drives the real argparse tree (``build_worktree_parser`` -> ``cmd_worktree``) against a
real git repository, and asserts on the repository afterwards: HEAD, the merge parents, a clean tree
and no ``MERGE_HEAD``. ``test_subagent_branch_round_trip`` runs the actual subagent worktree
lifecycle (create, commit, finalize/archive) and merges the branch the parent is handed.
"""

import argparse
import json
import os
import subprocess

import pytest

from hermes_cli.subcommands.worktree import build_worktree_parser

pytestmark = pytest.mark.skipif(subprocess.run(["git", "--version"], capture_output=True, check=False).returncode,
                                reason="git required")


def _git(cwd, *args):
    result = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", cwd=str(cwd),
                            check=False)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout.strip()


def _commit(repo, path, text, message):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    for key, value in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}.items():
        monkeypatch.setenv(key, value)
    root = tmp_path / "repo"
    _git(tmp_path, "init", "-q", "-b", "main", str(root))
    _commit(root, "README.md", "hello\n", "init")
    _commit(root, "src/app.py", "x = 1\n", "scaffold")
    return root


def _child(repo, files, name="child"):
    """A branch off main with one commit per (path, text); returns its tip."""
    _git(repo, "switch", "-q", "-c", name)
    tip = ""
    for path, text in files:
        tip = _commit(repo, path, text, f"edit {path}")
    _git(repo, "switch", "-q", "main")
    return tip


def _merge(repo, capsys, *argv):
    parser = argparse.ArgumentParser(prog="hermes")
    build_worktree_parser(parser.add_subparsers(dest="command"))
    ns = parser.parse_args(["worktree", "merge", *argv, "--repo", str(repo), "--json"])
    capsys.readouterr()
    code = ns.func(ns)
    return code, json.loads(capsys.readouterr().out)


def _unchanged(repo, head):
    assert _git(repo, "rev-parse", "HEAD") == head
    assert _git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert not (repo / ".git" / "MERGE_HEAD").exists()


def test_branch_inside_allow_and_tested_at_tip_merges_after_gate(repo, capsys):
    tip = _child(repo, [("src/app.py", "x = 2\n"), ("src/util.py", "y = 1\n")])
    before = _git(repo, "rev-parse", "HEAD")

    code, result = _merge(repo, capsys, "child", "--allow", "src/", "--tested", tip,
                          "--gate", "test -f src/util.py")

    assert code == 0 and result["merged"] is True
    assert _git(repo, "rev-parse", "HEAD^1") == before
    assert _git(repo, "rev-parse", "HEAD^2") == tip
    assert result["files"] == ["src/app.py", "src/util.py"]
    assert result["gate"]["exit_code"] == 0


def test_file_outside_allow_refuses_without_merging(repo, capsys):
    _child(repo, [("src/app.py", "x = 2\n"), ("README.md", "rewritten\n")])
    before = _git(repo, "rev-parse", "HEAD")

    code, result = _merge(repo, capsys, "child", "--allow", "src")

    assert code == 1 and result["merged"] is False
    assert result["outside_allowed"] == ["README.md"]
    _unchanged(repo, before)


def test_rename_into_allowed_directory_still_counts_the_removed_side(repo, capsys):
    _git(repo, "switch", "-q", "-c", "child")
    _git(repo, "mv", "README.md", "src/README.md")
    _git(repo, "commit", "-q", "-m", "move")
    _git(repo, "switch", "-q", "main")
    before = _git(repo, "rev-parse", "HEAD")

    code, result = _merge(repo, capsys, "child", "--allow", "src/")

    assert code == 1
    assert result["outside_allowed"] == ["README.md"]
    _unchanged(repo, before)


def test_commits_after_the_tested_commit_refuse(repo, capsys):
    _git(repo, "switch", "-q", "-c", "child")
    tested = _commit(repo, "src/app.py", "x = 2\n", "tested change")
    _commit(repo, "src/app.py", "x = 3\n", "one more small thing")
    _git(repo, "switch", "-q", "main")
    before = _git(repo, "rev-parse", "HEAD")

    code, result = _merge(repo, capsys, "child", "--tested", tested)

    assert code == 1
    assert [line.split(" ", 1)[1] for line in result["untested_commits"]] == ["one more small thing"]
    _unchanged(repo, before)


def test_tested_commit_not_on_the_branch_refuses(repo, capsys):
    _child(repo, [("src/app.py", "x = 2\n")])
    before = _git(repo, "rev-parse", "HEAD")
    elsewhere = _child(repo, [("src/other.py", "z = 1\n")], name="elsewhere")

    code, result = _merge(repo, capsys, "child", "--tested", elsewhere)

    assert code == 1 and "not on the branch" in result["error"]
    _unchanged(repo, before)


def test_failing_gate_resets_to_the_pre_merge_commit(repo, capsys):
    _child(repo, [("src/app.py", "x = 2\n")])
    before = _git(repo, "rev-parse", "HEAD")

    code, result = _merge(repo, capsys, "child", "--gate", "echo gate-said-no; exit 3")

    assert code == 1 and result["merged"] is False
    assert result["gate"]["exit_code"] == 3 and "gate-said-no" in result["gate"]["output_tail"]
    assert result["reset_to"] == before
    _unchanged(repo, before)


def _alive(pid):
    try:
        os.kill(pid, 0)  # windows-footgun: ok — the only caller is marked platforms("posix")
    except ProcessLookupError:
        return False
    return True


@pytest.mark.platforms("posix")
def test_gate_timeout_kills_the_gate_tree_and_resets(repo, capsys, tmp_path):
    from tests.waiting import wait_until

    _child(repo, [("src/app.py", "x = 2\n")])
    before = _git(repo, "rev-parse", "HEAD")
    pidfile = tmp_path / "grandchild.pid"

    code, result = _merge(repo, capsys, "child", "--gate", f"sleep 60 & echo $! > {pidfile}; wait",
                          "--gate-timeout", "2")

    assert code == 1 and result["gate"]["timed_out"] is True
    assert result["gate"]["duration_s"] < 15, "the merge waited for the gate's orphaned child"
    _unchanged(repo, before)
    grandchild = int(pidfile.read_text(encoding="utf-8-sig"))
    assert wait_until(lambda: not _alive(grandchild), timeout=5), "the gate's child outlived the timeout"


def test_branch_that_moves_after_the_checks_merges_only_the_checked_commit(repo, capsys, monkeypatch):
    from hermes_cli import worktree_merge

    checked = _child(repo, [("src/app.py", "x = 2\n")])
    real_check = worktree_merge._check_branch

    def check_then_child_commits(*args, **kwargs):
        real_check(*args, **kwargs)
        _git(repo, "switch", "-q", "child")
        _commit(repo, "README.md", "unchecked\n", "landed after the checks")
        _git(repo, "switch", "-q", "main")

    monkeypatch.setattr(worktree_merge, "_check_branch", check_then_child_commits)

    code, result = _merge(repo, capsys, "child", "--allow", "src/")

    assert code == 0 and result["tip"] == checked
    assert _git(repo, "rev-parse", "HEAD^2") == checked
    assert (repo / "README.md").read_text(encoding="utf-8-sig") == "hello\n"


def test_failing_gate_that_switches_branch_moves_only_the_original_branch_back(repo, capsys):
    _child(repo, [("src/app.py", "x = 2\n")])
    _git(repo, "branch", "elsewhere")
    before = _git(repo, "rev-parse", "HEAD")

    code, result = _merge(repo, capsys, "child", "--gate", "git switch -q elsewhere; exit 1")

    assert code == 1 and result["reset_to"] == before
    assert _git(repo, "rev-parse", "refs/heads/main") == before
    assert _git(repo, "symbolic-ref", "HEAD") == "refs/heads/elsewhere"


def test_ignored_untracked_file_the_branch_would_overwrite_refuses(repo, capsys):
    _commit(repo, ".gitignore", "notes.txt\n", "ignore notes")
    _git(repo, "switch", "-q", "-c", "child")
    (repo / "notes.txt").write_text("branch bytes\n", encoding="utf-8")
    _git(repo, "add", "-f", "notes.txt")
    _git(repo, "commit", "-q", "-m", "track an ignored path")
    _git(repo, "switch", "-q", "main")
    (repo / "notes.txt").write_text("USER WORK\n", encoding="utf-8")
    before = _git(repo, "rev-parse", "HEAD")

    code, result = _merge(repo, capsys, "child", "--gate", "exit 1")

    assert code == 1 and result["untracked_collisions"] == ["notes.txt"]
    assert (repo / "notes.txt").read_text(encoding="utf-8-sig") == "USER WORK\n"
    _unchanged(repo, before)


def test_failing_gate_that_stages_an_untracked_user_file_leaves_it_on_disk(repo, capsys):
    _child(repo, [("src/app.py", "x = 2\n")])
    (repo / "scratch.txt").write_text("USER WORK\n", encoding="utf-8")
    before = _git(repo, "rev-parse", "HEAD")

    code, result = _merge(repo, capsys, "child", "--gate", "git add -A; exit 1")

    assert code == 1 and result["reset_to"] == before
    assert (repo / "scratch.txt").read_text(encoding="utf-8-sig") == "USER WORK\n"
    _unchanged(repo, before)


def test_failing_gate_that_commits_is_reported_and_nothing_is_reset(repo, capsys):
    _child(repo, [("src/app.py", "x = 2\n")])
    (repo / "scratch.txt").write_text("USER WORK\n", encoding="utf-8")

    code, result = _merge(repo, capsys, "child", "--gate", "git add -A; git commit -q -m gate; exit 1")

    assert code == 1 and result["reset_to"] == "" and "nothing was reset" in result["error"]
    assert _git(repo, "log", "-1", "--format=%s") == "gate"
    assert (repo / "scratch.txt").read_text(encoding="utf-8-sig") == "USER WORK\n"


def test_assume_unchanged_edit_refuses_and_is_left_alone(repo, capsys):
    _child(repo, [("src/util.py", "y = 1\n")])
    _git(repo, "update-index", "--assume-unchanged", "README.md")
    (repo / "README.md").write_text("USER WORK\n", encoding="utf-8")
    before = _git(repo, "rev-parse", "HEAD")

    code, result = _merge(repo, capsys, "child", "--gate", "exit 1")

    assert code == 1 and "assume-unchanged" in result["error"]
    assert (repo / "README.md").read_text(encoding="utf-8-sig") == "USER WORK\n"
    assert _git(repo, "rev-parse", "HEAD") == before


def test_submodule_change_counts_even_when_config_hides_it(repo, capsys):
    first = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{first},vendor")
    _git(repo, "commit", "-q", "-m", "gitlink")
    second = _child(repo, [("src/other.py", "z = 1\n")], name="donor")
    _git(repo, "switch", "-q", "-c", "child")
    _git(repo, "update-index", "--cacheinfo", f"160000,{second},vendor")
    _git(repo, "commit", "-q", "-m", "bump gitlink")
    _git(repo, "switch", "-q", "main")
    _git(repo, "config", "diff.ignoreSubmodules", "all")
    before = _git(repo, "rev-parse", "HEAD")

    code, result = _merge(repo, capsys, "child", "--allow", "src/")

    assert code == 1 and result["outside_allowed"] == ["vendor"]
    _unchanged(repo, before)


@pytest.mark.platforms("posix")
def test_carriage_return_in_a_name_is_not_read_as_a_newline(repo, capsys):
    _child(repo, [("src/a\rb", "x\n")])
    before = _git(repo, "rev-parse", "HEAD")

    code, result = _merge(repo, capsys, "child", "--allow", "src/a\nb")

    assert code == 1 and result["outside_allowed"] == ["src/a\rb"]
    _unchanged(repo, before)


@pytest.mark.platforms("posix")
def test_gate_timeout_kills_a_detached_child_that_ignores_sigterm(repo, capsys, tmp_path):
    import sys

    from tests.waiting import wait_until

    _child(repo, [("src/app.py", "x = 2\n")])
    pidfile = tmp_path / "leaf.pid"
    leaf = (f"{sys.executable} -c \"import os, signal, sys, time; os.setsid(); "  # windows-footgun: ok — test is platforms("posix")
            f"signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"open(sys.argv[1], 'w', encoding='utf-8').write(str(os.getpid())); time.sleep(60)\" {pidfile}")

    code, result = _merge(repo, capsys, "child", "--gate", f"{leaf} & wait", "--gate-timeout", "2")

    assert code == 1 and result["gate"]["timed_out"] is True
    leaf_pid = int(pidfile.read_text(encoding="utf-8-sig"))
    assert wait_until(lambda: not _alive(leaf_pid), timeout=5), "the detached gate child outlived the timeout"


def test_conflict_aborts_and_lists_the_files(repo, capsys):
    _child(repo, [("src/app.py", "x = 2\n")])
    _commit(repo, "src/app.py", "x = 99\n", "parent moved on")
    before = _git(repo, "rev-parse", "HEAD")

    code, result = _merge(repo, capsys, "child")

    assert code == 1 and result["conflicts"] == ["src/app.py"]
    _unchanged(repo, before)


def test_dirty_tracked_file_refuses_and_is_left_alone(repo, capsys):
    _child(repo, [("src/util.py", "y = 1\n")])
    before = _git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("work in progress\n", encoding="utf-8")

    code, result = _merge(repo, capsys, "child")

    assert code == 1 and "uncommitted changes" in result["error"]
    assert _git(repo, "rev-parse", "HEAD") == before
    assert (repo / "README.md").read_text(encoding="utf-8-sig") == "work in progress\n"


def test_already_merged_branch_is_a_no_op_success(repo, capsys):
    _child(repo, [("src/app.py", "x = 2\n")])
    assert _merge(repo, capsys, "child")[0] == 0
    head = _git(repo, "rev-parse", "HEAD")

    code, result = _merge(repo, capsys, "child")

    assert code == 0 and result["already_merged"] is True
    assert _git(repo, "rev-parse", "HEAD") == head


def test_subagent_branch_round_trip(repo, capsys):
    from tools import subagent_worktree

    info = subagent_worktree.create_subagent_worktree(str(repo), "abc123")
    assert info is not None
    _git(repo, "add", ".gitignore")  # creation records .worktrees/ in the parent's .gitignore
    _git(repo, "commit", "-q", "-m", "ignore .worktrees")
    child = repo / ".worktrees" / "subagent-abc123"
    tested = _commit(child, "src/feature.py", "def f():\n    return 1\n", "feature")
    payload = subagent_worktree.finalize_subagent_worktree(info)
    assert payload["commits"] == 1 and payload["branch"] == info["branch"]

    code, result = _merge(repo, capsys, info["branch"], "--allow", "src/", "--tested", tested,
                          "--gate", "python -c \"import ast, pathlib; ast.parse(pathlib.Path('src/feature.py').read_bytes())\"")

    assert code == 0 and result["merged"] is True
    assert (repo / "src" / "feature.py").exists()
    assert os.path.isdir(child) is False or payload["pruned"] is False
