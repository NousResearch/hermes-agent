"""A throttled clone is retried without publishing a partial checkout."""
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent


@pytest.mark.parametrize("materialize_fails", [False, True])
def test_clone_retries_and_publishes_only_materialized_tree(tmp_path, materialize_fails):
    origin = tmp_path / "origin"
    origin.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(origin), *args], check=True, capture_output=True, text=True)
    git("init", "-b", "main")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    (origin / "README").write_text("complete checkout\n")
    git("add", "README")
    git("commit", "-m", "fixture")
    dest = tmp_path / "install"
    attempts = tmp_path / "attempts"
    env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=(tmp_path / "home").as_posix(),
               HERMES_INSTALL_DIR=dest.as_posix(), HERMES_REPO_URL=origin.as_posix())
    # Inject throttling at the network boundary; the fallback and checkout use real Git.
    script = f'''source {shlex.quote((ROOT / 'scripts/install.sh').as_posix())} --manifest
sleep() {{ :; }}
git() {{
    if [ "$1" = clone ]; then
        printf '%s\\n' "$*" >> {shlex.quote(attempts.as_posix())}
        case " $* " in *" --filter=blob:none --no-checkout "*) ;; *) return 1 ;; esac
    fi
    if [ "{int(materialize_fails)}" = 1 ] && [ "${{3:-}}" = reset ]; then return 1; fi
    command git "$@"
}}
stage_repository
'''
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)
    calls = attempts.read_text().splitlines()
    assert len(calls) > 1
    assert "--no-checkout" in calls[-1]
    if materialize_fails:
        assert result.returncode != 0
        assert not dest.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (dest / "README").read_text() == "complete checkout\n"
    assert not list(tmp_path.glob(".hermes-clone-*"))


def test_existing_treeless_checkout_is_refetched_before_update(tmp_path):
    origin = tmp_path / "origin.git"
    source = tmp_path / "source"
    install = tmp_path / "install"

    # An empty hooks dir keeps the host's global pre-push hook (which refuses
    # fixture emails) out of the fixture push; nothing here leaves the tmp dir.
    empty_hooks = tmp_path / "empty-hooks"
    empty_hooks.mkdir()
    fixture_git = ("-c", "user.email=t@t.invalid", "-c", "user.name=Fixture",
                   "-c", f"core.hooksPath={empty_hooks}")

    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)], check=True, capture_output=True)
    def git(cwd, *args):
        return subprocess.run(["git", "-C", str(cwd), *fixture_git, *args], check=True,
                              capture_output=True, text=True)

    source.mkdir()
    git(source, "init", "-b", "main")
    for index in range(5):
        (source / "history.txt").write_text(f"commit {index}\n")
        git(source, "add", "history.txt")
        git(source, "commit", "-m", f"history {index}")
    git(source, "remote", "add", "origin", origin.as_posix())
    git(source, "push", "origin", "main")
    # A bare origin with an unborn HEAD clones into "unable to checkout"; point
    # HEAD at the branch, and use file:// so git actually applies --filter
    # (plain local paths ignore it).
    git(origin, "symbolic-ref", "HEAD", "refs/heads/main")
    git(origin, "config", "uploadpack.allowFilter", "true")
    git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")

    subprocess.run(["git", "clone", "--filter=tree:0", origin.as_uri(), install.as_posix()],
                   check=True, capture_output=True, text=True)
    assert git(install, "config", "--get", "remote.origin.partialclonefilter").stdout.strip() == "tree:0"

    # The pre-fix behavior this guards against: on the unmigrated treeless
    # clone, a path-filtered rev-list needs the tree of every commit, all of
    # which are missing, so GIT_NO_LAZY_FETCH=1 makes it fail hard (#129514).
    probe = ["git", "-C", install.as_posix(), "rev-list", "HEAD", "--", "history.txt"]
    stale = subprocess.run(probe, capture_output=True, text=True,
                           env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    assert stale.returncode != 0, "treeless clone answered a path-filtered rev-list without lazy fetches"

    env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=(tmp_path / "home").as_posix(),
               HERMES_INSTALL_DIR=install.as_posix(), HERMES_REPO_URL=origin.as_uri())
    script = f'''source {shlex.quote((ROOT / 'scripts/install.sh').as_posix())} --manifest
stage_repository
GIT_NO_LAZY_FETCH=1 git -C {shlex.quote(install.as_posix())} rev-list HEAD -- history.txt
'''
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert git(install, "config", "--get", "remote.origin.partialclonefilter").stdout.strip() == "blob:none"
    # After the blobless migration every historical tree is local, so the same
    # probe answers locally with lazy fetches forbidden.
    fresh = subprocess.run(probe, capture_output=True, text=True,
                           env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    assert fresh.returncode == 0, fresh.stderr
