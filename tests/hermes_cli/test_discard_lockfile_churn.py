"""`hermes update` lockfile-churn cleanup reverts only npm's own rewrites (#112378).

The npm projects (``scripts/whatsapp-bridge``, the Photon sidecar) carry their own ``package-lock.json``;
npm rewrites them non-deterministically, so a lock whose sibling manifest is clean is churn. A dirty
sibling manifest protects its lock. ``pnpm-lock.yaml`` is never rewritten by pnpm's frozen installs:
a change there is the user's and survives the cleanup.
"""

import os
import subprocess

from hermes_cli.update_cmd_git import _discard_lockfile_churn

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True, env=_GIT_ENV).stdout


def _repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "package.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
    for rel in ("apps/desktop", "scripts/whatsapp-bridge"):
        (tmp_path / rel).mkdir(parents=True)
        (tmp_path / rel / "package.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "scripts/whatsapp-bridge/package-lock.json").write_text("lock v1\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initial")
    return tmp_path


def _still_dirty_after_cleanup(repo, *dirty):
    for rel in dirty:
        (repo / rel).write_text("dirty\n", encoding="utf-8")
    _discard_lockfile_churn(["git"], repo)
    return set(_git(repo, "diff", "--name-only").splitlines())


def test_npm_project_lock_churn_is_discarded(tmp_path):
    repo = _repo(tmp_path)
    assert _still_dirty_after_cleanup(repo, "scripts/whatsapp-bridge/package-lock.json") == set()


def test_dirty_sibling_manifest_preserves_its_lock(tmp_path):
    repo = _repo(tmp_path)
    assert _still_dirty_after_cleanup(
        repo, "scripts/whatsapp-bridge/package.json", "scripts/whatsapp-bridge/package-lock.json") == {
        "scripts/whatsapp-bridge/package.json", "scripts/whatsapp-bridge/package-lock.json",
    }


def test_another_manifest_does_not_protect_an_npm_project_lock(tmp_path):
    repo = _repo(tmp_path)
    assert _still_dirty_after_cleanup(
        repo, "apps/desktop/package.json", "scripts/whatsapp-bridge/package-lock.json") == {
        "apps/desktop/package.json",
    }


def test_pnpm_lock_changes_are_never_discarded(tmp_path):
    repo = _repo(tmp_path)
    assert _still_dirty_after_cleanup(repo, "pnpm-lock.yaml") == {"pnpm-lock.yaml"}
