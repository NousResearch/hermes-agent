"""Review commits preserve native Git identity without enabling hooks."""

import os
import subprocess

import pytest

from hermes_cli import web_git


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()


@pytest.fixture
def identity_repo(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("GIT_") or key == "EMAIL":
            monkeypatch.delenv(key)
    root = tmp_path.resolve()
    config = root / "global.gitconfig"
    config.write_text(
        "[user]\n name = Global Name\n email = global@example.invalid\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2001-02-03T04:05:06 +0230")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2002-03-04T05:06:07 -0430")
    repo = root / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "file.txt").write_text("content\n", encoding="utf-8")
    git(repo, "add", "file.txt")
    return repo, config


@pytest.mark.parametrize(
    "source",
    ["global", "local", "conditional", "explicit", "partial-env", "role-config"],
)
def test_review_commit_matches_native_effective_identity(
    identity_repo, monkeypatch, source
):
    repo, config = identity_repo
    if source in ("local", "explicit", "partial-env"):
        git(repo, "config", "user.name", "Local Name")
        git(repo, "config", "user.email", "local@example.invalid")
    if source == "conditional":
        included = repo.parent / "included.gitconfig"
        included.write_text(
            "[user]\n name = Conditional Name\n email = conditional@example.invalid\n",
            encoding="utf-8",
        )
        with config.open("a", encoding="utf-8") as stream:
            stream.write(
                f'[includeIf "gitdir:{repo.as_posix()}/"]\n path = {included.as_posix()}\n'
            )
    if source in ("explicit", "partial-env"):
        monkeypatch.setenv("GIT_AUTHOR_NAME", "Explicit Author")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "committer@example.invalid")
    if source == "explicit":
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "author@example.invalid")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "Explicit Committer")
    if source == "role-config":
        git(repo, "config", "author.name", "Configured Author")
        git(repo, "config", "committer.email", "configured-committer@example.invalid")
    # Git itself is the oracle, including dates and different author/committer fields.
    expected = [
        git(repo, "var", role) for role in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT")
    ]
    assert web_git.review_commit(str(repo), "identity", push=False) == {"ok": True}
    headers = git(repo, "cat-file", "commit", "HEAD").splitlines()
    assert [
        next(line[len(role) + 1 :] for line in headers if line.startswith(role + " "))
        for role in ("author", "committer")
    ] == expected


def test_review_commit_keeps_behavior_isolated_and_missing_identity_refused(
    identity_repo,
):
    repo, config = identity_repo
    hooks = repo.parent / "hooks"
    hooks.mkdir()
    marker = repo / "hook-ran"
    hook = hooks / "pre-commit"
    hook.write_text("#!/bin/sh\nprintf ran > hook-ran\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    git(repo, "config", "core.hooksPath", hooks.as_posix())
    git(repo, "config", "core.fsmonitor", hook.as_posix())
    # These commands must not run during the metadata queries or isolated commit.
    git(repo, "config", "core.pager", hook.as_posix())
    git(repo, "config", "core.editor", hook.as_posix())
    git(repo, "config", "filter.fixture.clean", hook.as_posix())
    git(repo, "config", "filter.fixture.smudge", hook.as_posix())
    assert web_git.review_commit(str(repo), "isolated", push=False) == {"ok": True}
    assert not marker.exists()
    # Even an active attribute driver must not run during identity metadata reads.
    attributes = repo / ".gitattributes"
    attributes.write_text("* filter=fixture\n", encoding="utf-8")
    web_git._review_commit_env(str(repo))
    assert not marker.exists()
    attributes.unlink()
    # Removing the only identity must not fall back to the operating-system user.
    config.write_text("[user]\n useConfigOnly = true\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity|auto-detection"):
        web_git.review_commit(str(repo), "missing", push=False)
    assert not marker.exists()
