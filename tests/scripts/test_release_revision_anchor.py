"""Changelog attribution is anchored to the selected release range."""

import subprocess


def _git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def test_get_commits_attributes_each_revision_not_current_head(tmp_path, monkeypatch):
    """A mailmap change after an older commit must not rewrite that commit's author."""
    from scripts import release
    from scripts.releases import authors

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)

    (repo / ".mailmap").write_text("", encoding="utf-8")
    subprocess.run(["git", "add", ".mailmap"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    subprocess.run(["git", "tag", "base"], cwd=repo, check=True)

    (repo / "payload.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "feat: payload"], cwd=repo, check=True)
    payload_sha = _git(repo, "rev-parse", "HEAD")

    (repo / ".mailmap").write_text("Mapped Person <mapped@example.com> Test <test@example.com>\n",
                                    encoding="utf-8")
    subprocess.run(["git", "add", ".mailmap"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "chore: add mailmap"], cwd=repo, check=True)

    monkeypatch.setattr(release, "REPO_ROOT", repo)
    assert _git(repo, "show", "-s", "--format=%aN", payload_sha) == "Mapped Person"
    assert _git(repo, "show", "-s", "--use-mailmap", "--format=%aN", payload_sha) == "Mapped Person"
    monkeypatch.setitem(authors.AUTHOR_MAP, "test@example.com", "test-user")
    commits = release.get_commits(since_tag="base")
    payload = next(commit for commit in commits if commit["sha"] == payload_sha)
    assert payload["author_name"] == "Test"
    assert payload["author_email"] == "test@example.com"
    assert payload["github_author"] == "@test-user"
