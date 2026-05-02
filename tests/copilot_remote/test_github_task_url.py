"""Tests for safe GitHub task URL construction."""

from copilot_remote.github_task_url import build_github_task_web_url


def test_build_github_task_web_url_encodes_handle(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repo-name"
    repo_dir.mkdir()

    monkeypatch.setattr(
        "copilot_remote.github_task_url._git_origin_url",
        lambda path: "https://github.com/RosenblattAI/repo-name.git",
    )

    url = build_github_task_web_url(
        str(repo_dir),
        "repo-name",
        "task-123<script>",
    )

    assert url == "https://github.com/RosenblattAI/repo-name/tasks/task-123%3Cscript%3E"


def test_build_github_task_web_url_rejects_invalid_slug(tmp_path):
    repo_dir = tmp_path / "repo-name"
    repo_dir.mkdir()

    assert build_github_task_web_url(str(repo_dir), "../repo-name", "task-123") is None


def test_build_github_task_web_url_requires_matching_repo_dir_name(monkeypatch, tmp_path):
    repo_dir = tmp_path / "not-repo-name"
    repo_dir.mkdir()

    monkeypatch.setattr(
        "copilot_remote.github_task_url._git_origin_url",
        lambda path: "https://github.com/RosenblattAI/repo-name.git",
    )

    assert build_github_task_web_url(str(repo_dir), "repo-name", "task-123") is None


def test_build_github_task_web_url_rejects_mismatched_origin_repo(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repo-name"
    repo_dir.mkdir()

    monkeypatch.setattr(
        "copilot_remote.github_task_url._git_origin_url",
        lambda path: "https://github.com/RosenblattAI/other-repo.git",
    )

    assert build_github_task_web_url(str(repo_dir), "repo-name", "task-123") is None


def test_build_github_task_web_url_supports_ssh_origin(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repo-name"
    repo_dir.mkdir()

    monkeypatch.setattr(
        "copilot_remote.github_task_url._git_origin_url",
        lambda path: "git@github.com:RosenblattAI/repo-name.git",
    )

    url = build_github_task_web_url(str(repo_dir), "repo-name", "task-ssh")

    assert url == "https://github.com/RosenblattAI/repo-name/tasks/task-ssh"


def test_build_github_task_web_url_accepts_case_insensitive_origin_repo(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repo-name"
    repo_dir.mkdir()

    monkeypatch.setattr(
        "copilot_remote.github_task_url._git_origin_url",
        lambda path: "https://github.com/RosenblattAI/repo-name.git",
    )

    url = build_github_task_web_url(str(repo_dir), "repo-name", "task-case")

    assert url == "https://github.com/RosenblattAI/repo-name/tasks/task-case"