"""Tests for safe GitHub task URL construction."""

from pathlib import Path

from copilot_remote.github_task_url import build_github_task_web_url


def test_build_github_task_web_url_encodes_handle(monkeypatch, tmp_path):
    repo_dir = tmp_path / "static-pages"
    repo_dir.mkdir()

    monkeypatch.setattr(
        "copilot_remote.github_task_url._git_origin_url",
        lambda path: "https://github.com/RosenblattAI/static-pages.git",
    )

    url = build_github_task_web_url(
        str(repo_dir),
        "static-pages",
        "task-123<script>",
    )

    assert url == "https://github.com/RosenblattAI/static-pages/tasks/task-123%3Cscript%3E"


def test_build_github_task_web_url_rejects_invalid_slug(tmp_path):
    repo_dir = tmp_path / "static-pages"
    repo_dir.mkdir()

    assert build_github_task_web_url(str(repo_dir), "../static-pages", "task-123") is None


def test_build_github_task_web_url_requires_matching_repo_dir_name(monkeypatch, tmp_path):
    repo_dir = tmp_path / "not-static-pages"
    repo_dir.mkdir()

    monkeypatch.setattr(
        "copilot_remote.github_task_url._git_origin_url",
        lambda path: "https://github.com/RosenblattAI/static-pages.git",
    )

    assert build_github_task_web_url(str(repo_dir), "static-pages", "task-123") is None