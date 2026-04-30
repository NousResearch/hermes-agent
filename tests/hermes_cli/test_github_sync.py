"""Tests for GitHub sync service."""

from __future__ import annotations

from hermes_cli.code.github_sync import GitHubSyncService
from hermes_state import SessionDB


class _MockClient:
    def __init__(self):
        self.calls = []

    def list_paginated(self, path, params=None, limit=100):
        self.calls.append((path, params or {}, limit))
        if path == "/installation/repositories":
            return [
                {"id": 1, "full_name": "acme/repo1", "name": "repo1", "owner": {"login": "acme"}, "default_branch": "main"},
                {"id": 2, "full_name": "acme/repo2", "name": "repo2", "owner": {"login": "acme"}, "default_branch": "main"},
            ]
        if path.endswith("/issues"):
            return [
                {"id": 10, "number": 1, "title": "bug", "state": "open", "user": {"login": "alice"}},
                {"id": 11, "number": 2, "title": "feature", "state": "closed", "user": {"login": "bob"}},
            ]
        if path.endswith("/pulls"):
            return [
                {"id": 20, "number": 3, "title": "PR", "state": "open", "user": {"login": "alice"}, "head": {"ref": "x", "sha": "abc"}, "base": {"ref": "main"}},
            ]
        if path.endswith("/branches"):
            return [{"name": "main", "protected": True, "commit": {"sha": "abc"}}]
        return []


def test_sync_repositories_dry_run(tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    mock = _MockClient()
    service = GitHubSyncService(db_path=db_path, api_client=mock)
    result = service.sync_repositories(dry_run=True, limit=50)
    assert result["dry_run"] is True
    assert result["synced"] == 0
    assert len(result["repositories"]) == 2


def test_sync_repositories_persist(tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    mock = _MockClient()
    service = GitHubSyncService(db_path=db_path, api_client=mock)
    result = service.sync_repositories(dry_run=False, limit=50)
    assert result["synced"] == 2


def test_sync_issues_and_pull_requests(tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    mock = _MockClient()
    service = GitHubSyncService(db_path=db_path, api_client=mock)
    issues = service.sync_issues("acme/repo1", limit=100)
    pulls = service.sync_pull_requests("acme/repo1", limit=100)
    assert issues["synced"] == 2
    assert pulls["synced"] == 1


def test_sync_branches(tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    mock = _MockClient()
    service = GitHubSyncService(db_path=db_path, api_client=mock)
    branches = service.sync_branches("acme/repo1", limit=100)
    assert branches["synced"] == 1
    assert branches["branches"][0]["name"] == "main"


def test_sync_calls_are_pagination_aware(tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    mock = _MockClient()
    service = GitHubSyncService(db_path=db_path, api_client=mock)
    service.sync_repositories(dry_run=True, limit=37)
    assert mock.calls[0][2] == 37
