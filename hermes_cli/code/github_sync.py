#!/usr/bin/env python3
"""GitHub metadata sync service for Hermes Code Mode (P1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from hermes_cli.code.github_integration import GitHubIntegrationService, GitHubIntegrationStore
from hermes_state import SessionDB


class GitHubSyncService:
    def __init__(
        self,
        db_path: Optional[Path] = None,
        *,
        api_client: Optional[Any] = None,
    ) -> None:
        self._db_path = db_path
        self._api_client = api_client

    def _store(self) -> GitHubIntegrationStore:
        return GitHubIntegrationStore(db_path=self._db_path)

    def _client(self, installation_id: Optional[int] = None):
        if self._api_client is not None:
            return self._api_client
        return GitHubIntegrationService(db_path=self._db_path).api_client(installation_id)

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        db = SessionDB(db_path=self._db_path) if self._db_path else SessionDB()
        try:
            db.append_code_event(
                event_type=event_type,
                payload={
                    "type": event_type,
                    "version": 1,
                    "payload": payload,
                },
                source="github_sync",
            )
        finally:
            db.close()

    def sync_repositories(
        self,
        *,
        installation_id: Optional[int] = None,
        dry_run: bool = False,
        limit: int = 100,
    ) -> dict[str, Any]:
        client = self._client(installation_id)
        repositories = client.list_paginated("/installation/repositories", limit=limit)

        if dry_run:
            return {"dry_run": True, "synced": 0, "repositories": repositories}

        store = self._store()
        try:
            synced = [store.upsert_repository(repo, installation_id=installation_id) for repo in repositories]
        finally:
            store.close()
        self._emit(
            "github.repository.synced",
            {"installation_id": installation_id, "synced": len(synced)},
        )
        return {"dry_run": False, "synced": len(synced), "repositories": synced}

    def sync_issues(self, repo_full_name: str, *, limit: int = 100) -> dict[str, Any]:
        client = self._client()
        issues = client.list_paginated(
            f"/repos/{repo_full_name}/issues",
            params={"state": "all"},
            limit=limit,
        )
        issues = [issue for issue in issues if not issue.get("pull_request")]
        store = self._store()
        try:
            synced = [store.upsert_issue(repo_full_name, issue) for issue in issues]
        finally:
            store.close()
        self._emit("github.issue.synced", {"repo_full_name": repo_full_name, "synced": len(synced)})
        return {"repo_full_name": repo_full_name, "synced": len(synced), "issues": synced}

    def sync_pull_requests(self, repo_full_name: str, *, limit: int = 100) -> dict[str, Any]:
        client = self._client()
        pulls = client.list_paginated(
            f"/repos/{repo_full_name}/pulls",
            params={"state": "all"},
            limit=limit,
        )
        store = self._store()
        try:
            synced = [store.upsert_pull_request(repo_full_name, pr) for pr in pulls]
        finally:
            store.close()
        self._emit("github.pull_request.synced", {"repo_full_name": repo_full_name, "synced": len(synced)})
        return {"repo_full_name": repo_full_name, "synced": len(synced), "pull_requests": synced}

    def sync_branches(self, repo_full_name: str, *, limit: int = 100) -> dict[str, Any]:
        client = self._client()
        branches = client.list_paginated(f"/repos/{repo_full_name}/branches", limit=limit)
        store = self._store()
        try:
            synced = [store.upsert_branch(repo_full_name, branch) for branch in branches]
        finally:
            store.close()
        return {"repo_full_name": repo_full_name, "synced": len(synced), "branches": synced}

    def sync_repository_details(self, repo_full_name: str, *, limit: int = 100) -> dict[str, Any]:
        return {
            "repo_full_name": repo_full_name,
            "issues": self.sync_issues(repo_full_name, limit=limit),
            "pull_requests": self.sync_pull_requests(repo_full_name, limit=limit),
            "branches": self.sync_branches(repo_full_name, limit=limit),
        }
