"""``GET /api/health`` ``commit`` contract (review G4).

The public health probe reports the backend's BOOT ``commit`` so an attaching client can refuse a
backend whose boot commit differs from its checkout: a ``serve`` that outlived ``hermes update``
is never re-adopted. The field must always be present
(``null`` off git) and must come from ``get_version_info().commit``.
"""

from __future__ import annotations

import pytest

from hermes_cli.version_info import VersionInfo


def _info(commit: str | None) -> VersionInfo:
    return VersionInfo(base_version="2026.10.1", derived_version="2026.10.1", distance=0,
                       commit=commit, branch="main", source="git")


@pytest.mark.parametrize("commit", ["0fbe50a9d88aa7e1c0b5f3d2e4a6b8c9d0e1f2a3", None])
def test_health_reports_the_boot_commit_from_version_info(monkeypatch, commit):
    from starlette.testclient import TestClient

    from hermes_cli.web_routers import status
    import hermes_cli.web_server as ws

    monkeypatch.setattr(status, "get_version_info", lambda: _info(commit))

    response = TestClient(ws.app).get("/api/health")  # public: no session token

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "commit" in body, "attaching clients key the boot-identity decision on this field"
    assert body["commit"] == commit


def test_health_commit_is_frozen_when_the_app_starts(tmp_path, monkeypatch):
    """Importing web_server reads only the release, so app startup must freeze the full identity:
    a checkout that moves after startup (``hermes update``) must not change the reported commit."""
    import asyncio
    import os
    import subprocess
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from hermes_cli import version_info, web_server_app
    from hermes_cli.web_routers import status

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
            env={"HOME": str(tmp_path), "PATH": os.environ["PATH"], "GIT_NO_LAZY_FETCH": "1"},
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes@example.invalid")
    git("commit", "-q", "--allow-empty", "-m", "boot")
    boot_commit = git("rev-parse", "HEAD")

    @asynccontextmanager
    async def no_services(app):
        yield

    monkeypatch.setattr(web_server_app, "standalone_lifespan", no_services)
    monkeypatch.setattr(version_info, "_resolve_stamp_file", lambda: None)
    monkeypatch.setattr(version_info, "_resolve_repo_dir", lambda: repo)
    monkeypatch.setattr(version_info, "_cached_version_info", version_info._cached_version_info)
    version_info._reset_version_info_cache()

    async def serve_across_an_update():
        async with web_server_app.app_lifespan(SimpleNamespace(state=SimpleNamespace())):
            git("commit", "-q", "--allow-empty", "-m", "update")
            return await status.get_health()

    try:
        body = asyncio.run(serve_across_an_update())
    finally:
        version_info._reset_version_info_cache()
    assert body["commit"] == boot_commit
