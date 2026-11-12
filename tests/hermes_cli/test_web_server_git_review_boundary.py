"""Tests: git review endpoints must agree on the managed-files boundary.

Issue #146744 — review/list had no boundary check while review/diff and
review/commit-context applied _hosted_fs_read_guard, so a locked-root
deployment (HERMES_HOME=/opt/data) listed files from a repo outside the
boundary but returned empty diffs with no error.
"""

import subprocess
from pathlib import Path

import pytest

pytest.importorskip("starlette.testclient")

from starlette.testclient import TestClient

from hermes_cli import web_server


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def locked_client(monkeypatch, tmp_path):
    """A test client whose managed-files root is locked to a specific directory."""
    root = tmp_path / "managed"
    root.mkdir()
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))

    prev_auth = getattr(web_server.app.state, "auth_required", None)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield client, root
    finally:
        if prev_auth is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = prev_auth
        if prev_host is None:
            if hasattr(web_server.app.state, "bound_host"):
                delattr(web_server.app.state, "bound_host")
        else:
            web_server.app.state.bound_host = prev_host


@pytest.fixture
def outside_repo(tmp_path):
    """A git repo at a path outside the managed root (sibling of tmp_path/managed)."""
    repo = tmp_path / "outside-repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    (repo / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    return repo


def test_review_list_honors_boundary_like_diff(locked_client, outside_repo):
    """review/list must refuse repos outside the managed root, just like review/diff."""
    client, _root = locked_client

    # diff correctly returns 403
    diff_resp = client.get(
        "/api/git/review/diff",
        params={"path": str(outside_repo), "file": "a.txt"},
    )
    assert diff_resp.status_code == 403, diff_resp.text

    # list MUST also return 403 — not silently list files
    list_resp = client.get(
        "/api/git/review/list",
        params={"path": str(outside_repo)},
    )
    assert list_resp.status_code == 403, (
        f"review/list returned {list_resp.status_code} for a repo outside the "
        f"managed root; expected 403. Body: {list_resp.text}"
    )


def test_review_list_works_inside_boundary(locked_client):
    """review/list must still work for repos inside the managed root."""
    client, root = locked_client

    repo = root / "my-repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    (repo / "a.txt").write_text("one\ntwo\n", encoding="utf-8")

    list_resp = client.get(
        "/api/git/review/list",
        params={"path": str(repo)},
    )
    assert list_resp.status_code == 200, list_resp.text
    files = list_resp.json().get("files", [])
    assert any(f["path"] == "a.txt" for f in files)


def test_review_list_no_lock_allows_any_repo(monkeypatch, tmp_path):
    """When no managed root is locked, review/list works for any valid repo."""
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)

    prev_auth = getattr(web_server.app.state, "auth_required", None)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

    try:
        repo = tmp_path / "free-repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "Test")
        (repo / "b.txt").write_text("hi\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "init")
        (repo / "b.txt").write_text("hello\n", encoding="utf-8")

        list_resp = client.get(
            "/api/git/review/list",
            params={"path": str(repo)},
        )
        assert list_resp.status_code == 200, list_resp.text
    finally:
        if prev_auth is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = prev_auth
        if prev_host is None:
            if hasattr(web_server.app.state, "bound_host"):
                delattr(web_server.app.state, "bound_host")
        else:
            web_server.app.state.bound_host = prev_host


def test_docker_opt_data_boundary_consistency(monkeypatch, tmp_path):
    """When locked by HERMES_HOME=/opt/data (hosted docker), list and diff must both return 403 for an outside repo."""
    opt_data = tmp_path / "opt_data"
    opt_data.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(opt_data))
    monkeypatch.setattr("hermes_cli.web_server_files._HOSTED_MANAGED_FILES_ROOT", opt_data)
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)

    prev_auth = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

    try:
        outside = tmp_path / "vault_repo"
        outside.mkdir()
        _git(outside, "init", "-q")
        _git(outside, "config", "user.email", "t@example.com")
        _git(outside, "config", "user.name", "Test")
        (outside / "file.txt").write_text("v1\n", encoding="utf-8")
        _git(outside, "add", "-A")
        _git(outside, "commit", "-qm", "init")
        (outside / "file.txt").write_text("v2\n", encoding="utf-8")

        diff_resp = client.get("/api/git/review/diff", params={"path": str(outside), "file": "file.txt"})
        assert diff_resp.status_code == 403

        list_resp = client.get("/api/git/review/list", params={"path": str(outside)})
        assert list_resp.status_code == 403
    finally:
        if prev_auth is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = prev_auth

