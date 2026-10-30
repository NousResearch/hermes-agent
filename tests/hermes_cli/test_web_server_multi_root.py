"""Tests for multi-root support in HERMES_DASHBOARD_FILES_ROOT (Issue #146744)."""

import os
import subprocess
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli import web_server_files


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _init_repo(repo: Path) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "a.txt").write_text("v1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    (repo / "a.txt").write_text("v2\n", encoding="utf-8")
    return repo


@pytest.fixture
def test_client():
    prev_auth = getattr(web_server.app.state, "auth_required", None)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield client
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


def test_multi_root_empty_value_is_unlocked(monkeypatch):
    """An empty or unset HERMES_DASHBOARD_FILES_ROOT leaves the file browser unlocked."""
    for empty_val in ("", "   "):
        monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", empty_val)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        policy = web_server_files._managed_files_policy(None, create_root=False)
        assert policy.locked_root is None
        assert policy.locked_roots is None
        assert policy.can_change_path is True


def test_multi_root_single_path_unchanged(monkeypatch, tmp_path):
    """A single path behaves exactly as before."""
    root = tmp_path / "single_root"
    root.mkdir()
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))

    policy = web_server_files._managed_files_policy(None, create_root=False)
    assert policy.locked_root == root
    assert policy.locked_roots == [root]
    assert policy.default_path == root
    assert policy.can_change_path is False

    inside = root / "file.txt"
    outside = tmp_path / "outside.txt"
    assert web_server_files._hosted_fs_path_allowed(policy.locked_root, inside) is True
    assert web_server_files._hosted_fs_path_allowed(policy.locked_root, outside) is False


def test_multi_root_two_valid_roots(monkeypatch, tmp_path, test_client):
    """Multiple roots separated by os.pathsep are parsed and both allowed."""
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    root1.mkdir()
    root2.mkdir()
    roots_str = f"{root1}{os.pathsep}{root2}"
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", roots_str)

    policy = web_server_files._managed_files_policy(None, create_root=False)
    assert policy.locked_root == [root1, root2]
    assert policy.locked_roots == [root1, root2]
    assert policy.default_path == root1
    assert policy.can_change_path is False

    # Path containment check under either root
    inside1 = root1 / "file1.txt"
    inside2 = root2 / "file2.txt"
    outside = tmp_path / "outside.txt"
    assert web_server_files._hosted_fs_path_allowed(policy.locked_root, inside1) is True
    assert web_server_files._hosted_fs_path_allowed(policy.locked_root, inside2) is True
    assert web_server_files._hosted_fs_path_allowed(policy.locked_root, outside) is False


def test_multi_root_path_under_second_root(monkeypatch, tmp_path, test_client):
    """A git repo located under the second root is allowed by review routes."""
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    repo2 = _init_repo(root2 / "project2")
    roots_str = f"{root1}{os.pathsep}{root2}"
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", roots_str)

    list_resp = test_client.get("/api/git/review/list", params={"path": str(repo2)})
    assert list_resp.status_code == 200, list_resp.text
    files = list_resp.json().get("files", [])
    assert any(f["path"] == "a.txt" for f in files)

    diff_resp = test_client.get("/api/git/review/diff", params={"path": str(repo2), "file": "a.txt"})
    assert diff_resp.status_code == 200, diff_resp.text
    assert "diff" in diff_resp.json()

    ctx_resp = test_client.get("/api/git/review/commit-context", params={"path": str(repo2)})
    assert ctx_resp.status_code == 200, ctx_resp.text


def test_multi_root_path_outside_all_roots(monkeypatch, tmp_path, test_client):
    """A repo outside all configured roots is denied (403) across review routes."""
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    root1.mkdir()
    root2.mkdir()
    outside_repo = _init_repo(tmp_path / "outside_project")
    roots_str = f"{root1}{os.pathsep}{root2}"
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", roots_str)

    list_resp = test_client.get("/api/git/review/list", params={"path": str(outside_repo)})
    assert list_resp.status_code == 403, list_resp.text

    diff_resp = test_client.get("/api/git/review/diff", params={"path": str(outside_repo), "file": "a.txt"})
    assert diff_resp.status_code == 403, diff_resp.text

    ctx_resp = test_client.get("/api/git/review/commit-context", params={"path": str(outside_repo)})
    assert ctx_resp.status_code == 403, ctx_resp.text


@pytest.mark.parametrize(
    "bad_value",
    [
        os.pathsep,
        f"{os.pathsep}something",
        f"something{os.pathsep}",
        f"foo{os.pathsep}{os.pathsep}bar",
        "relative/path",
        f"relative/a{os.pathsep}relative/b",
    ],
)
def test_multi_root_malformed_values_raise_error(monkeypatch, tmp_path, bad_value):
    """Malformed values (empty segments, relative paths) raise a validation error."""
    # Prefix relative paths with an absolute one for mixed cases
    if "something" in bad_value:
        bad_value = bad_value.replace("something", str(tmp_path))
    if "foo" in bad_value:
        bad_value = bad_value.replace("foo", str(tmp_path)).replace("bar", str(tmp_path / "bar"))

    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", bad_value)
    with pytest.raises(ValueError):
        web_server_files._managed_files_policy(None, create_root=False)


def test_multi_root_file_instead_of_directory_raises_error(monkeypatch, tmp_path):
    """An existing file passed as a root raises a validation error."""
    file_path = tmp_path / "some_file.txt"
    file_path.write_text("hello", encoding="utf-8")
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(file_path))
    with pytest.raises(ValueError):
        web_server_files._managed_files_policy(None, create_root=False)

