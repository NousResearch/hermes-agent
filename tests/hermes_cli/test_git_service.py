"""Tests for Phase 5 Hermes Code Mode: GitService."""

import json
import pytest
import subprocess
from pathlib import Path


class TestGitService:
    @pytest.fixture()
    def tmp_db(self, tmp_path):
        return tmp_path / "state.db"

    @pytest.fixture()
    def wdb(self, tmp_db):
        from hermes_state import WorkspaceDB

        d = WorkspaceDB(db_path=tmp_db)
        yield d
        d.close()

    @pytest.fixture()
    def sdb(self, tmp_db):
        from hermes_state import CodeSessionDB

        d = CodeSessionDB(db_path=tmp_db)
        yield d
        d.close()

    @pytest.fixture()
    def gdb(self, tmp_db):
        from hermes_state import GitSnapshotDB

        d = GitSnapshotDB(db_path=tmp_db)
        yield d
        d.close()

    @pytest.fixture()
    def git_service(self, tmp_db):
        from hermes_cli.code.git_service import GitService

        return GitService(db_path=tmp_db)

    def _init_git_repo(self, path: Path):
        subprocess.run(["git", "init"], cwd=str(path), check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=str(path),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=str(path),
            check=True,
            capture_output=True,
        )

    def _commit_file(
        self, path: Path, filename: str, content: str, message: str = "initial commit"
    ):
        (path / filename).write_text(content)
        subprocess.run(
            ["git", "add", filename], cwd=str(path), check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(path),
            check=True,
            capture_output=True,
        )

    # ------------------------------------------------------------------
    # 1. Not a Git repo
    # ------------------------------------------------------------------

    def test_get_status_not_git_repo(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "not_git"
        ws_path.mkdir()
        ws = wdb.upsert_workspace(path=str(ws_path), name="not_git")
        status = git_service.get_status(ws["id"])
        assert status["is_git_repo"] is False
        assert status["branch"] is None

    def test_get_branch_not_git_repo(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "not_git"
        ws_path.mkdir()
        ws = wdb.upsert_workspace(path=str(ws_path), name="not_git")
        with pytest.raises(ValueError, match="not a Git repository"):
            git_service.get_branch(ws["id"])

    # ------------------------------------------------------------------
    # 2. Branch detection
    # ------------------------------------------------------------------

    def test_get_branch_detected(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        result = git_service.get_branch(ws["id"])
        assert result["branch"] in ("master", "main")

    # ------------------------------------------------------------------
    # 3. Remote origin
    # ------------------------------------------------------------------

    def test_get_remote_detected(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/example/repo.git"],
            cwd=str(ws_path),
            check=True,
            capture_output=True,
        )
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        result = git_service.get_remote(ws["id"])
        assert result["remote_url"] == "https://github.com/example/repo.git"

    # ------------------------------------------------------------------
    # 4. Clean status
    # ------------------------------------------------------------------

    def test_get_status_clean(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        status = git_service.get_status(ws["id"])
        assert status["is_git_repo"] is True
        assert status["dirty"] is False
        assert status["files"] == []
        assert status["summary"]["modified"] == 0
        assert status["summary"]["untracked"] == 0

    # ------------------------------------------------------------------
    # 5. Modified file
    # ------------------------------------------------------------------

    def test_get_status_modified(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        (ws_path / "README.md").write_text("# hello world")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        status = git_service.get_status(ws["id"])
        assert status["dirty"] is True
        assert len(status["files"]) == 1
        assert status["files"][0]["status"] == "modified"
        assert status["files"][0]["path"] == "README.md"
        assert status["summary"]["modified"] == 1

    # ------------------------------------------------------------------
    # 6. Untracked file
    # ------------------------------------------------------------------

    def test_get_status_untracked(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        (ws_path / "new.py").write_text("print(1)")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        status = git_service.get_status(ws["id"])
        assert status["dirty"] is True
        assert any(
            f["status"] == "untracked" and f["path"] == "new.py"
            for f in status["files"]
        )
        assert status["summary"]["untracked"] == 1

    # ------------------------------------------------------------------
    # 7. Deleted file
    # ------------------------------------------------------------------

    def test_get_status_deleted(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        (ws_path / "README.md").unlink()
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        status = git_service.get_status(ws["id"])
        assert status["dirty"] is True
        assert any(
            f["status"] == "deleted" and f["path"] == "README.md"
            for f in status["files"]
        )
        assert status["summary"]["deleted"] == 1

    # ------------------------------------------------------------------
    # 8. Diff general
    # ------------------------------------------------------------------

    def test_get_diff_general(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        (ws_path / "README.md").write_text("# hello world\nmore lines\n")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        diff = git_service.get_diff(ws["id"])
        assert "hello world" in diff["diff"]
        assert diff["additions"] > 0

    # ------------------------------------------------------------------
    # 9. Diff per file
    # ------------------------------------------------------------------

    def test_get_diff_per_file(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "a.txt", "a")
        self._commit_file(ws_path, "b.txt", "b", message="add b")
        (ws_path / "a.txt").write_text("a modified")
        (ws_path / "b.txt").write_text("b modified")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        diff_a = git_service.get_diff(ws["id"], path="a.txt")
        diff_b = git_service.get_diff(ws["id"], path="b.txt")
        assert "a modified" in diff_a["diff"]
        assert "b modified" in diff_b["diff"]
        # diff_a should not contain b.txt changes
        assert "b modified" not in diff_a["diff"]

    # ------------------------------------------------------------------
    # 10. Snapshot
    # ------------------------------------------------------------------

    def test_create_snapshot(self, git_service, wdb, gdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        (ws_path / "new.py").write_text("print(1)")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        snapshot = git_service.create_snapshot(ws["id"])
        assert snapshot["workspace_id"] == ws["id"]
        assert snapshot["dirty"] is True
        assert snapshot["summary"]["untracked"] == 1
        assert len(snapshot["files"]) == 1

        # Verify persisted
        persisted = gdb.get_snapshot(snapshot["id"])
        assert persisted is not None
        assert persisted["dirty"] is True

    # ------------------------------------------------------------------
    # 11. Prepare branch clean -> safe
    # ------------------------------------------------------------------

    def test_prepare_branch_clean(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        result = git_service.prepare_branch(ws["id"], "feature/test")
        assert result["safety"] == "safe"
        assert result["branch_name"] == "feature/test"

    # ------------------------------------------------------------------
    # 12. Prepare branch dirty -> needs_approval
    # ------------------------------------------------------------------

    def test_prepare_branch_dirty(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        (ws_path / "new.py").write_text("print(1)")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        result = git_service.prepare_branch(ws["id"], "feature/test")
        assert result["safety"] == "needs_approval"
        assert "uncommitted" in result["reason"].lower()

    # ------------------------------------------------------------------
    # 13. Create branch clean -> creates branch
    # ------------------------------------------------------------------

    def test_create_branch_clean(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        result = git_service.create_branch(ws["id"], "feature/test")
        assert result["result"]["safety"] == "safe"
        assert result["result"]["executed"] is True
        branch = git_service.get_branch(ws["id"])
        assert branch["branch"] == "feature/test"

    # ------------------------------------------------------------------
    # 14. Create branch dirty -> does not execute
    # ------------------------------------------------------------------

    def test_create_branch_dirty(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        (ws_path / "new.py").write_text("print(1)")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        result = git_service.create_branch(ws["id"], "feature/test")
        assert result["result"]["safety"] == "needs_approval"
        assert result["result"]["executed"] is False
        branch = git_service.get_branch(ws["id"])
        assert branch["branch"] in ("master", "main")

    # ------------------------------------------------------------------
    # 15. Invalid branch name
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "bad_name",
        [
            "-bad",
            "bad..name",
            "bad name",
            "bad\\name",
            "bad~name",
            "bad^name",
            "bad:name",
        ],
    )
    def test_prepare_branch_invalid(self, git_service, wdb, tmp_path, bad_name):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        result = git_service.prepare_branch(ws["id"], bad_name)
        assert result["safety"] == "blocked"

    # ------------------------------------------------------------------
    # 16. Prepare commit
    # ------------------------------------------------------------------

    def test_prepare_commit(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        (ws_path / "new.py").write_text("print(1)")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        result = git_service.prepare_commit(ws["id"], "feat: add new.py")
        assert result["safety"] == "needs_approval"
        assert result["executed"] is False
        assert result["message"] == "feat: add new.py"
        assert len(result["files"]) == 1
        assert "diff_stat" in result

    # ------------------------------------------------------------------
    # 17. Timeline events
    # ------------------------------------------------------------------

    def test_timeline_events(self, git_service, wdb, sdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "README.md", "# hello")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        session = sdb.create_session(workspace_id=ws["id"])

        git_service.get_status(ws["id"], code_session_id=session["id"])
        git_service.create_snapshot(ws["id"], code_session_id=session["id"])
        git_service.prepare_branch(ws["id"], "feature/timeline")
        git_service.create_branch(
            ws["id"], "feature/timeline", code_session_id=session["id"]
        )
        git_service.prepare_commit(ws["id"], "test", code_session_id=session["id"])

        events = sdb.list_events(session["id"])
        event_types = [e["type"] for e in events]
        assert "git.status_checked" in event_types
        assert "git.snapshot.created" in event_types
        assert "git.branch.created" in event_types
        assert "git.commit.prepared" in event_types

    # ------------------------------------------------------------------
    # 18. List files
    # ------------------------------------------------------------------

    def test_list_files(self, git_service, wdb, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_git_repo(ws_path)
        self._commit_file(ws_path, "a.txt", "a")
        self._commit_file(ws_path, "b.txt", "b", message="add b")
        (ws_path / "a.txt").write_text("a modified")
        (ws_path / "c.txt").write_text("c new")
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        files = git_service.list_files(ws["id"])
        assert len(files) == 2
        statuses = {f["path"]: f["status"] for f in files}
        assert statuses["a.txt"] == "modified"
        assert statuses["c.txt"] == "untracked"


# =============================================================================
# HTTP endpoint tests
# =============================================================================


class TestGitEndpoints:
    @pytest.fixture()
    def client(self, tmp_path):
        from hermes_cli.web_server import app
        from fastapi.testclient import TestClient
        from hermes_state import WorkspaceDB, CodeSessionDB

        # Override DB path via monkeypatching the service imports is hard;
        # instead we use the default DB and isolate via tmp env.
        import os

        old_home = os.environ.get("HERMES_HOME")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        os.environ["HERMES_HOME"] = str(hermes_home)

        # Reset any cached default db path by forcing new connections
        db_path = hermes_home / "state.db"
        wdb = WorkspaceDB(db_path=db_path)
        sdb = CodeSessionDB(db_path=db_path)
        wdb.close()
        sdb.close()

        # Patch services to use this db path by default? The endpoints instantiate
        # GitService() without args, so they use default DB path. We need to make
        # default path resolve to our temp db.
        import hermes_state

        orig_default = hermes_state.DEFAULT_DB_PATH
        hermes_state.DEFAULT_DB_PATH = db_path

        from hermes_cli.web_server import _SESSION_TOKEN

        test_client = TestClient(app)
        test_client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"
        yield test_client

        hermes_state.DEFAULT_DB_PATH = orig_default
        if old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = old_home

    def _init_repo(self, path: Path):
        subprocess.run(["git", "init"], cwd=str(path), check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=str(path),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=str(path),
            check=True,
            capture_output=True,
        )
        (path / "README.md").write_text("# hello")
        subprocess.run(
            ["git", "add", "README.md"], cwd=str(path), check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(path),
            check=True,
            capture_output=True,
        )

    def test_http_git_status(self, client, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_repo(ws_path)
        # Need workspace in DB
        from hermes_state import WorkspaceDB

        wdb = WorkspaceDB()
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        wdb.close()

        resp = client.get(f"/api/code/workspaces/{ws['id']}/git/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"]["is_git_repo"] is True

    def test_http_git_branch(self, client, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_repo(ws_path)
        from hermes_state import WorkspaceDB

        wdb = WorkspaceDB()
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        wdb.close()

        resp = client.get(f"/api/code/workspaces/{ws['id']}/git/branch")
        assert resp.status_code == 200
        data = resp.json()
        assert data["branch"] in ("master", "main")

    def test_http_git_diff(self, client, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_repo(ws_path)
        (ws_path / "README.md").write_text("# hello world")
        from hermes_state import WorkspaceDB

        wdb = WorkspaceDB()
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        wdb.close()

        resp = client.get(f"/api/code/workspaces/{ws['id']}/git/diff")
        assert resp.status_code == 200
        data = resp.json()
        assert "hello world" in data["diff"]["diff"]

    def test_http_git_snapshot(self, client, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_repo(ws_path)
        (ws_path / "new.py").write_text("print(1)")
        from hermes_state import WorkspaceDB

        wdb = WorkspaceDB()
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        wdb.close()

        resp = client.post(f"/api/code/workspaces/{ws['id']}/git/snapshot", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["snapshot"]["dirty"] is True

    def test_http_git_prepare_branch(self, client, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_repo(ws_path)
        from hermes_state import WorkspaceDB

        wdb = WorkspaceDB()
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        wdb.close()

        resp = client.post(
            f"/api/code/workspaces/{ws['id']}/git/branch/prepare",
            json={"branch_name": "feature/test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety"] == "safe"

    def test_http_git_create_branch(self, client, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_repo(ws_path)
        from hermes_state import WorkspaceDB

        wdb = WorkspaceDB()
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        wdb.close()

        resp = client.post(
            f"/api/code/workspaces/{ws['id']}/git/branch",
            json={"branch_name": "feature/new"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["executed"] is True

    def test_http_git_prepare_commit(self, client, tmp_path):
        ws_path = tmp_path / "repo"
        ws_path.mkdir()
        self._init_repo(ws_path)
        (ws_path / "new.py").write_text("print(1)")
        from hermes_state import WorkspaceDB

        wdb = WorkspaceDB()
        ws = wdb.upsert_workspace(path=str(ws_path), name="repo")
        wdb.close()

        resp = client.post(
            f"/api/code/workspaces/{ws['id']}/git/commit/prepare",
            json={"message": "feat: add new.py"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety"] == "needs_approval"
        assert data["executed"] is False
