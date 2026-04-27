import sys
import pytest
from pathlib import Path
from hermes_cli.code.command_runner import CommandRunnerService, CommandSafety


class TestCommandRunnerService:
    @pytest.fixture()
    def wdb(self, tmp_path):
        from hermes_state import WorkspaceDB

        d = WorkspaceDB(db_path=tmp_path / "state.db")
        yield d
        d.close()

    @pytest.fixture()
    def sdb(self, tmp_path):
        from hermes_state import CodeSessionDB

        d = CodeSessionDB(db_path=tmp_path / "state.db")
        yield d
        d.close()

    @pytest.fixture()
    def runner(self, tmp_path):
        return CommandRunnerService(db_path=tmp_path / "state.db")

    @pytest.fixture()
    def session(self, wdb, sdb, tmp_path):
        ws_path = tmp_path / "myworkspace"
        ws_path.mkdir(exist_ok=True)
        ws = wdb.upsert_workspace(
            path=str(ws_path),
            name="myworkspace",
            is_git_repo=True,
            branch="main",
            detected_stack=["python"],
        )
        session = sdb.create_session(workspace_id=ws["id"])
        return session

    def test_classify_safe(self, runner):
        assert runner.classify_command("git status") == CommandSafety.SAFE
        assert runner.classify_command("npm run build") == CommandSafety.SAFE
        assert runner.classify_command("python -c 'print(1)'") == CommandSafety.SAFE

    def test_classify_needs_approval(self, runner):
        assert (
            runner.classify_command("git push origin main")
            == CommandSafety.NEEDS_APPROVAL
        )
        assert (
            runner.classify_command("npm install foo") == CommandSafety.NEEDS_APPROVAL
        )
        assert (
            runner.classify_command("pip install requests")
            == CommandSafety.NEEDS_APPROVAL
        )

    def test_classify_blocked(self, runner):
        assert runner.classify_command("sudo apt update") == CommandSafety.BLOCKED
        assert runner.classify_command("rm -rf /") == CommandSafety.BLOCKED
        assert (
            runner.classify_command("npm run build; rm -rf /") == CommandSafety.BLOCKED
        )

    def test_create_and_run_safe_command(self, runner, session):
        # A simple command that always works
        cmd = runner.create_command(
            code_session_id=session["id"],
            command=f"{sys.executable} -c 'print(\"hello runner\")'",
            timeout_seconds=5,
        )
        assert cmd["status"] == "pending"
        assert cmd["safety"] == CommandSafety.SAFE

        updated = runner.run_command_sync(cmd["id"])
        assert updated["status"] == "completed"
        assert updated["exit_code"] == 0
        assert "hello runner" in updated["stdout"]

        events = runner._session_db().list_events(session["id"])
        event_types = [e["type"] for e in events]
        assert "command.completed" in event_types

    def test_run_command_failed(self, runner, session):
        cmd = runner.create_command(
            code_session_id=session["id"],
            command=f"{sys.executable} -c 'raise SystemExit(3)'",
            timeout_seconds=5,
        )
        updated = runner.run_command_sync(cmd["id"])
        assert updated["status"] == "failed"
        assert updated["exit_code"] == 3

    def test_run_command_timeout(self, runner, session):
        cmd = runner.create_command(
            code_session_id=session["id"],
            command=f"{sys.executable} -c 'while True: pass'",
            timeout_seconds=1,
        )
        updated = runner.run_command_sync(cmd["id"])
        assert updated["status"] == "timeout"
        assert updated["completed_at"] is not None

    def test_blocked_command_is_not_run(self, runner, session):
        cmd = runner.create_command(
            code_session_id=session["id"], command="rm -rf /tmp/foo", timeout_seconds=5
        )
        assert cmd["status"] == "blocked"
        updated = runner.run_command_sync(cmd["id"])
        assert updated["status"] == "blocked"
        assert "blocked by safety policy" in updated["stderr"]

    def test_cwd_outside_workspace(self, runner, session):
        with pytest.raises(ValueError, match="outside the workspace"):
            runner.create_command(
                code_session_id=session["id"], command="ls", cwd="/tmp"
            )

    def test_cancel_command(self, runner, session):
        cmd = runner.create_command(
            code_session_id=session["id"],
            command="ls",
        )
        cancelled = runner.cancel_command(cmd["id"])
        assert cancelled["status"] == "cancelled"


from fastapi.testclient import TestClient
from hermes_cli.web_server import app, _SESSION_TOKEN


class TestCommandRunnerEndpoints:
    @pytest.fixture()
    def client(self):
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        # Point hermes state to temp dir
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
        import hermes_state

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
        from hermes_state import WorkspaceDB, CodeSessionDB

        WorkspaceDB(db_path=tmp_path / "state.db")._init_schema()
        CodeSessionDB(db_path=tmp_path / "state.db")._init_schema()

    @pytest.fixture()
    def session_id(self, tmp_path):
        from hermes_state import WorkspaceDB, CodeSessionDB

        db_path = tmp_path / "state.db"
        wdb = WorkspaceDB(db_path=db_path)
        ws_path = tmp_path / "myworkspace"
        ws_path.mkdir(exist_ok=True)
        ws = wdb.upsert_workspace(
            path=str(ws_path),
            name="myworkspace",
            is_git_repo=True,
            branch="main",
            detected_stack=["python"],
        )
        sdb = CodeSessionDB(db_path=db_path)
        session = sdb.create_session(workspace_id=ws["id"])
        wdb.close()
        sdb.close()
        return session["id"]

    def test_list_commands_empty(self, client, session_id):
        resp = client.get(
            f"/api/code/sessions/{session_id}/commands",
            headers={"Authorization": f"Bearer {_SESSION_TOKEN}"},
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_run_command_safe(self, client, session_id):
        resp = client.post(
            f"/api/code/sessions/{session_id}/commands/run",
            headers={"Authorization": f"Bearer {_SESSION_TOKEN}"},
            json={"command": f"{sys.executable} -c 'print(1)'", "timeout_seconds": 5},
        )
        assert resp.status_code == 200
        cmd = resp.json()["command"]
        assert cmd["status"] == "completed"
        assert cmd["exit_code"] == 0
        assert "1" in cmd["stdout"]

    def test_get_command(self, client, session_id):
        resp1 = client.post(
            f"/api/code/sessions/{session_id}/commands/run",
            headers={"Authorization": f"Bearer {_SESSION_TOKEN}"},
            json={"command": f"{sys.executable} -c 'print(1)'", "timeout_seconds": 5},
        )
        cmd_id = resp1.json()["command"]["id"]

        resp2 = client.get(
            f"/api/code/commands/{cmd_id}",
            headers={"Authorization": f"Bearer {_SESSION_TOKEN}"},
        )
        assert resp2.status_code == 200
        assert resp2.json()["command"]["id"] == cmd_id

    def test_cancel_command(self, client, session_id):
        resp1 = client.post(
            f"/api/code/sessions/{session_id}/commands/run",
            headers={"Authorization": f"Bearer {_SESSION_TOKEN}"},
            json={"command": f"{sys.executable} -c 'print(1)'", "timeout_seconds": 5},
        )
        cmd_id = resp1.json()["command"]["id"]

        resp2 = client.post(
            f"/api/code/commands/{cmd_id}/cancel",
            headers={"Authorization": f"Bearer {_SESSION_TOKEN}"},
        )
        assert resp2.status_code == 200
        # Given it's sync and finished, it will be returned as completed, and "Command is not running" message
        assert resp2.json()["message"] == "Command is not running"
