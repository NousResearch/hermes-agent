"""Dashboard update status reads the ROOT home's shared update.log, correlated by action id."""

import types

import pytest

import hermes_cli.web_server_gateway as _web_server_gateway


class TestUpdateStatusRootLog:
    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
            """Create a TestClient and isolate the state DB under the test HERMES_HOME."""
            try:
                from starlette.testclient import TestClient
            except ImportError:
                pytest.skip("fastapi/starlette not installed")

            import hermes_state
            from hermes_constants import get_hermes_home
            from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

            monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

            self.client = TestClient(app)
            self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def test_update_status_recovers_completed_result_after_dashboard_restart(self, monkeypatch, tmp_path):
        # The dashboard runs under a profile home; ``hermes update`` mirrors to the ROOT home's
        # update.log (main_dashboard), so the durable completion marker is read there.
        root = tmp_path / "root"
        (root / "logs").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
        action_id = "c" * 32
        (tmp_path / "hermes-update.log").write_text(
            f"=== hermes-update started 2026-08-17 11:19:34 {action_id} ===\n"
            "pulling updates...\n",
            encoding="utf-8",
        )
        (root / "logs" / "update.log").write_text(
            "=== hermes update started 2026-08-17T11:19:35 ===\n"
            "✓ Update complete!\n"
            f"=== hermes-update completed {action_id} ===\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_DIR", tmp_path)
        monkeypatch.setattr(_web_server_gateway, "_ACTION_PROCS", {})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_RESULTS", {})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_COMMANDS", {})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_IDS", {})

        status = self.client.get("/api/actions/hermes-update/status?lines=2000")

        assert status.status_code == 200
        data = status.json()
        assert data["running"] is False
        assert data["exit_code"] == 0
        assert data["action_id"] == action_id
        assert f"=== hermes-update completed {action_id} ===" in data["lines"]


    def test_update_status_ignores_another_profiles_completion_in_the_shared_root_log(self, monkeypatch, tmp_path):
        # Every profile's ``hermes update`` mirrors into the ROOT update.log; profile A's success
        # there must not certify the action profile B's dashboard started (and lost on restart).
        root = tmp_path / "root"
        (root / "logs").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "b"))
        b_logs = tmp_path / "b-logs"
        monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_DIR", b_logs)
        for registry in ("_ACTION_PROCS", "_ACTION_RESULTS", "_ACTION_COMMANDS", "_ACTION_IDS"):
            monkeypatch.setattr(_web_server_gateway, registry, {})
        monkeypatch.setattr(_web_server_gateway.subprocess, "Popen", lambda *a, **kw: types.SimpleNamespace(pid=7))
        b_id = "b" * 32
        _web_server_gateway._spawn_hermes_action(["update"], "hermes-update", env_overrides={"HERMES_ACTION_ID": b_id})
        _web_server_gateway._ACTION_PROCS.clear()  # B's dashboard restarted: in-memory result lost

        def status_after_root_completion(action_id):
            (root / "logs" / "update.log").write_text(
                "=== hermes update started 2026-08-17T11:19:35 ===\n"
                f"=== hermes-update completed {action_id} ===\n", encoding="utf-8")
            return self.client.get("/api/actions/hermes-update/status?lines=2000").json()

        foreign = status_after_root_completion("a" * 32)
        assert foreign["exit_code"] is None
        assert "action_id" not in foreign
        own = status_after_root_completion(b_id)
        assert own["exit_code"] == 0
        assert own["action_id"] == b_id
