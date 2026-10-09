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

    @pytest.mark.parametrize("log_shape,b_outcome", [
        ("short", "success"), ("long", "success"), ("rotated", "success"), ("long", "failed"), ("long", None)])
    def test_restarted_dashboard_keeps_its_action_identity_past_the_log_tail(
            self, monkeypatch, tmp_path, log_shape, b_outcome):
        # The status route reads at most 2,000 log lines: build output (or rotation) pushing the
        # start header out must not turn B's own success into exit null ("Action failed (exit ?)"),
        # nor let profile A's newer latest receipt answer for B.
        from hermes_cli import update_receipt

        root, b_id, a_id = tmp_path / "root", "b" * 32, "a" * 32
        (root / "logs").mkdir(parents=True)
        b_logs = tmp_path / "b-logs"
        monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_DIR", b_logs)
        for registry in ("_ACTION_PROCS", "_ACTION_RESULTS", "_ACTION_COMMANDS", "_ACTION_IDS"):
            monkeypatch.setattr(_web_server_gateway, registry, {})
        monkeypatch.setattr(_web_server_gateway.subprocess, "Popen", lambda *a, **kw: types.SimpleNamespace(pid=7))
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "b"))
        _web_server_gateway._spawn_hermes_action(["update"], "hermes-update", env_overrides={"HERMES_ACTION_ID": b_id})
        for profile, action_id, outcome in (("b", b_id, b_outcome), ("a", a_id, "success")):
            if outcome:
                monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / profile))
                monkeypatch.setenv("HERMES_ACTION_ID", action_id)
                update_receipt.begin_update_receipt()
                update_receipt.finalize_update_receipt(outcome)
        monkeypatch.delenv("HERMES_ACTION_ID")
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "b"))
        log = b_logs / "hermes-update.log"
        if log_shape == "long":
            with log.open("a", encoding="utf-8") as handle:
                handle.write("build output\n" * 2001)
        elif log_shape == "rotated":
            log.unlink()
        if b_outcome == "success":
            (root / "logs" / "update.log").write_text(f"=== hermes-update completed {b_id} ===\n", encoding="utf-8")
        _web_server_gateway._ACTION_PROCS.clear()  # dashboard restarted: in-memory registries lost
        _web_server_gateway._ACTION_IDS.clear()

        data = self.client.get("/api/actions/hermes-update/status?lines=2000").json()

        assert data["running"] is False
        assert data["exit_code"] == (0 if b_outcome == "success" else 1 if b_outcome == "failed" else None)
        assert data.get("action_id") == (b_id if b_outcome == "success" else None)
        assert data["receipt"]["action_id"] == (b_id if b_outcome else a_id)

    @pytest.mark.parametrize("b_outcome", [None, "failed", "success"])
    def test_root_latest_receipt_of_another_profile_never_certifies_this_action(self, monkeypatch, tmp_path, b_outcome):
        # The receipt store is root-wide too: A's later success in latest.json must not become B's
        # exit 0. Only a receipt naming B's action (found in the archive when A is latest) counts.
        from hermes_cli import update_receipt

        root = tmp_path / "root"
        (root / "logs").mkdir(parents=True)
        b_id, a_id = "b" * 32, "a" * 32
        for profile, action_id, outcome in (("b", b_id, b_outcome), ("a", a_id, "success")):
            if outcome:
                monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / profile))
                monkeypatch.setenv("HERMES_ACTION_ID", action_id)
                update_receipt.begin_update_receipt()
                update_receipt.finalize_update_receipt(outcome)
        monkeypatch.delenv("HERMES_ACTION_ID")
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "b"))
        assert update_receipt.read_latest_receipt()["action_id"] == a_id
        (tmp_path / "hermes-update.log").write_text(
            f"=== hermes-update started 2026-08-17 11:19:34 {b_id} ===\n", encoding="utf-8")
        (root / "logs" / "update.log").write_text(f"=== hermes-update completed {a_id} ===\n", encoding="utf-8")
        monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_DIR", tmp_path)
        for registry in ("_ACTION_PROCS", "_ACTION_RESULTS", "_ACTION_COMMANDS", "_ACTION_IDS"):
            monkeypatch.setattr(_web_server_gateway, registry, {})

        data = self.client.get("/api/actions/hermes-update/status?lines=2000").json()

        assert data["exit_code"] == (
            0 if b_outcome == "success" else 1 if b_outcome in ("failed", "partial") else None
        )
        assert "action_id" not in data  # A's root completion marker is not B's
        # The attached receipt names its writer: B's own when it exists, else A's (never B's).
        assert data["receipt"]["action_id"] == (b_id if b_outcome else a_id)
        assert data["receipt"]["outcome"] == (b_outcome or "success")

    @pytest.mark.parametrize("debt", ["none", "followups", "user_action"])
    def test_status_receipt_carries_owed_steps_of_a_committed_update(self, monkeypatch, tmp_path, debt):
        # C3: owed post-commit steps keep the run a success (exit 0); the summary must still name
        # them so the dashboard and Desktop report the debt instead of plain success.
        from hermes_cli import update_receipt

        root, action_id = tmp_path / "root", "c" * 32
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
        monkeypatch.setenv("HERMES_ACTION_ID", action_id)
        update_receipt.begin_update_receipt()
        if debt == "followups":
            update_receipt.record_followup("dependencies", "synthetic selected-interpreter sync failed")
            update_receipt.record_followup("config_migration", "disk full")
        elif debt == "user_action":
            update_receipt.record_user_action("autostash", "re-apply the parked stash")
        update_receipt.finalize_update_receipt("success")
        (tmp_path / "hermes-update.log").write_text(
            f"=== hermes-update started 2026-08-17 11:19:34 {action_id} ===\n", encoding="utf-8")
        monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_DIR", tmp_path)
        for registry in ("_ACTION_PROCS", "_ACTION_RESULTS", "_ACTION_COMMANDS", "_ACTION_IDS"):
            monkeypatch.setattr(_web_server_gateway, registry, {})

        data = self.client.get("/api/actions/hermes-update/status?lines=2000").json()

        receipt = data["receipt"]
        assert [f["step"] for f in receipt["followups"]] == (
            ["dependencies", "config_migration"] if debt == "followups" else [])
        assert receipt["user_action"] == (
            {"step": "autostash", "reason": "re-apply the parked stash"} if debt == "user_action" else None)
        # Debt never changes the C3 exit mapping: owed follow-ups stay exit 0, a user action is partial.
        assert data["exit_code"] == (1 if debt == "user_action" else 0)

    @pytest.mark.parametrize(
        ("outcome", "expected_exit"),
        [("failed", 1), ("partial", 1), ("refused", 2), ("interrupted", 130)],
    )
    def test_reconnected_action_status_reports_terminal_receipt_failure(
        self, monkeypatch, tmp_path, outcome, expected_exit,
    ):
        from hermes_cli import update_receipt

        root, action_id = tmp_path / "root", "d" * 32
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
        monkeypatch.setenv("HERMES_ACTION_ID", action_id)
        update_receipt.begin_update_receipt()
        update_receipt.finalize_update_receipt(outcome)
        log_dir = tmp_path / "dashboard-logs"
        monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_DIR", log_dir)
        for registry in ("_ACTION_PROCS", "_ACTION_RESULTS", "_ACTION_COMMANDS", "_ACTION_IDS"):
            monkeypatch.setattr(_web_server_gateway, registry, {})
        monkeypatch.setattr(
            _web_server_gateway.subprocess, "Popen", lambda *args, **kwargs: types.SimpleNamespace(pid=7)
        )
        _web_server_gateway._spawn_hermes_action(
            ["update"], "hermes-update", env_overrides={"HERMES_ACTION_ID": action_id}
        )
        _web_server_gateway._ACTION_PROCS.clear()  # backend restarted; only the receipt remains

        data = self.client.get("/api/actions/hermes-update/status?lines=2000").json()

        assert data["running"] is False
        assert data["exit_code"] == expected_exit
        assert data["receipt"]["outcome"] == outcome

    def test_running_receipt_is_not_terminal_status(self, monkeypatch, tmp_path):
        from hermes_cli import update_receipt

        root, action_id = tmp_path / "root", "e" * 32
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
        monkeypatch.setenv("HERMES_ACTION_ID", action_id)
        update_receipt.begin_update_receipt()
        log_dir = tmp_path / "dashboard-logs"
        monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_DIR", log_dir)
        for registry in ("_ACTION_PROCS", "_ACTION_RESULTS", "_ACTION_COMMANDS", "_ACTION_IDS"):
            monkeypatch.setattr(_web_server_gateway, registry, {})
        monkeypatch.setattr(
            _web_server_gateway.subprocess, "Popen", lambda *args, **kwargs: types.SimpleNamespace(pid=7)
        )
        _web_server_gateway._spawn_hermes_action(
            ["update"], "hermes-update", env_overrides={"HERMES_ACTION_ID": action_id}
        )
        _web_server_gateway._ACTION_PROCS.clear()

        data = self.client.get("/api/actions/hermes-update/status?lines=2000").json()

        assert data["running"] is False
        assert data["exit_code"] is None
        assert data["receipt"]["outcome"] == "running"

    def test_action_without_receipt_recovers_failure_marker_after_backend_restart(
        self, monkeypatch, tmp_path,
    ):
        root, action_id = tmp_path / "root", "f" * 32
        (root / "logs").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
        log_dir = tmp_path / "dashboard-logs"
        monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_DIR", log_dir)
        for registry in ("_ACTION_PROCS", "_ACTION_RESULTS", "_ACTION_COMMANDS", "_ACTION_IDS"):
            monkeypatch.setattr(_web_server_gateway, registry, {})
        monkeypatch.setattr(
            _web_server_gateway.subprocess, "Popen", lambda *args, **kwargs: types.SimpleNamespace(pid=7)
        )
        _web_server_gateway._spawn_hermes_action(
            ["update"], "hermes-update", env_overrides={"HERMES_ACTION_ID": action_id}
        )
        _web_server_gateway._ACTION_PROCS.clear()
        (root / "logs" / "update.log").write_text(
            "=== hermes update started 2026-10-08T10:00:00 ===\n"
            f"=== hermes-update failed {action_id} exit=3 ===\n",
            encoding="utf-8",
        )

        data = self.client.get("/api/actions/hermes-update/status?lines=2000").json()

        assert data["running"] is False
        assert data["exit_code"] == 3
        assert "action_id" not in data
