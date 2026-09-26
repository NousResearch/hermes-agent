"""Invariant tests for #107427: gateway /update must survive its own restart.

Half A — spawner escapes the gateway cgroup; half B — the watcher does not report
success from the pre-restart ``.update_exit_code`` write alone.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Half A — cgroup escape in _spawn_detached_update
# ---------------------------------------------------------------------------


class TestSpawnDetachedUpdateCgroupEscape:
    """_spawn_detached_update must escape the service cgroup when supervised."""

    @pytest.mark.platforms("linux")
    def test_escapes_with_systemd_run_when_supervised_and_probe_ok(self, tmp_path, monkeypatch):
        """Supervised + probe OK => argv wrapped in systemd-run, env carries the bus."""
        from types import SimpleNamespace

        import gateway.slash_commands as sc
        run_id = "run-107427"

        captured: dict = {}

        class FakePopen:
            def __init__(self, *a, **kw):
                captured["argv"] = a[0] if a else kw.get("args")
                captured["env"] = kw.get("env")
                captured["start_new_session"] = kw.get("start_new_session")
                self.pid = 9999

        monkeypatch.setattr("hermes_platform.resolver.locate_command", lambda name: SimpleNamespace(
            command=("/usr/bin/systemd-run",) if name == "systemd-run" else ()))
        monkeypatch.setattr("tools.process_registry._systemd_run_user_scope_available", lambda: True)
        monkeypatch.setattr("tools.process_registry.systemd_user_bus_env", lambda e=None: {**(e or {}), "DBUS_SESSION_BUS_ADDRESS": "unix:path=/fake/bus", "XDG_RUNTIME_DIR": "/run/user/1000"})
        monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setenv("INVOCATION_ID", "fake-invocation")

        sc._spawn_detached_update(
            ["hermes"], tmp_path / "out.txt", tmp_path / ".update_exit_code", run_id)

        assert captured["argv"][0] == "/usr/bin/systemd-run"
        assert "--scope" in captured["argv"]
        assert "--collect" in captured["argv"]
        unit_index = captured["argv"].index("--unit")
        unit = captured["argv"][unit_index + 1]
        assert unit.startswith("hermes-gateway-update-") and unit.endswith(".scope")
        assert any(f"--update-id={run_id}" in str(part) for part in captured["argv"])
        assert captured["env"]["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/fake/bus"
        assert captured["start_new_session"] is True

    @pytest.mark.platforms("posix")
    def test_falls_back_to_plain_setsid_when_probe_fails(self, tmp_path, monkeypatch):
        """Unsupervised or probe false => plain setsid spawn, no env override."""
        from types import SimpleNamespace

        import gateway.slash_commands as sc

        captured: dict = {}

        class FakePopen:
            def __init__(self, *a, **kw):
                captured["argv"] = a[0] if a else kw.get("args")
                captured["env"] = kw.get("env")
                self.pid = 9999

        monkeypatch.delenv("INVOCATION_ID", raising=False)
        monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: False)
        monkeypatch.setattr("tools.process_registry._systemd_run_user_scope_available", lambda: False)
        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setattr("hermes_platform.resolver.locate_command", lambda name: SimpleNamespace(
            command=("/usr/bin/setsid",) if name == "setsid" else ()))

        sc._spawn_detached_update(
            ["hermes"], tmp_path / "out.txt", tmp_path / ".update_exit_code", "run-107427")

        assert captured["argv"][0] != "systemd-run"
        assert captured["env"] is None


# ---------------------------------------------------------------------------
# Half B — finalized evidence gate in run_notifications
# ---------------------------------------------------------------------------


class TestGatewayUpdateFinalizedGate:
    """Only the invoked run's terminal receipt can authorize a successful /update notice."""

    def _paths(self, tmp_path, update_id="run-107427"):
        from gateway.run_notifications import GatewayNotificationsMixin

        (tmp_path / "logs" / "update_receipts").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".update_pending.json").write_text(json.dumps({"update_id": update_id}))
        (tmp_path / ".update_exit_code").write_text("0")
        paths = GatewayNotificationsMixin._UpdatePaths(
            pending=tmp_path / ".update_pending.json",
            claimed=tmp_path / ".update_pending.claimed.json",
            output=tmp_path / ".update_output.txt",
            exit_code=tmp_path / ".update_exit_code",
            prompt=tmp_path / ".update_prompt.json",
            response=tmp_path / ".update_response",
        )
        return paths, update_id

    def _write_receipt(self, tmp_path, update_id, **overrides):
        receipt = {
            "update_id": update_id,
            "finished_at": "2026-09-24T19:17:01Z",
            "outcome": "success",
            "exit_code": 0,
            "gateway_restart": {"incomplete": False, "phase_error": ""},
        }
        receipt.update(overrides)
        path = tmp_path / "logs" / "update_receipts" / f"update_20260924_191701_123_{update_id}.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        (tmp_path / "logs" / "update_receipts" / "latest.json").write_text(
            json.dumps(receipt), encoding="utf-8")
        return path

    def test_finalized_when_exact_terminal_receipt_and_obligation_absent(self, tmp_path, monkeypatch):
        from gateway.run_notifications import GatewayNotificationsMixin

        paths, update_id = self._paths(tmp_path)
        self._write_receipt(tmp_path, update_id)
        monkeypatch.setattr(
            "hermes_cli.update_cmd_fleet._fleet_restart_obligation_armed", lambda: False)

        assert GatewayNotificationsMixin._gateway_update_finalized(paths) is True

    def test_not_finalized_when_receipt_belongs_to_another_run(self, tmp_path, monkeypatch):
        from gateway.run_notifications import GatewayNotificationsMixin

        paths, update_id = self._paths(tmp_path)
        self._write_receipt(tmp_path, "different-update", finished_at="later", outcome="success")
        monkeypatch.setattr(
            "hermes_cli.update_cmd_fleet._fleet_restart_obligation_armed", lambda: False)

        assert GatewayNotificationsMixin._gateway_update_finalized(paths) is False
        assert update_id != "different-update"

    def test_not_finalized_when_exact_receipt_reports_failure(self, tmp_path, monkeypatch):
        from gateway.run_notifications import GatewayNotificationsMixin

        paths, update_id = self._paths(tmp_path)
        self._write_receipt(tmp_path, update_id, outcome="failed", exit_code=1)
        monkeypatch.setattr(
            "hermes_cli.update_cmd_fleet._fleet_restart_obligation_armed", lambda: False)

        assert GatewayNotificationsMixin._gateway_update_finalized(paths) is False

    def test_not_finalized_when_exact_receipt_omits_exit_code(self, tmp_path, monkeypatch):
        from gateway.run_notifications import GatewayNotificationsMixin

        paths, update_id = self._paths(tmp_path)
        self._write_receipt(tmp_path, update_id)
        receipt_path = tmp_path / "logs" / "update_receipts" / f"update_20260924_191701_123_{update_id}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt.pop("exit_code", None)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        monkeypatch.setattr(
            "hermes_cli.update_cmd_fleet._fleet_restart_obligation_armed", lambda: False)

        assert GatewayNotificationsMixin._gateway_update_finalized(paths) is False

    def test_not_finalized_when_exact_receipt_is_running(self, tmp_path, monkeypatch):
        from gateway.run_notifications import GatewayNotificationsMixin

        paths, update_id = self._paths(tmp_path)
        self._write_receipt(tmp_path, update_id, finished_at=None, outcome="running", exit_code=0)
        monkeypatch.setattr(
            "hermes_cli.update_cmd_fleet._fleet_restart_obligation_armed", lambda: False)

        assert GatewayNotificationsMixin._gateway_update_finalized(paths) is False

    def test_not_finalized_when_host_obligation_pending(self, tmp_path, monkeypatch):
        from gateway.run_notifications import GatewayNotificationsMixin

        paths, update_id = self._paths(tmp_path)
        self._write_receipt(tmp_path, update_id)
        monkeypatch.setattr(
            "hermes_cli.update_cmd_fleet._fleet_restart_obligation_armed", lambda: True)

        assert GatewayNotificationsMixin._gateway_update_finalized(paths) is False

    def test_not_finalized_when_exact_receipt_missing(self, tmp_path, monkeypatch):
        from gateway.run_notifications import GatewayNotificationsMixin

        paths, _update_id = self._paths(tmp_path)
        (tmp_path / "logs" / "update_receipts" / "latest.json").write_text(
            json.dumps({"update_id": "different-update", "finished_at": "later", "outcome": "success"}),
            encoding="utf-8")
        monkeypatch.setattr(
            "hermes_cli.update_cmd_fleet._fleet_restart_obligation_armed", lambda: False)

        assert GatewayNotificationsMixin._gateway_update_finalized(paths) is False
