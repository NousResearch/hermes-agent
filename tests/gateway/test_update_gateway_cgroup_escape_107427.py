"""Invariant tests for #107427: gateway /update must survive its own restart.

Half A — spawner escapes the gateway cgroup; half B — the watcher does not report
success from the pre-restart ``.update_exit_code`` write alone.
"""

import json
import os
import subprocess
import shutil
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Half A — cgroup escape in _spawn_detached_update
# ---------------------------------------------------------------------------


class TestSpawnDetachedUpdateCgroupEscape:
    """_spawn_detached_update must escape the service cgroup when supervised."""

    def test_escapes_with_systemd_run_when_supervised_and_probe_ok(self, tmp_path, monkeypatch):
        """Supervised + probe OK => argv wrapped in systemd-run, env carries the bus."""
        import gateway.slash_commands as sc

        captured: dict = {}

        class FakePopen:
            def __init__(self, *a, **kw):
                captured["argv"] = a[0] if a else kw.get("args")
                captured["env"] = kw.get("env")
                captured["start_new_session"] = kw.get("start_new_session")
                self.pid = 9999

        monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/systemd-run" if x == "systemd-run" else "/usr/bin/setsid" if x == "setsid" else None)
        monkeypatch.setattr("tools.process_registry._systemd_run_user_scope_available", lambda: True)
        monkeypatch.setattr("tools.process_registry.systemd_user_bus_env", lambda e=None: {**(e or {}), "DBUS_SESSION_BUS_ADDRESS": "unix:path=/fake/bus", "XDG_RUNTIME_DIR": "/run/user/1000"})
        monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setenv("INVOCATION_ID", "fake-invocation")

        sc._spawn_detached_update(["hermes"], tmp_path / "out.txt", tmp_path / ".update_exit_code")

        assert captured["argv"][0] == "/usr/bin/systemd-run"
        assert "--scope" in captured["argv"]
        assert "--collect" in captured["argv"]
        assert captured["env"]["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/fake/bus"
        assert captured["start_new_session"] is True

    def test_falls_back_to_plain_setsId_when_probe_fails(self, tmp_path, monkeypatch):
        """Unsupervised or probe false => plain setsid spawn, no env override."""
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
        monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/setsid" if x == "setsid" else None)

        sc._spawn_detached_update(["hermes"], tmp_path / "out.txt", tmp_path / ".update_exit_code")

        assert captured["argv"][0] != "systemd-run"
        assert captured["env"] is None


# ---------------------------------------------------------------------------
# Half B — finalized evidence gate in run_notifications
# ---------------------------------------------------------------------------


class TestGatewayUpdateFinalizedGate:
    """_gateway_update_finalized and its consumers must fail closed on stale/missing receipt."""

    def _paths(self, tmp_path):
        from gateway.run_notifications import GatewayNotificationsMixin

        (tmp_path / "logs" / "update_receipts").mkdir(parents=True, exist_ok=True)
        exit_code = tmp_path / ".update_exit_code"
        exit_code.write_text("0")
        time.sleep(0.02)
        latest = tmp_path / "logs" / "update_receipts" / "latest.json"
        latest.write_text(json.dumps({"finished_at": "now"}))
        try:
            se = exit_code.stat().st_mtime
            sl = latest.stat().st_mtime
            if sl <= se:
                os.utime(latest, (se + 1, se + 1))
        except OSError:
            pass
        Paths = GatewayNotificationsMixin._UpdatePaths(
            pending=tmp_path / ".update_pending.json",
            claimed=tmp_path / ".update_pending.claimed.json",
            output=tmp_path / ".update_output.txt",
            exit_code=exit_code,
            prompt=tmp_path / ".update_prompt.json",
            response=tmp_path / ".update_response",
        )
        return Paths

    def test_finalized_when_receipt_fresh_and_marker_absent(self, tmp_path):
        from gateway.run_notifications import GatewayNotificationsMixin

        paths = self._paths(tmp_path)
        assert GatewayNotificationsMixin._gateway_update_finalized(paths) is True

    def test_not_finalized_when_receipt_stale(self, tmp_path):
        from gateway.run_notifications import GatewayNotificationsMixin

        paths = self._paths(tmp_path)
        latest = tmp_path / "logs" / "update_receipts" / "latest.json"
        exit_code = tmp_path / ".update_exit_code"
        try:
            os.utime(latest, (time.time() - 100, time.time() - 100))
            os.utime(exit_code, None)
        except OSError:
            pass
        assert GatewayNotificationsMixin._gateway_update_finalized(paths) is False

    def test_not_finalized_when_marker_present(self, tmp_path):
        from gateway.run_notifications import GatewayNotificationsMixin

        paths = self._paths(tmp_path)
        (tmp_path / "fleet_restart_pending").write_text("started=now\n")
        assert GatewayNotificationsMixin._gateway_update_finalized(paths) is False

    def test_not_finalized_when_latest_missing(self, tmp_path):
        from gateway.run_notifications import GatewayNotificationsMixin

        paths = self._paths(tmp_path)
        (tmp_path / "logs" / "update_receipts" / "latest.json").unlink()
        assert GatewayNotificationsMixin._gateway_update_finalized(paths) is False
