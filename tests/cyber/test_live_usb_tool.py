"""Unit tests for the live USB tool (no root, no actual USB required)."""
from __future__ import annotations

import json
import sys
import types

for mod in ("tools.registry",):
    if mod not in sys.modules:
        stub = types.ModuleType(mod)
        stub.registry = types.SimpleNamespace(register=lambda **_kw: None)
        sys.modules[mod] = stub

from tools.cyber_live_usb import _handle  # noqa: E402


class TestLiveUsbTool:
    def test_status_returns_scripts_dir(self) -> None:
        out = json.loads(_handle({"action": "status"}))
        assert "scripts_dir" in out
        assert "live-usb" in out["scripts_dir"]

    def test_status_reports_build_deps(self) -> None:
        out = json.loads(_handle({"action": "status"}))
        assert "build_dependencies" in out
        assert "can_build" in out
        assert "can_write" in out

    def test_status_lists_available_isos(self) -> None:
        out = json.loads(_handle({"action": "status"}))
        assert "available_isos" in out
        assert isinstance(out["available_isos"], list)

    def test_list_usb_returns_removable_devices(self) -> None:
        out = json.loads(_handle({"action": "list_usb"}))
        # Either returns device list or an error if lsblk missing
        assert "removable_devices" in out or "error" in out

    def test_unknown_action_returns_error_and_valid_list(self) -> None:
        out = json.loads(_handle({"action": "nuke_everything"}))
        assert "error" in out
        assert "valid_actions" in out
        assert set(out["valid_actions"]) == {"build", "write", "provision", "list_usb", "status"}

    def test_write_missing_device_returns_error(self) -> None:
        # write with no device specified
        out = json.loads(_handle({"action": "write"}))
        assert "error" in out

    def test_write_nonexistent_device_returns_error(self) -> None:
        out = json.loads(_handle({"action": "write", "device": "/dev/hermes_no_such_dev"}))
        assert "error" in out

    def test_provision_missing_device_returns_error(self) -> None:
        out = json.loads(_handle({"action": "provision"}))
        assert "error" in out

    def test_no_action_returns_error(self) -> None:
        out = json.loads(_handle({}))
        assert "error" in out

    def test_write_with_persistence_and_encrypt_arguments(self, monkeypatch) -> None:
        from unittest.mock import patch, MagicMock
        from pathlib import Path

        # Mock root check and block device check
        monkeypatch.setattr("tools.cyber_live_usb._running_as_root", lambda: True)
        monkeypatch.setattr(Path, "is_block_device", lambda self: True)
        monkeypatch.setattr(Path, "exists", lambda self: True)

        run_calls = []
        def fake_run(cmd, timeout=300):
            run_calls.append(cmd)
            return {"rc": 0, "stdout": "ok", "stderr": ""}

        monkeypatch.setattr("tools.cyber_live_usb._run", fake_run)

        res = json.loads(_handle({
            "action": "write",
            "device": "/dev/sdb",
            "iso": "/tmp/hermes.iso",
            "persistence": "8G",
            "encrypt": True,
            "verify": True,
        }))

        assert res["success"] is True
        assert len(run_calls) == 1
        cmd = run_calls[0]
        assert "--persistence" in cmd
        assert "8G" in cmd
        assert "--encrypt" in cmd
        assert "--verify" in cmd

    def test_build_with_additional_arguments(self, monkeypatch) -> None:
        monkeypatch.setattr("tools.cyber_live_usb._running_as_root", lambda: True)

        run_calls = []
        def fake_run(cmd, timeout=1800):
            run_calls.append(cmd)
            return {"rc": 0, "stdout": "ok", "stderr": ""}

        monkeypatch.setattr("tools.cyber_live_usb._run", fake_run)

        res = json.loads(_handle({
            "action": "build",
            "mirror": "http://mirror.example.com",
            "no_bundle_source": True,
            "headless_scan": True,
        }))

        assert res["success"] is True
        cmd = run_calls[0]
        assert "--mirror" in cmd
        assert "http://mirror.example.com" in cmd
        assert "--no-bundle-source" in cmd
        assert "--headless-scan" in cmd

    def test_provision_with_env_file_and_model_name(self, monkeypatch) -> None:
        monkeypatch.setattr("tools.cyber_live_usb._running_as_root", lambda: True)

        run_calls = []
        def fake_run(cmd, timeout=60):
            run_calls.append(cmd)
            return {"rc": 0, "stdout": "ok", "stderr": ""}

        monkeypatch.setattr("tools.cyber_live_usb._run", fake_run)

        res = json.loads(_handle({
            "action": "provision",
            "device": "/dev/sdb",
            "env_file": "/tmp/.env",
            "model_name": "claude-opus-4-7-20251101",
            "audit": True,
        }))

        assert res["success"] is True
        cmd = run_calls[0]
        assert "--env-file" in cmd
        assert "/tmp/.env" in cmd
        assert "--model-name" in cmd
        assert "claude-opus-4-7-20251101" in cmd
        assert "--audit" in cmd
