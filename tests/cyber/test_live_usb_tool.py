"""Unit tests for the live USB tool (no root, no actual USB required)."""
from __future__ import annotations

from typing import Any

import json
import logging
import sys
import types

import pytest

for mod in ("tools.registry",):
    if mod not in sys.modules:
        stub = types.ModuleType(mod)
        stub.registry = types.SimpleNamespace(register=lambda **_kw: None)
        sys.modules[mod] = stub

import tools.cyber_live_usb as live_usb  # noqa: E402
from tools.cyber_live_usb import _handle  # noqa: E402


class TestLiveUsbTool:
    def test_run_redacts_secret_flags_from_log_but_preserves_command(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_subprocess_run(cmd: list[str], **kwargs) -> types.SimpleNamespace:  # noqa: ANN003
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

        raw_cmd = [
            "bash",
            "provision.sh",
            "--telegram-token",
            "telegram-secret-token",
            "--model-key=model-secret-key",
            "--config",
            "/tmp/config",
        ]
        original_cmd = list(raw_cmd)
        monkeypatch.setattr(live_usb.subprocess, "run", fake_subprocess_run)
        caplog.set_level(logging.INFO, logger=live_usb.logger.name)

        result = live_usb._run(raw_cmd, timeout=12)

        assert result["rc"] == 0
        assert raw_cmd == original_cmd
        assert captured["cmd"] == original_cmd
        assert "telegram-secret-token" in " ".join(captured["cmd"])
        assert "model-secret-key" in " ".join(captured["cmd"])
        assert captured["kwargs"]["timeout"] == 12
        assert "telegram-secret-token" not in caplog.text
        assert "model-secret-key" not in caplog.text
        assert "--telegram-token <redacted>" in caplog.text
        assert "--model-key=<redacted>" in caplog.text

    @pytest.mark.parametrize(
        ("raw_cmd", "expected"),
        [
            (["cmd", "--telegram-token", "sensitive-value"], ["cmd", "--telegram-token", "<redacted>"]),
            (["cmd", "--telegram-token=sensitive-value"], ["cmd", "--telegram-token=<redacted>"]),
            (["cmd", "--model-key", "sensitive-value"], ["cmd", "--model-key", "<redacted>"]),
            (["cmd", "--model-key=sensitive-value"], ["cmd", "--model-key=<redacted>"]),
            (["cmd", "--operator-approval", "sensitive-value"], ["cmd", "--operator-approval", "<redacted>"]),
            (["cmd", "--operator-approval=sensitive-value"], ["cmd", "--operator-approval=<redacted>"]),
            (["cmd", "--approval-token", "sensitive-value"], ["cmd", "--approval-token", "<redacted>"]),
            (["cmd", "--approval-token=sensitive-value"], ["cmd", "--approval-token=<redacted>"]),
            (["cmd", "--live-usb-approval", "sensitive-value"], ["cmd", "--live-usb-approval", "<redacted>"]),
            (["cmd", "--live-usb-approval=sensitive-value"], ["cmd", "--live-usb-approval=<redacted>"]),
            (["cmd", "--telegram-token"], ["cmd", "--telegram-token"]),
            (["cmd", "--model-key="], ["cmd", "--model-key=<redacted>"]),
        ],
    )
    def test_redacted_command_handles_approval_aliases_and_edge_cases(
        self,
        raw_cmd: list[str],
        expected: list[str],
    ) -> None:
        assert live_usb._redacted_command_for_log(raw_cmd) == expected
        assert "sensitive-value" not in " ".join(live_usb._redacted_command_for_log(raw_cmd))

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

    def test_build_requires_operator_approval_when_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.delenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", raising=False)
        monkeypatch.setattr(
            live_usb,
            "_run",
            lambda *_args, **_kw: pytest.fail("build must fail before running build_iso.sh without approval"),
        )

        out = json.loads(_handle({"action": "build"}))

        assert out["approved"] is False
        assert "operator approval" in out["error"]
        assert out["reason"] == "missing HERMES_AGENTCYBER_LIVE_USB_APPROVAL"

    def test_write_requires_operator_approval_before_touching_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = {"is_block_device": False}

        def fake_is_block_device(_path) -> bool:  # noqa: ANN001
            called["is_block_device"] = True
            return True

        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.delenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", raising=False)
        monkeypatch.setattr(live_usb.Path, "is_block_device", fake_is_block_device)
        monkeypatch.setattr(
            live_usb,
            "_run",
            lambda *_args, **_kw: pytest.fail("write must fail before invoking write_usb.sh without approval"),
        )

        out = json.loads(_handle({"action": "write", "device": "/dev/sdz", "iso": "/tmp/hermes.iso"}))

        assert out["approved"] is False
        assert "operator approval" in out["error"]
        assert called["is_block_device"] is False

    def test_provision_requires_operator_approval_when_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.delenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", raising=False)
        monkeypatch.setattr(
            live_usb,
            "_run",
            lambda *_args, **_kw: pytest.fail("provision must fail before invoking provision.sh without approval"),
        )

        out = json.loads(_handle({"action": "provision", "device": "/dev/sdz1"}))

        assert out["approved"] is False
        assert "operator approval" in out["error"]

    def test_write_approved_path_is_mocked_and_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_run(cmd: list[str], timeout: int = 300) -> dict:
            captured["cmd"] = cmd
            captured["timeout"] = timeout
            return {"rc": 0, "stdout": "mocked write ok", "stderr": ""}

        monkeypatch.setenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", "approved-live-usb-lane")
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.setattr(live_usb.Path, "is_block_device", lambda _path: True)
        monkeypatch.setattr(live_usb.Path, "exists", lambda _path: True)
        monkeypatch.setattr(live_usb, "_run", fake_run)

        out = json.loads(_handle({
            "action": "write",
            "device": "/dev/sdz",
            "iso": "/tmp/hermes.iso",
            "operator_approval": "approved-live-usb-lane",
            "verify": True,
        }))

        assert out["success"] is True
        assert captured["timeout"] == 600
        assert captured["cmd"] == [
            "bash",
            str(live_usb._SCRIPTS_DIR / "write_usb.sh"),
            "--iso",
            "/tmp/hermes.iso",
            "--device",
            "/dev/sdz",
            "--yes",
            "--verify",
        ]
