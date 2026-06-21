"""Unit tests for the live USB tool (no root, no actual USB required)."""
from __future__ import annotations

import json
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
