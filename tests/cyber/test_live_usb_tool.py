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
        assert "build_dependencies_ready" in out
        assert "write_dependencies_ready" in out
        assert "can_build" in out
        assert "can_write" in out
        assert "operation_gates" in out

    def test_status_can_flags_do_not_ignore_destructive_gates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(live_usb._APPROVAL_ENV_VAR, raising=False)
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: False)
        monkeypatch.setattr(live_usb.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

        out = json.loads(_handle({"action": "status"}))

        assert out["build_dependencies_ready"] is True
        assert out["write_dependencies_ready"] is True
        assert out["can_build"] is False
        assert out["can_write"] is False
        gates = out["operation_gates"]
        assert gates["root"] is False
        assert gates["operator_approval_env_configured"] is False
        assert gates["safe_actions"] == ["status", "list_usb"]
        assert "root plus exact operator approval" in gates["build"]
        assert "root plus exact operator approval" in gates["write"]
        assert "verifiable whole removable /dev disk metadata" in gates["write"]
        assert "dependency checks alone" in gates["build"]

    def test_status_write_dependencies_include_lsblk_for_target_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_which(cmd: str) -> str | None:
            if cmd == "lsblk":
                return None
            return f"/usr/bin/{cmd}"

        monkeypatch.setenv(live_usb._APPROVAL_ENV_VAR, "approved-live-usb-lane")
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.setattr(live_usb.shutil, "which", fake_which)

        out = json.loads(_handle({"action": "status"}))

        assert out["build_dependencies_ready"] is True
        assert out["write_dependencies_ready"] is False
        assert out["can_write"] is False
        assert "lsblk" in out["missing_for_write"]

    def test_status_lists_available_isos(self) -> None:
        out = json.loads(_handle({"action": "status"}))
        assert "available_isos" in out
        assert isinstance(out["available_isos"], list)

    def test_list_usb_returns_removable_devices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_subprocess_run(cmd: list[str], **kwargs) -> types.SimpleNamespace:  # noqa: ANN003
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "blockdevices": [
                            {
                                "name": "sdb",
                                "size": "15G",
                                "tran": "usb",
                                "model": "Fixture USB",
                                "rm": "1",
                                "vendor": "TestVendor",
                            },
                            {
                                "name": "nvme0n1",
                                "size": "1T",
                                "tran": "nvme",
                                "model": "Internal Disk",
                                "rm": "0",
                                "vendor": "TestVendor",
                            },
                        ]
                    }
                ),
                stderr="",
            )

        monkeypatch.setattr(live_usb.shutil, "which", lambda cmd: "/usr/bin/lsblk" if cmd == "lsblk" else None)
        monkeypatch.setattr(live_usb.subprocess, "run", fake_subprocess_run)

        out = json.loads(_handle({"action": "list_usb"}))

        assert captured["cmd"] == [
            "lsblk",
            "-d",
            "-o",
            "NAME,SIZE,TRAN,MODEL,RM,VENDOR",
            "--sort",
            "NAME",
            "--json",
        ]
        assert captured["kwargs"]["timeout"] == 10
        assert out["removable_devices"] == [
            {
                "device": "/dev/sdb",
                "size": "15G",
                "transport": "usb",
                "model": "Fixture USB",
                "removable": True,
                "vendor": "TestVendor",
            }
        ]
        assert len(out["all_block_devices"]) == 2

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

    @pytest.mark.parametrize(
        ("action", "payload"),
        [
            ("build", {"action": "build"}),
            ("write", {"action": "write", "device": "/dev/sdz", "iso": "/tmp/hermes.iso"}),
            ("provision", {"action": "provision", "device": "/dev/sdz"}),
        ],
    )
    def test_destructive_actions_non_root_guidance_requires_more_than_sudo(
        self,
        action: str,
        payload: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        called = {"is_block_device": False}

        def fake_is_block_device(_path) -> bool:  # noqa: ANN001
            called["is_block_device"] = True
            pytest.fail(f"{action} must fail root gate before checking block devices")

        monkeypatch.setattr(live_usb, "_running_as_root", lambda: False)
        monkeypatch.setattr(live_usb.Path, "is_block_device", fake_is_block_device)
        monkeypatch.setattr(
            live_usb,
            "_script",
            lambda *_args, **_kw: pytest.fail(f"{action} must fail root gate before resolving scripts"),
        )
        monkeypatch.setattr(
            live_usb,
            "_run",
            lambda *_args, **_kw: pytest.fail(f"{action} must fail root gate before running scripts"),
        )

        out = json.loads(_handle(payload))
        combined = json.dumps(out).lower()

        assert "root" in combined
        assert "operator approval" in combined
        assert "root alone" in combined
        assert "not sufficient" in combined
        assert "sudo live-usb/write_usb.sh --iso <path> --device <dev> --yes" not in combined
        assert called["is_block_device"] is False

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

    def test_build_rejects_dev_output_before_script_or_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", "approved-live-usb-lane")
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.setattr(
            live_usb,
            "_script",
            lambda *_args, **_kw: pytest.fail("build must reject unsafe output before resolving scripts"),
        )
        monkeypatch.setattr(
            live_usb,
            "_run",
            lambda *_args, **_kw: pytest.fail("build must reject unsafe output before running build_iso.sh"),
        )

        out = json.loads(_handle({
            "action": "build",
            "operator_approval": "approved-live-usb-lane",
            "output": "/dev/hermes-cyber-live.iso",
        }))

        assert out["error"] == "Unsafe ISO output target: /dev/hermes-cyber-live.iso"
        assert out["reason"] == "output target canonicalizes under /dev"
        assert "regular file path" in out["hint"]

    def test_build_rejects_existing_block_output_before_resolve_or_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output = "/tmp/operator-output-block"
        called = {"resolve": False}

        def fake_resolve(_path: Any, strict: bool = False) -> live_usb.Path:
            called["resolve"] = True
            pytest.fail("existing block output must fail before path resolution")

        monkeypatch.setenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", "approved-live-usb-lane")
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.setattr(live_usb.Path, "is_block_device", lambda path: str(path) == output)
        monkeypatch.setattr(live_usb.Path, "resolve", fake_resolve)
        monkeypatch.setattr(
            live_usb,
            "_script",
            lambda *_args, **_kw: pytest.fail("build must reject block output before resolving scripts"),
        )
        monkeypatch.setattr(
            live_usb,
            "_run",
            lambda *_args, **_kw: pytest.fail("build must reject block output before running build_iso.sh"),
        )

        out = json.loads(_handle({
            "action": "build",
            "operator_approval": "approved-live-usb-lane",
            "output": output,
        }))

        assert out["error"] == f"Unsafe ISO output target: {output}"
        assert out["reason"] == "output target is an existing block device"
        assert called["resolve"] is False

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
            lambda *_args, **_kw: pytest.fail("provision must fail before invoking provision.sh without approval"),
        )

        out = json.loads(_handle({"action": "provision", "device": "/dev/sdz"}))

        assert out["approved"] is False
        assert "operator approval" in out["error"]
        assert called["is_block_device"] is False

    @pytest.mark.parametrize(
        ("action", "payload"),
        [
            ("build", {"action": "build"}),
            ("write", {"action": "write", "device": "/dev/sdz", "iso": "/tmp/hermes.iso"}),
            ("provision", {"action": "provision", "device": "/dev/sdz"}),
        ],
    )
    @pytest.mark.parametrize(
        "operator_approval",
        [
            "wrong-live-usb-lane",
            "APPROVED-LIVE-USB-LANE",
            " approved-live-usb-lane",
            "approved-live-usb-lane ",
        ],
    )
    def test_destructive_actions_reject_non_exact_operator_approval_before_side_effects(
        self,
        action: str,
        payload: dict[str, object],
        operator_approval: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        called = {"is_block_device": False}

        def fake_is_block_device(_path) -> bool:  # noqa: ANN001
            called["is_block_device"] = True
            pytest.fail(f"{action} must fail approval before checking block devices")

        monkeypatch.setenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", "approved-live-usb-lane")
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.setattr(live_usb.Path, "is_block_device", fake_is_block_device)
        monkeypatch.setattr(
            live_usb,
            "_script",
            lambda *_args, **_kw: pytest.fail(f"{action} must fail approval before resolving scripts"),
        )
        monkeypatch.setattr(
            live_usb,
            "_run",
            lambda *_args, **_kw: pytest.fail(f"{action} must fail approval before running scripts"),
        )

        out = json.loads(_handle({**payload, "operator_approval": operator_approval}))

        assert out["approved"] is False
        assert out["reason"] == "operator_approval did not match"
        assert out["error"] == f"{action} requires explicit operator approval."
        assert called["is_block_device"] is False

    def test_provision_approved_non_block_device_returns_error_before_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", "approved-live-usb-lane")
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.setattr(live_usb.Path, "is_block_device", lambda _path: False)
        monkeypatch.setattr(
            live_usb,
            "_script",
            lambda *_args, **_kw: pytest.fail("provision must fail before resolving provision.sh"),
        )
        monkeypatch.setattr(
            live_usb,
            "_run",
            lambda *_args, **_kw: pytest.fail("provision must fail before invoking provision.sh"),
        )

        out = json.loads(_handle({
            "action": "provision",
            "device": "/dev/sdz",
            "operator_approval": "approved-live-usb-lane",
        }))

        assert out == {"error": "Not a block device: /dev/sdz"}

    def test_removable_guard_returns_canonical_dev_path_for_flag_one(
        self,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sysfs_root = tmp_path / "sys_class_block"
        removable_dir = sysfs_root / "sdz"
        removable_dir.mkdir(parents=True)
        (removable_dir / "removable").write_text("1\n", encoding="utf-8")

        def fake_resolve(path: Any, strict: bool = False) -> live_usb.Path:
            assert strict is True
            if str(path) == "/tmp/operator-usb-alias":
                return live_usb.Path("/dev/sdz")
            raise AssertionError(f"unexpected resolve path: {path}")

        monkeypatch.setattr(live_usb, "_REMOVABLE_SYSFS_ROOTS", (sysfs_root,))
        monkeypatch.setattr(live_usb, "_linux_block_device_type", lambda _device: "disk")
        monkeypatch.setattr(live_usb.Path, "resolve", fake_resolve)
        monkeypatch.setattr(live_usb.Path, "is_block_device", lambda path: str(path) == "/dev/sdz")

        assert live_usb._require_verifiably_removable_block_device("/tmp/operator-usb-alias") == "/dev/sdz"

    def test_removable_guard_resolve_failure_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_resolve(_path: Any, strict: bool = False) -> live_usb.Path:
            assert strict is True
            raise OSError("cannot resolve")

        monkeypatch.setattr(live_usb.Path, "resolve", fake_resolve)

        assert live_usb._require_verifiably_removable_block_device("/dev/sdz") == {
            "error": "Target block device is not verifiably removable: /dev/sdz",
            "reason": "could not resolve target block device",
        }

    def test_removable_guard_non_dev_resolved_target_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(live_usb.Path, "resolve", lambda _path, strict=False: live_usb.Path("/tmp/sdz"))

        assert live_usb._require_verifiably_removable_block_device("/tmp/operator-usb-alias") == {
            "error": "Target block device is not verifiably removable: /tmp/operator-usb-alias",
            "reason": "resolved target is not a /dev block device",
        }

    @pytest.mark.parametrize(
        ("action", "payload"),
        [
            ("write", {"action": "write", "device": "/dev/sdz1", "iso": "/tmp/hermes.iso"}),
            ("provision", {"action": "provision", "device": "/dev/sdz1"}),
        ],
    )
    @pytest.mark.parametrize("linux_type", ["part", "lvm"])
    def test_approved_non_disk_target_fails_before_script_and_run(
        self,
        action: str,
        payload: dict[str, object],
        linux_type: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_read_text(*_args: Any, **_kw: Any) -> str:
            pytest.fail(f"{action} must reject non-disk targets before checking removable flags")

        monkeypatch.setenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", "approved-live-usb-lane")
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.setattr(live_usb.Path, "is_block_device", lambda _path: True)
        monkeypatch.setattr(live_usb.Path, "resolve", lambda path, strict=False: path)
        monkeypatch.setattr(live_usb.Path, "read_text", fake_read_text)
        monkeypatch.setattr(live_usb, "_linux_block_device_type", lambda _device: linux_type)
        monkeypatch.setattr(
            live_usb,
            "_script",
            lambda *_args, **_kw: pytest.fail(f"{action} must fail before resolving scripts"),
        )
        monkeypatch.setattr(
            live_usb,
            "_run",
            lambda *_args, **_kw: pytest.fail(f"{action} must fail before running scripts"),
        )

        out = json.loads(_handle({**payload, "operator_approval": "approved-live-usb-lane"}))

        assert out == {
            "error": "Target block device is not verifiably removable: /dev/sdz1",
            "reason": f"target is not a whole disk (Linux block type is not disk: {linux_type})",
        }

    @pytest.mark.parametrize(
        ("action", "payload"),
        [
            ("write", {"action": "write", "device": "/dev/sdz", "iso": "/tmp/hermes.iso"}),
            ("provision", {"action": "provision", "device": "/dev/sdz"}),
        ],
    )
    def test_approved_non_removable_target_fails_before_script_and_run(
        self,
        action: str,
        payload: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_read_text(_path: Any, encoding: str | None = None) -> str:
            assert encoding == "utf-8"
            return "0\n"

        monkeypatch.setenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", "approved-live-usb-lane")
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.setattr(live_usb, "_linux_block_device_type", lambda _device: "disk")
        monkeypatch.setattr(live_usb.Path, "is_block_device", lambda _path: True)
        monkeypatch.setattr(live_usb.Path, "resolve", lambda path, strict=False: path)
        monkeypatch.setattr(live_usb.Path, "read_text", fake_read_text)
        monkeypatch.setattr(
            live_usb,
            "_script",
            lambda *_args, **_kw: pytest.fail(f"{action} must fail before resolving scripts"),
        )
        monkeypatch.setattr(
            live_usb,
            "_run",
            lambda *_args, **_kw: pytest.fail(f"{action} must fail before running scripts"),
        )

        out = json.loads(_handle({**payload, "operator_approval": "approved-live-usb-lane"}))

        assert out == {
            "error": "Target block device is not verifiably removable: /dev/sdz",
            "reason": "Linux removable flag is not 1",
        }

    @pytest.mark.parametrize(
        ("action", "payload"),
        [
            ("write", {"action": "write", "device": "/dev/sdz", "iso": "/tmp/hermes.iso"}),
            ("provision", {"action": "provision", "device": "/dev/sdz"}),
        ],
    )
    @pytest.mark.parametrize("read_error", [FileNotFoundError("missing"), PermissionError("denied")])
    def test_approved_missing_or_unreadable_removable_flag_fails_closed(
        self,
        action: str,
        payload: dict[str, object],
        read_error: OSError,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_read_text(_path: Any, encoding: str | None = None) -> str:
            assert encoding == "utf-8"
            raise read_error

        monkeypatch.setenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", "approved-live-usb-lane")
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.setattr(live_usb, "_linux_block_device_type", lambda _device: "disk")
        monkeypatch.setattr(live_usb.Path, "is_block_device", lambda _path: True)
        monkeypatch.setattr(live_usb.Path, "resolve", lambda path, strict=False: path)
        monkeypatch.setattr(live_usb.Path, "read_text", fake_read_text)
        monkeypatch.setattr(
            live_usb,
            "_script",
            lambda *_args, **_kw: pytest.fail(f"{action} must fail before resolving scripts"),
        )
        monkeypatch.setattr(
            live_usb,
            "_run",
            lambda *_args, **_kw: pytest.fail(f"{action} must fail before running scripts"),
        )

        out = json.loads(_handle({**payload, "operator_approval": "approved-live-usb-lane"}))

        assert out == {
            "error": "Target block device is not verifiably removable: /dev/sdz",
            "reason": "Linux removable flag is missing or unreadable",
        }

    def test_build_approved_path_uses_stdin_approval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_run(cmd: list[str], timeout: int = 300, input_text: str | None = None) -> dict:
            captured["cmd"] = cmd
            captured["timeout"] = timeout
            captured["input_text"] = input_text
            return {"rc": 0, "stdout": "mocked build ok", "stderr": ""}

        monkeypatch.setenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", "approved-live-usb-lane")
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.setattr(live_usb, "_run", fake_run)

        out = json.loads(_handle({
            "action": "build",
            "operator_approval": "approved-live-usb-lane",
            "arch": "arm64",
            "output": "/tmp/hermes.iso",
            "headless_scan": True,
        }))

        assert out["success"] is True
        assert captured["timeout"] == 1800
        assert captured["input_text"] == "approved-live-usb-lane\n"
        assert captured["cmd"] == [
            "bash",
            str(live_usb._SCRIPTS_DIR / "build_iso.sh"),
            "--arch",
            "arm64",
            "--output",
            "/tmp/hermes.iso",
            "--headless-scan",
            "--operator-approval-stdin",
        ]

    def test_provision_approved_path_is_mocked_and_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_run(cmd: list[str], timeout: int = 300, input_text: str | None = None) -> dict:
            captured["cmd"] = cmd
            captured["timeout"] = timeout
            captured["input_text"] = input_text
            return {"rc": 0, "stdout": "mocked provision ok", "stderr": ""}

        monkeypatch.setenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", "approved-live-usb-lane")
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.setattr(live_usb.Path, "is_block_device", lambda _path: True)
        monkeypatch.setattr(live_usb, "_require_verifiably_removable_block_device", lambda _device: "/dev/sdz")
        monkeypatch.setattr(live_usb, "_run", fake_run)

        out = json.loads(_handle({
            "action": "provision",
            "device": "/dev/sdz",
            "operator_approval": "approved-live-usb-lane",
            "config": "/tmp/hermes-config.tar.gz",
            "allowed_users": "12345",
            "model_provider": "openai",
            "audit": True,
        }))

        assert out["success"] is True
        assert captured["timeout"] == 60
        assert captured["input_text"] == "approved-live-usb-lane\n"
        assert captured["cmd"] == [
            "bash",
            str(live_usb._SCRIPTS_DIR / "provision.sh"),
            "--usb",
            "/dev/sdz",
            "--operator-approval-stdin",
            "--config",
            "/tmp/hermes-config.tar.gz",
            "--allowed-users",
            "12345",
            "--model-provider",
            "openai",
            "--audit",
        ]

    def test_write_approved_path_is_mocked_and_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_run(cmd: list[str], timeout: int = 300, input_text: str | None = None) -> dict:
            captured["cmd"] = cmd
            captured["timeout"] = timeout
            captured["input_text"] = input_text
            return {"rc": 0, "stdout": "mocked write ok", "stderr": ""}

        monkeypatch.setenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", "approved-live-usb-lane")
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.setattr(live_usb.Path, "is_block_device", lambda _path: True)
        monkeypatch.setattr(live_usb, "_require_verifiably_removable_block_device", lambda _device: "/dev/sdz")
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
        assert captured["input_text"] == "approved-live-usb-lane\n"
        assert captured["cmd"] == [
            "bash",
            str(live_usb._SCRIPTS_DIR / "write_usb.sh"),
            "--iso",
            "/tmp/hermes.iso",
            "--device",
            "/dev/sdz",
            "--yes",
            "--operator-approval-stdin",
            "--verify",
        ]

    @pytest.mark.parametrize(
        ("action", "payload", "device_flag", "script_name"),
        [
            (
                "write",
                {"action": "write", "device": "/tmp/operator-usb-alias", "iso": "/tmp/hermes.iso"},
                "--device",
                "write_usb.sh",
            ),
            (
                "provision",
                {"action": "provision", "device": "/tmp/operator-usb-alias"},
                "--usb",
                "provision.sh",
            ),
        ],
    )
    def test_approved_alias_input_is_canonicalized_before_run(
        self,
        action: str,
        payload: dict[str, object],
        device_flag: str,
        script_name: str,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}
        alias = "/tmp/operator-usb-alias"
        canonical_device = "/dev/sdz"
        sysfs_root = tmp_path / "sys_class_block"
        removable_dir = sysfs_root / "sdz"
        removable_dir.mkdir(parents=True)
        (removable_dir / "removable").write_text("1\n", encoding="utf-8")

        original_exists = live_usb.Path.exists

        def fake_exists(path: Any) -> bool:
            if str(path) == "/tmp/hermes.iso":
                return True
            return original_exists(path)

        def fake_resolve(path: Any, strict: bool = False) -> live_usb.Path:
            assert strict is True
            if str(path) == alias:
                return live_usb.Path(canonical_device)
            raise AssertionError(f"unexpected resolve path: {path}")

        def fake_run(cmd: list[str], timeout: int = 300, input_text: str | None = None) -> dict:
            captured["cmd"] = cmd
            captured["timeout"] = timeout
            captured["input_text"] = input_text
            return {"rc": 0, "stdout": f"mocked {action} ok", "stderr": ""}

        monkeypatch.setenv("HERMES_AGENTCYBER_LIVE_USB_APPROVAL", "approved-live-usb-lane")
        monkeypatch.setattr(live_usb, "_REMOVABLE_SYSFS_ROOTS", (sysfs_root,))
        monkeypatch.setattr(live_usb, "_running_as_root", lambda: True)
        monkeypatch.setattr(live_usb, "_linux_block_device_type", lambda _device: "disk")
        monkeypatch.setattr(
            live_usb.Path,
            "is_block_device",
            lambda path: str(path) in {alias, canonical_device},
        )
        monkeypatch.setattr(live_usb.Path, "resolve", fake_resolve)
        monkeypatch.setattr(live_usb.Path, "exists", fake_exists)
        monkeypatch.setattr(live_usb, "_run", fake_run)

        out = json.loads(_handle({**payload, "operator_approval": "approved-live-usb-lane"}))

        cmd = captured["cmd"]
        assert isinstance(cmd, list)
        assert out["success"] is True
        assert captured["input_text"] == "approved-live-usb-lane\n"
        assert cmd[:2] == ["bash", str(live_usb._SCRIPTS_DIR / script_name)]
        assert "--operator-approval-stdin" in cmd
        assert cmd[cmd.index(device_flag) + 1] == canonical_device
        assert alias not in cmd
