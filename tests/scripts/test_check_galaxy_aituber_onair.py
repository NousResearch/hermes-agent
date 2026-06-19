"""Tests for the Galaxy AITuber OnAir Windows helper script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "windows" / "check-galaxy-aituber-onair.ps1"

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Galaxy AITuber OnAir helper tests exercise Windows PowerShell and .cmd shims.",
)


def _extract_json(output: str) -> dict:
    start = output.find("{")
    end = output.rfind("}")
    assert start >= 0 and end > start, output
    return json.loads(output[start : end + 1])


def test_configure_device_uses_adb_stay_awake_and_opens_avatar(tmp_path):
    log = tmp_path / "adb.log"
    fake_adb = tmp_path / "adb.cmd"
    fake_adb.write_text(
        "\n".join(
                [
                    "@echo off",
                    ">>\"%~dp0adb.log\" echo %*",
                "if \"%1\"==\"devices\" (",
                "  echo List of devices attached",
                "  echo galaxy123 device product:starlte model:SM_G965U",
                "  exit /b 0",
                ")",
                "exit /b 0",
            ]
        ),
        encoding="utf-8",
    )
    pnp_snapshot = tmp_path / "pnp.json"
    pnp_snapshot.write_text("[]", encoding="utf-8")
    driver_snapshot = tmp_path / "driver.json"
    driver_snapshot.write_text("[]", encoding="utf-8")

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-AvatarUrl",
            "http://127.0.0.1:9/",
            "-AdbPath",
            str(fake_adb),
            "-PnpSnapshotPath",
            str(pnp_snapshot),
            "-DriverSnapshotPath",
            str(driver_snapshot),
            "-ConfigureDevice",
            "-OpenOnDevice",
        ],
        cwd=ROOT,
        text=True,
        encoding="cp932",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = _extract_json(result.stdout)
    assert payload["device_config"]["ok"] is True
    assert payload["open_on_device"]["ok"] is True
    adb_log = log.read_text(encoding="utf-8")
    assert "shell pm grant com.brave.browser android.permission.RECORD_AUDIO" in adb_log
    assert "shell settings put global stay_on_while_plugged_in 3" in adb_log
    assert "shell settings put system screen_off_timeout 2147483647" in adb_log
    assert "shell settings put secure lock_to_app_enabled 1" in adb_log
    assert (
        "shell am start -p com.brave.browser -a android.intent.action.VIEW -d http://127.0.0.1:9/"
        in adb_log
    )


def test_checks_firewall_only_when_avatar_url_is_unreachable(tmp_path):
    fake_adb = tmp_path / "adb.cmd"
    fake_adb.write_text(
        "\n".join(
            [
                "@echo off",
                "if \"%1\"==\"devices\" (",
                "  echo List of devices attached",
                "  exit /b 0",
                ")",
                "exit /b 0",
            ]
        ),
        encoding="utf-8",
    )
    pnp_snapshot = tmp_path / "pnp.json"
    pnp_snapshot.write_text("[]", encoding="utf-8")
    driver_snapshot = tmp_path / "driver.json"
    driver_snapshot.write_text("[]", encoding="utf-8")
    fake_netsh = tmp_path / "netsh.cmd"
    fake_netsh.write_text(
        "\n".join(
            [
                "@echo off",
                "echo Rule Name: Node.js JavaScript Runtime",
                "echo Enabled: Yes",
                "echo Direction: In",
                "echo Protocol: TCP",
                "echo Action: Allow",
                "exit /b 0",
            ]
        ),
        encoding="utf-8",
    )

    env = dict(**os.environ)
    env["PATH"] = f"{tmp_path};{env['PATH']}"
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-AvatarUrl",
            "http://127.0.0.1:9/",
            "-AdbPath",
            str(fake_adb),
            "-PnpSnapshotPath",
            str(pnp_snapshot),
            "-DriverSnapshotPath",
            str(driver_snapshot),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="cp932",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = _extract_json(result.stdout)
    assert payload["http"]["ok"] is False
    assert payload["firewall"]["checked"] is True
    assert payload["firewall"]["reason"] == "avatar_url_unreachable"
    assert "Node.js JavaScript Runtime" in "\n".join(payload["firewall"]["raw"])


def test_reports_samsung_mtp_without_adb_interface(tmp_path):
    fake_adb = tmp_path / "adb.cmd"
    fake_adb.write_text(
        "\n".join(
            [
                "@echo off",
                "if \"%1\"==\"devices\" (",
                "  echo List of devices attached",
                "  exit /b 0",
                ")",
                "exit /b 0",
            ]
        ),
        encoding="utf-8",
    )
    pnp_snapshot = tmp_path / "pnp.json"
    pnp_snapshot.write_text(
        json.dumps(
            [
                {
                    "Class": "WPD",
                    "FriendlyName": "Galaxy S9+",
                    "Status": "OK",
                    "InstanceId": "USB\\VID_04E8&PID_6860&MS_COMP_MTP&SAMSUNG_ANDROID",
                },
                {
                    "Class": "USB",
                    "FriendlyName": "SAMSUNG Mobile USB Composite Device",
                    "Status": "OK",
                    "InstanceId": "USB\\VID_04E8&PID_6860",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-AvatarUrl",
            "http://127.0.0.1:9/",
            "-AdbPath",
            str(fake_adb),
            "-PnpSnapshotPath",
            str(pnp_snapshot),
        ],
        cwd=ROOT,
        text=True,
        encoding="cp932",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = _extract_json(result.stdout)
    assert payload["usb"]["samsung_present"] is True
    assert payload["usb"]["adb_interface_present"] is False
    assert payload["usb"]["diagnosis"] == "samsung_mtp_present_adb_interface_missing"
    assert any("USB debugging" in action for action in payload["next_actions"])


def test_reports_missing_samsung_usb_driver_when_adb_interface_is_missing(tmp_path):
    fake_adb = tmp_path / "adb.cmd"
    fake_adb.write_text(
        "\n".join(
            [
                "@echo off",
                "if \"%1\"==\"devices\" (",
                "  echo List of devices attached",
                "  exit /b 0",
                ")",
                "exit /b 0",
            ]
        ),
        encoding="utf-8",
    )
    pnp_snapshot = tmp_path / "pnp.json"
    pnp_snapshot.write_text(
        json.dumps(
            [
                {
                    "Class": "WPD",
                    "FriendlyName": "Galaxy S9+",
                    "Status": "OK",
                    "InstanceId": "USB\\VID_04E8&PID_6860&MS_COMP_MTP&SAMSUNG_ANDROID",
                }
            ]
        ),
        encoding="utf-8",
    )
    driver_snapshot = tmp_path / "drivers.json"
    driver_snapshot.write_text("[]", encoding="utf-8")

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-AvatarUrl",
            "http://127.0.0.1:9/",
            "-AdbPath",
            str(fake_adb),
            "-PnpSnapshotPath",
            str(pnp_snapshot),
            "-DriverSnapshotPath",
            str(driver_snapshot),
        ],
        cwd=ROOT,
        text=True,
        encoding="cp932",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = _extract_json(result.stdout)
    assert payload["driver"]["samsung_usb_driver_installed"] is False
    assert payload["driver"]["official_url"] == "https://developer.samsung.com/android-usb-driver"
    assert payload["driver"]["recommendation"] == "install_official_samsung_usb_driver_after_usb_debugging"


def test_waits_for_adb_state_change(tmp_path):
    counter = tmp_path / "adb-count.txt"
    pnp_snapshot = tmp_path / "pnp.json"
    pnp_snapshot.write_text("[]", encoding="utf-8")
    driver_snapshot = tmp_path / "drivers.json"
    driver_snapshot.write_text("[]", encoding="utf-8")
    fake_adb = tmp_path / "adb.cmd"
    fake_adb.write_text(
        "\n".join(
            [
                "@echo off",
                "set COUNT=0",
                f"if exist \"{counter}\" set /p COUNT=<\"{counter}\"",
                "set /a COUNT=%COUNT%+1",
                f">\"{counter}\" echo %COUNT%",
                "if \"%1\"==\"devices\" (",
                "  echo List of devices attached",
                "  if %COUNT% GEQ 2 echo galaxy123 unauthorized",
                "  exit /b 0",
                ")",
                "exit /b 0",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-AvatarUrl",
            "http://127.0.0.1:9/",
            "-AdbPath",
            str(fake_adb),
            "-WaitForAdbSeconds",
            "8",
            "-PnpSnapshotPath",
            str(pnp_snapshot),
            "-DriverSnapshotPath",
            str(driver_snapshot),
        ],
        cwd=ROOT,
        text=True,
        encoding="cp932",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = _extract_json(result.stdout)
    assert payload["adb_wait"]["waited"] is True
    assert payload["adb_wait"]["final_state"] == "unauthorized"
    assert payload["adb"]["devices"][0]["state"] == "unauthorized"
    assert any("tap Allow" in action for action in payload["next_actions"])


def test_wait_reports_offline_adb_without_hanging(tmp_path):
    pnp_snapshot = tmp_path / "pnp.json"
    pnp_snapshot.write_text("[]", encoding="utf-8")
    driver_snapshot = tmp_path / "drivers.json"
    driver_snapshot.write_text("[]", encoding="utf-8")
    fake_adb = tmp_path / "adb.cmd"
    fake_adb.write_text(
        "\n".join(
            [
                "@echo off",
                "if \"%1\"==\"kill-server\" (",
                "  exit /b 0",
                ")",
                "if \"%1\"==\"start-server\" (",
                "  exit /b 0",
                ")",
                "if \"%1\"==\"devices\" (",
                "  echo List of devices attached",
                "  echo galaxy123 offline transport_id:1",
                "  exit /b 0",
                ")",
                "exit /b 0",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-AvatarUrl",
            "http://127.0.0.1:9/",
            "-AdbPath",
            str(fake_adb),
            "-WaitForAdbSeconds",
            "8",
            "-PnpSnapshotPath",
            str(pnp_snapshot),
            "-DriverSnapshotPath",
            str(driver_snapshot),
        ],
        cwd=ROOT,
        text=True,
        encoding="cp932",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = _extract_json(result.stdout)
    assert payload["adb_wait"]["reset_attempted"] is False
    assert payload["adb_wait"]["reset_recommended"] is True
    assert payload["adb_wait"]["final_state"] == "offline"
    assert payload["adb"]["devices"][0]["state"] == "offline"
    assert any("Reconnect USB" in action for action in payload["next_actions"])


def test_lock_task_uses_foreground_browser_task_id(tmp_path):
    log = tmp_path / "adb.log"
    fake_adb = tmp_path / "adb.cmd"
    fake_adb.write_text(
        "\n".join(
            [
                "@echo off",
                ">>\"%~dp0adb.log\" echo %*",
                "if \"%1\"==\"devices\" (",
                "  echo List of devices attached",
                "  echo galaxy123 device product:starlte model:SM_G965U",
                "  exit /b 0",
                ")",
                "if \"%1\"==\"shell\" if \"%2\"==\"dumpsys\" (",
                "  echo     * TaskRecord{abc123 #470 A=com.brave.browser U=0 StackId=17 sz=1}",
                "  echo     mResumedActivity: ActivityRecord{abc u0 com.brave.browser/org.chromium.chrome.browser.ChromeTabbedActivity t470}",
                "  exit /b 0",
                ")",
                "exit /b 0",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-AvatarUrl",
            "http://127.0.0.1:9/",
            "-AdbPath",
            str(fake_adb),
            "-OpenOnDevice",
            "-LockTask",
        ],
        cwd=ROOT,
        text=True,
        encoding="cp932",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = _extract_json(result.stdout)
    assert payload["lock_task"]["ok"] is True
    assert payload["lock_task"]["task_id"] == "470"
    adb_log = log.read_text(encoding="utf-8")
    assert "shell am task lock stop" in adb_log
    assert "shell am task lock 470" in adb_log


def test_adb_command_failure_is_reported_as_json(tmp_path):
    pnp_snapshot = tmp_path / "pnp.json"
    pnp_snapshot.write_text("[]", encoding="utf-8")
    driver_snapshot = tmp_path / "drivers.json"
    driver_snapshot.write_text("[]", encoding="utf-8")
    fake_adb = tmp_path / "adb.cmd"
    fake_adb.write_text(
        "\n".join(
            [
                "@echo off",
                "if \"%1\"==\"devices\" (",
                "  echo List of devices attached",
                "  echo galaxy123 device product:starlte model:SM_G965U",
                "  exit /b 0",
                ")",
                "if \"%1\"==\"shell\" if \"%2\"==\"am\" (",
                "  echo Error: Activity not started, unknown error code 101 1>&2",
                "  exit /b 1",
                ")",
                "exit /b 0",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-AvatarUrl",
            "http://127.0.0.1:9/",
            "-AdbPath",
            str(fake_adb),
            "-PnpSnapshotPath",
            str(pnp_snapshot),
            "-DriverSnapshotPath",
            str(driver_snapshot),
            "-OpenOnDevice",
        ],
        cwd=ROOT,
        text=True,
        encoding="cp932",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = _extract_json(result.stdout)
    assert payload["open_on_device"]["ok"] is False
    assert "Activity not started" in "\n".join(payload["open_on_device"]["output"])
