"""Tests for the QuestFrame FH6VR Hermes plugin bridge."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from plugins.questframe_fh6vr import core


def test_status_lists_019_tools():
    payload = json.loads(core.handle_status())
    tools = payload["available_tools"]
    assert "questframe_color_depth_pairing_selftest" in tools
    assert "questframe_openxr_presentation_selftest" in tools
    assert "questframe_cockpit_presence_selftest" in tools
    assert "questframe_pcvr_management_selftest" in tools
    assert "questframe_hermes_bridge_selftest" in tools
    assert "questframe_hmd_controller_input_selftest" in tools


def test_color_depth_pairing_dispatches_launcher():
    fake = {"ok": True, "command": "fh6-color-depth-pairing-selftest", "exit_code": 0}
    with patch.object(core, "run_launcher", return_value=fake) as run:
        payload = json.loads(
            core.handle_color_depth_pairing_selftest(
                {
                    "approve": True,
                    "attempt_window_capture": True,
                    "require_foreground": True,
                }
            )
        )
    assert payload["ok"] is True
    run.assert_called_once()
    assert run.call_args.args[0] == "fh6-color-depth-pairing-selftest"
    extra = run.call_args.kwargs["extra_args"]
    assert "--approve" in extra
    assert "--attempt-window-capture" in extra
    assert "--require-foreground" in extra
    assert "--json" in extra


def test_openxr_presentation_dispatches_launcher():
    fake = {"ok": True, "command": "openxr-presentation-selftest", "exit_code": 0}
    with patch.object(core, "run_launcher", return_value=fake) as run:
        payload = json.loads(
            core.handle_openxr_presentation_selftest(
                {
                    "approve": True,
                    "attempt_window_capture": True,
                    "require_foreground": True,
                    "require_pairing": True,
                    "require_hmd": True,
                    "min_hmd_width": 1980,
                    "min_hmd_height": 1280,
                }
            )
        )
    assert payload["ok"] is True
    run.assert_called_once()
    assert run.call_args.args[0] == "openxr-presentation-selftest"
    extra = run.call_args.kwargs["extra_args"]
    assert "--approve" in extra
    assert "--attempt-window-capture" in extra
    assert "--require-foreground" in extra
    assert "--require-pairing" in extra
    assert "--require-hmd" in extra
    assert "--min-hmd-width" in extra
    assert "1980" in extra
    assert "--min-hmd-height" in extra
    assert "1280" in extra


def test_slash_color_depth_pairing_alias():
    with patch.object(core, "handle_color_depth_pairing_selftest", return_value='{"ok":true}') as handler:
        out = core.handle_slash(
            "color-depth-pairing-selftest --approve --attempt-window-capture --require-foreground"
        )
    assert '"ok":true' in out.replace(" ", "")
    handler.assert_called_once()
    args = handler.call_args.args[0]
    assert args["approve"] is True
    assert args["attempt_window_capture"] is True
    assert args["require_foreground"] is True


def test_slash_live_capture_foreground_alias():
    with patch.object(core, "handle_live_capture_selftest", return_value='{"ok":true}') as handler:
        out = core.handle_slash("live-capture-selftest --attempt-window-capture --require-foreground")
    assert '"ok":true' in out.replace(" ", "")
    handler.assert_called_once()
    args = handler.call_args.args[0]
    assert args["attempt_window_capture"] is True
    assert args["require_foreground"] is True


def test_cockpit_presence_dispatches_launcher():
    fake = {"ok": True, "command": "cockpit-presence-selftest", "exit_code": 0}
    with patch.object(core, "run_launcher", return_value=fake) as run:
        payload = json.loads(
            core.handle_cockpit_presence_selftest(
                {
                    "approve": True,
                    "attempt_window_capture": True,
                    "seconds": 10,
                    "target_hz": 72,
                }
            )
        )
    assert payload["ok"] is True
    run.assert_called_once()
    assert run.call_args.args[0] == "cockpit-presence-selftest"
    extra = run.call_args.kwargs["extra_args"]
    assert "--approve" in extra
    assert "--attempt-window-capture" in extra
    assert "--seconds" in extra
    assert "10" in extra
    assert "--target-hz" in extra
    assert "72" in extra


def test_slash_cockpit_presence_alias():
    with patch.object(core, "handle_cockpit_presence_selftest", return_value='{"ok":true}') as handler:
        out = core.handle_slash("cockpit-presence-selftest --approve --seconds 10")
    assert '"ok":true' in out.replace(" ", "")
    handler.assert_called_once()
    args = handler.call_args.args[0]
    assert args["approve"] is True
    assert args["seconds"] == 10


def test_pcvr_management_dispatches_launcher():
    fake = {"ok": True, "command": "pcvr-management-selftest", "exit_code": 0}
    with patch.object(core, "run_launcher", return_value=fake) as run:
        payload = json.loads(
            core.handle_pcvr_management_selftest(
                {
                    "allow_missing_runtime": True,
                    "no_process_list": True,
                    "timeout_seconds": 30,
                }
            )
        )
    assert payload["ok"] is True
    run.assert_called_once()
    assert run.call_args.args[0] == "pcvr-management-selftest"
    extra = run.call_args.kwargs["extra_args"]
    assert "--allow-missing-runtime" in extra
    assert "--no-process-list" in extra
    assert "--json" in extra


def test_hermes_bridge_dispatches_launcher():
    fake = {"ok": True, "command": "hermes-bridge-selftest", "exit_code": 0}
    with patch.object(core, "run_launcher", return_value=fake) as run:
        payload = json.loads(
            core.handle_hermes_bridge_selftest(
                {
                    "timeout_seconds": 30,
                }
            )
        )
    assert payload["ok"] is True
    run.assert_called_once()
    assert run.call_args.args[0] == "hermes-bridge-selftest"
    assert run.call_args.kwargs["extra_args"] == ["--json"]
    assert run.call_args.kwargs["timeout_seconds"] == 30


def test_hmd_controller_input_dispatches_launcher():
    fake = {"ok": True, "command": "hmd-controller-input-selftest", "exit_code": 0}
    with patch.object(core, "run_launcher", return_value=fake) as run:
        payload = json.loads(
            core.handle_hmd_controller_input_selftest(
                {
                    "allow_missing_runtime": True,
                    "require_virtual_gamepad": True,
                    "no_process_list": True,
                    "timeout_seconds": 30,
                }
            )
        )
    assert payload["ok"] is True
    run.assert_called_once()
    assert run.call_args.args[0] == "hmd-controller-input-selftest"
    extra = run.call_args.kwargs["extra_args"]
    assert "--allow-missing-runtime" in extra
    assert "--require-virtual-gamepad" in extra
    assert "--no-process-list" in extra
    assert "--json" in extra


def test_slash_pcvr_management_alias():
    with patch.object(core, "handle_pcvr_management_selftest", return_value='{"ok":true}') as handler:
        out = core.handle_slash("pcvr-management-selftest --allow-missing-runtime")
    assert '"ok":true' in out.replace(" ", "")
    handler.assert_called_once()
    args = handler.call_args.args[0]
    assert args["allow_missing_runtime"] is True


def test_slash_hmd_controller_input_alias():
    with patch.object(core, "handle_hmd_controller_input_selftest", return_value='{"ok":true}') as handler:
        out = core.handle_slash(
            "hmd-controller-input-selftest --allow-missing-runtime --require-virtual-gamepad"
        )
    assert '"ok":true' in out.replace(" ", "")
    handler.assert_called_once()
    args = handler.call_args.args[0]
    assert args["allow_missing_runtime"] is True
    assert args["require_virtual_gamepad"] is True
