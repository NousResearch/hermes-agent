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


def test_color_depth_pairing_dispatches_launcher():
    fake = {"ok": True, "command": "fh6-color-depth-pairing-selftest", "exit_code": 0}
    with patch.object(core, "run_launcher", return_value=fake) as run:
        payload = json.loads(core.handle_color_depth_pairing_selftest({"approve": True}))
    assert payload["ok"] is True
    run.assert_called_once()
    assert run.call_args.args[0] == "fh6-color-depth-pairing-selftest"
    assert "--approve" in run.call_args.kwargs["extra_args"]
    assert "--json" in run.call_args.kwargs["extra_args"]


def test_openxr_presentation_dispatches_launcher():
    fake = {"ok": True, "command": "openxr-presentation-selftest", "exit_code": 0}
    with patch.object(core, "run_launcher", return_value=fake) as run:
        payload = json.loads(
            core.handle_openxr_presentation_selftest(
                {"approve": True, "require_pairing": True}
            )
        )
    assert payload["ok"] is True
    run.assert_called_once()
    assert run.call_args.args[0] == "openxr-presentation-selftest"
    extra = run.call_args.kwargs["extra_args"]
    assert "--approve" in extra
    assert "--require-pairing" in extra


def test_slash_color_depth_pairing_alias():
    with patch.object(core, "handle_color_depth_pairing_selftest", return_value='{"ok":true}') as handler:
        out = core.handle_slash("color-depth-pairing-selftest --approve")
    assert '"ok":true' in out.replace(" ", "")
    handler.assert_called_once()
    assert handler.call_args.args[0]["approve"] is True


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
