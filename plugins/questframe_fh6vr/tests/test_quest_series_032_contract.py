"""Contract tests for Quest series passthrough (no live launcher)."""

from __future__ import annotations

import json
from unittest.mock import patch

from plugins.questframe_fh6vr import core

FIXTURE_PCVR = {
    "Name": "questframe-pcvr-management-selftest",
    "Status": "Pass",
    "Assertions": [{"Name": "quest2-vd-120hz-guardrail", "Status": "Pass"}],
    "State": {
        "QuestSeries": [{
            "Id": "quest-2",
            "Rtx3060SafeRefreshHz": [72, 90],
            "KnownUnstableRefreshHz": [120],
        }]
    },
}


def test_pcvr_management_selftest_passthrough_includes_quest_series_guardrail():
    with patch.object(core, "run_launcher", return_value=FIXTURE_PCVR):
        payload = json.loads(core.handle_pcvr_management_selftest({"no_process_list": True}))
    assert payload["Status"] == "Pass"
    quest2 = next(q for q in payload["State"]["QuestSeries"] if q["Id"] == "quest-2")
    assert quest2["Rtx3060SafeRefreshHz"] == [72, 90]


FIXTURE_HMD = {
    "Name": "hmd-controller-input-bridge-selftest",
    "Status": "Pass",
    "State": {
        "MappingProfiles": [{"Id": "quest-pro-touch-pro-wheel"}],
        "ReadyForInGameDriving": False,
        "VirtualGamepad": {
            "Platform": "Windows",
            "PreferredBackend": "ViGEmBus XInput virtual gamepad",
            "Available": False,
            "Installed": False,
            "RequiresInstall": True,
            "ReadyForFh6Driving": False,
            "Candidates": [
                {
                    "Id": "windows-vigem-xinput",
                    "DisplayName": "ViGEmBus XInput virtual gamepad",
                    "Platform": "Windows",
                    "Installed": False,
                    "Available": False,
                    "CanExposeXInput": True,
                    "RequiresSteamInputTranslation": False,
                    "RequiresUserPermission": False,
                    "ReadyForFh6Driving": False,
                    "CandidateReason": "ViGEmBus driver not installed.",
                    "DetectionEvidence": "HKLM service key ViGEmBus was not found.",
                    "NextStep": (
                        "Install signed ViGEmBus from https://github.com/ViGEm/ViGEmBus/releases "
                        "outside the FH6 install folder, reboot if prompted, then rerun with "
                        "--require-virtual-gamepad."
                    ),
                }
            ],
        },
    },
}


def test_hmd_controller_selftest_passthrough_keeps_driving_boundary():
    with patch.object(core, "run_launcher", return_value=FIXTURE_HMD):
        payload = json.loads(
            core.handle_hmd_controller_input_selftest({"allow_missing_runtime": True})
        )
    assert payload["State"]["ReadyForInGameDriving"] is False


def test_hmd_controller_selftest_passthrough_includes_vigem_backend_status_fields():
    with patch.object(core, "run_launcher", return_value=FIXTURE_HMD):
        payload = json.loads(
            core.handle_hmd_controller_input_selftest({"allow_missing_runtime": True})
        )

    virtual_gamepad = payload["State"]["VirtualGamepad"]
    assert virtual_gamepad["Platform"] == "Windows"
    assert virtual_gamepad["ReadyForFh6Driving"] is False

    vigem = next(
        c for c in virtual_gamepad["Candidates"] if c["Id"] == "windows-vigem-xinput"
    )
    assert vigem["Installed"] is False
    assert vigem["Available"] is False
    assert vigem["CanExposeXInput"] is True
    assert vigem["ReadyForFh6Driving"] is False
    assert vigem["CandidateReason"] == "ViGEmBus driver not installed."
