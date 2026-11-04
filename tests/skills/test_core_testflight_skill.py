"""Regression tests for the TestFlight/device-plane decision contract."""

import sys
from pathlib import Path

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "devops"
    / "core-testflight"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import delivery_state


def test_paired_but_tunnel_unavailable_keeps_testflight_ready():
    result = delivery_state.classify_delivery(
        apple_ready=True,
        tester_eligible=True,
        device_paired=True,
        device_tunnel_available=False,
        target_build="96",
        installed_build=None,
    )

    assert result["device_control_available"] is False
    assert result["outcome"] == "TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE"
    assert result["installation_verified"] is False


def test_unpaired_device_has_same_testflight_outcome():
    result = delivery_state.classify_delivery(
        apple_ready=True,
        tester_eligible=True,
        device_paired=False,
        device_tunnel_available=False,
        target_build="96",
        installed_build=None,
    )

    assert result["outcome"] == "TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE"


def test_exact_installed_build_is_done_even_without_new_control():
    result = delivery_state.classify_delivery(
        apple_ready=True,
        tester_eligible=True,
        device_paired=False,
        device_tunnel_available=False,
        target_build="96",
        installed_build="96",
    )

    assert result["outcome"] == "DONE"
    assert result["installation_verified"] is True


def test_apple_failure_is_not_device_control_failure():
    result = delivery_state.classify_delivery(
        apple_ready=False,
        tester_eligible=True,
        device_paired=True,
        device_tunnel_available=False,
        target_build="96",
        installed_build=None,
    )

    assert result["outcome"] == "TESTFLIGHT_NOT_READY"
