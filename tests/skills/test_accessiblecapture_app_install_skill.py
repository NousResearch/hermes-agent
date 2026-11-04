"""Regression tests for AccessibleCapture's local-to-TestFlight handoff."""

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


def _local_unavailable_result():
    return delivery_state.classify_delivery(
        apple_ready=True,
        tester_eligible=True,
        device_paired=True,
        device_tunnel_available=False,
        target_build="96",
        installed_build=None,
    )


def test_accessiblecapture_acceptance_case_is_testflight_ready():
    result = _local_unavailable_result()

    assert result["outcome"] == "TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE"
    assert result["outcome"] != "BLOCKED"
    assert result["outcome"] != "FAILED"


def test_no_installed_claim_without_exact_build_readback():
    result = _local_unavailable_result()

    assert result["installed_build"] is None
    assert result["installation_verified"] is False


def test_device_tunnel_is_only_required_for_device_control():
    result = _local_unavailable_result()

    assert result["apple_ready"] is True
    assert result["tester_eligible"] is True
    assert result["device_control_available"] is False
