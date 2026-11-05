"""Regression tests for the TestFlight/device-plane decision contract."""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "devops"
    / "core-testflight"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import app_store_connect
import delivery_state


def _api_fixture_get(path):
    if path.startswith("/apps?"):
        return {
            "data": [
                {
                    "type": "apps",
                    "id": "app-1",
                    "attributes": {"bundleId": "com.accessiblecore.capture"},
                }
            ]
        }
    if path.startswith("/builds?"):
        return {"data": [{"type": "builds", "id": "build-96", "attributes": {"version": "96"}}]}
    if path.startswith("/builds/build-96?"):
        return {
            "data": {
                "type": "builds",
                "id": "build-96",
                "attributes": {
                    "version": "96",
                    "processingState": "VALID",
                    "expired": False,
                    "usesNonExemptEncryption": False,
                },
                "relationships": {
                    "preReleaseVersion": {
                        "data": {"type": "preReleaseVersions", "id": "pre-1"}
                    }
                },
            }
        }
    if path == "/preReleaseVersions/pre-1":
        return {"data": {"type": "preReleaseVersions", "id": "pre-1", "attributes": {"version": "1.0"}}}
    if path.startswith("/buildBetaDetails?"):
        return {"data": [{"type": "buildBetaDetails", "id": "beta-1", "attributes": {"internalBuildState": "IN_BETA_TESTING"}}]}
    if path.startswith("/betaGroups?"):
        return {"data": [{"type": "betaGroups", "id": "group-1", "attributes": {"name": "Accessible Core"}}]}
    if path == "/betaGroups/group-1/builds?limit=200":
        return {"data": [{"type": "builds", "id": "build-96"}]}
    if path == "/betaGroups/group-1/betaTesters?limit=200":
        return {"data": [{"type": "betaTesters", "id": "tester-1", "attributes": {"state": "ENABLED"}}]}
    raise AssertionError(f"unexpected API path: {path}")


def test_expired_browser_session_does_not_block_supported_api_verification(monkeypatch):
    class FakeApi:
        get = staticmethod(_api_fixture_get)

    monkeypatch.setattr(app_store_connect, "load_api_client", lambda: FakeApi())
    result = app_store_connect.verify_existing_build(
        bundle_id="com.accessiblecore.capture",
        build_number="96",
        beta_group="Accessible Core",
    )

    assert result["verification"] == "API"
    assert result["processing_state"] == "VALID"
    assert result["group_member"] is True
    assert result["eligible_tester_count"] == 1


def test_api_verification_reports_credential_failure_without_browser_fallback(monkeypatch):
    def unavailable_api():
        raise app_store_connect.AppStoreConnectError(
            "Core Secrets could not resolve apple/app-store-connect"
        )

    monkeypatch.setattr(app_store_connect, "load_api_client", unavailable_api)
    with pytest.raises(app_store_connect.AppStoreConnectError, match="Core Secrets"):
        app_store_connect.verify_existing_build(
            bundle_id="com.accessiblecore.capture",
            build_number="96",
            beta_group="Accessible Core",
        )


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
